#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

#[cfg(feature = "dx12")]
use gfx_backend_dx12 as back;

#[cfg(feature = "metal")]
use gfx_backend_metal as back;

#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as back;

use arrayvec::ArrayVec;
use core::mem::{size_of, ManuallyDrop};

use crate::primitives::Triangle;
use crate::{FRAGMENT_SOURCE, VERTEX_SOURCE, WINDOW_NAME};
use winit::Window;

use gfx_hal::{
    adapter::{Adapter, MemoryTypeId, PhysicalDevice},
    buffer::Usage as BufferUsage,
    command::{ClearColor, ClearValue, CommandBuffer, MultiShot, Primary},
    device::Device,
    format::{Aspects, ChannelType, Format, Swizzle},
    image::{Extent, Layout, SubresourceRange, Usage, ViewKind},
    memory::{Properties, Requirements},
    pass::{Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, Subpass, SubpassDesc},
    pool::{CommandPool, CommandPoolCreateFlags},
    pso::{
        AttributeDesc, BakedStates, BasePipeline, BlendDesc, BlendOp, BlendState, ColorBlendDesc,
        ColorMask, DepthStencilDesc, DepthTest, DescriptorSetLayoutBinding, ElemOffset, ElemStride,
        Element, EntryPoint, Face, Factor, FrontFace, GraphicsPipelineDesc, GraphicsShaderSet,
        InputAssemblerDesc, LogicOp, PipelineCreationFlags, PipelineStage, PolygonMode, Rasterizer,
        Rect, ShaderStageFlags, Specialization, StencilTest, VertexBufferDesc, Viewport,
    },
    queue::{family::QueueGroup, QueueType, Submission},
    window::{Backbuffer, Extent2D, FrameSync, PresentMode, Swapchain, SwapchainConfig},
    Backend, Gpu, Graphics, Instance, Primitive, QueueFamily, Surface,
};

pub struct HalState {
    buffer: ManuallyDrop<<back::Backend as Backend>::Buffer>,
    current_frame: usize,
    frames_in_flight: usize,
    in_flight_fences: Vec<<back::Backend as Backend>::Fence>,
    render_finished_semaphores: Vec<<back::Backend as Backend>::Semaphore>,
    image_available_semaphores: Vec<<back::Backend as Backend>::Semaphore>,
    command_buffers: Vec<CommandBuffer<back::Backend, Graphics, MultiShot, Primary>>,
    command_pool: ManuallyDrop<CommandPool<back::Backend, Graphics>>,
    framebuffers: Vec<<back::Backend as Backend>::Framebuffer>,
    image_views: Vec<(<back::Backend as Backend>::ImageView)>,
    render_area: Rect,
    render_pass: ManuallyDrop<<back::Backend as Backend>::RenderPass>,
    swapchain: ManuallyDrop<<back::Backend as Backend>::Swapchain>,
    device: ManuallyDrop<back::Device>,
    queue_group: QueueGroup<back::Backend, Graphics>,
    memory: ManuallyDrop<<back::Backend as Backend>::Memory>,
    requirements: Requirements,
    descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout>,
    graphics_pipeline: ManuallyDrop<<back::Backend as Backend>::GraphicsPipeline>,
    pipeline_layout: ManuallyDrop<<back::Backend as Backend>::PipelineLayout>,
    _adapter: Adapter<back::Backend>,
    _surface: <back::Backend as Backend>::Surface,
    _instance: ManuallyDrop<back::Instance>,
}

impl HalState {
    pub fn new(window: &Window) -> Result<Self, &'static str> {
        let instance = back::Instance::create(WINDOW_NAME, 1);
        let mut surface: <back::Backend as Backend>::Surface = instance.create_surface(window);

        let adapter: Adapter<back::Backend> = instance
            .enumerate_adapters()
            .into_iter()
            .find(|a| {
                a.queue_families
                    .iter()
                    .any(|qf| qf.supports_graphics() && surface.supports_queue_family(qf))
            })
            .ok_or("Couldn't find a graphical adapter!")?;
        info!("Using adapter: {:?}", &adapter.info);
        info!(
            "Supported queue family types: {:?}",
            &adapter
                .queue_families
                .iter()
                .map(|qf| qf.queue_type())
                .collect::<Vec<QueueType>>()
        );

        let (mut device, queue_group): (back::Device, QueueGroup<back::Backend, Graphics>) = {
            let queue_family = adapter
                .queue_families
                .iter()
                .find(|qf| qf.supports_graphics() && surface.supports_queue_family(qf))
                .ok_or("Couldn't find a QueueFamily with graphics")?;
            let Gpu { device, mut queues } = unsafe {
                adapter
                    .physical_device
                    .open(&[(&queue_family, &[1.0; 1])])
                    .map_err(|_| "Couldn't open the physical device")?
            };

            let queue_group = queues
                .take::<Graphics>(queue_family.id())
                .ok_or("Couldn't take ownership of the QueueGroup")?;

            if !queue_group.queues.is_empty() {
                Ok(())
            } else {
                Err("The QueueGroup did not have any CommandQueues available")
            }?;
            (device, queue_group)
        };

        // we want to setup the stuff we render to here
        let (swapchain, extent, backbuffer, format, frames_in_flight) = {
            // we get info from the surface about what it supports
            let (caps, preferred_formats, present_modes, composite_alphas) =
                surface.compatibility(&adapter.physical_device);
            info!("{:?}", caps);
            info!("Preferred Formats: {:?}", preferred_formats);
            info!("Present modes: {:?}", present_modes);
            info!("Composite Alphas: {:?}", composite_alphas);

            // and we try to choose what we think is best from what is available
            // this is the way frames are buffered, most recent from swapchain, double buffer, or monkey fiesta
            let present_mode = {
                use gfx_hal::window::PresentMode::*;
                [Mailbox, Fifo, Relaxed, Immediate]
                    .iter()
                    .cloned()
                    .find(|pm| present_modes.contains(pm))
                    .ok_or("No present mode values present")?
            };

            // how does the surface get composited with other windows in the OS?
            let composite_alpha = {
                use gfx_hal::window::CompositeAlpha::*;
                [Opaque, Inherit, PreMultiplied, PostMultiplied]
                    .iter()
                    .cloned()
                    .find(|ca| composite_alphas.contains(ca))
                    .ok_or("No composite alpha we can select")?
            };

            // what color format do we want?
            // we prefer sRGB if possible
            let format = match preferred_formats {
                None => Format::Rgba8Srgb,
                Some(formats) => match formats
                    .iter()
                    .find(|fmt| fmt.base_format().1 == ChannelType::Srgb)
                    .cloned()
                {
                    Some(srgb_fmt) => srgb_fmt,
                    None => formats
                        .get(0)
                        .cloned()
                        .ok_or("Preferred formats list is empty")?,
                },
            };

            // get window/screen size
            let extent = {
                let window_client_area = window
                    .get_inner_size()
                    .ok_or("Window doesn't exist")?
                    .to_physical(window.get_hidpi_factor());

                Extent2D {
                    width: caps.extents.end.width.min(window_client_area.width as u32),
                    height: caps
                        .extents
                        .end
                        .height
                        .min(window_client_area.height as u32),
                }
            };

            // how many frames are in our framebuffer?
            // if we can use mailbox (i.e. most recent from swapchain) then use 3 for triple buffer rendering
            // otherwise use double buffering
            let image_count = if present_mode == PresentMode::Mailbox {
                (caps.image_count.end - 1).min(caps.image_count.start.max(3))
            } else {
                (caps.image_count.end - 1).min(caps.image_count.start.max(2))
            };

            // only 1 image layer for now
            let image_layers = 1;

            // we will only be using color for now, no depth buffer or fancy stuff yet
            let image_usage = if caps.usage.contains(Usage::COLOR_ATTACHMENT) {
                Usage::COLOR_ATTACHMENT
            } else {
                Err("The surface isn't capable of surporting color")?
            };

            // build the swapchain config from the stuff we chose from
            let swapchain_config = SwapchainConfig {
                present_mode,
                composite_alpha,
                format,
                extent,
                image_count,
                image_layers,
                image_usage,
            };
            info!("{:?}", swapchain_config);

            // get the swapchain and backbuffer
            let (swapchain, backbuffer) = unsafe {
                device
                    .create_swapchain(&mut surface, swapchain_config, None)
                    .map_err(|_| "Failed to create swapchain and backbuffer")?
            };

            (swapchain, extent, backbuffer, format, image_count as usize)
        };

        let render_area = extent.to_extent().rect();

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) = {
            let mut image_available_semaphores: Vec<<back::Backend as Backend>::Semaphore> = vec![];
            let mut render_finished_semaphores: Vec<<back::Backend as Backend>::Semaphore> = vec![];
            let mut in_flight_fences: Vec<<back::Backend as Backend>::Fence> = vec![];
            for _ in 0..frames_in_flight {
                in_flight_fences.push(
                    device
                        .create_fence(true)
                        .map_err(|_| "Could not create a fence")?,
                );
                render_finished_semaphores.push(
                    device
                        .create_semaphore()
                        .map_err(|_| "Could not create a semaphore")?,
                );
                image_available_semaphores.push(
                    device
                        .create_semaphore()
                        .map_err(|_| "Could not create a semaphore")?,
                );
            }
            (
                image_available_semaphores,
                render_finished_semaphores,
                in_flight_fences,
            )
        };

        let render_pass = {
            let color_attachment = Attachment {
                format: Some(format),
                samples: 1,
                ops: AttachmentOps {
                    load: AttachmentLoadOp::Clear,
                    store: AttachmentStoreOp::Store,
                },
                stencil_ops: AttachmentOps::DONT_CARE,
                layouts: Layout::Undefined..Layout::Present,
            };
            let subpass = SubpassDesc {
                // 0 is an ID
                colors: &[(0, Layout::ColorAttachmentOptimal)],
                depth_stencil: None,
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };
            unsafe {
                device
                    .create_render_pass(&[color_attachment], &[subpass], &[])
                    .map_err(|_| "Couldn't create render pass")?
            }
        };

        let image_views: Vec<_> = match backbuffer {
            Backbuffer::Images(images) => images
                .into_iter()
                .map(|image| unsafe {
                    device
                        .create_image_view(
                            &image,
                            ViewKind::D2,
                            format,
                            Swizzle::NO,
                            SubresourceRange {
                                aspects: Aspects::COLOR,
                                levels: 0..1,
                                layers: 0..1,
                            },
                        )
                        .map_err(|_| "Couldn't create the image view for this image")
                })
                .collect::<Result<Vec<_>, &str>>()?,
            Backbuffer::Framebuffer(_) => {
                unimplemented!("Can't handle a framebuffer backbuffer yet")
            }
        };

        let framebuffers: Vec<<back::Backend as Backend>::Framebuffer> = {
            image_views
                .iter()
                .map(|image_view| unsafe {
                    device
                        .create_framebuffer(
                            &render_pass,
                            vec![image_view],
                            Extent {
                                width: extent.width as u32,
                                height: extent.height as u32,
                                depth: 1,
                            },
                        )
                        .map_err(|_| "Failed to create a framebuffer")
                })
                .collect::<Result<Vec<_>, &str>>()?
        };

        let mut command_pool = unsafe {
            device
                .create_command_pool_typed(&queue_group, CommandPoolCreateFlags::RESET_INDIVIDUAL)
                .map_err(|_| "Could not create command pool")?
        };

        let command_buffers: Vec<_> = framebuffers
            .iter()
            .map(|_| command_pool.acquire_command_buffer())
            .collect();

        let (descriptor_set_layouts, pipeline_layout, graphics_pipeline) =
            Self::create_pipeline(&mut device, extent, &render_pass)?;

        let (buffer, memory, requirements) = unsafe {
            const F32_XY_TRIANGLE: u64 = (size_of::<f32>() * 2 * 3) as u64;
            let mut buffer = device
                .create_buffer(F32_XY_TRIANGLE, BufferUsage::VERTEX)
                .map_err(|_| "Failed to create vertex buffer")?;
            let requirements = device.get_buffer_requirements(&buffer);
            let memory_type_id = adapter
                .physical_device
                .memory_properties()
                .memory_types
                .iter()
                .enumerate()
                .find(|&(id, memory_type)| {
                    requirements.type_mask & (1 << id) != 0
                        && memory_type.properties.contains(Properties::CPU_VISIBLE)
                })
                .map(|(id, _)| MemoryTypeId(id))
                .ok_or("Couldn't find a memory type to support the vertex buffer")?;
            let memory = device
                .allocate_memory(memory_type_id, requirements.size)
                .map_err(|_| "Couldn't allocate vertex buffer memory")?;
            device
                .bind_buffer_memory(&memory, 0, &mut buffer)
                .map_err(|_| "Couldn't bind the buffer memory")?;
            (buffer, memory, requirements)
        };

        Ok(Self {
            _instance: ManuallyDrop::new(instance),
            _surface: surface,
            _adapter: adapter,
            device: ManuallyDrop::new(device),
            queue_group,
            swapchain: ManuallyDrop::new(swapchain),
            render_area,
            render_pass: ManuallyDrop::new(render_pass),
            image_views,
            framebuffers,
            command_pool: ManuallyDrop::new(command_pool),
            command_buffers,
            frames_in_flight,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            current_frame: 0,
            descriptor_set_layouts,
            pipeline_layout: ManuallyDrop::new(pipeline_layout),
            graphics_pipeline: ManuallyDrop::new(graphics_pipeline),
            memory: ManuallyDrop::new(memory),
            buffer: ManuallyDrop::new(buffer),
            requirements,
        })
    }

    #[allow(clippy::type_complexity)]
    fn create_pipeline(
        device: &mut back::Device,
        extent: Extent2D,
        render_pass: &<back::Backend as Backend>::RenderPass,
    ) -> Result<
        (
            Vec<<back::Backend as Backend>::DescriptorSetLayout>,
            <back::Backend as Backend>::PipelineLayout,
            <back::Backend as Backend>::GraphicsPipeline,
        ),
        &'static str,
    > {
        let mut compiler = shaderc::Compiler::new().ok_or("shaderc not found")?;
        let vertex_compile_artifact = compiler
            .compile_into_spirv(
                VERTEX_SOURCE,
                shaderc::ShaderKind::Vertex,
                "vertex.vert",
                "main",
                None,
            )
            .map_err(|e| {
                error!("{}", e);
                "Couldn't compile vertex shader"
            })?;
        let fragment_compile_artifact = compiler
            .compile_into_spirv(
                FRAGMENT_SOURCE,
                shaderc::ShaderKind::Fragment,
                "fragment.frag",
                "main",
                None,
            )
            .map_err(|e| {
                error!("{}", e);
                "Couldn't compile fragment shader"
            })?;

        let vertex_shader_module = unsafe {
            device
                .create_shader_module(vertex_compile_artifact.as_binary_u8())
                .map_err(|_| "Couldn't create vertex module from SPIRV")
        }?;
        let fragment_shader_module = unsafe {
            device
                .create_shader_module(fragment_compile_artifact.as_binary_u8())
                .map_err(|_| "Couldn't create fragment module from SPIRV")
        }?;
        let (descriptor_set_layouts, pipeline_layout, gfx_pipeline) = {
            let (vs_entry, fs_entry) = (
                EntryPoint {
                    entry: "main",
                    module: &vertex_shader_module,
                    specialization: Specialization {
                        constants: &[],
                        data: &[],
                    },
                },
                EntryPoint {
                    entry: "main",
                    module: &fragment_shader_module,
                    specialization: Specialization {
                        constants: &[],
                        data: &[],
                    },
                },
            );

            let shaders = GraphicsShaderSet {
                vertex: vs_entry,
                hull: None,
                domain: None,
                geometry: None,
                fragment: Some(fs_entry),
            };

            let input_assembler = InputAssemblerDesc::new(Primitive::TriangleList);

            let vertex_buffers: Vec<VertexBufferDesc> = vec![VertexBufferDesc {
                binding: 0,
                stride: (size_of::<f32>() * 5) as ElemStride,
                rate: 0,
            }];
            let position_attribute = AttributeDesc {
                location: 0,
                binding: 0,
                element: Element {
                    format: Format::Rg32Float,
                    offset: 0,
                },
            };
            let color_attribute = AttributeDesc {
                location: 1,
                binding: 0,
                element: Element {
                    format: Format::Rgb32Float,
                    offset: (size_of::<f32>() * 2) as ElemOffset,
                },
            };

            let attributes: Vec<AttributeDesc> = vec![position_attribute, color_attribute];

            let rasterizer = Rasterizer {
                depth_clamping: false,
                polygon_mode: PolygonMode::Fill,
                cull_face: Face::NONE,
                front_face: FrontFace::Clockwise,
                depth_bias: None,
                conservative: false,
            };

            let depth_stencil = DepthStencilDesc {
                depth: DepthTest::Off,
                depth_bounds: false,
                stencil: StencilTest::Off,
            };

            let blender = {
                let blend_state = BlendState::On {
                    color: BlendOp::Add {
                        src: Factor::One,
                        dst: Factor::Zero,
                    },
                    alpha: BlendOp::Add {
                        src: Factor::One,
                        dst: Factor::Zero,
                    },
                };
                BlendDesc {
                    logic_op: Some(LogicOp::Copy),
                    targets: vec![ColorBlendDesc(ColorMask::ALL, blend_state)],
                }
            };

            let baked_states = BakedStates {
                viewport: Some(Viewport {
                    rect: extent.to_extent().rect(),
                    depth: (0.0..1.0),
                }),
                scissor: Some(extent.to_extent().rect()),
                blend_color: None,
                depth_bounds: None,
            };

            let bindings = Vec::<DescriptorSetLayoutBinding>::new();
            let immutable_samplers = Vec::<<back::Backend as Backend>::Sampler>::new();
            let descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout> =
                vec![unsafe {
                    device
                        .create_descriptor_set_layout(bindings, immutable_samplers)
                        .map_err(|_| "Couldn't make a DescriptorSetLayout")?
                }];
            let push_constants = Vec::<(ShaderStageFlags, core::ops::Range<u32>)>::new();
            let layout = unsafe {
                device
                    .create_pipeline_layout(&descriptor_set_layouts, push_constants)
                    .map_err(|_| "Couldn't create a pipeline layout")?
            };

            let gfx_pipeline = {
                let desc = GraphicsPipelineDesc {
                    shaders,
                    rasterizer,
                    vertex_buffers,
                    attributes,
                    input_assembler,
                    blender,
                    depth_stencil,
                    multisampling: None,
                    baked_states,
                    layout: &layout,
                    subpass: Subpass {
                        index: 0,
                        main_pass: render_pass,
                    },
                    flags: PipelineCreationFlags::empty(),
                    parent: BasePipeline::None,
                };
                unsafe {
                    device
                        .create_graphics_pipeline(&desc, None)
                        .map_err(|_| "Couldn't create a graphics pipline")?
                }
            };
            (descriptor_set_layouts, layout, gfx_pipeline)
        };

        // destroy shader modules
        unsafe {
            device.destroy_shader_module(vertex_shader_module);
            device.destroy_shader_module(fragment_shader_module);
        }

        Ok((descriptor_set_layouts, pipeline_layout, gfx_pipeline))
    }

    pub fn draw_clear_frame(&mut self, color: [f32; 4]) -> Result<(), &'static str> {
        let image_available = &self.image_available_semaphores[self.current_frame];
        let render_finished = &self.render_finished_semaphores[self.current_frame];

        self.current_frame = (self.current_frame + 1) % self.frames_in_flight;

        let (i_u32, i_usize) = unsafe {
            let image_index = self
                .swapchain
                .acquire_image(core::u64::MAX, FrameSync::Semaphore(image_available))
                .map_err(|_| "Couldn't acquire an image from the swapchain")?;
            (image_index, image_index as usize)
        };

        let flight_fence = &self.in_flight_fences[i_usize];
        unsafe {
            self.device
                .wait_for_fence(flight_fence, core::u64::MAX)
                .map_err(|_| "Failed to wait on the fence")?;
            self.device
                .reset_fence(flight_fence)
                .map_err(|_| "Couldn't reset the fence")?;
        }

        // Record Commands to put into command buffer
        unsafe {
            let buffer = &mut self.command_buffers[i_usize];
            let clear_values = [ClearValue::Color(ClearColor::Float(color))];
            buffer.begin(false);
            buffer.begin_render_pass_inline(
                &self.render_pass,
                &self.framebuffers[i_usize],
                self.render_area,
                clear_values.iter(),
            );
            buffer.finish();
        }

        // submit commands and present frame
        let command_buffers = &self.command_buffers[i_usize..=i_usize];
        let wait_semaphores: ArrayVec<[_; 1]> =
            [(image_available, PipelineStage::COLOR_ATTACHMENT_OUTPUT)].into();
        let signal_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
        let present_wait_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
        let submission = Submission {
            command_buffers,
            wait_semaphores,
            signal_semaphores,
        };

        let the_command_queue = &mut self.queue_group.queues[0];
        unsafe {
            the_command_queue.submit(submission, Some(flight_fence));
            self.swapchain
                .present(the_command_queue, i_u32, present_wait_semaphores)
                .map_err(|_| "Failed to present into the swapchain!")
        }
    }

    pub fn draw_triangle_frame(&mut self, triangle: Triangle) -> Result<(), &'static str> {
        let image_available = &self.image_available_semaphores[self.current_frame];
        let render_finished = &self.render_finished_semaphores[self.current_frame];

        self.current_frame = (self.current_frame + 1) % self.frames_in_flight;

        let (i_u32, i_usize) = unsafe {
            let image_index = self
                .swapchain
                .acquire_image(core::u64::MAX, FrameSync::Semaphore(image_available))
                .map_err(|_| "Couldn't acquire an image from the swapchain")?;
            (image_index, image_index as usize)
        };

        let flight_fence = &self.in_flight_fences[i_usize];
        unsafe {
            self.device
                .wait_for_fence(flight_fence, core::u64::MAX)
                .map_err(|_| "Failed to wait on the fence")?;
            self.device
                .reset_fence(flight_fence)
                .map_err(|_| "Couldn't reset the fence")?;
        }

        unsafe {
            let mut data_target = self
                .device
                .acquire_mapping_writer(&self.memory, 0..self.requirements.size)
                .map_err(|_| "Failed to acquire mapping writer")?;
            let points = triangle.vertex_attributes();
            data_target[..points.len()].copy_from_slice(&points);
            self.device
                .release_mapping_writer(data_target)
                .map_err(|_| "Failed to release mapping writer")?;
        }

        // Record Commands to put into command buffer
        unsafe {
            let buffer = &mut self.command_buffers[i_usize];
            const TRIANGLE_CLEAR: [ClearValue; 1] =
                [ClearValue::Color(ClearColor::Float([0.1, 0.1, 0.1, 0.8]))];
            buffer.begin(false);
            {
                let mut encoder = buffer.begin_render_pass_inline(
                    &self.render_pass,
                    &self.framebuffers[i_usize],
                    self.render_area,
                    TRIANGLE_CLEAR.iter(),
                );
                encoder.bind_graphics_pipeline(&self.graphics_pipeline);
                let buffer_ref: &<back::Backend as Backend>::Buffer = &self.buffer;
                let buffers: ArrayVec<[_; 1]> = [(buffer_ref, 0)].into();
                encoder.bind_vertex_buffers(0, buffers);
                encoder.draw(0..3, 0..1);
            }
            buffer.finish();
        }

        // submit commands and present frame
        let command_buffers = &self.command_buffers[i_usize..=i_usize];
        let wait_semaphores: ArrayVec<[_; 1]> =
            [(image_available, PipelineStage::COLOR_ATTACHMENT_OUTPUT)].into();
        let signal_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
        let present_wait_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
        let submission = Submission {
            command_buffers,
            wait_semaphores,
            signal_semaphores,
        };

        let the_command_queue = &mut self.queue_group.queues[0];
        unsafe {
            the_command_queue.submit(submission, Some(flight_fence));
            self.swapchain
                .present(the_command_queue, i_u32, present_wait_semaphores)
                .map_err(|_| "Failed to present into the swapchain!")
        }
    }
}

impl core::ops::Drop for HalState {
    fn drop(&mut self) {
        let _ = self.device.wait_idle();
        unsafe {
            for fence in self.in_flight_fences.drain(..) {
                self.device.destroy_fence(fence)
            }
            for semaphore in self.render_finished_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore)
            }
            for semaphore in self.image_available_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore)
            }
            for framebuffer in self.framebuffers.drain(..) {
                self.device.destroy_framebuffer(framebuffer)
            }
            for image_view in self.image_views.drain(..) {
                self.device.destroy_image_view(image_view)
            }
            for descriptor_set_layout in self.descriptor_set_layouts.drain(..) {
                self.device
                    .destroy_descriptor_set_layout(descriptor_set_layout)
            }

            // pretty memory unsafe
            use core::ptr::read;
            self.device
                .destroy_graphics_pipeline(ManuallyDrop::into_inner(read(&self.graphics_pipeline)));

            self.device
                .free_memory(ManuallyDrop::into_inner(read(&self.memory)));
            self.device
                .destroy_buffer(ManuallyDrop::into_inner(read(&self.buffer)));
            self.device
                .destroy_pipeline_layout(ManuallyDrop::into_inner(read(&self.pipeline_layout)));

            self.device.destroy_command_pool(
                ManuallyDrop::into_inner(read(&self.command_pool)).into_raw(),
            );

            self.device
                .destroy_render_pass(ManuallyDrop::into_inner(read(&self.render_pass)));

            self.device
                .destroy_swapchain(ManuallyDrop::into_inner(read(&self.swapchain)));

            ManuallyDrop::drop(&mut self.device);
            ManuallyDrop::drop(&mut self._instance);
        }
    }
}
