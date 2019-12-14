use arrayvec::ArrayVec;

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

#[cfg(feature = "dx12")]
use gfx_backend_dx12 as back;

#[cfg(feature = "metal")]
use gfx_backend_metal as back;

#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as back;

use core::{
    marker::PhantomData,
    mem::{size_of, size_of_val, ManuallyDrop},
    ops::Deref,
};

use crate::{
    primitives::{Vertex, CUBE_INDEXES, CUBE_VERTEXES},
    CREATURE_BYTES, FRAGMENT_SOURCE, VERTEX_SOURCE, WINDOW_NAME,
};
use winit::Window;

use gfx_hal::{
    adapter::{Adapter, MemoryTypeId, PhysicalDevice},
    buffer::{IndexBufferView, Usage as BufferUsage},
    command::{ClearColor, ClearValue, CommandBuffer, MultiShot, Primary},
    device::Device,
    format::{Aspects, ChannelType, Format, Swizzle},
    image::{Extent, Layout, SubresourceRange, Usage, ViewKind},
    memory::{Pod, Properties, Requirements},
    pass::{Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, Subpass, SubpassDesc},
    pool::{CommandPool, CommandPoolCreateFlags},
    pso::{
        AttributeDesc, BakedStates, BasePipeline, BlendDesc, BlendOp, BlendState, ColorBlendDesc,
        ColorMask, DepthStencilDesc, DepthTest, DescriptorSetLayoutBinding, ElemOffset, ElemStride,
        Element, EntryPoint, Face, Factor, FrontFace, GraphicsPipelineDesc, GraphicsShaderSet,
        InputAssemblerDesc, LogicOp, PipelineCreationFlags, PipelineStage, PolygonMode, Rasterizer,
        Rect, ShaderStageFlags, Specialization, StencilTest, VertexBufferDesc, Viewport,
    },
    queue::{
        capability::{Capability, Supports, Transfer},
        family::QueueGroup,
        CommandQueue, QueueType, Submission,
    },
    window::{Backbuffer, Extent2D, FrameSync, PresentMode, Swapchain, SwapchainConfig},
    Backend, DescriptorPool, Gpu, Graphics, IndexType, Instance, Primitive, QueueFamily, Surface,
};

use nalgebra_glm as glm;
use std::time::Instant;

/// DO NOT USE THE VERSION OF THIS FUNCTION THAT'S IN THE GFX-HAL CRATE.
///
/// It can trigger UB if you upcast from a low alignment to a higher alignment
/// type. You'll be sad.
pub fn cast_slice<T: Pod, U: Pod>(ts: &[T]) -> Option<&[U]> {
    use core::mem::align_of;
    // Handle ZST (this all const folds)
    if size_of::<T>() == 0 || size_of::<U>() == 0 {
        if size_of::<T>() == size_of::<U>() {
            unsafe {
                return Some(core::slice::from_raw_parts(
                    ts.as_ptr() as *const U,
                    ts.len(),
                ));
            }
        } else {
            return None;
        }
    }
    // Handle alignments (this const folds)
    if align_of::<U>() > align_of::<T>() {
        // possible mis-alignment at the new type (this is a real runtime check)
        if (ts.as_ptr() as usize) % align_of::<U>() != 0 {
            return None;
        }
    }
    if size_of::<T>() == size_of::<U>() {
        // same size, so we direct cast, keeping the old length
        unsafe {
            Some(core::slice::from_raw_parts(
                ts.as_ptr() as *const U,
                ts.len(),
            ))
        }
    } else {
        // we might have slop, which would cause us to fail
        let byte_size = size_of::<T>() * ts.len();
        let (new_count, new_overflow) = (byte_size / size_of::<U>(), byte_size % size_of::<U>());
        if new_overflow > 0 {
            None
        } else {
            unsafe {
                Some(core::slice::from_raw_parts(
                    ts.as_ptr() as *const U,
                    new_count,
                ))
            }
        }
    }
}

pub struct HalState {
    creation_instant: Instant,
    cube_indexes: BufferBundle<back::Backend, back::Device>,
    cube_vertices: BufferBundle<back::Backend, back::Device>,
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
    texture: LoadedImage<back::Backend, back::Device>,
    descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout>,
    descriptor_set: ManuallyDrop<<back::Backend as Backend>::DescriptorSet>,
    descriptor_pool: ManuallyDrop<<back::Backend as Backend>::DescriptorPool>,
    graphics_pipeline: ManuallyDrop<<back::Backend as Backend>::GraphicsPipeline>,
    pipeline_layout: ManuallyDrop<<back::Backend as Backend>::PipelineLayout>,
    _adapter: Adapter<back::Backend>,
    _surface: <back::Backend as Backend>::Surface,
    _instance: ManuallyDrop<back::Instance>,
}

impl HalState {
    //region new
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

        let (mut device, mut queue_group): (back::Device, QueueGroup<back::Backend, Graphics>) = {
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
                return Err("The surface isn't capable of surporting color");
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

        let (
            descriptor_set_layouts,
            descriptor_pool,
            descriptor_set,
            pipeline_layout,
            graphics_pipeline,
        ) = Self::create_pipeline(&mut device, extent, &render_pass)?;

        let cube_vertices = BufferBundle::new(
            &adapter,
            &device,
            size_of_val(&CUBE_VERTEXES),
            BufferUsage::VERTEX,
        )?;
        unsafe {
            let mut data_target = device
                .acquire_mapping_writer(&cube_vertices.memory, 0..cube_vertices.requirements.size)
                .map_err(|_| "Failed to acquire a vertex buffer mapping writer")?;
            data_target[..CUBE_VERTEXES.len()].copy_from_slice(&CUBE_VERTEXES);
            device
                .release_mapping_writer(data_target)
                .map_err(|_| "Failed to release vertex buffer mapping writer")?;
        }

        let cube_indexes = BufferBundle::new(
            &adapter,
            &device,
            size_of_val(&CUBE_INDEXES),
            BufferUsage::INDEX,
        )?;
        unsafe {
            let mut data_target = device
                .acquire_mapping_writer(&cube_indexes.memory, 0..cube_indexes.requirements.size)
                .map_err(|_| "Failed to acquire a index buffer mapping writer")?;
            data_target[..CUBE_INDEXES.len()].copy_from_slice(&CUBE_INDEXES);
            device
                .release_mapping_writer(data_target)
                .map_err(|_| "Failed to release index buffer mapping writer")?;
        }

        let texture = LoadedImage::new(
            &adapter,
            &device,
            &mut command_pool,
            &mut queue_group.queues[0],
            image::load_from_memory(CREATURE_BYTES)
                .expect("binary corrupted")
                .to_rgba(),
        )?;

        unsafe {
            device.write_descriptor_sets(vec![
                gfx_hal::pso::DescriptorSetWrite {
                    set: &descriptor_set,
                    binding: 0,
                    array_offset: 0,
                    descriptors: Some(gfx_hal::pso::Descriptor::Image(
                        texture.image_view.deref(),
                        Layout::ShaderReadOnlyOptimal,
                    )),
                },
                gfx_hal::pso::DescriptorSetWrite {
                    set: &descriptor_set,
                    binding: 1,
                    array_offset: 0,
                    descriptors: Some(gfx_hal::pso::Descriptor::Sampler(texture.sampler.deref())),
                },
            ]);
        }

        Ok(Self {
            _instance: ManuallyDrop::new(instance),
            _surface: surface,
            _adapter: adapter,
            creation_instant: Instant::now(),
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
            descriptor_set: ManuallyDrop::new(descriptor_set),
            descriptor_pool: ManuallyDrop::new(descriptor_pool),
            pipeline_layout: ManuallyDrop::new(pipeline_layout),
            graphics_pipeline: ManuallyDrop::new(graphics_pipeline),
            cube_vertices,
            cube_indexes,
            texture,
        })
    }
    //endregion
    //region create_pipeline
    #[allow(clippy::type_complexity)]
    fn create_pipeline(
        device: &mut back::Device,
        extent: Extent2D,
        render_pass: &<back::Backend as Backend>::RenderPass,
    ) -> Result<
        (
            Vec<<back::Backend as Backend>::DescriptorSetLayout>,
            <back::Backend as Backend>::DescriptorPool,
            <back::Backend as Backend>::DescriptorSet,
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
        let (descriptor_set_layouts, descriptor_pool, descriptor_set, layout, gfx_pipeline) = {
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
                stride: (size_of::<Vertex>()) as ElemStride,
                rate: 0,
            }];

            let attributes: Vec<AttributeDesc> = Vertex::attributes();

            let rasterizer = Rasterizer {
                depth_clamping: false,
                polygon_mode: PolygonMode::Fill,
                cull_face: Face::BACK,
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

            let descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout> =
                vec![unsafe {
                    device
                        .create_descriptor_set_layout(
                            &[
                                DescriptorSetLayoutBinding {
                                    binding: 0,
                                    ty: gfx_hal::pso::DescriptorType::SampledImage,
                                    count: 1,
                                    stage_flags: ShaderStageFlags::FRAGMENT,
                                    immutable_samplers: false,
                                },
                                DescriptorSetLayoutBinding {
                                    binding: 1,
                                    ty: gfx_hal::pso::DescriptorType::Sampler,
                                    count: 1,
                                    stage_flags: ShaderStageFlags::FRAGMENT,
                                    immutable_samplers: false,
                                },
                            ],
                            &[],
                        )
                        .map_err(|_| "Couldn't make a DescriptorSetLayout")?
                }];

            let mut descriptor_pool = unsafe {
                device
                    .create_descriptor_pool(
                        1, // sets
                        &[
                            gfx_hal::pso::DescriptorRangeDesc {
                                ty: gfx_hal::pso::DescriptorType::SampledImage,
                                count: 1,
                            },
                            gfx_hal::pso::DescriptorRangeDesc {
                                ty: gfx_hal::pso::DescriptorType::Sampler,
                                count: 1,
                            },
                        ],
                    )
                    .map_err(|_| "Couldn't create a descriptor pool")?
            };

            let descriptor_set = unsafe {
                descriptor_pool
                    .allocate_set(&descriptor_set_layouts[0])
                    .map_err(|_| "Couldn't create descriptor set")?
            };

            let push_constants = vec![(ShaderStageFlags::VERTEX, 0..16)];
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
            (
                descriptor_set_layouts,
                descriptor_pool,
                descriptor_set,
                layout,
                gfx_pipeline,
            )
        };

        // destroy shader modules
        unsafe {
            device.destroy_shader_module(vertex_shader_module);
            device.destroy_shader_module(fragment_shader_module);
        }

        Ok((
            descriptor_set_layouts,
            descriptor_pool,
            descriptor_set,
            layout,
            gfx_pipeline,
        ))
    }
    //endregion
    //region draw_cubes_frame
    pub fn draw_cubes_frame(
        &mut self,
        models: &[glm::TMat4<f32>],
        aspect_ratio: f32,
    ) -> Result<(), &'static str> {
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
                .map_err(|_| "Failed to wait on fence")?;
            self.device
                .reset_fence(flight_fence)
                .map_err(|_| "Failed to reset fence")?;
        }
        let view = glm::look_at_lh(
            &glm::make_vec3(&[0.0, 0.0, -5.0]),
            &glm::make_vec3(&[0.0, 0.0, 0.0]),
            &glm::make_vec3(&[0.0, 1.0, 0.0]).normalize(),
        );

        let projection = {
            let mut temp = glm::perspective_lh_zo(aspect_ratio, f32::to_radians(50.0), 0.1, 100.0);
            temp[(1, 1)] *= -1.0;
            temp
        };

        let vp = projection * view;
        unsafe {
            let buffer = &mut self.command_buffers[i_usize];
            const QUAD_CLEAR: [ClearValue; 1] =
                [ClearValue::Color(ClearColor::Float([0.1, 0.2, 0.3, 1.0]))];
            buffer.begin(false);
            {
                let mut encoder = buffer.begin_render_pass_inline(
                    &self.render_pass,
                    &self.framebuffers[i_usize],
                    self.render_area,
                    QUAD_CLEAR.iter(),
                );
                encoder.bind_graphics_pipeline(&self.graphics_pipeline);
                encoder.bind_vertex_buffers(0, Some((self.cube_vertices.buffer.deref(), 0)));
                encoder.bind_index_buffer(IndexBufferView {
                    buffer: &self.cube_indexes.buffer,
                    offset: 0,
                    index_type: IndexType::U16,
                });
                encoder.bind_graphics_descriptor_sets(
                    &self.pipeline_layout,
                    0,
                    Some(self.descriptor_set.deref()),
                    &[],
                );
                for model in models.iter() {
                    let mvp = vp * model;
                    encoder.push_graphics_constants(
                        &self.pipeline_layout,
                        ShaderStageFlags::VERTEX,
                        0,
                        cast_slice::<f32, u32>(&mvp.data)
                            .expect("this cast never fails for same-aligned same-size data"),
                    );
                    encoder.draw_indexed(0..36, 0, 0..1);
                }
            }
            buffer.finish();
        }
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
                .map_err(|_| "Failed to present into the swapchain")
        }
    }
    //endregion
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
            self.cube_vertices.manually_drop(self.device.deref());
            self.cube_indexes.manually_drop(self.device.deref());
            self.texture.manually_drop(self.device.deref());
            use core::ptr::read;
            self.device
                .destroy_descriptor_pool(ManuallyDrop::into_inner(read(&self.descriptor_pool)));
            self.device
                .destroy_pipeline_layout(ManuallyDrop::into_inner(read(&self.pipeline_layout)));
            self.device
                .destroy_graphics_pipeline(ManuallyDrop::into_inner(read(&self.graphics_pipeline)));

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

pub struct BufferBundle<B: Backend, D: Device<B>> {
    pub buffer: ManuallyDrop<B::Buffer>,
    pub requirements: Requirements,
    pub memory: ManuallyDrop<B::Memory>,
    pub phantom: PhantomData<D>,
}

impl<B: Backend, D: Device<B>> BufferBundle<B, D> {
    pub fn new(
        adapter: &Adapter<B>,
        device: &D,
        size: usize,
        usage: BufferUsage,
    ) -> Result<Self, &'static str> {
        unsafe {
            let mut buffer = device
                .create_buffer(size as u64, usage)
                .map_err(|_| "Couldn't create a buffer")?;
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
                .ok_or("Couldn't find a memory type to support the buffer")?;

            let memory = device
                .allocate_memory(memory_type_id, requirements.size)
                .map_err(|_| "Couldn't allocate buffer memory")?;
            device
                .bind_buffer_memory(&memory, 0, &mut buffer)
                .map_err(|_| "Couldn't bind the buffer memory")?;

            Ok(Self {
                buffer: ManuallyDrop::new(buffer),
                requirements,
                memory: ManuallyDrop::new(memory),
                phantom: PhantomData,
            })
        }
    }

    pub unsafe fn manually_drop(&self, device: &D) {
        use core::ptr::read;
        device.destroy_buffer(ManuallyDrop::into_inner(read(&self.buffer)));
        device.free_memory(ManuallyDrop::into_inner(read(&self.memory)));
    }
}

pub struct LoadedImage<B: Backend, D: Device<B>> {
    pub image: ManuallyDrop<B::Image>,
    pub requirements: Requirements,
    pub memory: ManuallyDrop<B::Memory>,
    pub image_view: ManuallyDrop<B::ImageView>,
    pub sampler: ManuallyDrop<B::Sampler>,
    pub phantom: PhantomData<D>,
}

impl<B: Backend, D: Device<B>> LoadedImage<B, D> {
    pub fn new<C: Capability + Supports<Transfer>>(
        adapter: &Adapter<B>,
        device: &D,
        command_pool: &mut CommandPool<B, C>,
        command_queue: &mut CommandQueue<B, C>,
        img: image::RgbaImage,
    ) -> Result<Self, &'static str> {
        unsafe {
            let pixel_size = size_of::<image::Rgba<u8>>();
            let row_size = pixel_size * (img.width() as usize);
            let limits = adapter.physical_device.limits();
            let row_alignment_mask = limits.min_buffer_copy_pitch_alignment as u32 - 1;
            let row_pitch = ((row_size as u32 + row_alignment_mask) & !row_alignment_mask) as usize;
            debug_assert!(row_pitch as usize >= row_size);
            let required_bytes = row_pitch * img.height() as usize;
            let staging_bundle =
                BufferBundle::new(&adapter, device, required_bytes, BufferUsage::TRANSFER_SRC)?;

            let mut writer = device
                .acquire_mapping_writer(&staging_bundle.memory, 0..staging_bundle.requirements.size)
                .map_err(|_| "Failed to acquire mapping writer to the staging buffer")?;

            for y in 0..img.height() as usize {
                let row = &(*img)[y * row_size..(y + 1) * row_size];
                let dest_base = y * row_pitch;
                writer[dest_base..dest_base + row.len()].copy_from_slice(row);
            }
            device
                .release_mapping_writer(writer)
                .map_err(|_| "Failed to release mapping writer for staging buffer")?;

            let mut the_image = device
                .create_image(
                    gfx_hal::image::Kind::D2(img.width(), img.height(), 1, 1),
                    1, // mip levels
                    Format::Rgba8Srgb,
                    gfx_hal::image::Tiling::Optimal,
                    gfx_hal::image::Usage::TRANSFER_DST | gfx_hal::image::Usage::SAMPLED,
                    gfx_hal::image::ViewCapabilities::empty(),
                )
                .map_err(|_| "Couldn't create the image")?;

            let requirements = device.get_image_requirements(&the_image);
            let memory_type_id = adapter
                .physical_device
                .memory_properties()
                .memory_types
                .iter()
                .enumerate()
                .find(|&(id, memory_type)| {
                    // device local and not cpu visible (?)
                    requirements.type_mask & (1 << id) != 0
                        && memory_type.properties.contains(Properties::DEVICE_LOCAL)
                })
                .map(|(id, _)| MemoryTypeId(id))
                .ok_or("Couldn't find a memory type to support the image")?;
            let memory = device
                .allocate_memory(memory_type_id, requirements.size)
                .map_err(|_| "Couldn't allocate memory for image")?;
            device
                .bind_image_memory(&memory, 0, &mut the_image)
                .map_err(|_| "Couldn't bind the image memory")?;

            let image_view = device
                .create_image_view(
                    &the_image,
                    gfx_hal::image::ViewKind::D2,
                    Format::Rgba8Srgb,
                    gfx_hal::format::Swizzle::NO,
                    SubresourceRange {
                        aspects: Aspects::COLOR,
                        levels: 0..1,
                        layers: 0..1,
                    },
                )
                .map_err(|_| "Couldn't create the image view")?;
            let sampler = device
                .create_sampler(gfx_hal::image::SamplerInfo::new(
                    gfx_hal::image::Filter::Nearest,
                    gfx_hal::image::WrapMode::Tile,
                ))
                .map_err(|_| "Couldn't create sampler")?;

            let mut cmd_buffer = command_pool.acquire_command_buffer::<gfx_hal::command::OneShot>();
            cmd_buffer.begin();

            let image_barrier = gfx_hal::memory::Barrier::Image {
                states: (gfx_hal::image::Access::empty(), Layout::Undefined)
                    ..(
                        gfx_hal::image::Access::TRANSFER_WRITE,
                        Layout::TransferDstOptimal,
                    ),
                target: &the_image,
                families: None,
                range: SubresourceRange {
                    aspects: Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                },
            };
            cmd_buffer.pipeline_barrier(
                PipelineStage::TOP_OF_PIPE..PipelineStage::TRANSFER,
                gfx_hal::memory::Dependencies::empty(),
                &[image_barrier],
            );

            cmd_buffer.copy_buffer_to_image(
                &staging_bundle.buffer,
                &the_image,
                Layout::TransferDstOptimal,
                &[gfx_hal::command::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_width: (row_pitch / pixel_size) as u32,
                    buffer_height: img.height(),
                    image_layers: gfx_hal::image::SubresourceLayers {
                        aspects: Aspects::COLOR,
                        level: 0,
                        layers: 0..1,
                    },
                    image_offset: gfx_hal::image::Offset { x: 0, y: 0, z: 0 },
                    image_extent: gfx_hal::image::Extent {
                        width: img.width(),
                        height: img.height(),
                        depth: 1,
                    },
                }],
            );

            let image_barrier = gfx_hal::memory::Barrier::Image {
                states: (
                    gfx_hal::image::Access::TRANSFER_WRITE,
                    Layout::TransferDstOptimal,
                )
                    ..(
                        gfx_hal::image::Access::SHADER_READ,
                        Layout::ShaderReadOnlyOptimal,
                    ),
                target: &the_image,
                families: None,
                range: SubresourceRange {
                    aspects: Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                },
            };
            cmd_buffer.pipeline_barrier(
                PipelineStage::TRANSFER..PipelineStage::FRAGMENT_SHADER,
                gfx_hal::memory::Dependencies::empty(),
                &[image_barrier],
            );

            cmd_buffer.finish();
            let upload_fence = device
                .create_fence(false)
                .map_err(|_| "Couldn't create and upload fence")?;
            command_queue.submit_nosemaphores(Some(&cmd_buffer), Some(&upload_fence));
            device
                .wait_for_fence(&upload_fence, core::u64::MAX)
                .map_err(|_| "Couldn't wait for fence")?;
            device.destroy_fence(upload_fence);

            staging_bundle.manually_drop(device);
            command_pool.free(Some(cmd_buffer));

            Ok(LoadedImage {
                image: ManuallyDrop::new(the_image),
                requirements,
                memory: ManuallyDrop::new(memory),
                image_view: ManuallyDrop::new(image_view),
                sampler: ManuallyDrop::new(sampler),
                phantom: PhantomData,
            })
        }
    }

    pub unsafe fn manually_drop(&self, device: &D) {
        use core::ptr::read;

        device.destroy_sampler(ManuallyDrop::into_inner(read(&self.sampler)));
        device.destroy_image_view(ManuallyDrop::into_inner(read(&self.image_view)));
        device.destroy_image(ManuallyDrop::into_inner(read(&self.image)));
        device.free_memory(ManuallyDrop::into_inner(read(&self.memory)));
    }
}
