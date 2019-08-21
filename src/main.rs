#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

use arrayvec::ArrayVec;
use core::mem::ManuallyDrop;

use gfx_hal::{
    adapter::{Adapter, PhysicalDevice},
    command::{ClearColor, ClearValue, CommandBuffer, MultiShot, Primary},
    device::Device,
    format::{Aspects, ChannelType, Format, Swizzle},
    image::{Extent, Layout, SubresourceRange, Usage, ViewKind},
    pass::{Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, SubpassDesc},
    pool::{CommandPool, CommandPoolCreateFlags},
    pso::{PipelineStage, Rect},
    queue::{family::QueueGroup, QueueType, Submission},
    window::{Backbuffer, FrameSync, PresentMode, Swapchain, SwapchainConfig},
    Backend, Gpu, Graphics, Instance, QueueFamily, Surface,
};

use winit::{
    dpi::LogicalSize, CreationError, Event, EventsLoop, Window, WindowBuilder, WindowEvent,
};

#[cfg(feature = "dx12")]
use gfx_backend_dx12 as back;

#[cfg(feature = "metal")]
use gfx_backend_metal as back;

#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as back;

pub const WINDOW_NAME: &str = "Hello Vulkan";

#[derive(Debug)]
pub struct WinitState {
    pub events_loop: EventsLoop,
    pub window: Window,
}

impl WinitState {
    pub fn new<T: Into<String>>(title: T, size: LogicalSize) -> Result<Self, CreationError> {
        let events_loop = EventsLoop::new();
        let output = WindowBuilder::new()
            .with_title(title)
            .with_dimensions(size)
            .build(&events_loop);

        output.map(|window| Self {
            events_loop,
            window,
        })
    }
}

impl Default for WinitState {
    fn default() -> Self {
        Self::new(
            WINDOW_NAME,
            LogicalSize {
                width: 800.0,
                height: 600.0,
            },
        )
        .expect("Could not create a window!")
    }
}

pub struct HalState {
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

        let (device, queue_group): (back::Device, QueueGroup<back::Backend, Graphics>) = {
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
            let extent = caps.extents.end;

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
        })
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

            // pretty memory unsafe
            use core::ptr::read;
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

#[derive(Debug, Clone, Default)]
pub struct UserInput {
    pub end_requested: bool,
    pub new_frame_size: Option<(f64, f64)>,
    pub new_mouse_position: Option<(f64, f64)>,
}

impl UserInput {
    pub fn poll_events(events_loop: &mut EventsLoop) -> Self {
        let mut output = UserInput::default();
        events_loop.poll_events(|event| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => output.end_requested = true,
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                output.new_mouse_position = Some((position.x, position.y));
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(logical),
                ..
            } => {
                output.new_frame_size = Some((logical.width, logical.height));
            }
            _ => (),
        });
        output
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LocalState {
    pub frame_width: f64,
    pub frame_height: f64,
    pub mouse_x: f64,
    pub mouse_y: f64,
}

impl LocalState {
    pub fn update_from_input(&mut self, inputs: UserInput) {
        if let Some(framesize) = inputs.new_frame_size {
            self.frame_width = framesize.0;
            self.frame_height = framesize.1;
        }

        if let Some(position) = inputs.new_mouse_position {
            self.mouse_x = position.0;
            self.mouse_y = position.1;
        }
    }
}

pub fn do_the_render(hal: &mut HalState, locals: &LocalState) -> Result<(), &'static str> {
    hal.draw_clear_frame([1.0, 0.0, 1.0, 1.0])
}

fn main() {
    simple_logger::init().unwrap();
    let mut winit_state = WinitState::default();
    let mut hal_state = HalState::new(&winit_state.window).unwrap();

    let (frame_width, frame_height) = winit_state
        .window
        .get_inner_size()
        .map(|logical| logical.into())
        .unwrap_or((0.0, 0.0));

    let mut local_state = LocalState {
        frame_height,
        frame_width,
        mouse_x: 0.0,
        mouse_y: 0.0,
    };

    loop {
        let inputs = UserInput::poll_events(&mut winit_state.events_loop);
        if inputs.end_requested {
            break;
        }

        local_state.update_from_input(inputs);
        if let Err(e) = do_the_render(&mut hal_state, &local_state) {
            error!("Rendering error: {:?}", e);
            break;
        }
    }
}
