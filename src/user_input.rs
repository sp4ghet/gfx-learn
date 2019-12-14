use std::time::Instant;
use winit::{Event, EventsLoop, WindowEvent};

#[derive(Debug, Clone, Default)]
pub struct UserInput {
    pub end_requested: bool,
    pub new_frame_size: Option<(f64, f64)>,
    pub new_mouse_position: Option<(f64, f64)>,
    pub seconds: f32,
}

impl UserInput {
    pub fn poll_events(events_loop: &mut EventsLoop, last_timestamp: &mut Instant) -> Self {
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
        output.seconds = {
            let now = Instant::now();
            let duration = now.duration_since(*last_timestamp);
            *last_timestamp = now;
            duration.as_secs() as f32 + duration.as_nanos() as f32 * 1e-9
        };
        output
    }
}
