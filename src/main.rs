mod hal_state;
mod primitives;
mod user_input;
mod winit_state;

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

use hal_state::HalState;
use primitives::Quad;
use user_input::UserInput;
use winit_state::WinitState;

pub const WINDOW_NAME: &str = "Hello Vulkan";
pub const VERTEX_SOURCE: &str = include_str!("./tri.vert");
pub const FRAGMENT_SOURCE: &str = include_str!("./tri.frag");
pub static CREATURE_BYTES: &[u8] = include_bytes!("./creature.png");

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
    let x1 = 100.0;
    let y1 = 100.0;
    let x2 = locals.mouse_x as f32;
    let y2 = locals.mouse_y as f32;
    let quad = Quad {
        x: (x1 / locals.frame_width as f32) * 2.0 - 1.0,
        y: (y1 / locals.frame_height as f32) * 2.0 - 1.0,
        w: ((x2 - x1) / locals.frame_width as f32) * 2.0,
        h: ((y2 - y1) / locals.frame_height as f32) * 2.0,
    };
    hal.draw_quad_frame(quad)
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

        if inputs.new_frame_size.is_some() {
            debug!("Window changed size");
            drop(hal_state);
            hal_state = match HalState::new(&winit_state.window) {
                Ok(state) => state,
                Err(e) => panic!(e),
            };
        }

        local_state.update_from_input(inputs);
        if let Err(e) = do_the_render(&mut hal_state, &local_state) {
            error!("Rendering error: {:?}", e);
            debug!("Auto-restarting hal_state...");
            drop(hal_state);
            hal_state = match HalState::new(&winit_state.window) {
                Ok(state) => state,
                Err(e) => panic!(e),
            };
        }
    }
}
