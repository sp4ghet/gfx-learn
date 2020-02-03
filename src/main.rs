mod hal_state;
mod primitives;
mod user_input;
mod winit_state;

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

use hal_state::HalState;
use nalgebra_glm as glm;
use std::time::Instant;
use user_input::{EulerFPSCamera, UserInput};
use winit_state::WinitState;

pub const WINDOW_NAME: &str = "Hello Vulkan";
pub const VERTEX_SOURCE: &str = include_str!("./tri.vert");
pub const FRAGMENT_SOURCE: &str = include_str!("./tri.frag");
pub static CREATURE_BYTES: &[u8] = include_bytes!("./icon.jpg");

#[derive(Debug, Clone)]
pub struct LocalState {
    pub frame_width: f64,
    pub frame_height: f64,
    pub mouse_x: f64,
    pub mouse_y: f64,
    pub cubes: Vec<glm::TMat4<f32>>,
    pub camera: EulerFPSCamera,
    pub perspective_projection: glm::TMat4<f32>,
    pub orthographic_projection: glm::TMat4<f32>,
    pub is_orthographic: bool,
    pub spare_time: f32,
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
        assert!(self.frame_width != 0.0 && self.frame_height != 0.0);
        let x_axis = (self.mouse_x / self.frame_width) as f32;
        let y_axis = (self.mouse_y / self.frame_height) as f32;
        self.spare_time += inputs.seconds;
        const ONE_SIXTIETH: f32 = 1.0 / 60.0;

        while self.spare_time > 0.0 {
            for (i, cube_mut) in self.cubes.iter_mut().enumerate() {
                let r = ONE_SIXTIETH * 30.0 * (i as f32 + 1.0);
                *cube_mut = glm::rotate(
                    &cube_mut,
                    f32::to_radians(r),
                    &glm::make_vec3(&[x_axis, y_axis, 0.3]).normalize(),
                );
            }
            self.spare_time -= ONE_SIXTIETH;
        }

        const MOUSE_SENSITIVITY: f32 = 0.05;
        let d_pitch = -inputs.orientation_change.1 * MOUSE_SENSITIVITY;
        let d_yaw = -inputs.orientation_change.0 * MOUSE_SENSITIVITY;
        self.camera.update_orientation(d_pitch, d_yaw);
        self.camera
            .update_position(&inputs.keys_held, 5.0 * inputs.seconds);
        self.is_orthographic = if inputs.swap_projection {
            !self.is_orthographic
        } else {
            self.is_orthographic
        };
    }
}

pub fn do_the_render(hal: &mut HalState, locals: &LocalState) -> Result<(), &'static str> {
    let aspect_ratio = locals.frame_width / locals.frame_height;
    let projection = if locals.is_orthographic {
        locals.orthographic_projection
    } else {
        locals.perspective_projection
    };
    let view_projection = projection * locals.camera.make_view_matrix();
    hal.draw_cubes_frame(&view_projection, &locals.cubes, aspect_ratio as f32)
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
        spare_time: 0.0,
        cubes: vec![
            glm::identity(),
            glm::translate(&glm::identity(), &glm::make_vec3(&[1.5, 0.1, 0.0])),
            glm::translate(&glm::identity(), &glm::make_vec3(&[-3.0, 2.0, 3.0])),
            glm::translate(&glm::identity(), &glm::make_vec3(&[0.5, -4.0, 4.0])),
            glm::translate(&glm::identity(), &glm::make_vec3(&[-3.4, -2.3, 1.0])),
            glm::translate(&glm::identity(), &glm::make_vec3(&[-2.8, -0.7, 5.0])),
        ],
        camera: EulerFPSCamera::at_position(glm::make_vec3(&[0.0, 0.0, -5.0])),
        orthographic_projection: {
            let mut temp = glm::ortho_lh_zo(-5.0, 5.0, -5.0, 5.0, 0.1, 100.0);
            temp[(1, 1)] *= -1.0;
            temp
        },
        perspective_projection: {
            let mut temp = glm::perspective_lh_zo(4.0 / 3.0, f32::to_radians(50.0), 0.1, 100.0);
            temp[(1, 1)] *= -1.0;
            temp
        },
        is_orthographic: false,
    };

    let mut last_timestamp = Instant::now();

    loop {
        let inputs = UserInput::poll_events(&mut winit_state, &mut last_timestamp);
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
