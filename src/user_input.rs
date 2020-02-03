#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

use std::collections::HashSet;
use std::time::Instant;
use winit::{
    DeviceEvent, ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent,
};

use crate::winit_state::WinitState;
use nalgebra_glm as glm;

#[derive(Debug, Clone, Copy)]
pub struct EulerFPSCamera {
    pub position: glm::TVec3<f32>,
    pitch_deg: f32,
    yaw_deg: f32,
}

impl EulerFPSCamera {
    const UP: [f32; 3] = [0.0, 1.0, 0.0];

    fn make_front(&self) -> glm::TVec3<f32> {
        let pitch_rad = f32::to_radians(self.pitch_deg);
        let yaw_rad = f32::to_radians(self.yaw_deg);
        glm::make_vec3(&[
            yaw_rad.sin() * pitch_rad.cos(),
            pitch_rad.sin(),
            yaw_rad.cos() * pitch_rad.cos(),
        ])
    }

    pub fn update_orientation(&mut self, d_pitch_deg: f32, d_yaw_deg: f32) {
        self.pitch_deg = (self.pitch_deg + d_pitch_deg).max(-89.0).min(89.0);
        self.yaw_deg = (self.yaw_deg + d_yaw_deg) % 360.0;
    }

    pub fn update_position(&mut self, keys: &HashSet<VirtualKeyCode>, distance: f32) {
        let up = glm::make_vec3(&Self::UP);
        let forward = self.make_front();
        let cross_normalized = glm::cross::<f32, glm::U3>(&forward, &up).normalize();
        let mut move_vector = keys
            .iter()
            .fold(glm::make_vec3(&[0.0, 0.0, 0.0]), |vec, key| match *key {
                VirtualKeyCode::W => vec + forward,
                VirtualKeyCode::S => vec - forward,
                VirtualKeyCode::A => vec + cross_normalized,
                VirtualKeyCode::D => vec - cross_normalized,
                VirtualKeyCode::E => vec + up,
                VirtualKeyCode::Q => vec - up,
                _ => vec,
            });

        if move_vector != glm::zero() {
            move_vector = move_vector.normalize();
            self.position += move_vector * distance;
        }
    }

    pub fn make_view_matrix(&self) -> glm::TMat4<f32> {
        glm::look_at_lh(
            &self.position,
            &(self.position + self.make_front()),
            &glm::make_vec3(&Self::UP),
        )
    }

    pub const fn at_position(position: glm::TVec3<f32>) -> Self {
        Self {
            position,
            pitch_deg: 0.0,
            yaw_deg: 0.0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct UserInput {
    pub end_requested: bool,
    pub new_frame_size: Option<(f64, f64)>,
    pub new_mouse_position: Option<(f64, f64)>,
    pub seconds: f32,
    pub swap_projection: bool,
    pub keys_held: HashSet<VirtualKeyCode>,
    pub orientation_change: (f32, f32),
}

impl UserInput {
    pub fn poll_events(winit_state: &mut WinitState, last_timestamp: &mut Instant) -> Self {
        let mut output = UserInput::default();
        let window = &mut winit_state.window;
        let events_loop = &mut winit_state.events_loop;
        let keys_held = &mut winit_state.keys_held;
        let grabbed = &mut winit_state.grabbed;

        events_loop.poll_events(|event| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => output.end_requested = true,
            Event::DeviceEvent {
                event:
                    DeviceEvent::Key(KeyboardInput {
                        virtual_keycode: Some(code),
                        state,
                        ..
                    }),
                ..
            } => drop(match state {
                ElementState::Pressed => keys_held.insert(code),
                ElementState::Released => keys_held.remove(&code),
            }),
            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state,
                                virtual_keycode: Some(code),
                                ..
                            },
                        ..
                    },
                ..
            } => {
                #[cfg(feature = "metal")]
                {
                    match state {
                        ElementState::Pressed => keys_held.insert(code),
                        ElementState::Released => keys_held.remove(code),
                    }
                };

                if state == ElementState::Pressed {
                    match code {
                        VirtualKeyCode::Tab => output.swap_projection = !output.swap_projection,
                        VirtualKeyCode::Escape => {
                            if *grabbed {
                                debug!("Escape pressed while grabbed, releasing the mouse");
                                window
                                    .grab_cursor(false)
                                    .expect("Failed to release the mouse grab");
                                window.hide_cursor(false);
                                *grabbed = false;
                            }
                        }
                        _ => (),
                    }
                }
            }
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta: (dx, dy) },
                ..
            } => {
                if *grabbed {
                    output.orientation_change.0 -= dx as f32;
                    output.orientation_change.1 += dy as f32;
                }
            }
            Event::WindowEvent {
                event:
                    WindowEvent::MouseInput {
                        state: ElementState::Pressed,
                        button: MouseButton::Left,
                        ..
                    },
                ..
            } => {
                if *grabbed {
                    debug!("Click while already grabbed");
                } else {
                    debug!("Click while not grabbed");
                    window.grab_cursor(true).expect("Failed to grab with mouse");
                    window.hide_cursor(true);
                    *grabbed = true;
                }
            }
            Event::WindowEvent {
                event: WindowEvent::Focused(false),
                ..
            } => {
                if *grabbed {
                    debug!("Lost focus, releasing grab");
                    window.grab_cursor(false).expect("Failed to release grab");
                    window.hide_cursor(false);
                    *grabbed = false;
                } else {
                    debug!("Lost focus while not grabbed");
                }
            }
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
        output.keys_held = if *grabbed {
            keys_held.clone()
        } else {
            HashSet::new()
        };
        output
    }
}
