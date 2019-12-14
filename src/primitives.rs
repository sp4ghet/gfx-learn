use gfx_hal::{
    pso::{AttributeDesc, Element, ElemOffset},
    format::Format
};
use core::mem::size_of;

#[cfg_attr(rustfmt, rustfmt_skip)]
pub const CUBE_VERTEXES: [Vertex; 24] = [
  // Face 1 (front)
  Vertex { xyz: [0.0, 0.0, 0.0], uv: [0.0, 1.0] }, /* bottom left */
  Vertex { xyz: [0.0, 1.0, 0.0], uv: [0.0, 0.0] }, /* top left */
  Vertex { xyz: [1.0, 0.0, 0.0], uv: [1.0, 1.0] }, /* bottom right */
  Vertex { xyz: [1.0, 1.0, 0.0], uv: [1.0, 0.0] }, /* top right */
  // Face 2 (top)
  Vertex { xyz: [0.0, 1.0, 0.0], uv: [0.0, 1.0] }, /* bottom left */
  Vertex { xyz: [0.0, 1.0, 1.0], uv: [0.0, 0.0] }, /* top left */
  Vertex { xyz: [1.0, 1.0, 0.0], uv: [1.0, 1.0] }, /* bottom right */
  Vertex { xyz: [1.0, 1.0, 1.0], uv: [1.0, 0.0] }, /* top right */
  // Face 3 (back)
  Vertex { xyz: [0.0, 0.0, 1.0], uv: [0.0, 1.0] }, /* bottom left */
  Vertex { xyz: [0.0, 1.0, 1.0], uv: [0.0, 0.0] }, /* top left */
  Vertex { xyz: [1.0, 0.0, 1.0], uv: [1.0, 1.0] }, /* bottom right */
  Vertex { xyz: [1.0, 1.0, 1.0], uv: [1.0, 0.0] }, /* top right */
  // Face 4 (bottom)
  Vertex { xyz: [0.0, 0.0, 0.0], uv: [0.0, 1.0] }, /* bottom left */
  Vertex { xyz: [0.0, 0.0, 1.0], uv: [0.0, 0.0] }, /* top left */
  Vertex { xyz: [1.0, 0.0, 0.0], uv: [1.0, 1.0] }, /* bottom right */
  Vertex { xyz: [1.0, 0.0, 1.0], uv: [1.0, 0.0] }, /* top right */
  // Face 5 (left)
  Vertex { xyz: [0.0, 0.0, 1.0], uv: [0.0, 1.0] }, /* bottom left */
  Vertex { xyz: [0.0, 1.0, 1.0], uv: [0.0, 0.0] }, /* top left */
  Vertex { xyz: [0.0, 0.0, 0.0], uv: [1.0, 1.0] }, /* bottom right */
  Vertex { xyz: [0.0, 1.0, 0.0], uv: [1.0, 0.0] }, /* top right */
  // Face 6 (right)
  Vertex { xyz: [1.0, 0.0, 0.0], uv: [0.0, 1.0] }, /* bottom left */
  Vertex { xyz: [1.0, 1.0, 0.0], uv: [0.0, 0.0] }, /* top left */
  Vertex { xyz: [1.0, 0.0, 1.0], uv: [1.0, 1.0] }, /* bottom right */
  Vertex { xyz: [1.0, 1.0, 1.0], uv: [1.0, 0.0] }, /* top right */
];

#[cfg_attr(rustfmt, rustfmt_skip)]
pub const CUBE_INDEXES: [u16; 36] = [
     0,  1,  2,  2,  1,  3,
     4,  5,  6,  7,  6,  5,
    10,  9,  8,  9, 10, 11,
    12, 14, 13, 15, 13, 14,
    16, 17, 18, 19, 18, 17,
    20, 21, 22, 23, 22, 21
];

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct Vertex{
    pub xyz: [f32; 3],
    pub uv: [f32; 2],
}

impl Vertex{
    pub fn attributes() -> Vec<AttributeDesc> {
        let position_attribute = AttributeDesc{
            location: 0,
            binding: 0,
            element: Element{
                format: Format::Rgb32Float,
                offset: 0,
            }
        };
    
        let uv_attribute = AttributeDesc{
            location: 1,
            binding: 0,
            element: Element{
                format: Format::Rg32Float,
                offset: size_of::<[f32; 3]>() as ElemOffset,
            }
        };
        vec![position_attribute, uv_attribute]
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Triangle {
    pub points: [[f32; 2]; 3],
}

#[allow(clippy::many_single_char_names)]
impl Triangle {
    pub fn points_flat(self) -> [f32; 6] {
        let [[a, b], [c, d], [e, f]] = self.points;
        [a, b, c, d, e, f]
    }

    pub fn vertex_attributes(self) -> [f32; 3 * (2 + 3)] {
        let [[a, b], [c, d], [e, f]] = self.points;
        [
            a, b, 1.0, 0.0, 0.0, //r
            c, d, 0.0, 1.0, 0.0, //g
            e, f, 0.0, 0.0, 1.0, //b
        ]
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Quad {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Quad {
    #[allow(clippy::many_single_char_names)]
    pub fn vertex_attributes(self) -> [f32; 4 * (2 + 3 + 2)] {
        let x = self.x;
        let y = self.y;
        let w = self.w;
        let h = self.h;

        // #[rustfmt::skip] once it's no longer experimental for expressions
        #[cfg_attr(rustfmt, rustfmt_skip)]
        [
            // X Y R G B U V
            x, y+h, 1.0, 0.0, 0.0, 0.0, 1.0,
            x, y, 0.0, 1.0, 1.0, 0.0, 0.0,
            x+w, y, 0.0, 0.0, 1.0, 1.0, 0.0,
            x+w, y+h, 1.0, 0.0, 1.0, 1.0, 1.0,  
        ]
    }
}
