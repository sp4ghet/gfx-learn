#[derive(Debug, Copy, Clone)]
pub struct Triangle {
    pub points: [[f32; 2]; 3],
}

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
    pub fn vertex_attributes(self) -> [f32; 4 * (2 + 3 + 2)] {
        let x = self.x;
        let y = self.y;
        let w = self.w;
        let h = self.h;

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
