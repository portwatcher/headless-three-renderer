use napi_derive::napi;

#[napi(object)]
#[derive(Clone, Debug, Default)]
pub struct Camera {
    /// Output width in pixels.
    pub width: Option<u32>,
    /// Output height in pixels.
    pub height: Option<u32>,
    /// Camera position. Defaults to `[2.5, 1.8, 3.2]`.
    pub eye: Option<Vec<f64>>,
    /// Look-at target. Defaults to `[0, 0, 0]`.
    pub target: Option<Vec<f64>>,
    /// Up direction. Defaults to `[0, 1, 0]`.
    pub up: Option<Vec<f64>>,
    /// Vertical field of view in degrees. Defaults to `45`.
    pub fov_y_degrees: Option<f64>,
    /// Near clipping plane. Defaults to `0.01`.
    pub near: Option<f64>,
    /// Far clipping plane. Defaults to `100`.
    pub far: Option<f64>,
    /// Optional column-major 4x4 view-projection matrix in WebGPU clip space.
    pub view_projection: Option<Vec<f64>>,
    /// Optional column-major 4x4 view matrix.
    pub view_matrix: Option<Vec<f64>>,
    /// Camera world position `[x, y, z]` for PBR lighting.
    pub camera_position: Option<Vec<f64>>,
}
