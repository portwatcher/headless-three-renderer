use napi_derive::napi;

#[napi(object)]
pub struct MtoonParameters {
    pub shade_color: Vec<f64>,
    pub shading_shift: f64,
    pub shading_toony: f64,
    pub shading_shift_texture_scale: f64,
    pub rim_color: Vec<f64>,
    pub rim_fresnel_power: f64,
    pub rim_lift: f64,
    pub matcap_color: Vec<f64>,
    pub v0_compat_shade: bool,
    pub outline_color: Vec<f64>,
    pub outline_lighting_mix: f64,
    pub outline_width: f64,
    pub outline_mode: u32,
}
