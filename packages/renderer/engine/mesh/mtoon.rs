use super::*;
use crate::types::MtoonParameters;
use crate::util::parse_vec3;

pub(super) fn prepare_mtoon(value: Option<&MtoonParameters>) -> Result<[[f32; 4]; 6]> {
    let Some(value) = value else {
        return Ok([[0.0; 4]; 6]);
    };
    let shade = parse_vec3(Some(&value.shade_color), [0.0; 3], "mtoon.shadeColor")?;
    let rim = parse_vec3(Some(&value.rim_color), [0.0; 3], "mtoon.rimColor")?;
    let matcap = parse_vec3(Some(&value.matcap_color), [0.0; 3], "mtoon.matcapColor")?;
    let outline = parse_vec3(Some(&value.outline_color), [0.0; 3], "mtoon.outlineColor")?;
    if value.outline_mode > 2 {
        bail!("mtoon.outlineMode must be 0, 1, or 2");
    }
    Ok([
        [
            shade.x,
            shade.y,
            shade.z,
            finite_f32(value.shading_shift, "mtoon.shadingShift")?,
        ],
        [
            rim.x,
            rim.y,
            rim.z,
            finite_f32(value.rim_fresnel_power, "mtoon.rimFresnelPower")?.max(0.0),
        ],
        [
            matcap.x,
            matcap.y,
            matcap.z,
            finite_f32(value.rim_lift, "mtoon.rimLift")?,
        ],
        [
            finite_f32(value.shading_toony, "mtoon.shadingToony")?.clamp(0.0, 1.0),
            finite_f32(
                value.shading_shift_texture_scale,
                "mtoon.shadingShiftTextureScale",
            )?,
            if value.v0_compat_shade { 1.0 } else { 0.0 },
            0.0,
        ],
        [
            outline.x,
            outline.y,
            outline.z,
            finite_f32(value.outline_lighting_mix, "mtoon.outlineLightingMix")?.clamp(0.0, 1.0),
        ],
        [
            value.outline_mode as f32,
            finite_f32(value.outline_width, "mtoon.outlineWidth")?.max(0.0),
            1.0,
            0.0,
        ],
    ])
}
