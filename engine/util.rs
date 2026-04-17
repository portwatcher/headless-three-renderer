use anyhow::{Context, Result, bail};
use glam::{Mat4, Vec3};
use image::{ColorType, ImageEncoder, codecs::png::PngEncoder};

pub fn parse_vec3(values: Option<&[f64]>, default: [f32; 3], field: &str) -> Result<Vec3> {
    let Some(values) = values else {
        return Ok(Vec3::from_array(default));
    };

    if values.len() != 3 {
        bail!("{field} must be an array of 3 numbers");
    }

    Ok(Vec3::new(
        finite_f32(values[0], field)?,
        finite_f32(values[1], field)?,
        finite_f32(values[2], field)?,
    ))
}

pub fn parse_color(values: Option<&[f64]>, default: [f64; 4], field: &str) -> Result<[f64; 4]> {
    let Some(values) = values else {
        return Ok(default);
    };

    match values.len() {
        3 => Ok([
            finite_color(values[0], field)?,
            finite_color(values[1], field)?,
            finite_color(values[2], field)?,
            default[3],
        ]),
        4 => Ok([
            finite_color(values[0], field)?,
            finite_color(values[1], field)?,
            finite_color(values[2], field)?,
            finite_color(values[3], field)?,
        ]),
        _ => bail!("{field} must be `[r, g, b]` or `[r, g, b, a]`"),
    }
}

pub fn parse_transform(values: Option<&[f64]>, mesh_index: usize) -> Result<Mat4> {
    let Some(values) = values else {
        return Ok(Mat4::IDENTITY);
    };

    parse_mat4(values, &format!("scene.meshes[{mesh_index}].transform"))
}

pub fn parse_mat4(values: &[f64], field: &str) -> Result<Mat4> {
    if values.len() != 16 {
        bail!("{field} must be a column-major 4x4 matrix");
    }
    let mut matrix = [0.0; 16];
    for (slot, value) in matrix.iter_mut().zip(values) {
        *slot = finite_f32(*value, field)?;
    }

    Ok(Mat4::from_cols_array(&matrix))
}

pub fn finite_positive(value: f64, field: &str) -> Result<f32> {
    let value = finite_f32(value, field)?;
    if value <= 0.0 {
        bail!("{field} must be greater than 0");
    }
    Ok(value)
}

pub fn finite_f32(value: f64, field: &str) -> Result<f32> {
    if !value.is_finite() || value < f32::MIN as f64 || value > f32::MAX as f64 {
        bail!("{field} must contain finite f32-compatible numbers");
    }
    Ok(value as f32)
}

pub fn finite_color(value: f64, field: &str) -> Result<f64> {
    if !value.is_finite() {
        bail!("{field} must contain finite numbers");
    }
    Ok(clamp01(value))
}

pub fn color_to_f32(color: [f64; 4]) -> [f32; 4] {
    [
        color[0] as f32,
        color[1] as f32,
        color[2] as f32,
        color[3] as f32,
    ]
}

pub fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

pub fn validate_dimension(name: &str, value: u32, device_limit: u32) -> Result<()> {
    if value == 0 {
        bail!("{name} must be greater than 0");
    }
    if value > super::MAX_DIMENSION {
        bail!("{name} must be <= {}", super::MAX_DIMENSION);
    }
    if value > device_limit {
        bail!("{name} {value} exceeds the selected GPU limit {device_limit}");
    }
    Ok(())
}

pub fn align_to(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

pub fn encode_png(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(rgba, width, height, ColorType::Rgba8.into())
        .context("failed to encode PNG")?;
    Ok(png)
}
