use super::*;

#[derive(Copy, Clone)]
pub enum OutputFormat {
    Png,
    Rgba,
}

impl OutputFormat {
    pub(super) fn from_scene(scene: &RenderScene) -> Result<Self> {
        let Some(format) = scene.format.as_deref() else {
            return Ok(Self::Png);
        };

        match format.to_ascii_lowercase().as_str() {
            "png" => Ok(Self::Png),
            "rgba" | "raw" | "raw-rgba" => Ok(Self::Rgba),
            other => bail!("unsupported scene.format `{other}`; expected `png` or `rgba`"),
        }
    }
}

pub(super) fn resolve_sample_count(value: Option<u32>) -> Result<u32> {
    match value.unwrap_or(1) {
        0 | 1 => Ok(1),
        4 => Ok(4),
        other => bail!(
            "unsupported scene.sampleCount `{other}`; expected 1 for single-sample rendering or 4 for 4x MSAA"
        ),
    }
}

pub(super) fn resolve_tone_mapping(value: Option<u32>) -> Result<f32> {
    match value.unwrap_or(4) {
        0 => Ok(0.0), // THREE.NoToneMapping
        1 => Ok(1.0), // THREE.LinearToneMapping
        2 => Ok(2.0), // THREE.ReinhardToneMapping
        3 => Ok(3.0), // THREE.CineonToneMapping
        4 => Ok(4.0), // THREE.ACESFilmicToneMapping
        5 => Ok(5.0), // THREE.CustomToneMapping
        6 => Ok(6.0), // THREE.AgXToneMapping
        7 => Ok(7.0), // THREE.NeutralToneMapping
        other => bail!(
            "unsupported scene.toneMapping `{other}`; expected NoToneMapping, LinearToneMapping, ReinhardToneMapping, CineonToneMapping, ACESFilmicToneMapping, CustomToneMapping, AgXToneMapping, or NeutralToneMapping"
        ),
    }
}

pub(super) fn resolve_shadow_map_type(value: Option<u32>) -> Result<f32> {
    match value.unwrap_or(1) {
        0 => Ok(0.0), // THREE.BasicShadowMap
        1 => Ok(1.0), // THREE.PCFShadowMap
        2 => Ok(2.0), // THREE.PCFSoftShadowMap
        3 => Ok(3.0), // THREE.VSMShadowMap
        other => bail!(
            "unsupported scene.shadowMapType `{other}`; expected BasicShadowMap, PCFShadowMap, PCFSoftShadowMap, or VSMShadowMap"
        ),
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OutputColorSpace {
    Srgb,
    LinearSrgb,
}

impl OutputColorSpace {
    pub(super) fn from_scene(scene: &RenderScene) -> Result<Self> {
        let Some(color_space) = scene.output_color_space.as_deref() else {
            return Ok(Self::Srgb);
        };

        match color_space.to_ascii_lowercase().as_str() {
            "srgb" => Ok(Self::Srgb),
            "srgb-linear" | "linear-srgb" | "linearsrgb" | "linear" => Ok(Self::LinearSrgb),
            other => bail!(
                "unsupported scene.outputColorSpace `{other}`; expected `srgb`, `srgb-linear`, `linear-srgb`, `linearsrgb`, or `linear`"
            ),
        }
    }

    pub fn is_linear(self) -> bool {
        matches!(self, Self::LinearSrgb)
    }
}

pub(super) fn parse_environment_color_space(value: Option<&str>) -> Result<bool> {
    let Some(color_space) = value else {
        return Ok(true);
    };

    match color_space.to_ascii_lowercase().as_str() {
        "srgb" => Ok(true),
        "srgb-linear" | "linear-srgb" | "linearsrgb" | "linear" => Ok(false),
        other => bail!(
            "unsupported scene.environmentMapColorSpace `{other}`; expected `srgb`, `srgb-linear`, `linear-srgb`, `linearsrgb`, or `linear`"
        ),
    }
}

pub(super) fn parse_texture_transform(values: Option<&[f64]>, label: &str) -> Result<[f32; 6]> {
    let Some(values) = values else {
        return Ok([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    };
    if values.len() != 6 {
        bail!("{label} must be an array of 6 numbers");
    }
    Ok([
        finite_f32(values[0], label)?,
        finite_f32(values[1], label)?,
        finite_f32(values[2], label)?,
        finite_f32(values[3], label)?,
        finite_f32(values[4], label)?,
        finite_f32(values[5], label)?,
    ])
}

pub(super) fn parse_rotation_columns(values: Option<&[f64]>, label: &str) -> Result<[[f32; 4]; 3]> {
    let Some(values) = values else {
        return Ok([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]);
    };
    if values.len() != 9 {
        bail!("{label} must be an array of 9 numbers");
    }
    Ok([
        [
            finite_f32(values[0], label)?,
            finite_f32(values[1], label)?,
            finite_f32(values[2], label)?,
            0.0,
        ],
        [
            finite_f32(values[3], label)?,
            finite_f32(values[4], label)?,
            finite_f32(values[5], label)?,
            0.0,
        ],
        [
            finite_f32(values[6], label)?,
            finite_f32(values[7], label)?,
            finite_f32(values[8], label)?,
            0.0,
        ],
    ])
}

pub(super) fn parse_light_probe(values: Option<&[f64]>) -> Result<([[f32; 3]; 9], bool)> {
    let Some(values) = values else {
        return Ok(([[0.0; 3]; 9], false));
    };
    if values.len() != 27 {
        bail!("scene.lightProbe must be an array of 27 numbers");
    }
    let mut coefficients = [[0.0; 3]; 9];
    for i in 0..9 {
        coefficients[i] = [
            finite_f32(values[i * 3], "scene.lightProbe")?,
            finite_f32(values[i * 3 + 1], "scene.lightProbe")?,
            finite_f32(values[i * 3 + 2], "scene.lightProbe")?,
        ];
    }
    Ok((coefficients, true))
}
