use super::*;

impl FogSettings {
    pub(super) fn from_scene(scene: &RenderScene, background: [f64; 4]) -> Result<Self> {
        let color = parse_color(
            scene.fog_color.as_deref(),
            [background[0], background[1], background[2], 1.0],
            "scene.fogColor",
        )?;
        let base = Self {
            active: false,
            mode: 0.0,
            color: [color[0] as f32, color[1] as f32, color[2] as f32],
            near: 1.0,
            far: 1000.0,
            density: 0.00025,
        };

        let Some(fog_type) = scene.fog_type.as_deref() else {
            return Ok(base);
        };

        match fog_type.to_ascii_lowercase().as_str() {
            "linear" | "fog" => {
                let near = finite_f32(scene.fog_near.unwrap_or(1.0), "scene.fogNear")?;
                let far = finite_f32(scene.fog_far.unwrap_or(1000.0), "scene.fogFar")?;
                if far <= near {
                    bail!("scene.fogFar must be greater than scene.fogNear");
                }
                Ok(Self {
                    active: true,
                    mode: 1.0,
                    near,
                    far,
                    ..base
                })
            }
            "exp2" | "fogexp2" => {
                let density = finite_f32(scene.fog_density.unwrap_or(0.00025), "scene.fogDensity")?;
                if density < 0.0 {
                    bail!("scene.fogDensity must be greater than or equal to 0");
                }
                Ok(Self {
                    active: density > 0.0,
                    mode: 2.0,
                    density,
                    ..base
                })
            }
            other => bail!("unsupported scene.fogType `{other}`; expected `linear` or `exp2`"),
        }
    }
}

impl PostProcessingSettings {
    pub(super) fn from_scene(scene: &RenderScene) -> Result<Self> {
        let exposure = finite_f32(
            scene.post_exposure.unwrap_or(0.0),
            "scene.postProcessing.exposure",
        )?
        .clamp(-16.0, 16.0);
        let contrast = finite_f32(
            scene.post_contrast.unwrap_or(1.0),
            "scene.postProcessing.contrast",
        )?
        .clamp(0.0, 8.0);
        let saturation = finite_f32(
            scene.post_saturation.unwrap_or(1.0),
            "scene.postProcessing.saturation",
        )?
        .clamp(0.0, 8.0);
        let vignette = finite_f32(
            scene.post_vignette.unwrap_or(0.0),
            "scene.postProcessing.vignette",
        )?
        .clamp(0.0, 1.0);
        let grayscale = finite_f32(
            scene.post_grayscale.unwrap_or(0.0),
            "scene.postProcessing.grayscale",
        )?
        .clamp(0.0, 1.0);
        let invert = finite_f32(
            scene.post_invert.unwrap_or(0.0),
            "scene.postProcessing.invert",
        )?
        .clamp(0.0, 1.0);
        let active = exposure.abs() > 0.0001
            || (contrast - 1.0).abs() > 0.0001
            || (saturation - 1.0).abs() > 0.0001
            || vignette > 0.0001
            || grayscale > 0.0001
            || invert > 0.0001;

        Ok(Self {
            active,
            exposure,
            contrast,
            saturation,
            vignette,
            grayscale,
            invert,
        })
    }
}
