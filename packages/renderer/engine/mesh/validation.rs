use super::*;

pub(super) fn parse_texture_transform(
    values: Option<&[f64]>,
    mesh_index: usize,
) -> Result<[f32; 6]> {
    let Some(values) = values else {
        return Ok([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    };
    if values.len() != 6 {
        bail!("scene.meshes[{mesh_index}].textureTransform must be an array of 6 numbers");
    }
    Ok([
        finite_f32(values[0], "mesh textureTransform")?,
        finite_f32(values[1], "mesh textureTransform")?,
        finite_f32(values[2], "mesh textureTransform")?,
        finite_f32(values[3], "mesh textureTransform")?,
        finite_f32(values[4], "mesh textureTransform")?,
        finite_f32(values[5], "mesh textureTransform")?,
    ])
}

pub(super) fn parse_optional_vec3(values: Option<&[f64]>, field: &str) -> Result<Option<[f32; 3]>> {
    let Some(values) = values else {
        return Ok(None);
    };
    if values.len() != 3 {
        bail!("{field} must be an array of 3 numbers");
    }
    Ok(Some([
        finite_f32(values[0], field)?,
        finite_f32(values[1], field)?,
        finite_f32(values[2], field)?,
    ]))
}

pub(super) fn parse_optional_vec2(
    values: Option<&[f64]>,
    default: [f32; 2],
    field: &str,
) -> Result<[f32; 2]> {
    let Some(values) = values else {
        return Ok(default);
    };
    if values.len() != 2 {
        bail!("{field} must be an array of 2 numbers");
    }
    Ok([finite_f32(values[0], field)?, finite_f32(values[1], field)?])
}

pub(super) fn parse_optional_clamped_color3(
    values: Option<&[f64]>,
    default: [f32; 3],
    field: &str,
) -> Result<[f32; 3]> {
    let Some(values) = values else {
        return Ok(default);
    };
    if values.len() != 3 {
        bail!("{field} must be an array of 3 numbers");
    }
    Ok([
        finite_clamp01(values[0], field)?,
        finite_clamp01(values[1], field)?,
        finite_clamp01(values[2], field)?,
    ])
}

pub(super) fn finite_clamp01(value: f64, field: &str) -> Result<f32> {
    Ok(finite_f32(value, field)?.clamp(0.0, 1.0))
}

pub(super) fn parse_clipping_planes(
    values: Option<&[f64]>,
    union_count: Option<u32>,
    mesh_index: usize,
) -> Result<([[f32; 4]; MAX_CLIPPING_PLANES], u32, u32)> {
    let Some(values) = values else {
        return Ok(([[0.0; 4]; MAX_CLIPPING_PLANES], 0, 0));
    };
    if values.len() % 4 != 0 {
        bail!(
            "scene.meshes[{mesh_index}].clippingPlanes must be an array of [nx, ny, nz, constant] planes"
        );
    }

    let count = (values.len() / 4).min(MAX_CLIPPING_PLANES);
    let mut planes = [[0.0; 4]; MAX_CLIPPING_PLANES];
    for plane_index in 0..count {
        let base = plane_index * 4;
        planes[plane_index] = [
            finite_f32(values[base], "mesh clippingPlanes")?,
            finite_f32(values[base + 1], "mesh clippingPlanes")?,
            finite_f32(values[base + 2], "mesh clippingPlanes")?,
            finite_f32(values[base + 3], "mesh clippingPlanes")?,
        ];
    }

    let count = count as u32;
    let union_count = union_count.unwrap_or(count).min(count);
    Ok((planes, count, union_count))
}

pub(super) fn parse_custom_blend_state(
    mesh: &SceneMesh,
    blending: BlendMode,
    mesh_index: usize,
) -> Result<Option<CustomBlendState>> {
    if blending != BlendMode::Custom {
        return Ok(None);
    }

    const ADD_EQUATION: u32 = 100;
    const SRC_ALPHA_FACTOR: u32 = 204;
    const ONE_MINUS_SRC_ALPHA_FACTOR: u32 = 205;

    let color_equation = parse_blend_equation(
        mesh.blend_equation.unwrap_or(ADD_EQUATION),
        mesh_index,
        "blendEquation",
    )?;
    let alpha_equation = parse_blend_equation(
        mesh.blend_equation_alpha
            .unwrap_or(mesh.blend_equation.unwrap_or(ADD_EQUATION)),
        mesh_index,
        "blendEquationAlpha",
    )?;
    let color_src_factor = parse_blend_factor(
        mesh.blend_src.unwrap_or(SRC_ALPHA_FACTOR),
        mesh_index,
        "blendSrc",
    )?;
    let color_dst_factor = parse_blend_factor(
        mesh.blend_dst.unwrap_or(ONE_MINUS_SRC_ALPHA_FACTOR),
        mesh_index,
        "blendDst",
    )?;
    let alpha_src_factor = parse_blend_factor(
        mesh.blend_src_alpha
            .unwrap_or(mesh.blend_src.unwrap_or(SRC_ALPHA_FACTOR)),
        mesh_index,
        "blendSrcAlpha",
    )?;
    let alpha_dst_factor = parse_blend_factor(
        mesh.blend_dst_alpha
            .unwrap_or(mesh.blend_dst.unwrap_or(ONE_MINUS_SRC_ALPHA_FACTOR)),
        mesh_index,
        "blendDstAlpha",
    )?;
    let blend_color = parse_color(
        mesh.blend_color.as_deref(),
        [0.0, 0.0, 0.0, 1.0],
        &format!("scene.meshes[{mesh_index}].blendColor"),
    )?;
    let blend_alpha = finite_clamp01(mesh.blend_alpha.unwrap_or(0.0), "mesh blendAlpha")?;

    Ok(Some(CustomBlendState {
        color_equation,
        alpha_equation,
        color_src_factor,
        color_dst_factor,
        alpha_src_factor,
        alpha_dst_factor,
        constant: [
            clamp01(blend_color[0]) as f32,
            clamp01(blend_color[1]) as f32,
            clamp01(blend_color[2]) as f32,
            blend_alpha,
        ],
    }))
}

pub(super) fn parse_blend_equation(
    value: u32,
    mesh_index: usize,
    field: &str,
) -> Result<BlendEquation> {
    match value {
        100 => Ok(BlendEquation::Add),
        101 => Ok(BlendEquation::Subtract),
        102 => Ok(BlendEquation::ReverseSubtract),
        103 => Ok(BlendEquation::Min),
        104 => Ok(BlendEquation::Max),
        _ => bail!("scene.meshes[{mesh_index}].{field} has unsupported blend equation {value}"),
    }
}

pub(super) fn parse_blend_factor(
    value: u32,
    mesh_index: usize,
    field: &str,
) -> Result<BlendFactor> {
    match value {
        200 => Ok(BlendFactor::Zero),
        201 => Ok(BlendFactor::One),
        202 => Ok(BlendFactor::SrcColor),
        203 => Ok(BlendFactor::OneMinusSrcColor),
        204 => Ok(BlendFactor::SrcAlpha),
        205 => Ok(BlendFactor::OneMinusSrcAlpha),
        206 => Ok(BlendFactor::DstAlpha),
        207 => Ok(BlendFactor::OneMinusDstAlpha),
        208 => Ok(BlendFactor::DstColor),
        209 => Ok(BlendFactor::OneMinusDstColor),
        210 => Ok(BlendFactor::SrcAlphaSaturate),
        211 => Ok(BlendFactor::ConstantColor),
        212 => Ok(BlendFactor::OneMinusConstantColor),
        213 => Ok(BlendFactor::ConstantAlpha),
        214 => Ok(BlendFactor::OneMinusConstantAlpha),
        _ => bail!("scene.meshes[{mesh_index}].{field} has unsupported blend factor {value}"),
    }
}

pub(super) fn parse_depth_func(value: Option<&str>, mesh_index: usize) -> Result<StencilCompare> {
    match value.unwrap_or("less-equal") {
        "never" => Ok(StencilCompare::Never),
        "less" => Ok(StencilCompare::Less),
        "equal" => Ok(StencilCompare::Equal),
        "less-equal" => Ok(StencilCompare::LessEqual),
        "greater" => Ok(StencilCompare::Greater),
        "not-equal" => Ok(StencilCompare::NotEqual),
        "greater-equal" => Ok(StencilCompare::GreaterEqual),
        "always" => Ok(StencilCompare::Always),
        other => {
            bail!("scene.meshes[{mesh_index}].depthFunc has unsupported compare function `{other}`")
        }
    }
}

pub(super) fn parse_shadow_side(value: Option<&str>, mesh_index: usize) -> Result<MeshSide> {
    match value {
        None => Ok(MeshSide::Double),
        Some("front") => Ok(MeshSide::Front),
        Some("back") => Ok(MeshSide::Back),
        Some("double") => Ok(MeshSide::Double),
        Some(other) => {
            bail!("scene.meshes[{mesh_index}].shadowSide has unsupported side `{other}`")
        }
    }
}

pub(super) fn parse_stencil_compare(
    value: u32,
    mesh_index: usize,
    field: &str,
) -> Result<StencilCompare> {
    match value {
        512 => Ok(StencilCompare::Never),
        513 => Ok(StencilCompare::Less),
        514 => Ok(StencilCompare::Equal),
        515 => Ok(StencilCompare::LessEqual),
        516 => Ok(StencilCompare::Greater),
        517 => Ok(StencilCompare::NotEqual),
        518 => Ok(StencilCompare::GreaterEqual),
        519 => Ok(StencilCompare::Always),
        _ => bail!("scene.meshes[{mesh_index}].{field} has unsupported stencil function {value}"),
    }
}

pub(super) fn parse_stencil_operation(
    value: u32,
    mesh_index: usize,
    field: &str,
) -> Result<StencilOperation> {
    match value {
        0 => Ok(StencilOperation::Zero),
        7680 => Ok(StencilOperation::Keep),
        7681 => Ok(StencilOperation::Replace),
        7682 => Ok(StencilOperation::IncrementClamp),
        7683 => Ok(StencilOperation::DecrementClamp),
        34055 => Ok(StencilOperation::IncrementWrap),
        34056 => Ok(StencilOperation::DecrementWrap),
        5386 => Ok(StencilOperation::Invert),
        _ => bail!("scene.meshes[{mesh_index}].{field} has unsupported stencil operation {value}"),
    }
}
