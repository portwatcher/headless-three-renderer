struct BackgroundUniforms {
  // transform1.xyz / transform2.xyz = texture transform rows.
  // transform1.w = background intensity.
  transform1: vec4<f32>,
  // transform2.w integer flags: +1 = texture is sRGB, +2 = LinearSRGBColorSpace output,
  // +4 = equirectangular mapping.
  // transform2.w fractional lane stores 2D background blur amount / 4.
  transform2: vec4<f32>,
  inverse_view_projection: mat4x4<f32>,
  // xyz = camera world position.
  camera_params: vec4<f32>,
  rotation1: vec4<f32>,
  rotation2: vec4<f32>,
  rotation3: vec4<f32>,
};

@group(0) @binding(0)
var t_background: texture_2d<f32>;
@group(0) @binding(1)
var s_background: sampler;
@group(0) @binding(2)
var<uniform> uniforms: BackgroundUniforms;

struct BackgroundVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_background(@builtin(vertex_index) vertex_index: u32) -> BackgroundVertexOutput {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>( 3.0,  1.0),
    vec2<f32>(-1.0,  1.0),
  );
  let pos = positions[vertex_index];
  var out: BackgroundVertexOutput;
  out.position = vec4<f32>(pos, 0.0, 1.0);
  out.uv = pos * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
  return out;
}

fn transform_background_uv(uv: vec2<f32>) -> vec2<f32> {
  let uv1 = vec3<f32>(uv, 1.0);
  return vec2<f32>(dot(uniforms.transform1.xyz, uv1), dot(uniforms.transform2.xyz, uv1));
}

fn background_srgb_to_linear_channel(value: f32) -> f32 {
  if value <= 0.04045 {
    return value / 12.92;
  }
  return pow((value + 0.055) / 1.055, 2.4);
}

fn background_srgb_to_linear(color: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    background_srgb_to_linear_channel(color.r),
    background_srgb_to_linear_channel(color.g),
    background_srgb_to_linear_channel(color.b),
  );
}

fn background_texture_is_srgb() -> bool {
  return background_flag_enabled(1.0);
}

fn background_output_is_linear() -> bool {
  return background_flag_enabled(2.0);
}

fn background_texture_is_equirectangular() -> bool {
  return background_flag_enabled(4.0);
}

fn background_flag_enabled(bit: f32) -> bool {
  let flags = floor(uniforms.transform2.w);
  let scaled = floor(flags / bit);
  let flag = scaled - floor(scaled * 0.5) * 2.0;
  return flag > 0.5;
}

fn apply_background_output_color_space(color: vec3<f32>) -> vec3<f32> {
  if background_output_is_linear() {
    return color;
  }
  return pow(color, vec3<f32>(1.0 / 2.2));
}

fn background_blur_amount() -> f32 {
  return fract(uniforms.transform2.w) * 4.0;
}

fn equirect_background_uv(screen_uv: vec2<f32>) -> vec2<f32> {
  let ndc = vec2<f32>(screen_uv.x * 2.0 - 1.0, 1.0 - screen_uv.y * 2.0);
  let world = uniforms.inverse_view_projection * vec4<f32>(ndc, 1.0, 1.0);
  let world_pos = world.xyz / world.w;
  let dir = rotate_background_direction(normalize(world_pos - uniforms.camera_params.xyz));
  let equirect_uv = vec2<f32>(
    atan2(dir.z, dir.x) * 0.15915494309189535 + 0.5,
    asin(clamp(dir.y, -1.0, 1.0)) * 0.3183098861837907 + 0.5,
  );
  return transform_background_uv(equirect_uv);
}

fn rotate_background_direction(dir: vec3<f32>) -> vec3<f32> {
  return normalize(
    uniforms.rotation1.xyz * dir.x +
    uniforms.rotation2.xyz * dir.y +
    uniforms.rotation3.xyz * dir.z
  );
}

fn background_sample_uv(uv: vec2<f32>) -> vec2<f32> {
  if background_texture_is_equirectangular() {
    return equirect_background_uv(uv);
  }
  return transform_background_uv(uv);
}

fn sample_background(uv: vec2<f32>) -> vec4<f32> {
  let transformed_uv = background_sample_uv(uv);
  let blur = background_blur_amount();
  if blur <= 0.001 {
    return textureSample(t_background, s_background, transformed_uv);
  }

  let dimensions = vec2<f32>(textureDimensions(t_background, 0));
  let texel = vec2<f32>(1.0) / max(dimensions, vec2<f32>(1.0));
  let offset = texel * (1.0 + blur * 8.0);
  var color = textureSample(t_background, s_background, transformed_uv) * 0.25;
  color += textureSample(t_background, s_background, transformed_uv + vec2<f32>( offset.x, 0.0)) * 0.125;
  color += textureSample(t_background, s_background, transformed_uv + vec2<f32>(-offset.x, 0.0)) * 0.125;
  color += textureSample(t_background, s_background, transformed_uv + vec2<f32>(0.0,  offset.y)) * 0.125;
  color += textureSample(t_background, s_background, transformed_uv + vec2<f32>(0.0, -offset.y)) * 0.125;
  color += textureSample(t_background, s_background, transformed_uv + vec2<f32>( offset.x,  offset.y)) * 0.0625;
  color += textureSample(t_background, s_background, transformed_uv + vec2<f32>(-offset.x,  offset.y)) * 0.0625;
  color += textureSample(t_background, s_background, transformed_uv + vec2<f32>( offset.x, -offset.y)) * 0.0625;
  color += textureSample(t_background, s_background, transformed_uv + vec2<f32>(-offset.x, -offset.y)) * 0.0625;
  return color;
}

@fragment
fn fs_background(input: BackgroundVertexOutput) -> @location(0) vec4<f32> {
  let sample = sample_background(input.uv);
  var color = sample.rgb;
  if background_texture_is_srgb() {
    color = background_srgb_to_linear(color);
  }
  color *= uniforms.transform1.w;
  color = apply_background_output_color_space(color);
  return vec4<f32>(color, sample.a);
}
