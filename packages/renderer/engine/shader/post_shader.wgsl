struct PostUniforms {
  // x = exposure stops, y = contrast, z = saturation, w = vignette
  params1: vec4<f32>,
  // x = grayscale blend, y = invert blend
  params2: vec4<f32>,
};

@group(0) @binding(0)
var t_color: texture_2d<f32>;
@group(0) @binding(1)
var s_color: sampler;
@group(0) @binding(2)
var<uniform> uniforms: PostUniforms;

struct PostVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_post(@builtin(vertex_index) vertex_index: u32) -> PostVertexOutput {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>( 3.0,  1.0),
    vec2<f32>(-1.0,  1.0),
  );
  let pos = positions[vertex_index];
  var out: PostVertexOutput;
  out.position = vec4<f32>(pos, 0.0, 1.0);
  out.uv = pos * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
  return out;
}

@fragment
fn fs_post(input: PostVertexOutput) -> @location(0) vec4<f32> {
  let sample = textureSample(t_color, s_color, input.uv);
  var color = sample.rgb;

  color = color * exp2(uniforms.params1.x);
  color = (color - vec3<f32>(0.5)) * max(uniforms.params1.y, 0.0) + vec3<f32>(0.5);

  let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
  color = mix(vec3<f32>(luma), color, max(uniforms.params1.z, 0.0));
  color = mix(color, vec3<f32>(luma), clamp(uniforms.params2.x, 0.0, 1.0));

  if uniforms.params1.w > 0.0 {
    let centered = input.uv * 2.0 - vec2<f32>(1.0);
    let radius = dot(centered, centered);
    let edge = clamp((radius - 0.15) / 1.2, 0.0, 1.0);
    let falloff = edge * edge * (3.0 - 2.0 * edge);
    let vignette = mix(1.0, 1.0 - falloff, clamp(uniforms.params1.w, 0.0, 1.0));
    color *= vignette;
  }

  color = mix(color, vec3<f32>(1.0) - color, clamp(uniforms.params2.y, 0.0, 1.0));
  return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), sample.a);
}
