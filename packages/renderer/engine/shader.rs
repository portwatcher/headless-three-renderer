pub const SHADER: &str = r#"
const PI: f32 = 3.14159265359;
const MAX_LIGHTS: u32 = 16u;

struct GpuLight {
  light_type: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  color_intensity: vec4<f32>,
  // xyz = position (point/spot) or ground_color (hemisphere), w = distance
  position: vec4<f32>,
  // xyz = direction, w = decay
  direction: vec4<f32>,
  // spot: x = cos(outer_angle), y = cos(inner_angle)
  params: vec4<f32>,
};

struct Uniforms {
  mvp: mat4x4<f32>,
  model: mat4x4<f32>,
  normal_matrix: mat4x4<f32>,
  camera_pos: vec4<f32>,
  base_color: vec4<f32>,
  emissive: vec4<f32>,  // xyz = emissive color, w = alpha test cutoff
  metallic: f32,
  roughness: f32,
  ambient_intensity: f32,
  num_lights: u32,
  ambient_color: vec4<f32>,
  // x = normal_scale.x, y = normal_scale.y, z = has_normal_map (1.0 or 0.0), w = has_ibl (1.0 or 0.0)
  normal_map_params: vec4<f32>,
  // x = env_intensity, y = shading_model (0=standard PBR, 1=basic/unlit, 2=lambert)
  ibl_params: vec4<f32>,
  // x = ao_map_intensity, y = has_ao_map (1.0 or 0.0)
  ao_params: vec4<f32>,
  // x = 1/width, y = 1/height, z = width, w = height
  render_params: vec4<f32>,
  light_space_matrices: array<mat4x4<f32>, 6>,
  // x = has_shadow, y = bias, z = normal_bias, w = receive_shadow
  shadow_params: vec4<f32>,
  // x = shadow light index (as f32), y = 1/map_size, z = shadow kind (0=2D, 1=point, 2=cascaded), w = layer count
  shadow_params2: vec4<f32>,
  // xyz/w = cascade split distances; missing splits are large sentinel values
  shadow_params3: vec4<f32>,
  // x = clearcoat, y = clearcoat roughness, z = transmission, w = ior
  physical_params1: vec4<f32>,
  // xyz = sheen color, w = sheen roughness
  physical_params2: vec4<f32>,
  // x = anisotropy, y = anisotropy rotation, z = thickness, w = attenuation distance
  physical_params3: vec4<f32>,
  // x/y = clearcoat normal scale
  physical_params4: vec4<f32>,
  // xyz = attenuation color
  attenuation_color: vec4<f32>,
  lights: array<GpuLight, 16>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(1) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(1) @binding(1)
var s_diffuse: sampler;

@group(2) @binding(0)
var t_normal: texture_2d<f32>;
@group(2) @binding(1)
var s_normal: sampler;

@group(3) @binding(0)
var t_metallic_roughness: texture_2d<f32>;
@group(3) @binding(1)
var s_metallic_roughness: sampler;

@group(4) @binding(0)
var t_emissive: texture_2d<f32>;
@group(4) @binding(1)
var s_emissive: sampler;

@group(5) @binding(0)
var t_irradiance: texture_cube<f32>;
@group(5) @binding(1)
var t_prefilter: texture_cube<f32>;
@group(5) @binding(2)
var t_brdf_lut: texture_2d<f32>;
@group(5) @binding(3)
var s_ibl: sampler;

@group(6) @binding(0)
var t_ao: texture_2d<f32>;
@group(6) @binding(1)
var t_physical_scalar: texture_2d<f32>;
@group(6) @binding(2)
var t_physical_sheen: texture_2d<f32>;
@group(6) @binding(3)
var t_physical_anisotropy: texture_2d<f32>;
@group(6) @binding(4)
var t_clearcoat_normal: texture_2d<f32>;
@group(6) @binding(5)
var s_ao: sampler;

@group(7) @binding(0)
var t_shadow: texture_depth_2d_array;
@group(7) @binding(1)
var s_shadow: sampler_comparison;
@group(7) @binding(2)
var t_scene_color: texture_2d<f32>;
@group(7) @binding(3)
var s_scene_color: sampler;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) tangent: vec4<f32>,
  @location(3) color: vec4<f32>,
  @location(4) uv: vec2<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) world_pos: vec3<f32>,
  @location(1) world_normal: vec3<f32>,
  @location(2) world_tangent: vec3<f32>,
  @location(3) tangent_w: f32,
  @location(4) color: vec4<f32>,
  @location(5) uv: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  let world_pos = uniforms.model * vec4<f32>(input.position, 1.0);
  output.position = uniforms.mvp * vec4<f32>(input.position, 1.0);
  output.world_pos = world_pos.xyz;
  output.world_normal = normalize((uniforms.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz);
  output.world_tangent = normalize((uniforms.model * vec4<f32>(input.tangent.xyz, 0.0)).xyz);
  output.tangent_w = input.tangent.w;
  output.color = input.color;
  output.uv = input.uv;
  return output;
}

fn shadow_position(input: VertexInput, layer: u32) -> vec4<f32> {
  let world_pos = uniforms.model * vec4<f32>(input.position, 1.0);
  return uniforms.light_space_matrices[layer] * world_pos;
}

@vertex
fn vs_shadow0(input: VertexInput) -> @builtin(position) vec4<f32> {
  return shadow_position(input, 0u);
}

@vertex
fn vs_shadow1(input: VertexInput) -> @builtin(position) vec4<f32> {
  return shadow_position(input, 1u);
}

@vertex
fn vs_shadow2(input: VertexInput) -> @builtin(position) vec4<f32> {
  return shadow_position(input, 2u);
}

@vertex
fn vs_shadow3(input: VertexInput) -> @builtin(position) vec4<f32> {
  return shadow_position(input, 3u);
}

@vertex
fn vs_shadow4(input: VertexInput) -> @builtin(position) vec4<f32> {
  return shadow_position(input, 4u);
}

@vertex
fn vs_shadow5(input: VertexInput) -> @builtin(position) vec4<f32> {
  return shadow_position(input, 5u);
}

fn sample_shadow_layer(world_pos: vec3<f32>, layer: u32, world_normal: vec3<f32>) -> f32 {
  let biased_pos = world_pos + world_normal * uniforms.shadow_params.z;
  let light_ndc = uniforms.light_space_matrices[layer] * vec4<f32>(biased_pos, 1.0);
  let proj = light_ndc.xyz / light_ndc.w;

  if proj.z > 1.0 || proj.z < 0.0 {
    return 1.0;
  }
  let uv = vec2<f32>(proj.x * 0.5 + 0.5, -proj.y * 0.5 + 0.5);
  if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
    return 1.0;
  }

  let reference = proj.z - uniforms.shadow_params.y;
  let texel = uniforms.shadow_params2.y;

  // 3x3 PCF.
  var sum: f32 = 0.0;
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      let offset = vec2<f32>(f32(dx), f32(dy)) * texel;
      sum = sum + textureSampleCompareLevel(t_shadow, s_shadow, uv + offset, layer, reference);
    }
  }
  return sum / 9.0;
}

fn point_shadow_layer(light_vec: vec3<f32>) -> u32 {
  let abs_vec = abs(light_vec);
  if abs_vec.x >= abs_vec.y && abs_vec.x >= abs_vec.z {
    return select(1u, 0u, light_vec.x >= 0.0);
  }
  if abs_vec.y >= abs_vec.x && abs_vec.y >= abs_vec.z {
    return select(3u, 2u, light_vec.y >= 0.0);
  }
  return select(5u, 4u, light_vec.z >= 0.0);
}

// 3x3 PCF shadow sampling. Returns the fraction of samples NOT in shadow
// (i.e. 1.0 = fully lit, 0.0 = fully occluded).
fn sample_shadow(world_pos: vec3<f32>, world_normal: vec3<f32>) -> f32 {
  if uniforms.shadow_params.x < 0.5 || uniforms.shadow_params.w < 0.5 {
    return 1.0;
  }
  if uniforms.shadow_params2.z > 0.5 {
    if uniforms.shadow_params2.z > 1.5 {
      let camera_dist = distance(world_pos, uniforms.camera_pos.xyz);
      var layer = 0u;
      if camera_dist > uniforms.shadow_params3.x {
        layer = 1u;
      }
      if camera_dist > uniforms.shadow_params3.y {
        layer = 2u;
      }
      if camera_dist > uniforms.shadow_params3.z {
        layer = 3u;
      }
      layer = min(layer, max(u32(uniforms.shadow_params2.w), 1u) - 1u);
      return sample_shadow_layer(world_pos, layer, world_normal);
    }
    let light_index = u32(uniforms.shadow_params2.x);
    let light_pos = uniforms.lights[light_index].position.xyz;
    let layer = point_shadow_layer(world_pos - light_pos);
    return sample_shadow_layer(world_pos, layer, world_normal);
  }
  return sample_shadow_layer(world_pos, 0u, world_normal);
}

// GGX/Trowbridge-Reitz normal distribution
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d + 0.0001);
}

// Schlick-GGX geometry function
fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
  let r = roughness + 1.0;
  let k = (r * r) / 8.0;
  return n_dot_v / (n_dot_v * (1.0 - k) + k + 0.0001);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
  return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

fn geometry_smith_correlated_anisotropic(
  alpha_t: f32,
  alpha_b: f32,
  dot_tv: f32,
  dot_bv: f32,
  dot_tl: f32,
  dot_bl: f32,
  n_dot_v: f32,
  n_dot_l: f32,
) -> f32 {
  let gv = n_dot_l * length(vec3<f32>(alpha_t * dot_tv, alpha_b * dot_bv, n_dot_v));
  let gl = n_dot_v * length(vec3<f32>(alpha_t * dot_tl, alpha_b * dot_bl, n_dot_l));
  return 0.5 / max(gv + gl, 0.0001);
}

fn distribution_ggx_anisotropic(
  alpha_t: f32,
  alpha_b: f32,
  n_dot_h: f32,
  dot_th: f32,
  dot_bh: f32,
) -> f32 {
  let a2 = alpha_t * alpha_b;
  let v = vec3<f32>(alpha_b * dot_th, alpha_t * dot_bh, a2 * n_dot_h);
  let v2 = max(dot(v, v), 0.0001);
  let w2 = a2 / v2;
  return (a2 * w2 * w2) / PI;
}

// Schlick Fresnel approximation
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
  return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Schlick Fresnel with roughness for IBL
fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
  return f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Estevez/Kulla Charlie distribution and Neubelt visibility for cloth sheen.
fn distribution_charlie(roughness: f32, n_dot_h: f32) -> f32 {
  let alpha = max(roughness * roughness, 0.0001);
  let inv_alpha = 1.0 / alpha;
  let cos2h = n_dot_h * n_dot_h;
  let sin2h = max(1.0 - cos2h, 0.0078125);
  return (2.0 + inv_alpha) * pow(sin2h, inv_alpha * 0.5) / (2.0 * PI);
}

fn visibility_neubelt(n_dot_v: f32, n_dot_l: f32) -> f32 {
  return saturate(1.0 / max(4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v), 0.0001));
}

fn brdf_sheen(
  L: vec3<f32>,
  V: vec3<f32>,
  N: vec3<f32>,
  sheen_color: vec3<f32>,
  sheen_roughness: f32,
) -> vec3<f32> {
  let H = normalize(V + L);
  let n_dot_l = max(dot(N, L), 0.0);
  let n_dot_v = max(dot(N, V), 0.0);
  let n_dot_h = max(dot(N, H), 0.0);
  let D = distribution_charlie(sheen_roughness, n_dot_h);
  let Vis = visibility_neubelt(n_dot_v, n_dot_l);
  return sheen_color * (D * Vis);
}

fn ibl_sheen_brdf(N: vec3<f32>, V: vec3<f32>, roughness: f32) -> f32 {
  let n_dot_v = max(dot(N, V), 0.0);
  let r2 = roughness * roughness;
  let r_inv = 1.0 / (roughness + 0.1);
  let a = -1.9362 + 1.0678 * roughness + 0.4573 * r2 - 0.8469 * r_inv;
  let b = -0.6014 + 0.5538 * roughness - 0.4670 * r2 - 0.1255 * r_inv;
  return saturate(exp(a * n_dot_v + b));
}

fn tangent_basis(N: vec3<f32>, tangent: vec3<f32>, tangent_w: f32) -> mat3x3<f32> {
  var T = tangent;
  if dot(T, T) < 0.0001 {
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(N.y) > 0.999);
    T = normalize(cross(up, N));
  } else {
    T = normalize(T - N * dot(N, T));
  }
  let B = normalize(cross(N, T) * select(1.0, tangent_w, abs(tangent_w) > 0.0));
  return mat3x3<f32>(T, B, N);
}

fn volume_attenuation(distance: f32, attenuation_color: vec3<f32>, attenuation_distance: f32) -> vec3<f32> {
  if attenuation_distance >= 1.0e19 {
    return vec3<f32>(1.0);
  }
  let safe_color = max(attenuation_color, vec3<f32>(0.0001));
  let coefficient = -log(safe_color) / max(attenuation_distance, 0.0001);
  return exp(-coefficient * distance);
}

// Three.js-compatible distance attenuation
fn get_distance_attenuation(light_distance: f32, cutoff_distance: f32, decay_exponent: f32) -> f32 {
  var falloff = 1.0 / max(pow(light_distance, decay_exponent), 0.01);
  if cutoff_distance > 0.0 {
    let ratio = light_distance / cutoff_distance;
    let ratio2 = ratio * ratio;
    let ratio4 = ratio2 * ratio2;
    let window = saturate(1.0 - ratio4);
    falloff *= window * window;
  }
  return falloff;
}

// Three.js-compatible spot attenuation
fn get_spot_attenuation(cone_cos: f32, penumbra_cos: f32, angle_cos: f32) -> f32 {
  return smoothstep(cone_cos, penumbra_cos, angle_cos);
}

@fragment
fn fs_main(input: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
  let uv = vec2<f32>(input.uv.x, 1.0 - input.uv.y);
  let screen_uv = clamp(input.position.xy * uniforms.render_params.xy, vec2<f32>(0.0), vec2<f32>(1.0));
  let tex_color = textureSample(t_diffuse, s_diffuse, uv);
  let albedo = tex_color.rgb * input.color.rgb * uniforms.base_color.rgb;
  let alpha = tex_color.a * input.color.a * uniforms.base_color.a;

  // Alpha test: discard fragments below the cutoff threshold
  let alpha_cutoff = uniforms.emissive.w;
  if alpha_cutoff > 0.0 && alpha < alpha_cutoff {
    discard;
  }

  let shading_model = u32(uniforms.ibl_params.y);

  // Ambient occlusion: sample red channel, blend toward 1.0 by intensity.
  // Matches three.js: ao = (texture.r - 1.0) * aoMapIntensity + 1.0
  var ao: f32 = 1.0;
  if uniforms.ao_params.y > 0.5 {
    let ao_sample = textureSample(t_ao, s_ao, uv).r;
    ao = (ao_sample - 1.0) * uniforms.ao_params.x + 1.0;
  }

  // MeshBasicMaterial: unlit. Output = albedo * ao, then emissive + tone map + gamma.
  if shading_model == 1u {
    var unlit = albedo * ao;
    let emissive_basic = textureSample(t_emissive, s_emissive, uv).rgb;
    unlit = unlit + uniforms.emissive.rgb * emissive_basic;
    let mapped_basic = aces_filmic_tone_mapping(unlit);
    let gamma_basic = pow(mapped_basic, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(gamma_basic, alpha);
  }

  let use_specular = shading_model == 0u;

  let mr_sample = textureSample(t_metallic_roughness, s_metallic_roughness, uv);
  let metallic = uniforms.metallic * mr_sample.b;
  let roughness = max(uniforms.roughness * mr_sample.g, 0.04);
  let physical_scalar_sample = textureSample(t_physical_scalar, s_ao, uv);
  let physical_sheen_sample = textureSample(t_physical_sheen, s_ao, uv);
  let physical_anisotropy_sample = textureSample(t_physical_anisotropy, s_ao, uv);
  let clearcoat = clamp(uniforms.physical_params1.x * physical_scalar_sample.r, 0.0, 1.0);
  let clearcoat_roughness = max(uniforms.physical_params1.y * physical_scalar_sample.g, 0.0525);
  let transmission = clamp(uniforms.physical_params1.z * physical_scalar_sample.b, 0.0, 1.0);
  let ior = clamp(uniforms.physical_params1.w, 1.0, 2.333);
  let sheen_color = clamp(uniforms.physical_params2.rgb * physical_sheen_sample.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let sheen_roughness = clamp(uniforms.physical_params2.w * physical_sheen_sample.a, 0.0001, 1.0);
  let anisotropy = clamp(uniforms.physical_params3.x, 0.0, 1.0);
  let anisotropy_rotation = uniforms.physical_params3.y;
  let thickness = max(uniforms.physical_params3.z * physical_scalar_sample.a, 0.0);
  let attenuation_distance = max(uniforms.physical_params3.w, 0.0);

  // Normal mapping via TBN matrix
  var N = normalize(input.world_normal);
  // Flip normal when shading back-facing fragments (BackSide / DoubleSide).
  // For FrontSide meshes, back faces are culled so front_facing is always true.
  if !front_facing {
    N = -N;
  }
  var tbn = tangent_basis(N, input.world_tangent, input.tangent_w);
  if uniforms.normal_map_params.z > 0.5 {
    let normal_sample = textureSample(t_normal, s_normal, uv).rgb;
    var tangent_normal = normal_sample * 2.0 - vec3<f32>(1.0);
    tangent_normal.x *= uniforms.normal_map_params.x;
    tangent_normal.y *= uniforms.normal_map_params.y;
    N = normalize(tbn * tangent_normal);
    tbn = tangent_basis(N, tbn[0], input.tangent_w);
  }
  let T = normalize(tbn[0]);
  let B = normalize(tbn[1]);
  let clearcoat_normal_sample = textureSample(t_clearcoat_normal, s_ao, uv).rgb;
  var clearcoat_tangent_normal = clearcoat_normal_sample * 2.0 - vec3<f32>(1.0);
  clearcoat_tangent_normal.x *= uniforms.physical_params4.x;
  clearcoat_tangent_normal.y *= uniforms.physical_params4.y;
  let Ncc = normalize(tbn * clearcoat_tangent_normal);
  let anisotropy_map_raw = physical_anisotropy_sample.rg * 2.0 - vec2<f32>(1.0);
  let anisotropy_map_dir = select(
    vec2<f32>(1.0, 0.0),
    normalize(anisotropy_map_raw),
    dot(anisotropy_map_raw, anisotropy_map_raw) > 0.0001,
  );
  let anisotropy_rot_c = cos(anisotropy_rotation);
  let anisotropy_rot_s = sin(anisotropy_rotation);
  let anisotropy_vec = vec2<f32>(
    anisotropy_rot_c * anisotropy_map_dir.x - anisotropy_rot_s * anisotropy_map_dir.y,
    anisotropy_rot_s * anisotropy_map_dir.x + anisotropy_rot_c * anisotropy_map_dir.y,
  ) * anisotropy * physical_anisotropy_sample.b;
  let anisotropy_strength = clamp(length(anisotropy_vec), 0.0, 1.0);
  let anisotropy_dir = select(
    vec2<f32>(1.0, 0.0),
    anisotropy_vec / max(anisotropy_strength, 0.0001),
    anisotropy_strength > 0.0001,
  );

  let V = normalize(uniforms.camera_pos.xyz - input.world_pos);
  let n_dot_v = max(dot(N, V), 0.0);

  // Dielectric F0 from IOR (1.5 -> 0.04), metallic F0 = albedo.
  let dielectric_f0_scalar = pow((ior - 1.0) / (ior + 1.0), 2.0);
  let f0 = mix(vec3<f32>(dielectric_f0_scalar), albedo, metallic);

  var lo = vec3<f32>(0.0);

  let has_ibl = uniforms.normal_map_params.w > 0.5;

  if uniforms.num_lights == 0u && !has_ibl {
    // No lights or IBL: render with a basic hemispherical ambient
    let ambient = uniforms.ambient_color.rgb * uniforms.ambient_intensity;
    let sky_factor = 0.5 + 0.5 * N.y;
    let fallback_ambient = mix(vec3<f32>(0.1, 0.1, 0.12), vec3<f32>(0.4, 0.45, 0.5), sky_factor);
    let total_ambient = max(ambient, fallback_ambient);
    lo = albedo * total_ambient * ao;
  } else {
    // Direct lighting from scene lights
    let shadow_factor = sample_shadow(input.world_pos, N);
    let shadow_light_index = u32(uniforms.shadow_params2.x);
    for (var i = 0u; i < uniforms.num_lights && i < MAX_LIGHTS; i = i + 1u) {
      let light = uniforms.lights[i];

      if light.light_type == 3u {
        // Hemisphere light
        let up = normalize(light.direction.xyz);
        let sky_color = light.color_intensity.rgb * light.color_intensity.w;
        let ground_color = light.position.xyz * light.color_intensity.w;
        let hemi_factor = 0.5 + 0.5 * dot(N, up);
        lo = lo + albedo * mix(ground_color, sky_color, hemi_factor);
        continue;
      }

      var L: vec3<f32>;
      var attenuation: f32 = 1.0;

      if light.light_type == 0u {
        // Directional
        L = normalize(-light.direction.xyz);
        if i == shadow_light_index {
          attenuation *= shadow_factor;
        }
      } else {
        // Point or Spot
        let light_vec = light.position.xyz - input.world_pos;
        let dist = length(light_vec);
        L = light_vec / max(dist, 0.0001);
        let cutoff_distance = light.position.w;
        let decay_exponent = light.direction.w;
        attenuation = get_distance_attenuation(dist, cutoff_distance, decay_exponent);

        // Spot cone attenuation
        if light.light_type == 2u {
          let cos_angle = dot(normalize(-light_vec), normalize(light.direction.xyz));
          let cone_cos = light.params.x;
          let penumbra_cos = light.params.y;
          attenuation *= get_spot_attenuation(cone_cos, penumbra_cos, cos_angle);
        }
        if i == shadow_light_index {
          attenuation *= shadow_factor;
        }
      }

      let H = normalize(V + L);
      let n_dot_l = max(dot(N, L), 0.0);
      let n_dot_h = max(dot(N, H), 0.0);
      let h_dot_v = max(dot(H, V), 0.0);

      let radiance = light.color_intensity.rgb * light.color_intensity.w * attenuation;

      if use_specular {
        // Cook-Torrance BRDF (MeshStandardMaterial / MeshPhysicalMaterial)
        var D = distribution_ggx(n_dot_h, roughness);
        var G = geometry_smith(n_dot_v, n_dot_l, roughness);
        if anisotropy_strength > 0.0001 {
          let anisotropy_t = normalize(T * anisotropy_dir.x + B * anisotropy_dir.y);
          let anisotropy_b = normalize(B * anisotropy_dir.x - T * anisotropy_dir.y);
          let alpha_b = max(roughness * roughness, 0.0001);
          let alpha_t = mix(alpha_b, 1.0, anisotropy_strength * anisotropy_strength);
          D = distribution_ggx_anisotropic(
            alpha_t,
            alpha_b,
            n_dot_h,
            dot(anisotropy_t, H),
            dot(anisotropy_b, H),
          );
          G = geometry_smith_correlated_anisotropic(
            alpha_t,
            alpha_b,
            dot(anisotropy_t, V),
            dot(anisotropy_b, V),
            dot(anisotropy_t, L),
            dot(anisotropy_b, L),
            n_dot_v,
            n_dot_l,
          );
        }
        let F = fresnel_schlick(h_dot_v, f0);

        let specular = (D * G * F) / (4.0 * n_dot_v * n_dot_l + 0.0001);

        let k_s = F;
        let k_d = (vec3<f32>(1.0) - k_s) * (1.0 - metallic);

        var physical_specular = specular;
        if clearcoat > 0.0001 {
          let cc_n_dot_v = max(dot(Ncc, V), 0.0);
          let cc_n_dot_l = max(dot(Ncc, L), 0.0);
          let cc_n_dot_h = max(dot(Ncc, H), 0.0);
          let Dcc = distribution_ggx(cc_n_dot_h, clearcoat_roughness);
          let Gcc = geometry_smith(cc_n_dot_v, cc_n_dot_l, clearcoat_roughness);
          let Fcc = fresnel_schlick(h_dot_v, vec3<f32>(0.04));
          physical_specular += clearcoat * (Dcc * Gcc * Fcc) / (4.0 * cc_n_dot_v * cc_n_dot_l + 0.0001);
        }
        if max(max(sheen_color.r, sheen_color.g), sheen_color.b) > 0.0001 {
          physical_specular += brdf_sheen(L, V, N, sheen_color, sheen_roughness);
        }

        lo = lo + (k_d * albedo / PI + physical_specular) * radiance * n_dot_l;
      } else {
        // MeshLambertMaterial: diffuse-only
        lo = lo + albedo / PI * radiance * n_dot_l;
      }
    }

    // Image-Based Lighting (split-sum approximation)
    if has_ibl {
      let env_intensity = uniforms.ibl_params.x;

      // Diffuse IBL: irradiance cubemap
      let irradiance = textureSample(t_irradiance, s_ibl, N).rgb;

      if use_specular {
        let F_ibl = fresnel_schlick_roughness(n_dot_v, f0, roughness);
        let k_s_ibl = F_ibl;
        let k_d_ibl = (vec3<f32>(1.0) - k_s_ibl) * (1.0 - metallic);
        let diffuse_ibl = k_d_ibl * irradiance * albedo;

        // Specular IBL: prefiltered env map + BRDF LUT
        let R = reflect(-V, N);
        let max_lod = 4.0; // PREFILTER_MIP_LEVELS - 1
        let prefiltered_color = textureSampleLevel(t_prefilter, s_ibl, R, roughness * max_lod).rgb;
        let brdf_sample = textureSample(t_brdf_lut, s_ibl, vec2<f32>(n_dot_v, roughness)).rg;
        var specular_ibl = prefiltered_color * (F_ibl * brdf_sample.x + brdf_sample.y);

        if clearcoat > 0.0001 {
          let Rcc = reflect(-V, Ncc);
          let cc_n_dot_v = max(dot(Ncc, V), 0.0);
          let clearcoat_radiance = textureSampleLevel(t_prefilter, s_ibl, Rcc, clearcoat_roughness * max_lod).rgb;
          let brdf_clearcoat = textureSample(t_brdf_lut, s_ibl, vec2<f32>(cc_n_dot_v, clearcoat_roughness)).rg;
          let F_clearcoat = fresnel_schlick_roughness(cc_n_dot_v, vec3<f32>(0.04), clearcoat_roughness);
          specular_ibl += clearcoat * clearcoat_radiance * (F_clearcoat * brdf_clearcoat.x + brdf_clearcoat.y);
        }
        if max(max(sheen_color.r, sheen_color.g), sheen_color.b) > 0.0001 {
          specular_ibl += irradiance * sheen_color * ibl_sheen_brdf(N, V, sheen_roughness) / PI;
        }

        lo = lo + (diffuse_ibl + specular_ibl) * env_intensity * ao;
      } else {
        // Lambert: diffuse IBL only
        lo = lo + irradiance * albedo * env_intensity * ao;
      }
    } else {
      // Ambient (non-IBL fallback when lights are present)
      let ambient = uniforms.ambient_color.rgb * uniforms.ambient_intensity * albedo;
      lo = lo + ambient * ao;
    }
  }

  if use_specular && transmission > 0.0001 {
    let refracted_dir = refract(-V, N, 1.0 / ior);
    let transmittance = volume_attenuation(thickness, uniforms.attenuation_color.rgb, attenuation_distance);
    let scene_offset = refracted_dir.xy * thickness * 0.04;
    let scene_uv = clamp(screen_uv + scene_offset, vec2<f32>(0.0), vec2<f32>(1.0));
    var transmitted_light = textureSample(t_scene_color, s_scene_color, scene_uv).rgb * transmittance;
    if has_ibl {
      let max_lod = 4.0;
      let environment_refraction = textureSampleLevel(t_prefilter, s_ibl, refracted_dir, roughness * max_lod).rgb * transmittance;
      transmitted_light = mix(transmitted_light, environment_refraction, 0.35);
    }
    lo = mix(lo, transmitted_light, transmission);
  }

  // Emissive
  let emissive_sample = textureSample(t_emissive, s_emissive, uv).rgb;
  lo = lo + uniforms.emissive.rgb * emissive_sample;

  // Tone mapping (ACES Filmic, matches three.js) and gamma correction
  let mapped = aces_filmic_tone_mapping(lo);
  let gamma_corrected = pow(mapped, vec3<f32>(1.0 / 2.2));

  return vec4<f32>(gamma_corrected, alpha);
}

// ACES Filmic tone mapping, ported from three.js (Narkowicz fit with
// input/output matrices). Includes the 1/0.6 exposure pre-scale that
// three.js applies so output matches ACESFilmicToneMapping there.
fn rrt_and_odt_fit(v: vec3<f32>) -> vec3<f32> {
  let a = v * (v + vec3<f32>(0.0245786)) - vec3<f32>(0.000090537);
  let b = v * (0.983729 * v + vec3<f32>(0.4329510)) + vec3<f32>(0.238081);
  return a / b;
}

fn aces_filmic_tone_mapping(color_in: vec3<f32>) -> vec3<f32> {
  // WGSL mat3x3 constructor takes columns.
  let aces_input = mat3x3<f32>(
    vec3<f32>(0.59719, 0.07600, 0.02840),
    vec3<f32>(0.35458, 0.90834, 0.13383),
    vec3<f32>(0.04823, 0.01566, 0.83777),
  );
  let aces_output = mat3x3<f32>(
    vec3<f32>( 1.60475, -0.10208, -0.00327),
    vec3<f32>(-0.53108,  1.10813, -0.07276),
    vec3<f32>(-0.07367, -0.00605,  1.07602),
  );
  var color = color_in * (1.0 / 0.6);
  color = aces_input * color;
  color = rrt_and_odt_fit(color);
  color = aces_output * color;
  return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
}
"#;

pub fn custom_shader_source(fragment_body: &str) -> String {
    CUSTOM_FRAGMENT_SHADER.replace("__CUSTOM_FRAGMENT_BODY__", fragment_body)
}

const CUSTOM_FRAGMENT_SHADER: &str = r#"
const MAX_LIGHTS: u32 = 16u;

struct GpuLight {
  light_type: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  color_intensity: vec4<f32>,
  position: vec4<f32>,
  direction: vec4<f32>,
  params: vec4<f32>,
};

struct Uniforms {
  mvp: mat4x4<f32>,
  model: mat4x4<f32>,
  normal_matrix: mat4x4<f32>,
  camera_pos: vec4<f32>,
  base_color: vec4<f32>,
  emissive: vec4<f32>,
  metallic: f32,
  roughness: f32,
  ambient_intensity: f32,
  num_lights: u32,
  ambient_color: vec4<f32>,
  normal_map_params: vec4<f32>,
  ibl_params: vec4<f32>,
  ao_params: vec4<f32>,
  render_params: vec4<f32>,
  light_space_matrices: array<mat4x4<f32>, 6>,
  shadow_params: vec4<f32>,
  shadow_params2: vec4<f32>,
  shadow_params3: vec4<f32>,
  physical_params1: vec4<f32>,
  physical_params2: vec4<f32>,
  physical_params3: vec4<f32>,
  physical_params4: vec4<f32>,
  attenuation_color: vec4<f32>,
  lights: array<GpuLight, 16>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(1) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(1) @binding(1)
var s_diffuse: sampler;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) tangent: vec4<f32>,
  @location(3) color: vec4<f32>,
  @location(4) uv: vec2<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) world_pos: vec3<f32>,
  @location(1) world_normal: vec3<f32>,
  @location(2) world_tangent: vec3<f32>,
  @location(3) tangent_w: f32,
  @location(4) color: vec4<f32>,
  @location(5) uv: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  let world_pos = uniforms.model * vec4<f32>(input.position, 1.0);
  output.position = uniforms.mvp * vec4<f32>(input.position, 1.0);
  output.world_pos = world_pos.xyz;
  output.world_normal = normalize((uniforms.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz);
  output.world_tangent = normalize((uniforms.model * vec4<f32>(input.tangent.xyz, 0.0)).xyz);
  output.tangent_w = input.tangent.w;
  output.color = input.color;
  output.uv = input.uv;
  return output;
}

@fragment
fn fs_main(input: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
  let uv = vec2<f32>(input.uv.x, 1.0 - input.uv.y);
  let texture_color = textureSample(t_diffuse, s_diffuse, uv);
  let base_color = texture_color * input.color * uniforms.base_color;
  let alpha = base_color.a;
  let alpha_cutoff = uniforms.emissive.w;
  if alpha_cutoff > 0.0 && alpha < alpha_cutoff {
    discard;
  }
  var normal = normalize(input.world_normal);
  if !front_facing {
    normal = -normal;
  }
  __CUSTOM_FRAGMENT_BODY__
}
"#;

pub const POST_SHADER: &str = r#"
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
    let vignette = mix(1.0, smoothstep(1.35, 0.15, radius), clamp(uniforms.params1.w, 0.0, 1.0));
    color *= vignette;
  }

  color = mix(color, vec3<f32>(1.0) - color, clamp(uniforms.params2.y, 0.0, 1.0));
  return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), sample.a);
}
"#;
