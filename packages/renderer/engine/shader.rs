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
  light_space_matrix: mat4x4<f32>,
  // x = has_shadow, y = bias, z = normal_bias, w = receive_shadow
  shadow_params: vec4<f32>,
  // x = shadow light index (as f32), y = 1/map_size
  shadow_params2: vec4<f32>,
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
var s_ao: sampler;

@group(7) @binding(0)
var t_shadow: texture_depth_2d;
@group(7) @binding(1)
var s_shadow: sampler_comparison;

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

// Depth-only vertex shader used for the directional shadow map pass.
// Transforms the vertex by `light_space_matrix * model` so only the
// (orthographic) light-view depth is rasterized.
@vertex
fn vs_shadow(input: VertexInput) -> @builtin(position) vec4<f32> {
  let world_pos = uniforms.model * vec4<f32>(input.position, 1.0);
  return uniforms.light_space_matrix * world_pos;
}

// 3x3 PCF shadow sampling. Returns the fraction of samples NOT in shadow
// (i.e. 1.0 = fully lit, 0.0 = fully occluded).
fn sample_shadow(world_pos: vec3<f32>, world_normal: vec3<f32>) -> f32 {
  if uniforms.shadow_params.x < 0.5 || uniforms.shadow_params.w < 0.5 {
    return 1.0;
  }

  // Normal-offset bias: push the receiver position slightly along its normal
  // before transforming to light space to reduce shadow acne / peter-panning.
  let normal_bias = uniforms.shadow_params.z;
  let biased_pos = world_pos + world_normal * normal_bias;
  let light_ndc = uniforms.light_space_matrix * vec4<f32>(biased_pos, 1.0);
  // Orthographic projection: w is effectively 1.0, but divide anyway.
  let proj = light_ndc.xyz / light_ndc.w;

  // Outside the light's frustum: treat as fully lit.
  if proj.z > 1.0 || proj.z < 0.0 {
    return 1.0;
  }
  // Light space NDC [-1,1] to texture UV [0,1]; flip Y for texture coords.
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
      sum = sum + textureSampleCompareLevel(t_shadow, s_shadow, uv + offset, reference);
    }
  }
  return sum / 9.0;
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

// Schlick Fresnel approximation
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
  return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Schlick Fresnel with roughness for IBL
fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
  return f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
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

  // Normal mapping via TBN matrix
  var N = normalize(input.world_normal);
  // Flip normal when shading back-facing fragments (BackSide / DoubleSide).
  // For FrontSide meshes, back faces are culled so front_facing is always true.
  if !front_facing {
    N = -N;
  }
  if uniforms.normal_map_params.z > 0.5 {
    let T = normalize(input.world_tangent);
    let B = normalize(cross(N, T) * input.tangent_w);
    let tbn = mat3x3<f32>(T, B, N);
    let normal_sample = textureSample(t_normal, s_normal, uv).rgb;
    var tangent_normal = normal_sample * 2.0 - vec3<f32>(1.0);
    tangent_normal.x *= uniforms.normal_map_params.x;
    tangent_normal.y *= uniforms.normal_map_params.y;
    N = normalize(tbn * tangent_normal);
  }

  let V = normalize(uniforms.camera_pos.xyz - input.world_pos);
  let n_dot_v = max(dot(N, V), 0.0);

  // Dielectric F0 = 0.04, metallic F0 = albedo
  let f0 = mix(vec3<f32>(0.04), albedo, metallic);

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
      }

      let H = normalize(V + L);
      let n_dot_l = max(dot(N, L), 0.0);
      let n_dot_h = max(dot(N, H), 0.0);
      let h_dot_v = max(dot(H, V), 0.0);

      let radiance = light.color_intensity.rgb * light.color_intensity.w * attenuation;

      if use_specular {
        // Cook-Torrance BRDF (MeshStandardMaterial / MeshPhysicalMaterial)
        let D = distribution_ggx(n_dot_h, roughness);
        let G = geometry_smith(n_dot_v, n_dot_l, roughness);
        let F = fresnel_schlick(h_dot_v, f0);

        let specular = (D * G * F) / (4.0 * n_dot_v * n_dot_l + 0.0001);

        let k_s = F;
        let k_d = (vec3<f32>(1.0) - k_s) * (1.0 - metallic);

        lo = lo + (k_d * albedo / PI + specular) * radiance * n_dot_l;
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
        let specular_ibl = prefiltered_color * (F_ibl * brdf_sample.x + brdf_sample.y);

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
