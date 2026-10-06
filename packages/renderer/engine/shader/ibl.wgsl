// Image-based lighting and specular BRDF terms ported from Three.js r180 (MIT):
// bsdfs, lights_physical_pars_fragment, envmap_physical_pars_fragment, and
// cube_uv_reflection_fragment. The CubeUV atlas comes from the CPU port of PMREMGenerator.

const RECIPROCAL_PI: f32 = 0.3183098861837907;
const BRDF_EPSILON: f32 = 1e-6;
const CUBE_UV_MIN_MIP_LEVEL: f32 = 4.0;
const CUBE_UV_MIN_TILE_SIZE: f32 = 16.0;

fn f_schlick(f0: vec3<f32>, f90: f32, dot_vh: f32) -> vec3<f32> {
  // Optimized variant (presented by Epic at SIGGRAPH '13), as in Three.js F_Schlick.
  let fresnel = exp2((-5.55473 * dot_vh - 6.98316) * dot_vh);
  return f0 * (1.0 - fresnel) + vec3<f32>(f90 * fresnel);
}

fn f_schlick_scalar(f0: f32, f90: f32, dot_vh: f32) -> f32 {
  let fresnel = exp2((-5.55473 * dot_vh - 6.98316) * dot_vh);
  return f0 * (1.0 - fresnel) + f90 * fresnel;
}

fn v_ggx_smith_correlated(alpha: f32, dot_nl: f32, dot_nv: f32) -> f32 {
  let a2 = alpha * alpha;
  let gv = dot_nl * sqrt(a2 + (1.0 - a2) * dot_nv * dot_nv);
  let gl = dot_nv * sqrt(a2 + (1.0 - a2) * dot_nl * dot_nl);
  return 0.5 / max(gv + gl, BRDF_EPSILON);
}

fn d_ggx(alpha: f32, dot_nh: f32) -> f32 {
  let a2 = alpha * alpha;
  let denom = dot_nh * dot_nh * (a2 - 1.0) + 1.0;
  return RECIPROCAL_PI * a2 / (denom * denom);
}

// Three.js BRDF_GGX without anisotropy: F * (V * D), alpha = roughness^2.
fn brdf_ggx(L: vec3<f32>, V: vec3<f32>, N: vec3<f32>, f0: vec3<f32>, f90: f32, roughness: f32) -> vec3<f32> {
  let alpha = roughness * roughness;
  let H = normalize(L + V);
  let dot_nl = saturate(dot(N, L));
  let dot_nv = saturate(dot(N, V));
  let dot_nh = saturate(dot(N, H));
  let dot_vh = saturate(dot(V, H));
  return f_schlick(f0, f90, dot_vh) * (v_ggx_smith_correlated(alpha, dot_nl, dot_nv) * d_ggx(alpha, dot_nh));
}

fn v_ggx_smith_correlated_anisotropic(
  alpha_t: f32,
  alpha_b: f32,
  dot_tv: f32,
  dot_bv: f32,
  dot_tl: f32,
  dot_bl: f32,
  dot_nv: f32,
  dot_nl: f32,
) -> f32 {
  let gv = dot_nl * length(vec3<f32>(alpha_t * dot_tv, alpha_b * dot_bv, dot_nv));
  let gl = dot_nv * length(vec3<f32>(alpha_t * dot_tl, alpha_b * dot_bl, dot_nl));
  return saturate(0.5 / (gv + gl));
}

fn d_ggx_anisotropic(alpha_t: f32, alpha_b: f32, dot_nh: f32, dot_th: f32, dot_bh: f32) -> f32 {
  let a2 = alpha_t * alpha_b;
  let v = vec3<f32>(alpha_b * dot_th, alpha_t * dot_bh, a2 * dot_nh);
  let w2 = a2 / dot(v, v);
  return RECIPROCAL_PI * a2 * w2 * w2;
}

// Three.js BRDF_GGX with USE_ANISOTROPY: alphaB is roughness^2, T/B are the anisotropy directions.
fn brdf_ggx_anisotropic(
  L: vec3<f32>,
  V: vec3<f32>,
  N: vec3<f32>,
  f0: vec3<f32>,
  f90: f32,
  roughness: f32,
  alpha_t: f32,
  anisotropy_t: vec3<f32>,
  anisotropy_b: vec3<f32>,
) -> vec3<f32> {
  let alpha = roughness * roughness;
  let H = normalize(L + V);
  let dot_nl = saturate(dot(N, L));
  let dot_nv = saturate(dot(N, V));
  let dot_nh = saturate(dot(N, H));
  let dot_vh = saturate(dot(V, H));
  let visibility = v_ggx_smith_correlated_anisotropic(
    alpha_t, alpha,
    dot(anisotropy_t, V), dot(anisotropy_b, V),
    dot(anisotropy_t, L), dot(anisotropy_b, L),
    dot_nv, dot_nl,
  );
  let distribution = d_ggx_anisotropic(alpha_t, alpha, dot_nh, dot(anisotropy_t, H), dot(anisotropy_b, H));
  return f_schlick(f0, f90, dot_vh) * (visibility * distribution);
}

// Three.js D_Charlie, V_Neubelt, and BRDF_Sheen.
fn d_charlie(roughness: f32, dot_nh: f32) -> f32 {
  let alpha = max(roughness * roughness, 0.0001);
  let inv_alpha = 1.0 / alpha;
  let cos2h = dot_nh * dot_nh;
  let sin2h = max(1.0 - cos2h, 0.0078125);
  return (2.0 + inv_alpha) * pow(sin2h, inv_alpha * 0.5) / (2.0 * PI);
}

fn v_neubelt(dot_nv: f32, dot_nl: f32) -> f32 {
  return saturate(1.0 / (4.0 * (dot_nl + dot_nv - dot_nl * dot_nv)));
}

fn brdf_sheen(L: vec3<f32>, V: vec3<f32>, N: vec3<f32>, sheen_color: vec3<f32>, sheen_roughness: f32) -> vec3<f32> {
  let H = normalize(L + V);
  let dot_nl = saturate(dot(N, L));
  let dot_nv = saturate(dot(N, V));
  let dot_nh = saturate(dot(N, H));
  return sheen_color * (d_charlie(sheen_roughness, dot_nh) * v_neubelt(dot_nv, dot_nl));
}

fn ibl_sheen_brdf(N: vec3<f32>, V: vec3<f32>, roughness: f32) -> f32 {
  let dot_nv = saturate(dot(N, V));
  let r2 = roughness * roughness;
  let a = select(-8.48 * r2 + 14.3 * roughness - 9.95, -339.2 * r2 + 161.4 * roughness - 25.9, roughness < 0.25);
  let b = select(1.97 * r2 - 3.27 * roughness + 0.72, 44.0 * r2 - 23.7 * roughness + 3.26, roughness < 0.25);
  let dg = exp(a * dot_nv + b) + select(0.1 * (roughness - 0.25), 0.0, roughness < 0.25);
  return saturate(dg * RECIPROCAL_PI);
}

// Analytical DFG approximation (Three.js DFGApprox).
fn dfg_approx(N: vec3<f32>, V: vec3<f32>, roughness: f32) -> vec2<f32> {
  let dot_nv = saturate(dot(N, V));
  let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
  let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
  let r = roughness * c0 + c1;
  let a004 = min(r.x * r.x, exp2(-9.28 * dot_nv)) * r.x + r.y;
  return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

fn environment_brdf(N: vec3<f32>, V: vec3<f32>, specular_color: vec3<f32>, specular_f90: f32, roughness: f32) -> vec3<f32> {
  let fab = dfg_approx(N, V, roughness);
  return specular_color * fab.x + specular_f90 * fab.y;
}

// Three.js computeMultiscattering: x = single scattering, y = multiple scattering.
struct Multiscattering {
  single: vec3<f32>,
  multi: vec3<f32>,
};

fn compute_multiscattering(
  N: vec3<f32>,
  V: vec3<f32>,
  specular_color: vec3<f32>,
  specular_f90: f32,
  roughness: f32,
) -> Multiscattering {
  let fab = dfg_approx(N, V, roughness);
  let fss_ess = specular_color * fab.x + specular_f90 * fab.y;
  let ess = fab.x + fab.y;
  let ems = 1.0 - ess;
  let favg = specular_color + (vec3<f32>(1.0) - specular_color) * 0.047619;
  let fms = fss_ess * favg / (vec3<f32>(1.0) - ems * favg);
  return Multiscattering(fss_ess, fms * ems);
}

fn compute_specular_occlusion(dot_nv: f32, ambient_occlusion: f32, roughness: f32) -> f32 {
  return saturate(pow(dot_nv + ambient_occlusion, exp2(-16.0 * roughness - 1.0)) - 1.0 + ambient_occlusion);
}

// Three.js geometryViewDir in world space: the camera's +Z axis for orthographic cameras.
fn view_direction(world_pos: vec3<f32>) -> vec3<f32> {
  if uniforms.camera_pos.w > 0.5 {
    return normalize(vec3<f32>(uniforms.view[0].z, uniforms.view[1].z, uniforms.view[2].z));
  }
  return normalize(uniforms.camera_pos.xyz - world_pos);
}

// Three.js envMapRotation applied to a world-space lookup direction.
fn rotate_environment_direction(direction: vec3<f32>) -> vec3<f32> {
  return uniforms.env_rotation[0].xyz * direction.x
    + uniforms.env_rotation[1].xyz * direction.y
    + uniforms.env_rotation[2].xyz * direction.z;
}

fn cube_uv_face(direction: vec3<f32>) -> f32 {
  let a = abs(direction);
  if a.x > a.z {
    if a.x > a.y {
      return select(3.0, 0.0, direction.x > 0.0);
    }
    return select(4.0, 1.0, direction.y > 0.0);
  }
  if a.z > a.y {
    return select(5.0, 2.0, direction.z > 0.0);
  }
  return select(4.0, 1.0, direction.y > 0.0);
}

// RH coordinate system; PMREM face-indexing convention.
fn cube_uv_face_uv(direction: vec3<f32>, face: f32) -> vec2<f32> {
  var uv: vec2<f32>;
  if face == 0.0 {
    uv = vec2<f32>(direction.z, direction.y) / abs(direction.x);
  } else if face == 1.0 {
    uv = vec2<f32>(-direction.x, -direction.z) / abs(direction.y);
  } else if face == 2.0 {
    uv = vec2<f32>(-direction.x, direction.y) / abs(direction.z);
  } else if face == 3.0 {
    uv = vec2<f32>(-direction.z, direction.y) / abs(direction.x);
  } else if face == 4.0 {
    uv = vec2<f32>(-direction.x, direction.z) / abs(direction.y);
  } else {
    uv = vec2<f32>(direction.x, direction.y) / abs(direction.z);
  }
  return 0.5 * (uv + vec2<f32>(1.0));
}

fn bilinear_cube_uv(direction: vec3<f32>, mip_int_in: f32) -> vec3<f32> {
  var face = cube_uv_face(direction);
  let filter_int = max(CUBE_UV_MIN_MIP_LEVEL - mip_int_in, 0.0);
  let mip_int = max(mip_int_in, CUBE_UV_MIN_MIP_LEVEL);
  let face_size = exp2(mip_int);
  var uv = cube_uv_face_uv(direction, face) * (face_size - 2.0) + vec2<f32>(1.0);
  if face > 2.0 {
    uv.y += face_size;
    face -= 3.0;
  }
  uv.x += face * face_size;
  uv.x += filter_int * 3.0 * CUBE_UV_MIN_TILE_SIZE;
  uv.y += 4.0 * (exp2(uniforms.env_params.x) - face_size);
  let dimensions = vec2<f32>(textureDimensions(t_env_cube_uv, 0));
  return textureSampleLevel(t_env_cube_uv, s_ibl, uv / dimensions, 0.0).rgb;
}

fn roughness_to_mip(roughness: f32) -> f32 {
  if roughness >= 0.8 {
    return (1.0 - roughness) * (-1.0 + 2.0) / (1.0 - 0.8) - 2.0;
  }
  if roughness >= 0.4 {
    return (0.8 - roughness) * (2.0 + 1.0) / (0.8 - 0.4) - 1.0;
  }
  if roughness >= 0.305 {
    return (0.4 - roughness) * (3.0 - 2.0) / (0.4 - 0.305) + 2.0;
  }
  if roughness >= 0.21 {
    return (0.305 - roughness) * (4.0 - 3.0) / (0.305 - 0.21) + 3.0;
  }
  return -2.0 * log2(1.16 * roughness);
}

fn texture_cube_uv(direction: vec3<f32>, roughness: f32) -> vec3<f32> {
  let mip = clamp(roughness_to_mip(roughness), -2.0, uniforms.env_params.x);
  let mip_f = fract(mip);
  let mip_int = floor(mip);
  let color0 = bilinear_cube_uv(direction, mip_int);
  if mip_f == 0.0 {
    return color0;
  }
  let color1 = bilinear_cube_uv(direction, mip_int + 1.0);
  return mix(color0, color1, mip_f);
}

// Three.js getIBLIrradiance: PI * envMapColor(roughness 1) * envMapIntensity.
fn ibl_irradiance(N: vec3<f32>) -> vec3<f32> {
  return PI * texture_cube_uv(rotate_environment_direction(N), 1.0) * uniforms.ibl_params.x;
}

// Three.js getIBLRadiance: the reflection vector leans toward the normal with roughness^2.
fn ibl_radiance(V: vec3<f32>, N: vec3<f32>, roughness: f32) -> vec3<f32> {
  let reflect_vec = normalize(mix(reflect(-V, N), N, roughness * roughness));
  return texture_cube_uv(rotate_environment_direction(reflect_vec), roughness) * uniforms.ibl_params.x;
}

// Three.js getIBLAnisotropyRadiance: the bent normal for anisotropic reflections.
fn ibl_anisotropy_radiance(V: vec3<f32>, N: vec3<f32>, roughness: f32, bitangent: vec3<f32>, anisotropy: f32) -> vec3<f32> {
  var bent_normal = cross(bitangent, V);
  bent_normal = normalize(cross(bent_normal, bitangent));
  let weight = 1.0 - anisotropy * (1.0 - roughness);
  bent_normal = normalize(mix(bent_normal, N, weight * weight * weight * weight));
  return ibl_radiance(V, bent_normal, roughness);
}

// Unblurred environment for legacy MeshBasic/Lambert/Phong envMap lookups (WebGLCubeMaps).
fn legacy_environment_color(direction: vec3<f32>) -> vec3<f32> {
  return textureSampleLevel(t_env_cube, s_ibl, rotate_environment_direction(direction), 0.0).rgb;
}
