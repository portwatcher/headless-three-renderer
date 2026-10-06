// Tone mapping ported from Three.js r180 tonemapping_pars_fragment (MIT). Shared by the material
// and background shaders.
// ACES Filmic tone mapping, ported from three.js (Narkowicz fit with
// input/output matrices). Includes the 1/0.6 exposure pre-scale that
// three.js applies so output matches ACESFilmicToneMapping there.
fn rrt_and_odt_fit(v: vec3<f32>) -> vec3<f32> {
  let a = v * (v + vec3<f32>(0.0245786)) - vec3<f32>(0.000090537);
  let b = v * (0.983729 * v + vec3<f32>(0.4329510)) + vec3<f32>(0.238081);
  return a / b;
}

fn linear_tone_mapping(color: vec3<f32>, exposure: f32) -> vec3<f32> {
  return clamp(exposure * color, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn reinhard_tone_mapping(color_in: vec3<f32>, exposure: f32) -> vec3<f32> {
  let color = color_in * exposure;
  return clamp(color / (vec3<f32>(1.0) + color), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn cineon_tone_mapping(color_in: vec3<f32>, exposure: f32) -> vec3<f32> {
  var color = color_in * exposure;
  color = max(vec3<f32>(0.0), color - vec3<f32>(0.004));
  return pow(
    (color * (6.2 * color + vec3<f32>(0.5))) /
      (color * (6.2 * color + vec3<f32>(1.7)) + vec3<f32>(0.06)),
    vec3<f32>(2.2)
  );
}

fn aces_filmic_tone_mapping(color_in: vec3<f32>, exposure: f32) -> vec3<f32> {
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
  var color = color_in * (exposure / 0.6);
  color = aces_input * color;
  color = rrt_and_odt_fit(color);
  color = aces_output * color;
  return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn agx_default_contrast_approx(x: vec3<f32>) -> vec3<f32> {
  let x2 = x * x;
  let x4 = x2 * x2;
  return 15.5 * x4 * x2
    - 40.14 * x4 * x
    + 31.96 * x4
    - 6.868 * x2 * x
    + 0.4298 * x2
    + 0.1191 * x
    - vec3<f32>(0.00232);
}

fn agx_tone_mapping(color_in: vec3<f32>, exposure: f32) -> vec3<f32> {
  let linear_rec2020_to_linear_srgb = mat3x3<f32>(
    vec3<f32>(1.6605, -0.1246, -0.0182),
    vec3<f32>(-0.5876, 1.1329, -0.1006),
    vec3<f32>(-0.0728, -0.0083, 1.1187)
  );
  let linear_srgb_to_linear_rec2020 = mat3x3<f32>(
    vec3<f32>(0.6274, 0.0691, 0.0164),
    vec3<f32>(0.3293, 0.9195, 0.0880),
    vec3<f32>(0.0433, 0.0113, 0.8956)
  );
  let agx_inset = mat3x3<f32>(
    vec3<f32>(0.856627153315983, 0.137318972929847, 0.11189821299995),
    vec3<f32>(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
    vec3<f32>(0.0482516061458583, 0.101439036467562, 0.811302368396859)
  );
  let agx_outset = mat3x3<f32>(
    vec3<f32>(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
    vec3<f32>(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
    vec3<f32>(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405)
  );
  let agx_min_ev = -12.47393;
  let agx_max_ev = 4.026069;

  var color = color_in * exposure;
  color = linear_srgb_to_linear_rec2020 * color;
  color = agx_inset * color;
  color = max(color, vec3<f32>(0.0000000001));
  color = log2(color);
  color = (color - vec3<f32>(agx_min_ev)) / (agx_max_ev - agx_min_ev);
  color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
  color = agx_default_contrast_approx(color);
  color = agx_outset * color;
  color = pow(max(vec3<f32>(0.0), color), vec3<f32>(2.2));
  color = linear_rec2020_to_linear_srgb * color;
  return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn neutral_tone_mapping(color_in: vec3<f32>, exposure: f32) -> vec3<f32> {
  let start_compression = 0.8 - 0.04;
  let desaturation = 0.15;
  var color = color_in * exposure;

  let x = min(color.r, min(color.g, color.b));
  var offset = 0.04;
  if x < 0.08 {
    offset = x - 6.25 * x * x;
  }
  color = color - vec3<f32>(offset);

  let peak = max(color.r, max(color.g, color.b));
  if peak < start_compression {
    return color;
  }

  let d = 1.0 - start_compression;
  let new_peak = 1.0 - d * d / (peak + d - start_compression);
  color = color * (new_peak / peak);
  let g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
  return mix(color, vec3<f32>(new_peak), g);
}

// Three.js tone mapping: mode is the toneMapping constant (0 = none).
fn tone_map(color: vec3<f32>, mode: f32, exposure: f32) -> vec3<f32> {
  if mode < 0.5 {
    return color;
  }
  if abs(mode - 1.0) < 0.5 {
    return linear_tone_mapping(color, exposure);
  }
  if abs(mode - 2.0) < 0.5 {
    return reinhard_tone_mapping(color, exposure);
  }
  if abs(mode - 3.0) < 0.5 {
    return cineon_tone_mapping(color, exposure);
  }
  if abs(mode - 5.0) < 0.5 {
    return color;
  }
  if abs(mode - 6.0) < 0.5 {
    return agx_tone_mapping(color, exposure);
  }
  if abs(mode - 7.0) < 0.5 {
    return neutral_tone_mapping(color, exposure);
  }
  return aces_filmic_tone_mapping(color, exposure);
}
