  let anisotropy = clamp(uniforms.physical_params3.x, 0.0, 1.0);
  let anisotropy_rotation = uniforms.physical_params3.y;
  let thickness = max(uniforms.physical_params3.z * thickness_sample, 0.0);
  let attenuation_distance = max(uniforms.physical_params3.w, 0.0);

  if use_shadow_material {
    let shadow_alpha = alpha * (1.0 - sample_combined_shadow(input.world_pos, N));
    let mapped_shadow = apply_output_color_space(apply_material_tone_mapping(albedo));
    let fogged_shadow = apply_fog(mapped_shadow, fog_depth(input.world_pos));
    return output_color(fogged_shadow, shadow_alpha);
  }
  let T = normalize(tbn[0]);
  let B = normalize(tbn[1]);
  let clearcoat_normal_sample = textureSample(t_clearcoat_normal, s_clearcoat_normal_map, transform_clearcoat_normal_map_uv(uv, uv2)).rgb;
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

  // Dielectric F0 from IOR (1.5 -> 0.04), modulated by MeshPhysicalMaterial specular extensions.
  let dielectric_f0_scalar = pow((ior - 1.0) / (ior + 1.0), 2.0);
  let physical_specular_color = clamp(uniforms.physical_specular.rgb * physical_specular_color_sample, vec3<f32>(0.0), vec3<f32>(1.0));
  let physical_specular_intensity = clamp(uniforms.physical_specular.w * physical_specular_intensity_sample, 0.0, 1.0);
  let dielectric_f0 = min(vec3<f32>(dielectric_f0_scalar) * physical_specular_color, vec3<f32>(1.0)) * physical_specular_intensity;
  let specular_f90 = mix(physical_specular_intensity, 1.0, metallic);
  let iridescence_strength = clamp(uniforms.iridescence_params.x * iridescence_sample, 0.0, 1.0) * (1.0 - metallic);
  let iridescence_thickness = mix(uniforms.iridescence_params.z, uniforms.iridescence_params.w, iridescence_thickness_sample);
  let iridescence_f0 = iridescence_fresnel_color(
    n_dot_v,
    clamp(uniforms.iridescence_params.y, 1.0, 2.333),
    iridescence_thickness,
    iridescence_thickness,
  ) * physical_specular_intensity;
  let f0 = mix(mix(dielectric_f0, iridescence_f0, iridescence_strength), albedo, metallic);
  let phong_specular_color = clamp(uniforms.physical_params2.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let phong_shininess = max(uniforms.physical_params2.w, 0.0001);
  var phong_specular_strength = 1.0;
  if use_phong && uniforms.physical_params4.w > 0.5 {
    phong_specular_strength = decode_specular_map_sample(textureSample(t_physical_layers, s_specular_map, transform_specular_map_uv(uv, uv2), 0).r);
  }

  var lo = vec3<f32>(0.0);

  let has_ibl = uniforms.normal_map_params.w > 0.5;
  let has_light_probe = uniforms.light_probe_params.x > 0.5;
  var light_probe_diffuse = vec3<f32>(0.0);
  if has_light_probe {
    light_probe_diffuse = light_probe_irradiance(N);
  }

  if uniforms.num_lights == 0u && !has_ibl && !has_light_probe {
    // No lights or IBL: render with a basic hemispherical ambient
    let ambient = uniforms.ambient_color.rgb * uniforms.ambient_intensity;
    let sky_factor = 0.5 + 0.5 * N.y;
    let fallback_ambient = mix(vec3<f32>(0.1, 0.1, 0.12), vec3<f32>(0.4, 0.45, 0.5), sky_factor);
    let total_ambient = max(ambient, fallback_ambient);
    lo = albedo * total_ambient * ao + light_map_diffuse;
  } else {
    // Direct lighting from scene lights
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
        attenuation *= sample_shadow_for_light(i, input.world_pos, N);
      } else if light.light_type == 4u {
        // RectAreaLight approximation: finite one-sided area emitter from the
        // light center. This is intentionally cheaper than Three.js' LUT path.
        let light_vec = light.position.xyz - input.world_pos;
        let dist = length(light_vec);
        L = light_vec / max(dist, 0.0001);
        let width = max(light.params.x, 0.0);
        let height = max(light.params.y, 0.0);
        let area = max(width * height, 0.0001);
        let light_dir = normalize(light.direction.xyz);
        let facing = max(dot(light_dir, -L), 0.0);
        attenuation = facing * area / max(dist * dist + area, 0.0001);
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
        attenuation *= sample_shadow_for_light(i, input.world_pos, N);
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
        let F = fresnel_schlick_f90(h_dot_v, f0, specular_f90);

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
      } else if use_phong {
        // MeshPhongMaterial: non-physical Blinn-Phong direct lighting.
        let phong_f = fresnel_schlick(h_dot_v, phong_specular_color);
        let phong_d = (phong_shininess * 0.5 + 1.0) * pow(n_dot_h, phong_shininess) / PI;
        let phong_specular = phong_f * (0.25 * phong_d) * phong_specular_strength;
        lo = lo + (albedo / PI + phong_specular) * radiance * n_dot_l;
      } else if use_toon {
        // MeshToonMaterial: gradientMap samples the red ramp channel at dot(N, L) * 0.5 + 0.5.
        let toon_coord = dot(N, L) * 0.5 + 0.5;
        var toon_irradiance: f32;
        if uniforms.light_probe_params.y > 0.5 {
          toon_irradiance = decode_toon_gradient_map_sample(textureSample(t_physical_sheen, s_physical_sheen_map, vec2<f32>(toon_coord, 0.0))).r;
        } else {
          let toon_width = fwidth(toon_coord) * 0.5;
          toon_irradiance = mix(0.7, 1.0, smoothstep(0.7 - toon_width, 0.7 + toon_width, toon_coord));
        }
        lo = lo + toon_irradiance * albedo / PI * radiance;
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
        let F_ibl = fresnel_schlick_roughness_f90(n_dot_v, f0, specular_f90, roughness);
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
      } else if use_phong {
        let R = reflect(-V, N);
        let max_lod = 4.0; // PREFILTER_MIP_LEVELS - 1
        let phong_roughness = clamp(sqrt(2.0 / (phong_shininess + 2.0)), 0.0, 1.0);
        let reflected = textureSampleLevel(t_prefilter, s_ibl, R, phong_roughness * max_lod).rgb;
        let diffuse_ibl = irradiance * albedo * ao;
        let specular_ibl = reflected * phong_specular_color * phong_specular_strength * legacy_env_reflectivity;
        lo = lo + (diffuse_ibl + specular_ibl) * env_intensity;
      } else {
        // Lambert: diffuse IBL only
        lo = lo + irradiance * albedo * env_intensity * ao * legacy_env_reflectivity;
      }
    } else {
      // Ambient (non-IBL fallback when lights are present)
      let ambient = uniforms.ambient_color.rgb * uniforms.ambient_intensity * albedo;
      lo = lo + ambient * ao;
    }
    if has_light_probe {
      lo = lo + albedo * light_probe_diffuse * (1.0 / PI) * ao;
    }
    lo = lo + light_map_diffuse;
  }

  if use_specular && transmission > 0.0001 {
    let dispersion = max(uniforms.attenuation_color.w, 0.0);
    let refracted_dir = refract(-V, N, 1.0 / ior);
    let transmittance = volume_attenuation(thickness, uniforms.attenuation_color.rgb, attenuation_distance);
    let scene_offset = refracted_dir.xy * thickness * 0.04;
    let scene_uv = clamp(screen_uv + scene_offset, vec2<f32>(0.0), vec2<f32>(1.0));
    let transmitted_sample = sample_transmission_scene_color(scene_uv, roughness, ior);
    var transmitted_light = transmitted_sample * transmittance;
    if dispersion > 0.0001 {
      let half_spread = max(ior - 1.0, 0.0) * 0.025 * dispersion;
      let ior_r = clamp(ior - half_spread, 1.0, 2.333);
      let ior_b = clamp(ior + half_spread, 1.0, 2.333);
      let refracted_r = refract(-V, N, 1.0 / ior_r);
      let refracted_b = refract(-V, N, 1.0 / ior_b);
      let uv_r = clamp(screen_uv + refracted_r.xy * thickness * 0.04, vec2<f32>(0.0), vec2<f32>(1.0));
      let uv_b = clamp(screen_uv + refracted_b.xy * thickness * 0.04, vec2<f32>(0.0), vec2<f32>(1.0));
      transmitted_light = vec3<f32>(
        sample_transmission_scene_color(uv_r, roughness, ior_r).r,
        transmitted_sample.g,
        sample_transmission_scene_color(uv_b, roughness, ior_b).b,
      ) * transmittance;
    }
    if has_ibl {
      let max_lod = 4.0;
      let environment_sample = textureSampleLevel(t_prefilter, s_ibl, refracted_dir, roughness * max_lod).rgb;
      var environment_refraction = environment_sample * transmittance;
      if dispersion > 0.0001 {
        let half_spread = max(ior - 1.0, 0.0) * 0.025 * dispersion;
        let ior_r = clamp(ior - half_spread, 1.0, 2.333);
        let ior_b = clamp(ior + half_spread, 1.0, 2.333);
        let refracted_r = refract(-V, N, 1.0 / ior_r);
        let refracted_b = refract(-V, N, 1.0 / ior_b);
        environment_refraction = vec3<f32>(
          textureSampleLevel(t_prefilter, s_ibl, refracted_r, roughness * max_lod).r,
          environment_sample.g,
          textureSampleLevel(t_prefilter, s_ibl, refracted_b, roughness * max_lod).b,
        ) * transmittance;
      }
      transmitted_light = mix(transmitted_light, environment_refraction, 0.35);
    }
    lo = mix(lo, transmitted_light, transmission);
  }

  // Emissive
  let emissive_sample = decode_emissive_map_sample(textureSample(t_emissive, s_emissive, transform_emissive_map_uv(uv, uv2))).rgb;
  lo = lo + uniforms.emissive.rgb * emissive_sample;

  if legacy_material_env {
    let combine = u32(uniforms.env_map_params.x + 0.5);
    if combine != 0u || abs(legacy_env_reflectivity - 1.0) > 0.0001 {
      let legacy_env_mode = u32(uniforms.env_map_params.z + 0.5);
      var legacy_env_dir = reflect(-V, N);
      if legacy_env_mode == 2u {
        legacy_env_dir = refract(-V, N, uniforms.env_map_params.w);
      }
      let legacy_env_color = textureSampleLevel(t_prefilter, s_ibl, legacy_env_dir, 0.0).rgb * uniforms.ibl_params.x;
      let legacy_strength = legacy_env_reflectivity * select(1.0, phong_specular_strength, use_phong);
      if combine == 2u {
        lo = lo + legacy_env_color * legacy_strength;
      } else if combine == 1u {
        lo = mix(lo, legacy_env_color, legacy_strength);
      } else {
        lo = mix(lo, lo * legacy_env_color, legacy_strength);
      }
    }
  }

  // Tone mapping (ACES Filmic, matches three.js) and output color conversion.
  let mapped = apply_material_tone_mapping(lo);
  let output_mapped = apply_output_color_space(mapped);
  let fogged = apply_fog(output_mapped, fog_depth(input.world_pos));

  return output_color(fogged, alpha);
}

// ACES Filmic tone mapping, ported from three.js (Narkowicz fit with
// input/output matrices). Includes the 1/0.6 exposure pre-scale that
// three.js applies so output matches ACESFilmicToneMapping there.
fn rrt_and_odt_fit(v: vec3<f32>) -> vec3<f32> {
  let a = v * (v + vec3<f32>(0.0245786)) - vec3<f32>(0.000090537);
  let b = v * (0.983729 * v + vec3<f32>(0.4329510)) + vec3<f32>(0.238081);
  return a / b;
}

fn linear_tone_mapping(color: vec3<f32>) -> vec3<f32> {
  return clamp(uniforms.output_params.w * color, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn reinhard_tone_mapping(color_in: vec3<f32>) -> vec3<f32> {
  let color = color_in * uniforms.output_params.w;
  return clamp(color / (vec3<f32>(1.0) + color), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn cineon_tone_mapping(color_in: vec3<f32>) -> vec3<f32> {
  var color = color_in * uniforms.output_params.w;
  color = max(vec3<f32>(0.0), color - vec3<f32>(0.004));
  return pow(
    (color * (6.2 * color + vec3<f32>(0.5))) /
      (color * (6.2 * color + vec3<f32>(1.7)) + vec3<f32>(0.06)),
    vec3<f32>(2.2)
  );
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
  var color = color_in * (uniforms.output_params.w / 0.6);
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

fn agx_tone_mapping(color_in: vec3<f32>) -> vec3<f32> {
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

  var color = color_in * uniforms.output_params.w;
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

fn neutral_tone_mapping(color_in: vec3<f32>) -> vec3<f32> {
  let start_compression = 0.8 - 0.04;
  let desaturation = 0.15;
  var color = color_in * uniforms.output_params.w;

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

fn apply_material_tone_mapping(color: vec3<f32>) -> vec3<f32> {
  let mode = uniforms.output_params.y;
  if mode < 0.5 {
    return color;
  }
  if abs(mode - 1.0) < 0.5 {
    return linear_tone_mapping(color);
  }
  if abs(mode - 2.0) < 0.5 {
    return reinhard_tone_mapping(color);
  }
  if abs(mode - 3.0) < 0.5 {
    return cineon_tone_mapping(color);
  }
  if abs(mode - 5.0) < 0.5 {
    return color;
  }
  if abs(mode - 6.0) < 0.5 {
    return agx_tone_mapping(color);
  }
  if abs(mode - 7.0) < 0.5 {
    return neutral_tone_mapping(color);
  }
  return aces_filmic_tone_mapping(color);
}

fn apply_output_color_space(color: vec3<f32>) -> vec3<f32> {
  if uniforms.output_params.x > 0.5 {
    return color;
  }
  return pow(color, vec3<f32>(1.0 / 2.2));
}
