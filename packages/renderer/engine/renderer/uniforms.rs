use super::*;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Uniforms {
    pub mvp: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
    pub normal_matrix: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub base_color: [f32; 4],
    pub emissive: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub ambient_intensity: f32,
    pub num_lights: u32,
    pub ambient_color: [f32; 4],
    /// xyz = LightProbe SH coefficient, w = reserved.
    pub light_probe: [[f32; 4]; 9],
    /// x = has LightProbe, y = has toon gradient map, z = depth packing, w = has matcap color map.
    pub light_probe_params: [f32; 4],
    // x/y = normalScale or bumpScale, z = normal mode (0=none, 1=tangent normalMap, 2=bumpMap, 3=object normalMap), w = has_ibl
    pub normal_map_params: [f32; 4],
    /// x = env_intensity, y = shading_model, z = camera near, w = camera far
    pub ibl_params: [f32; 4],
    /// x = legacy env combine, y = reflectivity, z = material env mode (0=off, 1=reflect, 2=refract), w = refraction ratio
    pub env_map_params: [f32; 4],
    /// x = ao_map_intensity, y = has_ao_map, z = has_alpha_map, w = has_light_map
    pub ao_params: [f32; 4],
    /// x = 1/width, y = 1/height, z = width, w = height
    pub render_params: [f32; 4],
    /// x = 1 for LinearSRGBColorSpace output, 0 for SRGBColorSpace output; y = tone-mapping mode (0 when material toneMapped=false); z = alpha-to-coverage active; w = toneMappingExposure.
    pub output_params: [f32; 4],
    /// x/y/z = base-color texture transform row 0 (`u' = x*u + y*v + z`), w = reserved
    pub texture_transform1: [f32; 4],
    /// x/y/z = base-color texture transform row 1 (`v' = x*u + y*v + z`), w = base texture is sRGB
    pub texture_transform2: [f32; 4],
    /// x/y/z = alpha-map texture transform row 0 (`u' = x*u + y*v + z`), w = alpha map is sRGB
    pub alpha_map_transform1: [f32; 4],
    /// x/y/z = alpha-map texture transform row 1 (`v' = x*u + y*v + z`), w = reserved
    pub alpha_map_transform2: [f32; 4],
    /// Row pairs for normal, metallic-roughness, emissive, AO, light, and specular map transforms.
    /// Metallic-roughness/emissive/AO/light/specular-map row 0 w lanes flag sRGB decode; row 1 w lanes retain UV selection.
    pub map_transform_rows: [[f32; 4]; 12],
    /// Row pairs for current physical-extension map transforms.
    pub physical_map_transform_rows: [[f32; 4]; 24],
    /// World-space clipping planes `[nx, ny, nz, constant]`.
    pub clipping_planes: [[f32; 4]; MAX_CLIPPING_PLANES],
    /// x = union plane count, y = total plane count, z = alpha hash enabled, w = premultiplied alpha.
    pub clipping_params: [f32; 4],
    /// xyz = fog color, w = reserved
    pub fog_color: [f32; 4],
    /// x = mode (0=off, 1=linear, 2=exp2), y = near, z = far, w = density
    pub fog_params: [f32; 4],
    pub light_space_matrices: [[[f32; 4]; 4]; MAX_SHADOW_LAYERS],
    /// x = shadow count, y = first bias, z = first normal_bias, w = receive_shadow
    pub shadow_params: [f32; 4],
    /// x = first shadow light index, y = 1/map_width, z = 1/map_height, w = shadow-map type.
    pub shadow_params2: [f32; 4],
    /// x/y/z = first cascade split distances, w = shadow layer count.
    pub shadow_params3: [f32; 4],
    /// x = PCF radius multiplier, y = clip shadow caster fragments by clipping planes, z = explicit shadow side (0 double/no-cull, 1 front, 2 back), w = shadow-only alpha cutoff.
    pub shadow_params4: [f32; 4],
    /// x = light index, y = layer base, z = layer count, w = shadow kind.
    pub shadow_infos: [[f32; 4]; MAX_SHADOW_LAYERS],
    /// x = bias, y = normal_bias, z = PCF radius multiplier, w = shadow intensity.
    pub shadow_biases: [[f32; 4]; MAX_SHADOW_LAYERS],
    /// x/y/z = cascade split distances, w = reserved.
    pub shadow_cascade_splits: [[f32; 4]; MAX_SHADOW_LAYERS],
    /// x = clearcoat, y = clearcoat roughness, z = transmission, w = ior
    pub physical_params1: [f32; 4],
    /// xyz = sheen color, w = sheen roughness
    pub physical_params2: [f32; 4],
    /// x = anisotropy, y = anisotropy rotation, z/w = thickness/attenuation distance or distance near/far.
    pub physical_params3: [f32; 4],
    /// x/y = clearcoat normal scale, z = light_map_intensity, w = has_specular_map, matcap map sRGB flag, or toon gradient map sRGB flag depending on shading model.
    pub physical_params4: [f32; 4],
    /// xyz = attenuation color or distance reference position, w = dispersion for physical materials.
    pub attenuation_color: [f32; 4],
    /// xyz = MeshPhysicalMaterial specular color factor, w = specular intensity.
    pub physical_specular: [f32; 4],
    /// x = iridescence, y = iridescence IOR, z/w = iridescence thickness range in nanometers.
    pub iridescence_params: [f32; 4],
    pub lights: [GpuLight; MAX_LIGHTS],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PostUniforms {
    pub params1: [f32; 4],
    pub params2: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BackgroundUniforms {
    pub transform1: [f32; 4],
    pub transform2: [f32; 4],
    pub inverse_view_projection: [[f32; 4]; 4],
    pub camera_params: [f32; 4],
    pub rotation1: [f32; 4],
    pub rotation2: [f32; 4],
    pub rotation3: [f32; 4],
}
