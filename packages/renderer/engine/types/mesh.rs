use napi::bindgen_prelude::Buffer;
use napi_derive::napi;

#[napi(object)]
#[derive(Default)]
pub struct SceneMesh {
    /// Stable renderer-local key for reusing cached native mesh buffers.
    pub native_mesh_key: Option<u32>,
    /// Vertex count for cached native mesh-buffer references.
    pub native_vertex_count: Option<u32>,
    /// Index count for cached native mesh-buffer references.
    pub native_index_count: Option<u32>,
    /// Flat xyz positions: `[x0, y0, z0, x1, y1, z1, ...]`.
    pub positions: Vec<f64>,
    /// Optional uint32 triangle-list indices.
    pub indices: Option<Vec<u32>>,
    /// Optional per-vertex normals: `[nx0, ny0, nz0, ...]`.
    pub normals: Option<Vec<f64>>,
    /// Optional per-vertex colors. Accepts one `[r,g,b,a]` color or one color per vertex.
    pub colors: Option<Vec<f64>>,
    /// Mesh material color used when `colors` is omitted.
    pub color: Option<Vec<f64>>,
    /// Optional column-major 4x4 model matrix.
    pub transform: Option<Vec<f64>>,
    /// Optional per-vertex UV coordinates: `[u0, v0, u1, v1, ...]`.
    pub uvs: Option<Vec<f64>>,
    /// Optional secondary UV coordinates, used by AO/light-style maps.
    pub uvs2: Option<Vec<f64>>,
    /// Use per-face normals for flat-shaded triangle materials.
    pub flat_shading: Option<bool>,
    /// Whether this material is affected by scene fog. Defaults to true.
    pub fog: Option<bool>,
    /// Optional texture image data (raw RGBA8 bytes or encoded PNG/JPEG/WebP).
    pub texture: Option<Buffer>,
    /// Texture width in pixels (required when `texture` is raw RGBA8 bytes).
    pub texture_width: Option<u32>,
    /// Texture height in pixels (required when `texture` is raw RGBA8 bytes).
    pub texture_height: Option<u32>,
    /// Texture horizontal wrap mode: `"repeat"`, `"clamp"`, or `"mirror"`. Defaults to `"clamp"`.
    pub texture_wrap_s: Option<String>,
    /// Texture vertical wrap mode: `"repeat"`, `"clamp"`, or `"mirror"`. Defaults to `"clamp"`.
    pub texture_wrap_t: Option<String>,
    /// Texture magnification filter: `"nearest"` or `"linear"`. Defaults to `"linear"`.
    pub texture_mag_filter: Option<String>,
    /// Texture minification filter: `"nearest"` or `"linear"`. Defaults to `"linear"`.
    pub texture_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub texture_anisotropy: Option<f64>,
    /// Base-color map UV transform `[a, c, tx, b, d, ty]`.
    pub texture_transform: Option<Vec<f64>>,
    /// Base-color texture color space. `"srgb"` is decoded to linear before shading.
    pub texture_color_space: Option<String>,
    /// Whether base-color map sampling uses the secondary UV stream.
    pub texture_uses_uv2: Option<bool>,
    /// Optional metallic-roughness texture (glTF convention: G=roughness, B=metallic).
    pub metallic_roughness_texture: Option<Buffer>,
    /// Metallic-roughness texture width (required when texture is raw RGBA8 bytes).
    pub metallic_roughness_texture_width: Option<u32>,
    /// Metallic-roughness texture height (required when texture is raw RGBA8 bytes).
    pub metallic_roughness_texture_height: Option<u32>,
    /// Metallic-roughness texture horizontal wrap mode.
    pub metallic_roughness_texture_wrap_s: Option<String>,
    /// Metallic-roughness texture vertical wrap mode.
    pub metallic_roughness_texture_wrap_t: Option<String>,
    /// Metallic-roughness texture magnification filter.
    pub metallic_roughness_texture_mag_filter: Option<String>,
    /// Metallic-roughness texture minification filter.
    pub metallic_roughness_texture_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub metallic_roughness_texture_anisotropy: Option<f64>,
    /// Metallic-roughness map UV transform `[a, c, tx, b, d, ty]`.
    pub metallic_roughness_texture_transform: Option<Vec<f64>>,
    /// Metallic-roughness texture color space. `"srgb"` is decoded to linear before shading.
    pub metallic_roughness_texture_color_space: Option<String>,
    /// Whether metallic-roughness map sampling uses the secondary UV stream.
    pub metallic_roughness_texture_uses_uv2: Option<bool>,
    /// Optional specular strength map. Red channel multiplies MeshPhongMaterial specular.
    pub specular_map: Option<Buffer>,
    /// Specular map width (required when map is raw RGBA8 bytes).
    pub specular_map_width: Option<u32>,
    /// Specular map height (required when map is raw RGBA8 bytes).
    pub specular_map_height: Option<u32>,
    /// Specular map horizontal wrap mode.
    pub specular_map_wrap_s: Option<String>,
    /// Specular map vertical wrap mode.
    pub specular_map_wrap_t: Option<String>,
    /// Specular map magnification filter.
    pub specular_map_mag_filter: Option<String>,
    /// Specular map minification filter.
    pub specular_map_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub specular_map_anisotropy: Option<f64>,
    /// Specular map UV transform `[a, c, tx, b, d, ty]`.
    pub specular_map_transform: Option<Vec<f64>>,
    /// Specular map color space. `"srgb"` is decoded to linear before shading.
    pub specular_map_color_space: Option<String>,
    /// Whether specular map sampling uses the secondary UV stream.
    pub specular_map_uses_uv2: Option<bool>,
    /// Metallic factor (0..1). Defaults to 0.
    pub metallic: Option<f64>,
    /// Roughness factor (0..1). Defaults to 1.
    pub roughness: Option<f64>,
    /// Clearcoat intensity (0..1) for MeshPhysicalMaterial. Defaults to 0.
    pub clearcoat: Option<f64>,
    /// Optional clearcoat intensity map. Red channel multiplies `clearcoat`.
    pub clearcoat_map: Option<Buffer>,
    /// Clearcoat map width (required when map is raw RGBA8 bytes).
    pub clearcoat_map_width: Option<u32>,
    /// Clearcoat map height (required when map is raw RGBA8 bytes).
    pub clearcoat_map_height: Option<u32>,
    /// Clearcoat map horizontal wrap mode.
    pub clearcoat_map_wrap_s: Option<String>,
    /// Clearcoat map vertical wrap mode.
    pub clearcoat_map_wrap_t: Option<String>,
    /// Clearcoat map magnification filter.
    pub clearcoat_map_mag_filter: Option<String>,
    /// Clearcoat map minification filter.
    pub clearcoat_map_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub clearcoat_map_anisotropy: Option<f64>,
    /// Clearcoat map UV transform `[a, c, tx, b, d, ty]`.
    pub clearcoat_map_transform: Option<Vec<f64>>,
    /// Clearcoat map color space. `"srgb"` is decoded to linear before shading.
    pub clearcoat_map_color_space: Option<String>,
    /// Whether clearcoat map sampling uses the secondary UV stream.
    pub clearcoat_map_uses_uv2: Option<bool>,
    /// Clearcoat roughness (0..1). Defaults to 0.
    pub clearcoat_roughness: Option<f64>,
    /// Optional clearcoat roughness map. Green channel multiplies `clearcoatRoughness`.
    pub clearcoat_roughness_map: Option<Buffer>,
    /// Clearcoat roughness map width (required when map is raw RGBA8 bytes).
    pub clearcoat_roughness_map_width: Option<u32>,
    /// Clearcoat roughness map height (required when map is raw RGBA8 bytes).
    pub clearcoat_roughness_map_height: Option<u32>,
    /// Clearcoat roughness map horizontal wrap mode.
    pub clearcoat_roughness_map_wrap_s: Option<String>,
    /// Clearcoat roughness map vertical wrap mode.
    pub clearcoat_roughness_map_wrap_t: Option<String>,
    /// Clearcoat roughness map magnification filter.
    pub clearcoat_roughness_map_mag_filter: Option<String>,
    /// Clearcoat roughness map minification filter.
    pub clearcoat_roughness_map_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub clearcoat_roughness_map_anisotropy: Option<f64>,
    /// Clearcoat roughness map UV transform `[a, c, tx, b, d, ty]`.
    pub clearcoat_roughness_map_transform: Option<Vec<f64>>,
    /// Clearcoat roughness map color space. `"srgb"` is decoded to linear before shading.
    pub clearcoat_roughness_map_color_space: Option<String>,
    /// Whether clearcoat roughness map sampling uses the secondary UV stream.
    pub clearcoat_roughness_map_uses_uv2: Option<bool>,
    /// Optional clearcoat normal map.
    pub clearcoat_normal_map: Option<Buffer>,
    /// Clearcoat normal map width (required when map is raw RGBA8 bytes).
    pub clearcoat_normal_map_width: Option<u32>,
    /// Clearcoat normal map height (required when map is raw RGBA8 bytes).
    pub clearcoat_normal_map_height: Option<u32>,
    /// Clearcoat normal map horizontal wrap mode.
    pub clearcoat_normal_map_wrap_s: Option<String>,
    /// Clearcoat normal map vertical wrap mode.
    pub clearcoat_normal_map_wrap_t: Option<String>,
    /// Clearcoat normal map magnification filter.
    pub clearcoat_normal_map_mag_filter: Option<String>,
    /// Clearcoat normal map minification filter.
    pub clearcoat_normal_map_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub clearcoat_normal_map_anisotropy: Option<f64>,
    /// Clearcoat normal map UV transform `[a, c, tx, b, d, ty]`.
    pub clearcoat_normal_map_transform: Option<Vec<f64>>,
    /// Clearcoat normal map color space. `"srgb"` decodes RGB channels before normal sampling.
    pub clearcoat_normal_map_color_space: Option<String>,
    /// Whether clearcoat normal map sampling uses the secondary UV stream.
    pub clearcoat_normal_map_uses_uv2: Option<bool>,
    /// Clearcoat normal map scale `[x, y]`. Defaults to `[1, 1]`.
    pub clearcoat_normal_scale: Option<Vec<f64>>,
    /// Sheen color/intensity `[r, g, b]` in 0..1 range.
    pub sheen_color: Option<Vec<f64>>,
    /// Optional sheen color map. RGB channels multiply `sheenColor`.
    pub sheen_color_map: Option<Buffer>,
    /// Sheen color map width (required when map is raw RGBA8 bytes).
    pub sheen_color_map_width: Option<u32>,
    /// Sheen color map height (required when map is raw RGBA8 bytes).
    pub sheen_color_map_height: Option<u32>,
    /// Sheen color map horizontal wrap mode.
    pub sheen_color_map_wrap_s: Option<String>,
    /// Sheen color map vertical wrap mode.
    pub sheen_color_map_wrap_t: Option<String>,
    /// Sheen color map magnification filter.
    pub sheen_color_map_mag_filter: Option<String>,
    /// Sheen color map minification filter.
    pub sheen_color_map_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub sheen_color_map_anisotropy: Option<f64>,
    /// Sheen color map UV transform `[a, c, tx, b, d, ty]`.
    pub sheen_color_map_transform: Option<Vec<f64>>,
    /// Sheen color texture color space. `"srgb"` is decoded to linear before shading.
    pub sheen_color_map_color_space: Option<String>,
    /// Whether sheen color map sampling uses the secondary UV stream.
    pub sheen_color_map_uses_uv2: Option<bool>,
    /// Sheen roughness (0..1). Defaults to 1 when sheen is present.
    pub sheen_roughness: Option<f64>,
    /// Optional sheen roughness map. Alpha channel multiplies `sheenRoughness`.
    pub sheen_roughness_map: Option<Buffer>,
    /// Sheen roughness map width (required when map is raw RGBA8 bytes).
    pub sheen_roughness_map_width: Option<u32>,
    /// Sheen roughness map height (required when map is raw RGBA8 bytes).
    pub sheen_roughness_map_height: Option<u32>,
    /// Sheen roughness map horizontal wrap mode.
    pub sheen_roughness_map_wrap_s: Option<String>,
    /// Sheen roughness map vertical wrap mode.
    pub sheen_roughness_map_wrap_t: Option<String>,
    /// Sheen roughness map magnification filter.
    pub sheen_roughness_map_mag_filter: Option<String>,
    /// Sheen roughness map minification filter.
    pub sheen_roughness_map_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub sheen_roughness_map_anisotropy: Option<f64>,
    /// Sheen roughness map UV transform `[a, c, tx, b, d, ty]`.
    pub sheen_roughness_map_transform: Option<Vec<f64>>,
    /// Sheen roughness map color space metadata. Alpha roughness sampling is unchanged.
    pub sheen_roughness_map_color_space: Option<String>,
    /// Whether sheen roughness map sampling uses the secondary UV stream.
    pub sheen_roughness_map_uses_uv2: Option<bool>,
    /// Anisotropy strength (0..1). Defaults to 0.
    pub anisotropy: Option<f64>,
    /// Anisotropy rotation in tangent space, radians.
    pub anisotropy_rotation: Option<f64>,
    /// Optional anisotropy map. RG encode direction and B encodes strength.
    pub anisotropy_map: Option<Buffer>,
    /// Anisotropy map width (required when map is raw RGBA8 bytes).
    pub anisotropy_map_width: Option<u32>,
    /// Anisotropy map height (required when map is raw RGBA8 bytes).
    pub anisotropy_map_height: Option<u32>,
    /// Anisotropy map horizontal wrap mode.
    pub anisotropy_map_wrap_s: Option<String>,
    /// Anisotropy map vertical wrap mode.
    pub anisotropy_map_wrap_t: Option<String>,
    /// Anisotropy map magnification filter.
    pub anisotropy_map_mag_filter: Option<String>,
    /// Anisotropy map minification filter.
    pub anisotropy_map_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub anisotropy_map_anisotropy: Option<f64>,
    /// Anisotropy map UV transform `[a, c, tx, b, d, ty]`.
    pub anisotropy_map_transform: Option<Vec<f64>>,
    /// Anisotropy map color space. `"srgb"` is decoded to linear before shading.
    pub anisotropy_map_color_space: Option<String>,
    /// Whether anisotropy map sampling uses the secondary UV stream.
    pub anisotropy_map_uses_uv2: Option<bool>,
    /// MeshPhysicalMaterial iridescence factor (0..1). Defaults to 0.
    pub iridescence: Option<f64>,
    /// Optional iridescence factor map. Red channel multiplies `iridescence`.
    pub iridescence_map: Option<Buffer>,
    /// Iridescence factor map width (required when map is raw RGBA8 bytes).
    pub iridescence_map_width: Option<u32>,
    /// Iridescence factor map height (required when map is raw RGBA8 bytes).
    pub iridescence_map_height: Option<u32>,
    /// Iridescence factor map horizontal wrap mode.
    pub iridescence_map_wrap_s: Option<String>,
    /// Iridescence factor map vertical wrap mode.
    pub iridescence_map_wrap_t: Option<String>,
    /// Iridescence factor map magnification filter.
    pub iridescence_map_mag_filter: Option<String>,
    /// Iridescence factor map minification filter.
    pub iridescence_map_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub iridescence_map_anisotropy: Option<f64>,
    /// Iridescence factor map UV transform `[a, c, tx, b, d, ty]`.
    pub iridescence_map_transform: Option<Vec<f64>>,
    /// Iridescence factor map color space. `"srgb"` is decoded to linear before shading.
    pub iridescence_map_color_space: Option<String>,
    /// Whether iridescence factor map sampling uses the secondary UV stream.
    pub iridescence_map_uses_uv2: Option<bool>,
    /// Iridescence film IOR. Defaults to 1.3.
    pub iridescence_ior: Option<f64>,
    /// Iridescence film thickness range minimum in nanometers. Defaults to 100.
    pub iridescence_thickness_min: Option<f64>,
    /// Iridescence film thickness range maximum in nanometers. Defaults to 400.
    pub iridescence_thickness_max: Option<f64>,
    /// Optional iridescence thickness map. Green channel interpolates the configured thickness range.
    pub iridescence_thickness_map: Option<Buffer>,
    /// Iridescence thickness map width (required when map is raw RGBA8 bytes).
    pub iridescence_thickness_map_width: Option<u32>,
    /// Iridescence thickness map height (required when map is raw RGBA8 bytes).
    pub iridescence_thickness_map_height: Option<u32>,
    /// Iridescence thickness map horizontal wrap mode.
    pub iridescence_thickness_map_wrap_s: Option<String>,
    /// Iridescence thickness map vertical wrap mode.
    pub iridescence_thickness_map_wrap_t: Option<String>,
    /// Iridescence thickness map magnification filter.
    pub iridescence_thickness_map_mag_filter: Option<String>,
    /// Iridescence thickness map minification filter.
    pub iridescence_thickness_map_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub iridescence_thickness_map_anisotropy: Option<f64>,
    /// Iridescence thickness map UV transform `[a, c, tx, b, d, ty]`.
    pub iridescence_thickness_map_transform: Option<Vec<f64>>,
    /// Iridescence thickness map color space. `"srgb"` is decoded to linear before shading.
    pub iridescence_thickness_map_color_space: Option<String>,
    /// Whether iridescence thickness map sampling uses the secondary UV stream.
    pub iridescence_thickness_map_uses_uv2: Option<bool>,
    /// Physical transmission factor (0..1). Defaults to 0.
    pub transmission: Option<f64>,
    /// Optional transmission map. Red channel multiplies `transmission`.
    pub transmission_map: Option<Buffer>,
    /// Transmission map width (required when map is raw RGBA8 bytes).
    pub transmission_map_width: Option<u32>,
    /// Transmission map height (required when map is raw RGBA8 bytes).
    pub transmission_map_height: Option<u32>,
    /// Transmission map horizontal wrap mode.
    pub transmission_map_wrap_s: Option<String>,
    /// Transmission map vertical wrap mode.
    pub transmission_map_wrap_t: Option<String>,
    /// Transmission map magnification filter.
    pub transmission_map_mag_filter: Option<String>,
    /// Transmission map minification filter.
    pub transmission_map_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub transmission_map_anisotropy: Option<f64>,
    /// Transmission map UV transform `[a, c, tx, b, d, ty]`.
    pub transmission_map_transform: Option<Vec<f64>>,
    /// Transmission map color space. `"srgb"` is decoded to linear before shading.
    pub transmission_map_color_space: Option<String>,
    /// Whether transmission map sampling uses the secondary UV stream.
    pub transmission_map_uses_uv2: Option<bool>,
    /// MeshPhysicalMaterial dispersion factor. Defaults to 0.
    pub dispersion: Option<f64>,
    /// Index of refraction for physical materials. Defaults to 1.5.
    pub ior: Option<f64>,
    /// Transmission volume thickness in model units. Defaults to 0.
    pub thickness: Option<f64>,
    /// Optional thickness map. Green channel multiplies `thickness`.
    pub thickness_map: Option<Buffer>,
    /// Thickness map width (required when map is raw RGBA8 bytes).
    pub thickness_map_width: Option<u32>,
    /// Thickness map height (required when map is raw RGBA8 bytes).
    pub thickness_map_height: Option<u32>,
    /// Thickness map horizontal wrap mode.
    pub thickness_map_wrap_s: Option<String>,
    /// Thickness map vertical wrap mode.
    pub thickness_map_wrap_t: Option<String>,
    /// Thickness map magnification filter.
    pub thickness_map_mag_filter: Option<String>,
    /// Thickness map minification filter.
    pub thickness_map_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub thickness_map_anisotropy: Option<f64>,
    /// Thickness map UV transform `[a, c, tx, b, d, ty]`.
    pub thickness_map_transform: Option<Vec<f64>>,
    /// Thickness map color space. `"srgb"` is decoded to linear before shading.
    pub thickness_map_color_space: Option<String>,
    /// Whether thickness map sampling uses the secondary UV stream.
    pub thickness_map_uses_uv2: Option<bool>,
    /// Transmission attenuation distance. Defaults to a very large distance.
    pub attenuation_distance: Option<f64>,
    /// Transmission attenuation color `[r, g, b]` in 0..1 range.
    pub attenuation_color: Option<Vec<f64>>,
    /// MeshPhysicalMaterial specular color factor `[r, g, b]` in 0..1 range.
    pub physical_specular_color: Option<Vec<f64>>,
    /// MeshPhysicalMaterial specular intensity factor (0..1). Defaults to 1.
    pub physical_specular_intensity: Option<f64>,
    /// Optional MeshPhysicalMaterial specular color map. RGB channels multiply `specularColor`.
    pub specular_color_map: Option<Buffer>,
    /// Specular color map width (required when map is raw RGBA8 bytes).
    pub specular_color_map_width: Option<u32>,
    /// Specular color map height (required when map is raw RGBA8 bytes).
    pub specular_color_map_height: Option<u32>,
    /// Specular color map horizontal wrap mode.
    pub specular_color_map_wrap_s: Option<String>,
    /// Specular color map vertical wrap mode.
    pub specular_color_map_wrap_t: Option<String>,
    /// Specular color map magnification filter.
    pub specular_color_map_mag_filter: Option<String>,
    /// Specular color map minification filter.
    pub specular_color_map_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub specular_color_map_anisotropy: Option<f64>,
    /// Specular color map UV transform `[a, c, tx, b, d, ty]`.
    pub specular_color_map_transform: Option<Vec<f64>>,
    /// Specular color texture color space. `"srgb"` is decoded to linear before shading.
    pub specular_color_map_color_space: Option<String>,
    /// Whether specular color map sampling uses the secondary UV stream.
    pub specular_color_map_uses_uv2: Option<bool>,
    /// Optional MeshPhysicalMaterial specular intensity map. Alpha channel multiplies `specularIntensity`.
    pub specular_intensity_map: Option<Buffer>,
    /// Specular intensity map width (required when map is raw RGBA8 bytes).
    pub specular_intensity_map_width: Option<u32>,
    /// Specular intensity map height (required when map is raw RGBA8 bytes).
    pub specular_intensity_map_height: Option<u32>,
    /// Specular intensity map horizontal wrap mode.
    pub specular_intensity_map_wrap_s: Option<String>,
    /// Specular intensity map vertical wrap mode.
    pub specular_intensity_map_wrap_t: Option<String>,
    /// Specular intensity map magnification filter.
    pub specular_intensity_map_mag_filter: Option<String>,
    /// Specular intensity map minification filter.
    pub specular_intensity_map_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub specular_intensity_map_anisotropy: Option<f64>,
    /// Specular intensity map UV transform `[a, c, tx, b, d, ty]`.
    pub specular_intensity_map_transform: Option<Vec<f64>>,
    /// Specular intensity map color space metadata. Alpha intensity sampling is unchanged.
    pub specular_intensity_map_color_space: Option<String>,
    /// Whether specular intensity map sampling uses the secondary UV stream.
    pub specular_intensity_map_uses_uv2: Option<bool>,
    /// MeshPhongMaterial specular color `[r, g, b]` in 0..1 range.
    pub specular_color: Option<Vec<f64>>,
    /// MeshPhongMaterial shininess. Defaults to 30.
    pub shininess: Option<f64>,
    /// Emissive color `[r, g, b]` in 0..1 range.
    pub emissive: Option<Vec<f64>>,
    /// Emissive intensity multiplier. Defaults to 1.
    pub emissive_intensity: Option<f64>,
    /// Optional normal map image data (raw RGBA8 bytes or encoded PNG/JPEG/WebP).
    pub normal_map: Option<Buffer>,
    /// Normal map width in pixels (required when `normal_map` is raw RGBA8 bytes).
    pub normal_map_width: Option<u32>,
    /// Normal map height in pixels (required when `normal_map` is raw RGBA8 bytes).
    pub normal_map_height: Option<u32>,
    /// Normal map horizontal wrap mode.
    pub normal_map_wrap_s: Option<String>,
    /// Normal map vertical wrap mode.
    pub normal_map_wrap_t: Option<String>,
    /// Normal map magnification filter.
    pub normal_map_mag_filter: Option<String>,
    /// Normal map minification filter.
    pub normal_map_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub normal_map_anisotropy: Option<f64>,
    /// Normal map UV transform `[a, c, tx, b, d, ty]`.
    pub normal_map_transform: Option<Vec<f64>>,
    /// Normal map color space. `"srgb"` decodes RGB channels before normal sampling.
    pub normal_map_color_space: Option<String>,
    /// Whether normal map sampling uses the secondary UV stream.
    pub normal_map_uses_uv2: Option<bool>,
    /// Normal map space: "tangent" (default) or "object".
    pub normal_map_type: Option<String>,
    /// Normal map scale `[x, y]`. Defaults to `[1, 1]`.
    pub normal_scale: Option<Vec<f64>>,
    /// Optional bump map image data. Red channel is used as height.
    pub bump_map: Option<Buffer>,
    /// Bump map width (required when `bump_map` is raw RGBA8 bytes).
    pub bump_map_width: Option<u32>,
    /// Bump map height (required when `bump_map` is raw RGBA8 bytes).
    pub bump_map_height: Option<u32>,
    /// Bump map horizontal wrap mode.
    pub bump_map_wrap_s: Option<String>,
    /// Bump map vertical wrap mode.
    pub bump_map_wrap_t: Option<String>,
    /// Bump map magnification filter.
    pub bump_map_mag_filter: Option<String>,
    /// Bump map minification filter.
    pub bump_map_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub bump_map_anisotropy: Option<f64>,
    /// Bump map UV transform `[a, c, tx, b, d, ty]`.
    pub bump_map_transform: Option<Vec<f64>>,
    /// Bump map color space. `"srgb"` decodes RGB channels before bump sampling.
    pub bump_map_color_space: Option<String>,
    /// Whether bump map sampling uses the secondary UV stream.
    pub bump_map_uses_uv2: Option<bool>,
    /// Bump map scale. Defaults to 1.
    pub bump_scale: Option<f64>,
    /// Optional MeshMatcapMaterial color map image data.
    pub matcap_map: Option<Buffer>,
    /// Matcap color map width (required when `matcap_map` is raw RGBA8 bytes).
    pub matcap_map_width: Option<u32>,
    /// Matcap color map height (required when `matcap_map` is raw RGBA8 bytes).
    pub matcap_map_height: Option<u32>,
    /// Matcap color map horizontal wrap mode.
    pub matcap_map_wrap_s: Option<String>,
    /// Matcap color map vertical wrap mode.
    pub matcap_map_wrap_t: Option<String>,
    /// Matcap color map magnification filter.
    pub matcap_map_mag_filter: Option<String>,
    /// Matcap color map minification filter.
    pub matcap_map_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub matcap_map_anisotropy: Option<f64>,
    /// Matcap color map UV transform `[a, c, tx, b, d, ty]`.
    pub matcap_map_transform: Option<Vec<f64>>,
    /// Matcap color map color space: `"srgb"` or `"linear"`.
    pub matcap_map_color_space: Option<String>,
    /// Whether matcap color map sampling uses the secondary UV stream.
    pub matcap_map_uses_uv2: Option<bool>,
    /// MeshDepthMaterial packing mode. Uses Three.js constants: 3200 basic, 3201 RGBA, 3202 RGB, 3203 RG.
    pub depth_packing: Option<u32>,
    /// Optional MeshDistanceMaterial world-space reference position. Defaults to camera position.
    pub distance_reference_position: Option<Vec<f64>>,
    /// Optional MeshDistanceMaterial near distance. Defaults to camera near plane.
    pub distance_near: Option<f64>,
    /// Optional MeshDistanceMaterial far distance. Defaults to camera far plane.
    pub distance_far: Option<f64>,
    /// Optional MeshToonMaterial gradient ramp map. Red channel controls diffuse ramp strength.
    pub gradient_map: Option<Buffer>,
    /// Gradient map width (required when `gradient_map` is raw RGBA8 bytes).
    pub gradient_map_width: Option<u32>,
    /// Gradient map height (required when `gradient_map` is raw RGBA8 bytes).
    pub gradient_map_height: Option<u32>,
    /// Gradient map horizontal wrap mode.
    pub gradient_map_wrap_s: Option<String>,
    /// Gradient map vertical wrap mode.
    pub gradient_map_wrap_t: Option<String>,
    /// Gradient map magnification filter.
    pub gradient_map_mag_filter: Option<String>,
    /// Gradient map minification filter.
    pub gradient_map_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub gradient_map_anisotropy: Option<f64>,
    /// Gradient map color space: `"srgb"` or `"linear"`.
    pub gradient_map_color_space: Option<String>,
    /// Optional displacement map image data (raw RGBA8 bytes or encoded PNG/JPEG/WebP).
    /// Red channel displaces vertices along their object-space normals.
    pub displacement_map: Option<Buffer>,
    /// Displacement map width in pixels (required when `displacement_map` is raw RGBA8 bytes).
    pub displacement_map_width: Option<u32>,
    /// Displacement map height in pixels (required when `displacement_map` is raw RGBA8 bytes).
    pub displacement_map_height: Option<u32>,
    /// Displacement map horizontal wrap mode.
    pub displacement_map_wrap_s: Option<String>,
    /// Displacement map vertical wrap mode.
    pub displacement_map_wrap_t: Option<String>,
    /// Displacement map magnification filter.
    pub displacement_map_mag_filter: Option<String>,
    /// Displacement map minification filter.
    pub displacement_map_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub displacement_map_anisotropy: Option<f64>,
    /// Displacement map UV transform `[a, c, tx, b, d, ty]`.
    pub displacement_map_transform: Option<Vec<f64>>,
    /// Displacement map color space. `"srgb"` decodes the red channel before displacement.
    pub displacement_map_color_space: Option<String>,
    /// Whether displacement map sampling uses the secondary UV stream.
    pub displacement_map_uses_uv2: Option<bool>,
    /// Displacement scale multiplier. Defaults to 1.
    pub displacement_scale: Option<f64>,
    /// Displacement bias added after map sampling. Defaults to 0.
    pub displacement_bias: Option<f64>,
    /// Optional emissive map image data (raw RGBA8 bytes or encoded PNG/JPEG/WebP).
    pub emissive_map: Option<Buffer>,
    /// Emissive map width in pixels (required when `emissive_map` is raw RGBA8 bytes).
    pub emissive_map_width: Option<u32>,
    /// Emissive map height in pixels (required when `emissive_map` is raw RGBA8 bytes).
    pub emissive_map_height: Option<u32>,
    /// Emissive map horizontal wrap mode.
    pub emissive_map_wrap_s: Option<String>,
    /// Emissive map vertical wrap mode.
    pub emissive_map_wrap_t: Option<String>,
    /// Emissive map magnification filter.
    pub emissive_map_mag_filter: Option<String>,
    /// Emissive map minification filter.
    pub emissive_map_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub emissive_map_anisotropy: Option<f64>,
    /// Emissive map UV transform `[a, c, tx, b, d, ty]`.
    pub emissive_map_transform: Option<Vec<f64>>,
    /// Emissive texture color space. `"srgb"` is decoded to linear before shading.
    pub emissive_map_color_space: Option<String>,
    /// Whether emissive map sampling uses the secondary UV stream.
    pub emissive_map_uses_uv2: Option<bool>,
    /// Optional ambient occlusion map image data (raw RGBA8 bytes or encoded PNG/JPEG/WebP).
    /// Red channel is used as the occlusion factor (Three.js / glTF convention).
    pub ao_map: Option<Buffer>,
    /// AO map width in pixels (required when `ao_map` is raw RGBA8 bytes).
    pub ao_map_width: Option<u32>,
    /// AO map height in pixels (required when `ao_map` is raw RGBA8 bytes).
    pub ao_map_height: Option<u32>,
    /// AO map horizontal wrap mode.
    pub ao_map_wrap_s: Option<String>,
    /// AO map vertical wrap mode.
    pub ao_map_wrap_t: Option<String>,
    /// AO map magnification filter.
    pub ao_map_mag_filter: Option<String>,
    /// AO map minification filter.
    pub ao_map_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub ao_map_anisotropy: Option<f64>,
    /// AO map UV transform `[a, c, tx, b, d, ty]`.
    pub ao_map_transform: Option<Vec<f64>>,
    /// AO map color space. `"srgb"` decodes the red channel to linear before shading.
    pub ao_map_color_space: Option<String>,
    /// Whether AO map sampling uses the secondary UV stream.
    pub ao_map_uses_uv2: Option<bool>,
    /// AO map intensity multiplier (0..1). Defaults to 1.
    pub ao_map_intensity: Option<f64>,
    /// Optional light map image data (raw RGBA8 bytes or encoded PNG/JPEG/WebP).
    /// RGB channels add baked indirect diffuse lighting.
    pub light_map: Option<Buffer>,
    /// Light map width in pixels (required when `light_map` is raw RGBA8 bytes).
    pub light_map_width: Option<u32>,
    /// Light map height in pixels (required when `light_map` is raw RGBA8 bytes).
    pub light_map_height: Option<u32>,
    /// Light map horizontal wrap mode.
    pub light_map_wrap_s: Option<String>,
    /// Light map vertical wrap mode.
    pub light_map_wrap_t: Option<String>,
    /// Light map magnification filter.
    pub light_map_mag_filter: Option<String>,
    /// Light map minification filter.
    pub light_map_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub light_map_anisotropy: Option<f64>,
    /// Light map UV transform `[a, c, tx, b, d, ty]`.
    pub light_map_transform: Option<Vec<f64>>,
    /// Light-map texture color space. `"srgb"` is decoded to linear before shading.
    pub light_map_color_space: Option<String>,
    /// Whether light-map sampling uses the secondary UV stream.
    pub light_map_uses_uv2: Option<bool>,
    /// Light map intensity multiplier. Defaults to 1.
    pub light_map_intensity: Option<f64>,
    /// Optional alpha map image data (raw RGBA8 bytes or encoded PNG/JPEG/WebP).
    /// Green channel multiplies material opacity, matching Three.js.
    pub alpha_map: Option<Buffer>,
    /// Alpha map width in pixels (required when `alpha_map` is raw RGBA8 bytes).
    pub alpha_map_width: Option<u32>,
    /// Alpha map height in pixels (required when `alpha_map` is raw RGBA8 bytes).
    pub alpha_map_height: Option<u32>,
    /// Alpha map horizontal wrap mode.
    pub alpha_map_wrap_s: Option<String>,
    /// Alpha map vertical wrap mode.
    pub alpha_map_wrap_t: Option<String>,
    /// Alpha map magnification filter.
    pub alpha_map_mag_filter: Option<String>,
    /// Alpha map minification filter.
    pub alpha_map_min_filter: Option<String>,
    /// Texture anisotropy clamp requested for this sampler. Defaults to 1.
    pub alpha_map_anisotropy: Option<f64>,
    /// Alpha map UV transform `[a, c, tx, b, d, ty]`.
    pub alpha_map_transform: Option<Vec<f64>>,
    /// Alpha map color space. `"srgb"` decodes the green channel to linear before opacity.
    pub alpha_map_color_space: Option<String>,
    /// Whether alpha map sampling uses the secondary UV stream.
    pub alpha_map_uses_uv2: Option<bool>,
    /// Alpha test cutoff threshold (0..1). Fragments with alpha below this are discarded.
    pub alpha_test: Option<f64>,
    /// Whether to apply stochastic alpha hashing before depth/color writes.
    pub alpha_hash: Option<bool>,
    /// Whether to enable MSAA alpha-to-coverage when the render sample count is greater than 1.
    pub alpha_to_coverage: Option<bool>,
    /// Whether fragment RGB output is multiplied by output alpha before blending.
    pub premultiplied_alpha: Option<bool>,
    /// Whether material output is tone mapped before output color-space conversion. Defaults to true.
    pub tone_mapped: Option<bool>,
    /// Flattened world-space clipping planes `[nx, ny, nz, constant, ...]`, up to 8 planes.
    /// Fragments with `dot(normal, worldPosition) + constant < 0` are clipped.
    pub clipping_planes: Option<Vec<f64>>,
    /// Number of leading clipping planes evaluated as union planes. Remaining planes use
    /// intersection semantics. Defaults to all planes as union planes.
    pub clipping_union_count: Option<u32>,
    /// Whether clipping planes affect this mesh in the shadow pass.
    pub clip_shadows: Option<bool>,
    /// Whether the mesh is transparent (sorted back-to-front, no depth write).
    pub transparent: Option<bool>,
    /// Material blending mode: `"none"`, `"normal"`, `"additive"`, `"subtractive"`, `"multiply"`, or `"custom"`.
    pub blending: Option<String>,
    /// Custom blending equation constants when `blending` is `"custom"`.
    pub blend_equation: Option<u32>,
    pub blend_src: Option<u32>,
    pub blend_dst: Option<u32>,
    pub blend_equation_alpha: Option<u32>,
    pub blend_src_alpha: Option<u32>,
    pub blend_dst_alpha: Option<u32>,
    /// Custom blend constant color `[r, g, b]`.
    pub blend_color: Option<Vec<f64>>,
    /// Custom blend constant alpha.
    pub blend_alpha: Option<f64>,
    /// Whether depth testing is enabled. Defaults to true.
    pub depth_test: Option<bool>,
    /// Depth comparison function. Defaults to `"less-equal"`.
    pub depth_func: Option<String>,
    /// Whether passing fragments write depth. Defaults to true for opaque meshes and false for transparent meshes.
    pub depth_write: Option<bool>,
    /// Whether fragments write color. Defaults to true.
    pub color_write: Option<bool>,
    /// Whether to apply material polygon offset as depth bias in the main pass.
    pub polygon_offset: Option<bool>,
    /// Slope-scaled polygon offset factor.
    pub polygon_offset_factor: Option<f64>,
    /// Constant polygon offset units.
    pub polygon_offset_units: Option<f64>,
    /// Whether stencil testing/writes are enabled for this material.
    pub stencil_write: Option<bool>,
    /// Stencil write bit mask. Defaults to 0xff.
    pub stencil_write_mask: Option<u32>,
    /// Stencil comparison function constant. Defaults to AlwaysStencilFunc.
    pub stencil_func: Option<u32>,
    /// Stencil reference value. Defaults to 0.
    pub stencil_ref: Option<u32>,
    /// Stencil comparison bit mask. Defaults to 0xff.
    pub stencil_func_mask: Option<u32>,
    /// Operation when the stencil test fails. Defaults to KeepStencilOp.
    pub stencil_fail: Option<u32>,
    /// Operation when the stencil test passes but depth fails. Defaults to KeepStencilOp.
    pub stencil_z_fail: Option<u32>,
    /// Operation when both stencil and depth pass. Defaults to KeepStencilOp.
    pub stencil_z_pass: Option<u32>,
    /// Which faces to render: `"front"` (default), `"back"`, or `"double"`.
    pub side: Option<String>,
    /// Optional shadow-pass face filter: `"front"`, `"back"`, or `"double"`. Unset preserves no-cull shadow casters.
    pub shadow_side: Option<String>,
    /// Shading model: `"standard"` (default PBR), `"basic"` (unlit / MeshBasicMaterial),
    /// or `"lambert"` (diffuse-only / MeshLambertMaterial).
    pub shading_model: Option<String>,
    /// Whether this mesh samples the scene/material environment map when one exists.
    pub use_environment_map: Option<bool>,
    /// Per-mesh environment intensity. Defaults to the scene environment intensity.
    pub environment_map_intensity: Option<f64>,
    /// Legacy environment blend operation: 0=multiply, 1=mix, 2=add.
    pub environment_map_combine: Option<u32>,
    /// Legacy material environment reflectivity multiplier.
    pub environment_map_reflectivity: Option<f64>,
    /// Whether the material environment map uses refraction instead of reflection.
    pub environment_map_refraction: Option<bool>,
    /// Legacy material environment refraction ratio.
    pub environment_map_refraction_ratio: Option<f64>,
    /// Primitive topology: `"triangles"` (default), `"lines"` (LineList), or `"points"`.
    pub topology: Option<String>,
    /// WGSL fragment shader body for the custom material path.
    pub custom_fragment_shader: Option<String>,
    /// Whether this mesh casts shadows in the shadow pass. Defaults to false.
    pub cast_shadow: Option<bool>,
    /// Whether this mesh receives shadows in the main pass. Defaults to false.
    pub receive_shadow: Option<bool>,
    /// Three.js groupOrder inherited from the nearest Group.renderOrder. Defaults to 0.
    pub group_order: Option<f64>,
    /// Explicit Three.js object renderOrder. Defaults to 0.
    pub render_order: Option<f64>,
    /// Projected camera-space sort key provided by the Three.js adapter.
    pub sort_z: Option<f64>,
    /// Stable object id / adapter insertion-order tie-breaker.
    pub sort_index: Option<u32>,
    /// WebGLRenderer material variant sort key: skinned meshes add 1, instanced meshes add 2.
    pub material_variant: Option<u32>,
    /// Material id sort key used for opaque draw ordering.
    pub material_sort_key: Option<u32>,
}
