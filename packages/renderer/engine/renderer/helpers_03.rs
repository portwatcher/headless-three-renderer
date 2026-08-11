use super::*;

pub(super) fn create_cubemap_with_mips(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    base_size: u32,
    mip_levels: u32,
    faces: &[Vec<u8>],
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("prefiltered cubemap"),
        size: wgpu::Extent3d {
            width: base_size,
            height: base_size,
            depth_or_array_layers: 6,
        },
        mip_level_count: mip_levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COLOR_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for mip in 0..mip_levels {
        let mip_size = (base_size >> mip).max(1);
        for face in 0..6u32 {
            let idx = (mip * 6 + face) as usize;
            if idx < faces.len() {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: mip,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: face,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &faces[idx],
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * mip_size),
                        rows_per_image: Some(mip_size),
                    },
                    wgpu::Extent3d {
                        width: mip_size,
                        height: mip_size,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
    }
    texture
}

#[cfg(test)]
mod tests {
    use super::{
        AoPhysicalBindGroupKey, BackgroundBindGroupKey, BackgroundUniforms, CustomBlendPipelineKey,
        IblBindGroupKey, MeshBufferCacheKey, PhysicalLayersTextureCacheKey, PostUniforms,
        SamplerKey, TextureBindGroupKey, TextureBindGroupKind, TextureCacheKey,
        UniformBindGroupKey, Uniforms, downsample_rgba_mip, f32_key, insert_bounded_cache,
        post_uniforms, texture_mip_level_count,
    };
    use crate::ibl::IblMaps;
    use crate::mesh::{
        BlendEquation, BlendFactor, CustomBlendState, MipmapFilter, PreparedTexture,
        PreparedTextureMipLevel, TextureFilter, WrapMode,
    };
    use crate::settings::PostProcessingSettings;
    use bytemuck::Zeroable;
    use std::collections::HashMap;

    pub(super) fn single_pixel_texture(rgba: [u8; 4]) -> PreparedTexture {
        solid_texture(1, 1, rgba)
    }

    pub(super) fn solid_texture(width: u32, height: u32, rgba: [u8; 4]) -> PreparedTexture {
        let mut data = Vec::with_capacity((width * height * 4) as usize);
        for _ in 0..(width * height) {
            data.extend_from_slice(&rgba);
        }
        PreparedTexture {
            rgba: data,
            width,
            height,
            mipmaps: Vec::new(),
            wrap_s: WrapMode::ClampToEdge,
            wrap_t: WrapMode::ClampToEdge,
            mag_filter: TextureFilter::Linear,
            min_filter: TextureFilter::Linear,
            mipmap_filter: MipmapFilter::None,
            anisotropy: 1,
        }
    }

    pub(super) fn single_pixel_ibl_maps(red: u8) -> IblMaps {
        let face = vec![red, 0, 0, 255];
        IblMaps {
            irradiance_faces: vec![face.clone(); 6],
            irradiance_size: 1,
            prefilter_faces: vec![face; 6],
            prefilter_base_size: 1,
            prefilter_mip_levels: 1,
            brdf_lut: vec![0, red, 0, 255],
            brdf_lut_size: 1,
        }
    }

    #[test]
    pub(super) fn bounded_cache_insert_reuses_existing_and_preserves_new_entry() {
        let mut cache = HashMap::new();
        assert_eq!(insert_bounded_cache(&mut cache, 1, "one", 2), "one");
        assert_eq!(insert_bounded_cache(&mut cache, 2, "two", 2), "two");
        assert_eq!(insert_bounded_cache(&mut cache, 1, "replacement", 2), "one");

        assert_eq!(insert_bounded_cache(&mut cache, 3, "three", 2), "three");
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&3), Some(&"three"));

        let uncached = insert_bounded_cache(&mut cache, 4, "four", 0);
        assert_eq!(uncached, "four");
        assert_eq!(cache.get(&4), None);
    }

    #[test]
    pub(super) fn mip_level_count_tracks_min_filter_mode() {
        assert_eq!(
            texture_mip_level_count(8, 4, MipmapFilter::None),
            1,
            "non-mipmap filters keep a single uploaded level",
        );
        assert_eq!(texture_mip_level_count(8, 4, MipmapFilter::Nearest), 4);
        assert_eq!(texture_mip_level_count(3, 5, MipmapFilter::Linear), 3);
        assert_eq!(texture_mip_level_count(1, 1, MipmapFilter::Linear), 1);
    }

    #[test]
    pub(super) fn downsample_rgba_mip_averages_source_regions() {
        let rgba = vec![
            0, 0, 0, 255, 10, 20, 30, 255, 200, 0, 0, 255, 210, 20, 30, 255,
        ];
        let (mip, width, height) = downsample_rgba_mip(&rgba, 4, 1);
        assert_eq!((width, height), (2, 1));
        assert_eq!(
            mip,
            vec![5, 10, 15, 255, 205, 10, 15, 255],
            "each output texel averages its covered source span",
        );
    }

    #[test]
    pub(super) fn downsample_rgba_mip_covers_odd_dimensions() {
        let rgba = vec![
            0, 0, 0, 255, 60, 0, 0, 255, 120, 0, 0, 255, 180, 0, 0, 255, 240, 0, 0, 255,
        ];
        let (mip, width, height) = downsample_rgba_mip(&rgba, 5, 1);
        assert_eq!((width, height), (2, 1));
        assert_eq!(mip, vec![30, 0, 0, 255, 180, 0, 0, 255]);
    }

    #[test]
    pub(super) fn sampler_key_normalizes_effective_sampler_descriptor() {
        let default = SamplerKey::new(
            WrapMode::ClampToEdge,
            WrapMode::ClampToEdge,
            TextureFilter::Linear,
            TextureFilter::Linear,
            MipmapFilter::None,
            1,
        );
        assert!(default.is_default());
        assert_eq!(default.lod_max_clamp(), 0.0);

        let anisotropic = SamplerKey::new(
            WrapMode::Repeat,
            WrapMode::MirrorRepeat,
            TextureFilter::Linear,
            TextureFilter::Linear,
            MipmapFilter::None,
            32,
        );
        assert_eq!(anisotropic.anisotropy_clamp, 16);
        assert_eq!(anisotropic.mipmap_filter, MipmapFilter::Linear);
        assert_eq!(anisotropic.lod_max_clamp(), 32.0);

        let nearest = SamplerKey::new(
            WrapMode::Repeat,
            WrapMode::ClampToEdge,
            TextureFilter::Nearest,
            TextureFilter::Linear,
            MipmapFilter::Linear,
            8,
        );
        assert_eq!(nearest.anisotropy_clamp, 1);
        assert_eq!(nearest.mipmap_filter, MipmapFilter::Linear);
        assert_eq!(nearest.lod_max_clamp(), 32.0);
    }

    #[test]
    pub(super) fn texture_cache_keys_track_pixels_and_mip_generation() {
        let base = single_pixel_texture([255, 0, 0, 255]);
        let mut sampler_variant = single_pixel_texture([255, 0, 0, 255]);
        sampler_variant.wrap_s = WrapMode::Repeat;
        sampler_variant.mag_filter = TextureFilter::Nearest;

        assert_eq!(
            TextureCacheKey::from_texture(&base),
            TextureCacheKey::from_texture(&sampler_variant),
            "sampler-only changes reuse the same uploaded texture",
        );

        let mut mipmapped = single_pixel_texture([255, 0, 0, 255]);
        mipmapped.mipmap_filter = MipmapFilter::Linear;
        assert_ne!(
            TextureCacheKey::from_texture(&base),
            TextureCacheKey::from_texture(&mipmapped),
            "generated mip-chain textures need their own cache entry",
        );

        let different_pixels = single_pixel_texture([0, 255, 0, 255]);
        assert_ne!(
            TextureCacheKey::from_texture(&base),
            TextureCacheKey::from_texture(&different_pixels),
        );
    }

    #[test]
    pub(super) fn texture_cache_keys_include_explicit_mipmaps() {
        let mut first = PreparedTexture {
            rgba: vec![
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
            ],
            width: 2,
            height: 2,
            mipmaps: vec![PreparedTextureMipLevel {
                rgba: vec![128, 128, 128, 255],
                width: 1,
                height: 1,
            }],
            wrap_s: WrapMode::ClampToEdge,
            wrap_t: WrapMode::ClampToEdge,
            mag_filter: TextureFilter::Linear,
            min_filter: TextureFilter::Linear,
            mipmap_filter: MipmapFilter::Linear,
            anisotropy: 1,
        };
        let second = PreparedTexture {
            rgba: vec![
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
            ],
            width: 2,
            height: 2,
            mipmaps: vec![PreparedTextureMipLevel {
                rgba: vec![64, 64, 64, 255],
                width: 1,
                height: 1,
            }],
            wrap_s: WrapMode::ClampToEdge,
            wrap_t: WrapMode::ClampToEdge,
            mag_filter: TextureFilter::Linear,
            min_filter: TextureFilter::Linear,
            mipmap_filter: MipmapFilter::Linear,
            anisotropy: 1,
        };
        assert_ne!(
            TextureCacheKey::from_texture(&first),
            TextureCacheKey::from_texture(&second),
        );

        first.mipmaps.clear();
        assert_ne!(
            TextureCacheKey::from_texture(&first),
            TextureCacheKey::from_texture(&second),
            "explicit mipmaps are distinct from generated mip-chain uploads",
        );
    }

    #[test]
    pub(super) fn mesh_buffer_cache_keys_track_vertex_and_index_bytes() {
        let base = MeshBufferCacheKey::from_bytes(
            &[1, 2, 3, 4, 5, 6, 7, 8],
            Some(&[0, 0, 0, 0, 1, 0, 0, 0]),
        );
        assert_eq!(
            base,
            MeshBufferCacheKey::from_bytes(
                &[1, 2, 3, 4, 5, 6, 7, 8],
                Some(&[0, 0, 0, 0, 1, 0, 0, 0]),
            ),
        );
        assert_ne!(
            base,
            MeshBufferCacheKey::from_bytes(
                &[8, 7, 6, 5, 4, 3, 2, 1],
                Some(&[0, 0, 0, 0, 1, 0, 0, 0]),
            ),
            "vertex data changes need a distinct buffer entry",
        );
        assert_ne!(
            base,
            MeshBufferCacheKey::from_bytes(&[1, 2, 3, 4, 5, 6, 7, 8], None),
            "indexed and non-indexed geometry cannot share a buffer entry",
        );
        assert_ne!(
            base,
            MeshBufferCacheKey::from_bytes(
                &[1, 2, 3, 4, 5, 6, 7, 8],
                Some(&[1, 0, 0, 0, 0, 0, 0, 0]),
            ),
            "index data changes need a distinct buffer entry",
        );
    }

    #[test]
    pub(super) fn physical_layers_cache_keys_track_effective_layer_uploads() {
        let scalar = solid_texture(2, 2, [255, 0, 0, 255]);
        let anisotropy = solid_texture(2, 2, [0, 128, 255, 255]);
        let iridescence = solid_texture(2, 2, [255, 255, 128, 255]);
        let mismatched_anisotropy = single_pixel_texture([0, 255, 0, 255]);

        assert_eq!(
            PhysicalLayersTextureCacheKey::from_layers(&scalar, None, None),
            PhysicalLayersTextureCacheKey::from_layers(&scalar, Some(&mismatched_anisotropy), None),
            "mismatched optional layers use generated defaults during upload",
        );
        assert_ne!(
            PhysicalLayersTextureCacheKey::from_layers(&scalar, None, None),
            PhysicalLayersTextureCacheKey::from_layers(&scalar, Some(&anisotropy), None),
        );
        assert_ne!(
            PhysicalLayersTextureCacheKey::from_layers(&scalar, Some(&anisotropy), None),
            PhysicalLayersTextureCacheKey::from_layers(
                &scalar,
                Some(&anisotropy),
                Some(&iridescence),
            ),
        );
    }

    #[test]
    pub(super) fn texture_bind_group_keys_track_slot_texture_and_sampler() {
        let base = single_pixel_texture([255, 0, 0, 255]);
        let same = single_pixel_texture([255, 0, 0, 255]);
        assert_eq!(
            TextureBindGroupKey::new(TextureBindGroupKind::BaseColor, &base),
            TextureBindGroupKey::new(TextureBindGroupKind::BaseColor, &same),
        );
        assert_ne!(
            TextureBindGroupKey::new(TextureBindGroupKind::BaseColor, &base),
            TextureBindGroupKey::new(TextureBindGroupKind::NormalOrBump, &same),
            "different bind group layouts cannot share entries",
        );

        let different_texture = single_pixel_texture([0, 255, 0, 255]);
        assert_ne!(
            TextureBindGroupKey::new(TextureBindGroupKind::BaseColor, &base),
            TextureBindGroupKey::new(TextureBindGroupKind::BaseColor, &different_texture),
        );

        let mut repeat_sampler = single_pixel_texture([255, 0, 0, 255]);
        repeat_sampler.wrap_s = WrapMode::Repeat;
        assert_ne!(
            TextureBindGroupKey::new(TextureBindGroupKind::BaseColor, &base),
            TextureBindGroupKey::new(TextureBindGroupKind::BaseColor, &repeat_sampler),
            "sampler state is part of the bind group resource set",
        );
    }

    #[test]
    pub(super) fn background_bind_group_keys_track_texture_and_sampler() {
        let base = single_pixel_texture([255, 0, 0, 255]);
        let same = single_pixel_texture([255, 0, 0, 255]);
        let uniforms = BackgroundUniforms::zeroed();
        assert_eq!(
            BackgroundBindGroupKey::new(&base),
            BackgroundBindGroupKey::new(&same)
        );

        let different_texture = single_pixel_texture([0, 255, 0, 255]);
        assert_ne!(
            BackgroundBindGroupKey::new(&base),
            BackgroundBindGroupKey::new(&different_texture),
        );

        let mut repeated_texture = single_pixel_texture([255, 0, 0, 255]);
        repeated_texture.wrap_t = WrapMode::Repeat;
        assert_ne!(
            BackgroundBindGroupKey::new(&base),
            BackgroundBindGroupKey::new(&repeated_texture),
            "background sampler state is part of the cached bind group",
        );

        let mut changed_uniforms = uniforms;
        changed_uniforms.camera_params[0] = 1.0;
        assert_ne!(
            bytemuck::bytes_of(&uniforms),
            bytemuck::bytes_of(&changed_uniforms),
        );
        assert_eq!(
            BackgroundBindGroupKey::new(&base),
            BackgroundBindGroupKey::new(&base),
            "camera and background uniform content update the cached buffer instead of splitting bind groups",
        );
    }

    pub(super) fn ao_physical_key(ao: Option<&PreparedTexture>) -> AoPhysicalBindGroupKey {
        let default_sampler = SamplerKey::default_texture();
        AoPhysicalBindGroupKey {
            ao: ao.map(TextureCacheKey::from_texture),
            physical_layers: None,
            physical_sheen: None,
            physical_specular: None,
            clearcoat_normal: None,
            alpha: None,
            light: None,
            ao_sampler: ao.map(SamplerKey::from_texture).unwrap_or(default_sampler),
            alpha_sampler: default_sampler,
            light_sampler: default_sampler,
            specular_sampler: default_sampler,
            physical_layers_sampler: default_sampler,
            physical_sheen_sampler: default_sampler,
            physical_specular_sampler: default_sampler,
            clearcoat_normal_sampler: default_sampler,
        }
    }

    #[test]
    pub(super) fn ao_physical_bind_group_keys_track_resources_and_samplers() {
        let ao = single_pixel_texture([255, 0, 0, 255]);
        let same_ao = single_pixel_texture([255, 0, 0, 255]);
        assert_eq!(ao_physical_key(Some(&ao)), ao_physical_key(Some(&same_ao)));

        let different_ao = single_pixel_texture([0, 255, 0, 255]);
        assert_ne!(
            ao_physical_key(Some(&ao)),
            ao_physical_key(Some(&different_ao)),
        );

        let mut repeated_ao = single_pixel_texture([255, 0, 0, 255]);
        repeated_ao.wrap_s = WrapMode::Repeat;
        assert_ne!(
            ao_physical_key(Some(&ao)),
            ao_physical_key(Some(&repeated_ao)),
            "AO sampler state is part of the combined bind group",
        );
    }

    #[test]
    pub(super) fn uniform_bind_group_keys_track_uniform_bytes() {
        let first = Uniforms::zeroed();
        let second = Uniforms::zeroed();
        assert_eq!(
            UniformBindGroupKey::from_uniforms(&first),
            UniformBindGroupKey::from_uniforms(&second),
        );

        let mut changed = first;
        changed.base_color[0] = 1.0;
        assert_ne!(
            UniformBindGroupKey::from_uniforms(&first),
            UniformBindGroupKey::from_uniforms(&changed),
        );
    }

    #[test]
    pub(super) fn post_uniforms_match_reusable_buffer_layout() {
        let uniforms = post_uniforms(PostProcessingSettings {
            active: true,
            exposure: 1.25,
            contrast: 2.0,
            saturation: 0.5,
            vignette: 0.25,
            grayscale: 0.75,
            invert: 1.0,
        });

        assert_eq!(std::mem::size_of::<PostUniforms>(), 32);
        assert_eq!(uniforms.params1, [1.25, 2.0, 0.5, 0.25]);
        assert_eq!(uniforms.params2, [0.75, 1.0, 0.0, 0.0]);
    }

    #[test]
    pub(super) fn ibl_bind_group_keys_track_uploaded_map_bytes() {
        let base = single_pixel_ibl_maps(32);
        let same = single_pixel_ibl_maps(32);
        assert_eq!(
            IblBindGroupKey::from_maps(&base),
            IblBindGroupKey::from_maps(&same),
        );

        let different_face = single_pixel_ibl_maps(64);
        assert_ne!(
            IblBindGroupKey::from_maps(&base),
            IblBindGroupKey::from_maps(&different_face),
            "IBL face bytes are part of the uploaded resource cache key",
        );

        let mut different_brdf = single_pixel_ibl_maps(32);
        different_brdf.brdf_lut[1] = 96;
        assert_ne!(
            IblBindGroupKey::from_maps(&base),
            IblBindGroupKey::from_maps(&different_brdf),
            "BRDF LUT bytes are part of the uploaded resource cache key",
        );

        let mut different_mips = single_pixel_ibl_maps(32);
        different_mips.prefilter_mip_levels = 2;
        assert_ne!(
            IblBindGroupKey::from_maps(&base),
            IblBindGroupKey::from_maps(&different_mips),
            "prefilter dimensions are part of the uploaded resource cache key",
        );
    }

    #[test]
    pub(super) fn state_pipeline_keys_ignore_dynamic_blend_constants() {
        let first = CustomBlendState {
            color_equation: BlendEquation::Add,
            alpha_equation: BlendEquation::Max,
            color_src_factor: BlendFactor::ConstantColor,
            color_dst_factor: BlendFactor::OneMinusSrcAlpha,
            alpha_src_factor: BlendFactor::One,
            alpha_dst_factor: BlendFactor::OneMinusConstantAlpha,
            constant: [0.1, 0.2, 0.3, 0.4],
        };
        let second = CustomBlendState {
            constant: [0.9, 0.8, 0.7, 0.6],
            ..first
        };

        assert_eq!(
            CustomBlendPipelineKey::from(first),
            CustomBlendPipelineKey::from(second),
            "blend constants are set on the render pass, not baked into the pipeline",
        );
    }

    #[test]
    pub(super) fn state_pipeline_float_keys_normalize_signed_zero() {
        assert_eq!(f32_key(0.0), f32_key(-0.0));
        assert_ne!(f32_key(0.0), f32_key(0.25));
    }
}
