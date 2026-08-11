use super::*;

impl GpuRenderer {
    pub(super) fn ao_physical_bind_group_for(
        &self,
        mesh: &PreparedMesh,
    ) -> AoPhysicalBindGroupResources {
        let key = AoPhysicalBindGroupKey::new(mesh);
        let ao_texture = mesh
            .ao_map
            .as_ref()
            .map(|tex| self.upload_texture("headless-three-renderer ao map", tex));
        let light_texture = mesh
            .light_map
            .as_ref()
            .map(|tex| self.upload_texture("headless-three-renderer light map", tex));
        let alpha_texture = mesh
            .alpha_map
            .as_ref()
            .map(|tex| self.upload_texture("headless-three-renderer alpha map", tex));
        let physical_layers_texture =
            match (mesh.physical_maps.as_ref(), mesh.specular_map.as_ref()) {
                (Some(maps), _) => Some(self.upload_physical_layers_texture(
                    "headless-three-renderer physical layers map",
                    &maps.scalar_map,
                    Some(&maps.anisotropy_map),
                    Some(&maps.iridescence_map),
                )),
                (None, Some(tex)) => Some(self.upload_physical_layers_texture(
                    "headless-three-renderer specular and physical layers map",
                    tex,
                    None,
                    None,
                )),
                (None, None) => None,
            };
        let physical_sheen_texture = match (
            mesh.matcap_map.as_ref(),
            mesh.gradient_map.as_ref(),
            mesh.physical_maps.as_ref(),
        ) {
            (Some(tex), _, _) => {
                Some(self.upload_texture("headless-three-renderer matcap color map", tex))
            }
            (None, Some(tex), _) => {
                Some(self.upload_texture("headless-three-renderer toon gradient map", tex))
            }
            (None, None, Some(maps)) => Some(self.upload_texture(
                "headless-three-renderer physical sheen map",
                &maps.sheen_map,
            )),
            (None, None, None) => None,
        };
        let physical_specular_texture = mesh.physical_maps.as_ref().map(|maps| {
            self.upload_texture(
                "headless-three-renderer physical specular map",
                &maps.specular_map,
            )
        });
        let clearcoat_normal_texture = mesh
            .clearcoat_normal_map
            .as_ref()
            .map(|tex| self.upload_texture("headless-three-renderer clearcoat normal map", tex));

        if let Some(bind_group) = self
            .ao_physical_bind_group_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
        {
            return AoPhysicalBindGroupResources {
                bind_group,
                ao_texture,
                light_texture,
                alpha_texture,
                physical_layers_texture,
                physical_sheen_texture,
                physical_specular_texture,
                clearcoat_normal_texture,
            };
        }

        let default_white_view = self
            ._default_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let default_normal_view = self
            ._default_normal_map_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let default_physical_layers_view =
            self._default_physical_layers_texture
                .create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    ..Default::default()
                });
        let ao_view = ao_texture
            .as_ref()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
        let light_view = light_texture
            .as_ref()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
        let alpha_view = alpha_texture
            .as_ref()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
        let physical_layers_view = physical_layers_texture.as_ref().map(|texture| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        });
        let physical_sheen_view = physical_sheen_texture
            .as_ref()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
        let physical_specular_view = physical_specular_texture
            .as_ref()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
        let clearcoat_normal_view = clearcoat_normal_texture
            .as_ref()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
        let ao_sampler = mesh
            .ao_map
            .as_ref()
            .map(|tex| {
                self.sampler_for_texture(
                    tex.wrap_s,
                    tex.wrap_t,
                    tex.mag_filter,
                    tex.min_filter,
                    tex.mipmap_filter,
                    tex.anisotropy,
                )
            })
            .unwrap_or_else(|| self.sampler.clone());
        let alpha_sampler = mesh
            .alpha_map
            .as_ref()
            .map(|tex| {
                self.sampler_for_texture(
                    tex.wrap_s,
                    tex.wrap_t,
                    tex.mag_filter,
                    tex.min_filter,
                    tex.mipmap_filter,
                    tex.anisotropy,
                )
            })
            .unwrap_or_else(|| self.sampler.clone());
        let light_sampler = mesh
            .light_map
            .as_ref()
            .map(|tex| {
                self.sampler_for_texture(
                    tex.wrap_s,
                    tex.wrap_t,
                    tex.mag_filter,
                    tex.min_filter,
                    tex.mipmap_filter,
                    tex.anisotropy,
                )
            })
            .unwrap_or_else(|| self.sampler.clone());
        let specular_sampler = match (mesh.physical_maps.as_ref(), mesh.specular_map.as_ref()) {
            (None, Some(tex)) => self.sampler_for_texture(
                tex.wrap_s,
                tex.wrap_t,
                tex.mag_filter,
                tex.min_filter,
                tex.mipmap_filter,
                tex.anisotropy,
            ),
            _ => self.sampler.clone(),
        };
        let physical_layers_sampler = mesh
            .physical_maps
            .as_ref()
            .map(|maps| self.sampler_for_settings(maps.physical_layers_sampler))
            .unwrap_or_else(|| self.sampler.clone());
        let physical_sheen_sampler = match (
            mesh.matcap_map.as_ref(),
            mesh.gradient_map.as_ref(),
            mesh.physical_maps.as_ref(),
        ) {
            (Some(tex), _, _) | (None, Some(tex), _) => self.sampler_for_texture(
                tex.wrap_s,
                tex.wrap_t,
                tex.mag_filter,
                tex.min_filter,
                tex.mipmap_filter,
                tex.anisotropy,
            ),
            (None, None, Some(maps)) => self.sampler_for_settings(maps.sheen_sampler),
            (None, None, None) => self.sampler.clone(),
        };
        let physical_specular_sampler = mesh
            .physical_maps
            .as_ref()
            .map(|maps| self.sampler_for_settings(maps.specular_sampler))
            .unwrap_or_else(|| self.sampler.clone());
        let clearcoat_normal_sampler = mesh
            .clearcoat_normal_map
            .as_ref()
            .map(|tex| {
                self.sampler_for_texture(
                    tex.wrap_s,
                    tex.wrap_t,
                    tex.mag_filter,
                    tex.min_filter,
                    tex.mipmap_filter,
                    tex.anisotropy,
                )
            })
            .unwrap_or_else(|| self.sampler.clone());

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("headless-three-renderer ao and physical maps bind group"),
            layout: &self.ao_map_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        ao_view.as_ref().unwrap_or(&default_white_view),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        physical_layers_view
                            .as_ref()
                            .unwrap_or(&default_physical_layers_view),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        physical_sheen_view.as_ref().unwrap_or(&default_white_view),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        physical_specular_view
                            .as_ref()
                            .unwrap_or(&default_white_view),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(
                        clearcoat_normal_view
                            .as_ref()
                            .unwrap_or(&default_normal_view),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(
                        alpha_view.as_ref().unwrap_or(&default_white_view),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(
                        light_view.as_ref().unwrap_or(&default_white_view),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(&ao_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::Sampler(&alpha_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&light_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::Sampler(&specular_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::Sampler(&physical_layers_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::Sampler(&physical_sheen_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: wgpu::BindingResource::Sampler(&physical_specular_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: wgpu::BindingResource::Sampler(&clearcoat_normal_sampler),
                },
            ],
        });
        let bind_group = insert_bounded_cache(
            &mut self
                .ao_physical_bind_group_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            key,
            bind_group,
            MAX_AO_PHYSICAL_BIND_GROUP_CACHE_ENTRIES,
        );

        AoPhysicalBindGroupResources {
            bind_group,
            ao_texture,
            light_texture,
            alpha_texture,
            physical_layers_texture,
            physical_sheen_texture,
            physical_specular_texture,
            clearcoat_normal_texture,
        }
    }

    pub(super) fn upload_physical_layers_texture(
        &self,
        label: &'static str,
        scalar: &PreparedTexture,
        anisotropy: Option<&PreparedTexture>,
        iridescence: Option<&PreparedTexture>,
    ) -> wgpu::Texture {
        let key = PhysicalLayersTextureCacheKey::from_layers(scalar, anisotropy, iridescence);
        if let Some(texture) = self
            .physical_layers_texture_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
        {
            return texture;
        }

        let texture =
            self.upload_physical_layers_texture_uncached(label, scalar, anisotropy, iridescence);
        insert_bounded_cache(
            &mut self
                .physical_layers_texture_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            key,
            texture,
            MAX_PHYSICAL_LAYERS_TEXTURE_CACHE_ENTRIES,
        )
    }

    pub(super) fn upload_physical_layers_texture_uncached(
        &self,
        label: &'static str,
        scalar: &PreparedTexture,
        anisotropy: Option<&PreparedTexture>,
        iridescence: Option<&PreparedTexture>,
    ) -> wgpu::Texture {
        let tex_size = wgpu::Extent3d {
            width: scalar.width,
            height: scalar.height,
            depth_or_array_layers: 3,
        };
        let mip_level_count =
            texture_mip_level_count(scalar.width, scalar.height, scalar.mipmap_filter);
        let gpu_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: tex_size,
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.write_texture_mip_chain(
            &gpu_texture,
            0,
            &scalar.rgba,
            scalar.width,
            scalar.height,
            mip_level_count,
        );

        let mut default_anisotropy = Vec::new();
        let anisotropy_rgba = match anisotropy {
            Some(tex) if tex.width == scalar.width && tex.height == scalar.height => {
                tex.rgba.as_slice()
            }
            _ => {
                default_anisotropy.reserve_exact((scalar.width * scalar.height * 4) as usize);
                for _ in 0..(scalar.width * scalar.height) {
                    default_anisotropy.extend_from_slice(&[255u8, 128, 255, 255]);
                }
                default_anisotropy.as_slice()
            }
        };
        self.write_texture_mip_chain(
            &gpu_texture,
            1,
            anisotropy_rgba,
            scalar.width,
            scalar.height,
            mip_level_count,
        );

        let mut default_iridescence = Vec::new();
        let iridescence_rgba = match iridescence {
            Some(tex) if tex.width == scalar.width && tex.height == scalar.height => {
                tex.rgba.as_slice()
            }
            _ => {
                default_iridescence.reserve_exact((scalar.width * scalar.height * 4) as usize);
                for _ in 0..(scalar.width * scalar.height) {
                    default_iridescence.extend_from_slice(&[255u8, 255, 255, 255]);
                }
                default_iridescence.as_slice()
            }
        };
        self.write_texture_mip_chain(
            &gpu_texture,
            2,
            iridescence_rgba,
            scalar.width,
            scalar.height,
            mip_level_count,
        );
        gpu_texture
    }

    pub(super) fn write_texture_mip_chain(
        &self,
        gpu_texture: &wgpu::Texture,
        array_layer: u32,
        rgba: &[u8],
        width: u32,
        height: u32,
        mip_level_count: u32,
    ) {
        self.write_texture_mip(gpu_texture, 0, array_layer, rgba, width, height);
        if mip_level_count <= 1 {
            return;
        }

        let mut previous = rgba.to_vec();
        let mut previous_width = width;
        let mut previous_height = height;
        for mip_level in 1..mip_level_count {
            let (next, next_width, next_height) =
                downsample_rgba_mip(&previous, previous_width, previous_height);
            self.write_texture_mip(
                gpu_texture,
                mip_level,
                array_layer,
                &next,
                next_width,
                next_height,
            );
            previous = next;
            previous_width = next_width;
            previous_height = next_height;
        }
    }

    pub(super) fn write_prepared_texture_mip_chain(
        &self,
        gpu_texture: &wgpu::Texture,
        array_layer: u32,
        texture: &PreparedTexture,
        mip_level_count: u32,
    ) {
        self.write_texture_mip(
            gpu_texture,
            0,
            array_layer,
            &texture.rgba,
            texture.width,
            texture.height,
        );
        if texture.mipmaps.is_empty() {
            if mip_level_count <= 1 {
                return;
            }
            let mut previous = texture.rgba.clone();
            let mut previous_width = texture.width;
            let mut previous_height = texture.height;
            for mip_level in 1..mip_level_count {
                let (next, next_width, next_height) =
                    downsample_rgba_mip(&previous, previous_width, previous_height);
                self.write_texture_mip(
                    gpu_texture,
                    mip_level,
                    array_layer,
                    &next,
                    next_width,
                    next_height,
                );
                previous = next;
                previous_width = next_width;
                previous_height = next_height;
            }
            return;
        }

        for (index, mip) in texture.mipmaps.iter().enumerate() {
            self.write_texture_mip(
                gpu_texture,
                (index + 1) as u32,
                array_layer,
                &mip.rgba,
                mip.width,
                mip.height,
            );
        }
    }
}
