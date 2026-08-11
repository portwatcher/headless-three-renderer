use super::*;

#[test]
fn rejects_non_finite_post_processing_inputs() {
    let cases: Vec<(&str, Box<dyn Fn(&mut RenderScene)>)> = vec![
        (
            "exposure",
            Box::new(|scene| scene.post_exposure = Some(f64::NAN)),
        ),
        (
            "contrast",
            Box::new(|scene| scene.post_contrast = Some(f64::INFINITY)),
        ),
        (
            "saturation",
            Box::new(|scene| scene.post_saturation = Some(f64::NEG_INFINITY)),
        ),
        (
            "vignette",
            Box::new(|scene| scene.post_vignette = Some(f64::NAN)),
        ),
        (
            "grayscale",
            Box::new(|scene| scene.post_grayscale = Some(f64::INFINITY)),
        ),
        (
            "invert",
            Box::new(|scene| scene.post_invert = Some(f64::NEG_INFINITY)),
        ),
    ];
    let camera = Camera::default();
    let limits = render_limits();

    for (label, mutate) in cases {
        let mut scene = RenderScene::default();
        mutate(&mut scene);
        let error = match RenderSettings::from_scene(&scene, &camera, limits.clone()) {
            Ok(_) => panic!("{label} should fail"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("must contain finite f32-compatible numbers"),
            "{label} should fail with a finite scalar error, got: {error}",
        );
    }
}

#[test]
fn rejects_bad_uv_length() {
    let scene = RenderScene {
        meshes: Some(vec![SceneMesh {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            uvs: Some(vec![0.0, 0.0, 1.0]), // wrong length
            ..SceneMesh::default()
        }]),
        ..RenderScene::default()
    };

    assert!(prepare_meshes(&scene).is_err());
}

#[test]
fn decodes_raw_rgba_texture() {
    let rgba = vec![
        255u8, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255,
    ];
    let tex = decode_texture(&rgba, Some(2), Some(2), 0).unwrap();
    assert_eq!(tex.width, 2);
    assert_eq!(tex.height, 2);
    assert_eq!(tex.rgba.len(), 16);
}

#[test]
fn accepts_mesh_with_metallic_roughness_texture() {
    let scene = RenderScene {
        meshes: Some(vec![SceneMesh {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            uvs: Some(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
            metallic: Some(1.0),
            roughness: Some(0.5),
            metallic_roughness_texture: Some(vec![0u8, 128, 255, 255].into()),
            metallic_roughness_texture_width: Some(1),
            metallic_roughness_texture_height: Some(1),
            ..SceneMesh::default()
        }]),
        ..RenderScene::default()
    };
    let meshes = prepare_meshes(&scene).unwrap();
    assert!(meshes[0].metallic_roughness_texture.is_some());
    let mr_tex = meshes[0].metallic_roughness_texture.as_ref().unwrap();
    assert_eq!(mr_tex.width, 1);
    assert_eq!(mr_tex.height, 1);
}

#[test]
fn accepts_mesh_with_multiple_builtin_texture_maps() {
    let scene = RenderScene {
        meshes: Some(vec![SceneMesh {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            uvs: Some(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
            texture: Some(vec![255u8, 128, 64, 255].into()),
            texture_width: Some(1),
            texture_height: Some(1),
            normal_map: Some(vec![128u8, 128, 255, 255].into()),
            normal_map_width: Some(1),
            normal_map_height: Some(1),
            bump_map: Some(vec![64u8, 64, 64, 255].into()),
            bump_map_width: Some(1),
            bump_map_height: Some(1),
            displacement_map: Some(vec![0u8, 0, 0, 255].into()),
            displacement_map_width: Some(1),
            displacement_map_height: Some(1),
            displacement_scale: Some(0.0),
            metallic_roughness_texture: Some(vec![0u8, 128, 255, 255].into()),
            metallic_roughness_texture_width: Some(1),
            metallic_roughness_texture_height: Some(1),
            emissive_map: Some(vec![20u8, 40, 60, 255].into()),
            emissive_map_width: Some(1),
            emissive_map_height: Some(1),
            ao_map: Some(vec![180u8, 0, 0, 255].into()),
            ao_map_width: Some(1),
            ao_map_height: Some(1),
            light_map: Some(vec![20u8, 30, 40, 255].into()),
            light_map_width: Some(1),
            light_map_height: Some(1),
            alpha_map: Some(vec![0u8, 0, 0, 200].into()),
            alpha_map_width: Some(1),
            alpha_map_height: Some(1),
            specular_map: Some(vec![90u8, 0, 0, 255].into()),
            specular_map_width: Some(1),
            specular_map_height: Some(1),
            ..SceneMesh::default()
        }]),
        ..RenderScene::default()
    };

    let meshes = prepare_meshes(&scene).unwrap();
    let mesh = &meshes[0];
    assert!(mesh.texture.is_some());
    assert!(mesh.normal_map.is_some());
    assert!(mesh.bump_map.is_some());
    assert!(mesh.metallic_roughness_texture.is_some());
    assert!(mesh.emissive_map.is_some());
    assert!(mesh.ao_map.is_some());
    assert!(mesh.light_map.is_some());
    assert!(mesh.alpha_map.is_some());
    assert!(mesh.specular_map.is_some());
}

#[test]
fn accepts_line_and_point_meshes_with_base_and_alpha_maps() {
    let scene = RenderScene {
        meshes: Some(vec![
            SceneMesh {
                positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                uvs: Some(vec![0.0, 0.0, 1.0, 0.0]),
                topology: Some("lines".into()),
                texture: Some(vec![255u8, 128, 64, 255].into()),
                texture_width: Some(1),
                texture_height: Some(1),
                alpha_map: Some(vec![0u8, 0, 0, 200].into()),
                alpha_map_width: Some(1),
                alpha_map_height: Some(1),
                ..SceneMesh::default()
            },
            SceneMesh {
                positions: vec![0.0, 0.0, 0.0],
                uvs: Some(vec![0.5, 0.5]),
                topology: Some("points".into()),
                texture: Some(vec![64u8, 128, 255, 255].into()),
                texture_width: Some(1),
                texture_height: Some(1),
                alpha_map: Some(vec![0u8, 0, 0, 180].into()),
                alpha_map_width: Some(1),
                alpha_map_height: Some(1),
                ..SceneMesh::default()
            },
        ]),
        ..RenderScene::default()
    };

    let meshes = prepare_meshes(&scene).unwrap();
    assert_eq!(meshes.len(), 2);
    for mesh in meshes {
        assert!(mesh.texture.is_some());
        assert!(mesh.alpha_map.is_some());
        assert!(mesh.normal_map.is_none());
        assert!(mesh.physical_maps.is_none());
    }
}

#[test]
fn accepts_mesh_with_physical_extension_maps() {
    let scene = RenderScene {
        meshes: Some(vec![SceneMesh {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            uvs: Some(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
            clearcoat_map: Some(vec![255u8, 0, 0, 255].into()),
            clearcoat_map_width: Some(1),
            clearcoat_map_height: Some(1),
            clearcoat_normal_map: Some(vec![128u8, 128, 255, 255].into()),
            clearcoat_normal_map_width: Some(1),
            clearcoat_normal_map_height: Some(1),
            transmission_map: Some(vec![128u8, 0, 0, 255].into()),
            transmission_map_width: Some(1),
            transmission_map_height: Some(1),
            thickness_map: Some(vec![0u8, 64, 0, 255].into()),
            thickness_map_width: Some(1),
            thickness_map_height: Some(1),
            specular_color_map: Some(vec![10u8, 20, 30, 255].into()),
            specular_color_map_width: Some(1),
            specular_color_map_height: Some(1),
            specular_intensity_map: Some(vec![0u8, 0, 0, 77].into()),
            specular_intensity_map_width: Some(1),
            specular_intensity_map_height: Some(1),
            ..SceneMesh::default()
        }]),
        ..RenderScene::default()
    };

    let meshes = prepare_meshes(&scene).unwrap();
    let mesh = &meshes[0];
    assert!(mesh.clearcoat_normal_map.is_some());
    let physical_maps = mesh.physical_maps.as_ref().unwrap();
    assert_eq!(physical_maps.scalar_map.width, 1);
    assert_eq!(physical_maps.scalar_map.height, 1);
    assert_eq!(physical_maps.scalar_map.rgba[0], 255);
    assert_eq!(physical_maps.scalar_map.rgba[2], 128);
    assert_eq!(physical_maps.scalar_map.rgba[3], 64);
    assert_eq!(&physical_maps.specular_map.rgba[..4], &[10, 20, 30, 77]);
}

/// End-to-end smoke test for directional shadow maps. Renders a ground
/// quad that receives shadows from a box above it under a downward-
/// pointing directional light. We can't pixel-match without a reference,
/// but this exercises the shadow pipeline, WGSL compilation, and PCF
/// sampling path end-to-end.
#[test]
fn renders_scene_with_directional_shadow() {
    let renderer = match GpuRenderer::new() {
        Ok(r) => r,
        Err(err) => {
            eprintln!("skipping: no wgpu adapter available ({err})");
            return;
        }
    };

    // Ground plane (receives shadow)
    let ground = SceneMesh {
        positions: vec![
            -5.0, 0.0, -5.0, 5.0, 0.0, -5.0, 5.0, 0.0, 5.0, -5.0, 0.0, 5.0,
        ],
        indices: Some(vec![0, 1, 2, 0, 2, 3]),
        normals: Some(vec![
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        ]),
        color: Some(vec![0.8, 0.8, 0.8, 1.0]),
        receive_shadow: Some(true),
        ..SceneMesh::default()
    };
    // Occluder box (casts shadow)
    let occluder = SceneMesh {
        positions: vec![
            -0.5, 1.0, -0.5, 0.5, 1.0, -0.5, 0.5, 2.0, -0.5, -0.5, 2.0, -0.5, -0.5, 1.0, 0.5, 0.5,
            1.0, 0.5, 0.5, 2.0, 0.5, -0.5, 2.0, 0.5,
        ],
        indices: Some(vec![
            0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2, 3, 2, 6, 3, 6,
            7, 0, 4, 5, 0, 5, 1,
        ]),
        color: Some(vec![0.9, 0.2, 0.2, 1.0]),
        cast_shadow: Some(true),
        ..SceneMesh::default()
    };

    let light = SceneLight {
        light_type: "directional".into(),
        color: Some(vec![1.0, 1.0, 1.0]),
        intensity: Some(1.0),
        position: Some(vec![3.0, 5.0, 3.0]),
        direction: Some(vec![-3.0, -5.0, -3.0]),
        cast_shadow: Some(true),
        shadow_map_size: Some(256),
        shadow_bias: Some(-0.0005),
        shadow_normal_bias: Some(0.02),
        shadow_camera_left: Some(-5.0),
        shadow_camera_right: Some(5.0),
        shadow_camera_top: Some(5.0),
        shadow_camera_bottom: Some(-5.0),
        shadow_camera_near: Some(0.1),
        shadow_camera_far: Some(20.0),
        ..SceneLight::default()
    };

    let scene = RenderScene {
        width: Some(64),
        height: Some(64),
        format: Some("rgba".into()),
        meshes: Some(vec![ground, occluder]),
        lights: Some(vec![light]),
        ..RenderScene::default()
    };
    let camera = Camera {
        eye: Some(vec![4.0, 3.0, 6.0]),
        target: Some(vec![0.0, 0.5, 0.0]),
        ..Camera::default()
    };

    let rgba = renderer
        .render(&scene, &camera)
        .expect("render should succeed");
    assert_eq!(rgba.len(), 64 * 64 * 4);
}

#[test]
fn renders_scene_with_spot_shadow() {
    let renderer = match GpuRenderer::new() {
        Ok(r) => r,
        Err(err) => {
            eprintln!("skipping: no wgpu adapter available ({err})");
            return;
        }
    };

    let ground = SceneMesh {
        positions: vec![
            -4.0, 0.0, -4.0, 4.0, 0.0, -4.0, 4.0, 0.0, 4.0, -4.0, 0.0, 4.0,
        ],
        indices: Some(vec![0, 1, 2, 0, 2, 3]),
        normals: Some(vec![
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        ]),
        color: Some(vec![0.75, 0.75, 0.75, 1.0]),
        receive_shadow: Some(true),
        ..SceneMesh::default()
    };
    let caster = SceneMesh {
        positions: vec![
            -0.5, 0.7, -0.5, 0.5, 0.7, -0.5, 0.5, 1.7, -0.5, -0.5, 1.7, -0.5, -0.5, 0.7, 0.5, 0.5,
            0.7, 0.5, 0.5, 1.7, 0.5, -0.5, 1.7, 0.5,
        ],
        indices: Some(vec![
            0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2, 3, 2, 6, 3, 6,
            7, 0, 4, 5, 0, 5, 1,
        ]),
        color: Some(vec![0.9, 0.25, 0.2, 1.0]),
        cast_shadow: Some(true),
        ..SceneMesh::default()
    };
    let light = SceneLight {
        light_type: "spot".into(),
        color: Some(vec![1.0, 1.0, 1.0]),
        intensity: Some(8.0),
        position: Some(vec![0.0, 5.0, 3.0]),
        direction: Some(vec![0.0, -5.0, -3.0]),
        distance: Some(10.0),
        decay: Some(2.0),
        angle: Some(0.7),
        penumbra: Some(0.2),
        cast_shadow: Some(true),
        shadow_map_size: Some(256),
        shadow_bias: Some(-0.0005),
        shadow_normal_bias: Some(0.02),
        shadow_camera_near: Some(0.1),
        shadow_camera_far: Some(12.0),
        ..SceneLight::default()
    };

    let scene = RenderScene {
        width: Some(64),
        height: Some(64),
        format: Some("rgba".into()),
        meshes: Some(vec![ground, caster]),
        lights: Some(vec![light]),
        ..RenderScene::default()
    };
    let camera = Camera {
        eye: Some(vec![4.0, 3.0, 6.0]),
        target: Some(vec![0.0, 0.5, 0.0]),
        ..Camera::default()
    };

    let rgba = renderer
        .render(&scene, &camera)
        .expect("render should succeed");
    assert_eq!(rgba.len(), 64 * 64 * 4);
}

#[test]
fn renders_scene_with_point_shadow() {
    let renderer = match GpuRenderer::new() {
        Ok(r) => r,
        Err(err) => {
            eprintln!("skipping: no wgpu adapter available ({err})");
            return;
        }
    };

    let ground = SceneMesh {
        positions: vec![
            -4.0, 0.0, -4.0, 4.0, 0.0, -4.0, 4.0, 0.0, 4.0, -4.0, 0.0, 4.0,
        ],
        indices: Some(vec![0, 1, 2, 0, 2, 3]),
        normals: Some(vec![
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        ]),
        color: Some(vec![0.75, 0.75, 0.75, 1.0]),
        receive_shadow: Some(true),
        ..SceneMesh::default()
    };
    let caster = SceneMesh {
        positions: vec![
            -0.5, 0.7, -0.5, 0.5, 0.7, -0.5, 0.5, 1.7, -0.5, -0.5, 1.7, -0.5, -0.5, 0.7, 0.5, 0.5,
            0.7, 0.5, 0.5, 1.7, 0.5, -0.5, 1.7, 0.5,
        ],
        indices: Some(vec![
            0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2, 3, 2, 6, 3, 6,
            7, 0, 4, 5, 0, 5, 1,
        ]),
        color: Some(vec![0.9, 0.25, 0.2, 1.0]),
        cast_shadow: Some(true),
        ..SceneMesh::default()
    };
    let light = SceneLight {
        light_type: "point".into(),
        color: Some(vec![1.0, 1.0, 1.0]),
        intensity: Some(45.0),
        position: Some(vec![0.0, 4.0, 2.0]),
        distance: Some(12.0),
        decay: Some(2.0),
        cast_shadow: Some(true),
        shadow_map_size: Some(256),
        shadow_bias: Some(-0.0005),
        shadow_normal_bias: Some(0.02),
        shadow_camera_near: Some(0.1),
        shadow_camera_far: Some(12.0),
        ..SceneLight::default()
    };

    let scene = RenderScene {
        width: Some(64),
        height: Some(64),
        format: Some("rgba".into()),
        meshes: Some(vec![ground, caster]),
        lights: Some(vec![light]),
        ..RenderScene::default()
    };
    let camera = Camera {
        eye: Some(vec![4.0, 3.0, 6.0]),
        target: Some(vec![0.0, 0.5, 0.0]),
        ..Camera::default()
    };

    let rgba = renderer
        .render(&scene, &camera)
        .expect("render should succeed");
    assert_eq!(rgba.len(), 64 * 64 * 4);
}
