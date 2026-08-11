use super::*;

#[test]
fn empty_scene_prepares_no_meshes() {
    let scene = RenderScene::default();
    let meshes = prepare_meshes(&scene).unwrap();
    assert!(meshes.is_empty());
}

#[test]
fn rejects_non_triangle_unindexed_mesh() {
    let scene = RenderScene {
        meshes: Some(vec![SceneMesh {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            ..SceneMesh::default()
        }]),
        ..RenderScene::default()
    };

    assert!(prepare_meshes(&scene).is_err());
}

#[test]
fn accepts_indexed_quad() {
    let scene = RenderScene {
        meshes: Some(vec![SceneMesh {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            indices: Some(vec![0, 1, 2, 0, 2, 3]),
            color: Some(vec![1.0, 0.0, 0.0, 1.0]),
            ..SceneMesh::default()
        }]),
        ..RenderScene::default()
    };

    let meshes = prepare_meshes(&scene).unwrap();
    assert_eq!(meshes[0].vertices.len(), 4);
    assert_eq!(meshes[0].indices.as_ref().unwrap().len(), 6);
}

#[test]
fn prepares_many_meshes_in_source_order() {
    let meshes = (0..12)
        .map(|index| SceneMesh {
            positions: vec![0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.04, 0.0],
            color: Some(vec![index as f64 / 12.0, 0.2, 0.6, 1.0]),
            sort_index: Some(index),
            ..SceneMesh::default()
        })
        .collect();
    let scene = RenderScene {
        meshes: Some(meshes),
        ..RenderScene::default()
    };

    let prepared = prepare_meshes(&scene).unwrap();
    assert_eq!(prepared.len(), 12);
    for (index, mesh) in prepared.iter().enumerate() {
        assert_eq!(mesh.sort_index, index as u32);
        assert!((mesh.base_color[0] - index as f32 / 12.0).abs() < 0.0001);
    }
}

#[test]
fn accepts_mesh_with_uvs() {
    let scene = RenderScene {
        meshes: Some(vec![SceneMesh {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            uvs: Some(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
            ..SceneMesh::default()
        }]),
        ..RenderScene::default()
    };

    let meshes = prepare_meshes(&scene).unwrap();
    assert_eq!(meshes[0].vertices[0].uv, [0.0, 0.0]);
    assert_eq!(meshes[0].vertices[1].uv, [1.0, 0.0]);
    assert_eq!(meshes[0].vertices[2].uv, [0.0, 1.0]);
}

#[test]
fn rejects_non_finite_geometry_attribute_inputs() {
    let cases: Vec<(&str, Box<dyn Fn(&mut SceneMesh)>)> = vec![
        (
            "normals",
            Box::new(|mesh| {
                mesh.normals = Some(vec![0.0, 0.0, 1.0, f64::NAN, 0.0, 1.0, 0.0, 0.0, 1.0]);
            }),
        ),
        (
            "uvs",
            Box::new(|mesh| {
                mesh.uvs = Some(vec![0.0, 0.0, 1.0, f64::INFINITY, 0.0, 1.0]);
            }),
        ),
        (
            "uvs2",
            Box::new(|mesh| {
                mesh.uvs2 = Some(vec![0.0, 0.0, 1.0, 0.0, f64::NEG_INFINITY, 1.0]);
            }),
        ),
        (
            "colors",
            Box::new(|mesh| {
                mesh.colors = Some(vec![1.0, 0.0, 0.0, 0.0, f64::NAN, 0.0, 0.0, 0.0, 1.0]);
            }),
        ),
    ];

    for (label, mutate) in cases {
        let mut mesh = SceneMesh {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ..SceneMesh::default()
        };
        mutate(&mut mesh);
        let scene = RenderScene {
            meshes: Some(vec![mesh]),
            ..RenderScene::default()
        };
        let error = match prepare_meshes(&scene) {
            Ok(_) => panic!("{label} should fail"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("must contain finite"),
            "{label} should fail with a finite attribute error, got: {error}",
        );
    }
}

#[test]
fn rejects_non_finite_texture_anisotropy_inputs() {
    let cases: Vec<(&str, &str, Box<dyn Fn(&mut SceneMesh)>)> = vec![
        (
            "base map",
            "scene.meshes[0].textureAnisotropy",
            Box::new(|mesh| {
                mesh.texture = Some(vec![255u8, 255, 255, 255].into());
                mesh.texture_width = Some(1);
                mesh.texture_height = Some(1);
                mesh.texture_anisotropy = Some(f64::NAN);
            }),
        ),
        (
            "normal map",
            "scene.meshes[0].normalMapAnisotropy",
            Box::new(|mesh| {
                mesh.normal_map = Some(vec![128u8, 128, 255, 255].into());
                mesh.normal_map_width = Some(1);
                mesh.normal_map_height = Some(1);
                mesh.normal_map_anisotropy = Some(f64::INFINITY);
            }),
        ),
        (
            "physical map",
            "scene.meshes[0].clearcoatMapAnisotropy",
            Box::new(|mesh| {
                mesh.clearcoat_map = Some(vec![255u8, 255, 255, 255].into());
                mesh.clearcoat_map_width = Some(1);
                mesh.clearcoat_map_height = Some(1);
                mesh.clearcoat_map_anisotropy = Some(f64::NEG_INFINITY);
            }),
        ),
    ];

    for (label, expected_field, mutate) in cases {
        let mut mesh = SceneMesh {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            uvs: Some(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
            ..SceneMesh::default()
        };
        mutate(&mut mesh);
        let scene = RenderScene {
            meshes: Some(vec![mesh]),
            ..RenderScene::default()
        };
        let error = match prepare_meshes(&scene) {
            Ok(_) => panic!("{label} should fail"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("must contain finite f32-compatible numbers"),
            "{label} should fail with a finite anisotropy error, got: {error}",
        );
        assert!(
            error.contains(expected_field),
            "{label} should name {expected_field}, got: {error}",
        );
    }
}

#[test]
fn rejects_non_finite_material_scalar_inputs() {
    let cases: Vec<(&str, Box<dyn Fn(&mut SceneMesh)>)> = vec![
        ("metallic", Box::new(|mesh| mesh.metallic = Some(f64::NAN))),
        (
            "roughness",
            Box::new(|mesh| mesh.roughness = Some(f64::INFINITY)),
        ),
        (
            "clearcoat",
            Box::new(|mesh| mesh.clearcoat = Some(f64::NEG_INFINITY)),
        ),
        (
            "clearcoatNormalScale",
            Box::new(|mesh| mesh.clearcoat_normal_scale = Some(vec![1.0, f64::NAN])),
        ),
        (
            "sheenColor",
            Box::new(|mesh| mesh.sheen_color = Some(vec![0.5, f64::NAN, 0.5])),
        ),
        (
            "sheenRoughness",
            Box::new(|mesh| mesh.sheen_roughness = Some(f64::INFINITY)),
        ),
        (
            "anisotropy",
            Box::new(|mesh| mesh.anisotropy = Some(f64::NAN)),
        ),
        (
            "iridescence",
            Box::new(|mesh| mesh.iridescence = Some(f64::INFINITY)),
        ),
        (
            "transmission",
            Box::new(|mesh| mesh.transmission = Some(f64::NAN)),
        ),
        ("ior", Box::new(|mesh| mesh.ior = Some(f64::INFINITY))),
        (
            "thickness",
            Box::new(|mesh| mesh.thickness = Some(f64::NAN)),
        ),
        (
            "attenuationDistance",
            Box::new(|mesh| mesh.attenuation_distance = Some(f64::INFINITY)),
        ),
        (
            "attenuationColor",
            Box::new(|mesh| mesh.attenuation_color = Some(vec![1.0, f64::NAN, 1.0])),
        ),
        (
            "physicalSpecularColor",
            Box::new(|mesh| mesh.physical_specular_color = Some(vec![1.0, 1.0, f64::NAN])),
        ),
        (
            "physicalSpecularIntensity",
            Box::new(|mesh| mesh.physical_specular_intensity = Some(f64::NAN)),
        ),
        (
            "specularColor",
            Box::new(|mesh| mesh.specular_color = Some(vec![f64::NAN, 0.0, 0.0])),
        ),
        (
            "emissiveIntensity",
            Box::new(|mesh| mesh.emissive_intensity = Some(f64::INFINITY)),
        ),
        (
            "emissive",
            Box::new(|mesh| mesh.emissive = Some(vec![0.0, f64::NAN, 0.0])),
        ),
        (
            "alphaTest",
            Box::new(|mesh| mesh.alpha_test = Some(f64::NAN)),
        ),
        (
            "normalScale",
            Box::new(|mesh| mesh.normal_scale = Some(vec![f64::NAN, 1.0])),
        ),
        (
            "aoMapIntensity",
            Box::new(|mesh| mesh.ao_map_intensity = Some(f64::INFINITY)),
        ),
        (
            "blendAlpha",
            Box::new(|mesh| {
                mesh.blending = Some("custom".into());
                mesh.blend_alpha = Some(f64::NAN);
            }),
        ),
    ];

    for (label, mutate) in cases {
        let mut mesh = SceneMesh {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ..SceneMesh::default()
        };
        mutate(&mut mesh);
        let scene = RenderScene {
            meshes: Some(vec![mesh]),
            ..RenderScene::default()
        };
        let error = match prepare_meshes(&scene) {
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
fn rejects_non_finite_light_scalar_inputs() {
    let cases: Vec<(&str, &str, Box<dyn Fn(&mut SceneLight)>)> = vec![
        (
            "intensity",
            "directional",
            Box::new(|light| light.intensity = Some(f64::NAN)),
        ),
        (
            "distance",
            "point",
            Box::new(|light| light.distance = Some(f64::INFINITY)),
        ),
        (
            "decay",
            "point",
            Box::new(|light| light.decay = Some(f64::NEG_INFINITY)),
        ),
        (
            "angle",
            "spot",
            Box::new(|light| light.angle = Some(f64::NAN)),
        ),
        (
            "penumbra",
            "spot",
            Box::new(|light| light.penumbra = Some(f64::INFINITY)),
        ),
        (
            "width",
            "rectArea",
            Box::new(|light| light.width = Some(f64::NAN)),
        ),
        (
            "height",
            "rectArea",
            Box::new(|light| light.height = Some(f64::NEG_INFINITY)),
        ),
    ];

    for (label, light_type, mutate) in cases {
        let mut light = SceneLight {
            light_type: light_type.into(),
            ..SceneLight::default()
        };
        mutate(&mut light);
        let scene = scene_with_light(light);
        let error = match prepare_lights(&scene) {
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
fn rejects_non_finite_light_settings_inputs() {
    let valid_cascade_bounds = || {
        vec![
            -5.0, 5.0, 5.0, -5.0, 0.1, 10.0, -4.0, 4.0, 4.0, -4.0, 0.2, 20.0,
        ]
    };
    let cases: Vec<(&str, Box<dyn Fn(&mut RenderScene)>)> = vec![
        (
            "ambientIntensity",
            Box::new(|scene| scene.ambient_intensity = Some(f64::NAN)),
        ),
        (
            "environmentIntensity",
            Box::new(|scene| scene.environment_map_intensity = Some(f64::INFINITY)),
        ),
        (
            "shadow.camera.near",
            Box::new(|scene| {
                let mut light = directional_shadow_light();
                light.shadow_camera_near = Some(f64::NAN);
                scene.lights = Some(vec![light]);
            }),
        ),
        (
            "shadow.camera.far",
            Box::new(|scene| {
                let mut light = directional_shadow_light();
                light.shadow_camera_far = Some(f64::INFINITY);
                scene.lights = Some(vec![light]);
            }),
        ),
        (
            "shadow.camera.left",
            Box::new(|scene| {
                let mut light = directional_shadow_light();
                light.shadow_camera_left = Some(f64::NEG_INFINITY);
                scene.lights = Some(vec![light]);
            }),
        ),
        (
            "shadow.bias",
            Box::new(|scene| {
                let mut light = directional_shadow_light();
                light.shadow_bias = Some(f64::NAN);
                scene.lights = Some(vec![light]);
            }),
        ),
        (
            "shadow.normalBias",
            Box::new(|scene| {
                let mut light = directional_shadow_light();
                light.shadow_normal_bias = Some(f64::INFINITY);
                scene.lights = Some(vec![light]);
            }),
        ),
        (
            "shadow.cascades",
            Box::new(move |scene| {
                let mut light = directional_shadow_light();
                let mut bounds = valid_cascade_bounds();
                bounds[11] = f64::NAN;
                light.shadow_cascade_bounds = Some(bounds);
                scene.lights = Some(vec![light]);
            }),
        ),
        (
            "shadow.cascadeSplits",
            Box::new(|scene| {
                let mut light = directional_shadow_light();
                light.shadow_cascade_bounds = Some(vec![
                    -5.0, 5.0, 5.0, -5.0, 0.1, 10.0, -4.0, 4.0, 4.0, -4.0, 0.2, 20.0,
                ]);
                light.shadow_cascade_splits = Some(vec![f64::INFINITY]);
                scene.lights = Some(vec![light]);
            }),
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
