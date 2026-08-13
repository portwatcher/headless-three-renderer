#[derive(Clone, Debug)]
pub(super) struct EncoderPrerequisiteState {
    pub ready: bool,
    pub detail: String,
}

pub(super) async fn open_renderer_device(
    adapter: &wgpu::Adapter,
    descriptor: &wgpu::DeviceDescriptor<'_>,
) -> anyhow::Result<(wgpu::Device, wgpu::Queue, EncoderPrerequisiteState)> {
    #[cfg(target_os = "linux")]
    if adapter.get_info().backend == wgpu::Backend::Vulkan {
        match try_open_linux_vulkan_device(adapter, descriptor) {
            Ok((device, queue)) => {
                return Ok((
                    device,
                    queue,
                    EncoderPrerequisiteState {
                        ready: true,
                        detail: "single-device Vulkan external-memory, DRM-modifier/device-match, foreign-queue, and semaphore-fd extensions enabled"
                            .to_owned(),
                    },
                ));
            }
            Err(detail) => {
                let (device, queue) = adapter
                    .request_device(descriptor)
                    .await
                    .map_err(|error| anyhow::anyhow!("failed to create wgpu device: {error}"))?;
                return Ok((
                    device,
                    queue,
                    EncoderPrerequisiteState {
                        ready: false,
                        detail,
                    },
                ));
            }
        }
    }

    let (device, queue) = adapter
        .request_device(descriptor)
        .await
        .map_err(|error| anyhow::anyhow!("failed to create wgpu device: {error}"))?;
    Ok((
        device,
        queue,
        EncoderPrerequisiteState {
            ready: false,
            detail: "requires Linux with the Vulkan backend".to_owned(),
        },
    ))
}

#[cfg(target_os = "linux")]
fn try_open_linux_vulkan_device(
    adapter: &wgpu::Adapter,
    descriptor: &wgpu::DeviceDescriptor<'_>,
) -> Result<(wgpu::Device, wgpu::Queue), String> {
    use std::ffi::CStr;

    const REQUIRED_EXTENSIONS: [&CStr; 6] = [
        ash::khr::external_memory_fd::NAME,
        ash::ext::external_memory_dma_buf::NAME,
        ash::ext::image_drm_format_modifier::NAME,
        ash::ext::physical_device_drm::NAME,
        ash::ext::queue_family_foreign::NAME,
        ash::khr::external_semaphore_fd::NAME,
    ];

    // SAFETY: The guard is scoped to this adapter, and no HAL object escapes it.
    let hal_adapter = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }
        .ok_or_else(|| "the selected Vulkan adapter has no wgpu-hal Vulkan handle".to_owned())?;
    let capabilities = hal_adapter.physical_device_capabilities();
    let missing = REQUIRED_EXTENSIONS
        .iter()
        .filter(|extension| !capabilities.supports_extension(extension))
        .map(|extension| extension.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "Vulkan adapter is missing encoder-surface prerequisite extensions: {}",
            missing.join(", ")
        ));
    }

    // SAFETY: Every added extension was verified against this physical device.
    // The callback only appends extensions and leaves wgpu's features intact.
    let open_device = unsafe {
        hal_adapter.open_with_callback(
            descriptor.required_features,
            &descriptor.required_limits,
            &descriptor.memory_hints,
            Some(Box::new(|args| {
                for extension in REQUIRED_EXTENSIONS {
                    if !args.extensions.contains(&extension) {
                        args.extensions.push(extension);
                    }
                }
            })),
        )
    }
    .map_err(|error| format!("failed to enable Vulkan encoder prerequisites: {error:?}"))?;

    // SAFETY: `open_device` was created from this exact adapter and descriptor.
    unsafe {
        adapter
            .create_device_from_hal::<wgpu::hal::api::Vulkan>(open_device, descriptor)
            .map_err(|error| format!("failed to wrap Vulkan encoder-prerequisite device: {error}"))
    }
}
