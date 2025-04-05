use ash::{vk, Instance};
use anyhow::{anyhow, Result};

use crate::vulkan::error::VulkanError;

pub fn select_physical_device(
    instance: &Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32)> {
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };

    for physical_device in physical_devices {
        if let Some(queue_family_index) = find_queue_family(instance, physical_device, surface_loader, surface)? {
            return Ok((physical_device, queue_family_index));
        }
    }

    Err(anyhow!(VulkanError::NoSuitableDevice))
}

pub fn find_queue_family(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<Option<u32>> {
    let queue_family_properties = unsafe {
        instance.get_physical_device_queue_family_properties(physical_device)
    };

    for (index, queue_family) in queue_family_properties.iter().enumerate() {
        let index = index as u32;

        // Check for graphics support
        if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            // Check if this queue family supports presentation
            let presentation_support = unsafe {
                surface_loader.get_physical_device_surface_support(
                    physical_device,
                    index,
                    surface,
                )?
            };

            if presentation_support {
                return Ok(Some(index));
            }
        }
    }

    Ok(None)
}

pub fn create_logical_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
) -> Result<(ash::Device, vk::Queue)> {
    let queue_priorities = [1.0];
    let queue_create_info = vk::DeviceQueueCreateInfo {
        queue_family_index,
        queue_count: 1,
        p_queue_priorities: queue_priorities.as_ptr(),
        ..Default::default()
    };

    // Required device extensions
    let device_extension_names_raw = [
        ash::khr::swapchain::NAME.as_ptr(),
    ];

    let device_features = vk::PhysicalDeviceFeatures::default();

    let create_info = vk::DeviceCreateInfo {
        queue_create_info_count: 1,
        p_queue_create_infos: &queue_create_info,
        enabled_extension_count: device_extension_names_raw.len() as u32,
        pp_enabled_extension_names: device_extension_names_raw.as_ptr(),
        p_enabled_features: &device_features,
        ..Default::default()
    };

    // Create the logical device
    let device = unsafe {
        instance.create_device(physical_device, &create_info, None)?
    };

    // Get the graphics queue
    let graphics_queue = unsafe {
        device.get_device_queue(queue_family_index, 0)
    };

    Ok((device, graphics_queue))
}