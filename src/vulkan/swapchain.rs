use ash::{vk, Entry, Instance};
use anyhow::Result;
use winit::window::Window;

pub fn create_swapchain(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    device: &ash::Device,
    surface: vk::SurfaceKHR,
    queue_family_index: u32,
    window: &Window,
) -> Result<(ash::khr::swapchain::Device, vk::SwapchainKHR, Vec<vk::Image>, vk::Format, vk::Extent2D)> {
    let surface_loader = ash::khr::surface::Instance::new(&unsafe { Entry::load()? }, instance);

    // Query surface capabilities
    let capabilities = unsafe {
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?
    };

    // Choose surface format
    let formats = unsafe {
        surface_loader.get_physical_device_surface_formats(physical_device, surface)?
    };

    let format = formats
        .iter()
        .find(|format| {
            format.format == vk::Format::B8G8R8A8_SRGB
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or(&formats[0])
        .clone();

    // Choose present mode (prefer mailbox/triple buffering if available)
    let present_modes = unsafe {
        surface_loader.get_physical_device_surface_present_modes(physical_device, surface)?
    };

    let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
        vk::PresentModeKHR::MAILBOX
    } else {
        vk::PresentModeKHR::FIFO
    };

    // Choose swap extent
    let window_size = window.inner_size();
    let extent = vk::Extent2D {
        width: window_size.width,
        height: window_size.height,
    };

    // Create swapchain
    let swapchain_loader = ash::khr::swapchain::Device::new(instance, device);

    let queue_family_indices = [queue_family_index];
    let swapchain_create_info = vk::SwapchainCreateInfoKHR {
        surface,
        min_image_count: capabilities.min_image_count + 1,
        image_format: format.format,
        image_color_space: format.color_space,
        image_extent: extent,
        image_array_layers: 1,
        image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        image_sharing_mode: vk::SharingMode::EXCLUSIVE,
        queue_family_index_count: queue_family_indices.len() as u32,
        p_queue_family_indices: queue_family_indices.as_ptr(),
        pre_transform: capabilities.current_transform,
        composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
        present_mode,
        clipped: vk::TRUE,
        ..Default::default()
    };

    let swapchain = unsafe {
        swapchain_loader.create_swapchain(&swapchain_create_info, None)?
    };

    let swapchain_images = unsafe {
        swapchain_loader.get_swapchain_images(swapchain)?
    };

    Ok((swapchain_loader, swapchain, swapchain_images, format.format, extent))
}

pub fn create_image_views(
    device: &ash::Device,
    swapchain_images: &[vk::Image],
    format: vk::Format,
) -> Result<Vec<vk::ImageView>> {
    let mut image_views = Vec::with_capacity(swapchain_images.len());

    for &image in swapchain_images {
        let create_info = vk::ImageViewCreateInfo {
            image,
            view_type: vk::ImageViewType::TYPE_2D,
            format,
            components: vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            },
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };

        let image_view = unsafe {
            device.create_image_view(&create_info, None)?
        };

        image_views.push(image_view);
    }

    Ok(image_views)
}