use ash::{vk, Entry, Instance};
use anyhow::Result;
use raw_window_handle::HasDisplayHandle;
use std::{
    ffi::{CString, c_char},
};
use winit::window::Window;

#[cfg(debug_assertions)]
use crate::vulkan::debug;

pub fn create_instance(entry: &Entry, window: &Window) -> Result<Instance> {
    let app_name = CString::new("Vulkan Editor")?;
    let engine_name = CString::new("No Engine")?;

    let app_info = vk::ApplicationInfo {
        p_application_name: app_name.as_ptr(),
        application_version: vk::make_api_version(0, 1, 0, 0),
        p_engine_name: engine_name.as_ptr(),
        engine_version: vk::make_api_version(0, 1, 0, 0),
        api_version: vk::API_VERSION_1_0,
        ..Default::default()
    };

    // Get required instance extensions for windowing
    let display_handle = window.display_handle()?.as_raw();
    let mut extension_names = ash_window::enumerate_required_extensions(display_handle)?;

    // Add debug utils extension in debug mode
    #[cfg(debug_assertions)]
    {
        extension_names.push(ash::extensions::ext::DebugUtils::name().as_ptr());
    }

    // No need to convert as they are already *const i8 (same as *const c_char)
    let extension_names_raw: Vec<*const c_char> = extension_names.to_vec();

    // Add validation layers in debug mode
    #[cfg(debug_assertions)]
    let (layer_names, layer_names_raw) = debug::setup_validation_layers(entry)?;

    #[cfg(debug_assertions)]
    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT {
        message_severity:
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE |
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO |
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING |
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        message_type:
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL |
            vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE |
            vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        pfn_user_callback: Some(debug::vulkan_debug_callback),
        ..Default::default()
    };

    // Create the instance
    let create_info = vk::InstanceCreateInfo {
        p_application_info: &app_info,
        pp_enabled_extension_names: extension_names_raw.as_ptr(),
        enabled_extension_count: extension_names_raw.len() as u32,
        ..Default::default()
    };

    #[cfg(debug_assertions)]
    let create_info = vk::InstanceCreateInfo {
        p_application_info: &app_info,
        pp_enabled_extension_names: extension_names_raw.as_ptr(),
        enabled_extension_count: extension_names_raw.len() as u32,
        pp_enabled_layer_names: layer_names_raw.as_ptr(),
        enabled_layer_count: layer_names_raw.len() as u32,
        p_next: &debug_info as *const _ as *const std::ffi::c_void,
        ..Default::default()
    };

    let instance = unsafe { entry.create_instance(&create_info, None)? };
    Ok(instance)
}