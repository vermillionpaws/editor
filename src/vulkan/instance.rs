use ash::{vk, Entry, Instance};
use ash::ext;
use anyhow::Result;
use raw_window_handle::HasDisplayHandle;
use std::ffi::{CString, c_char};
use winit::window::Window;

#[cfg(debug_assertions)]
use crate::vulkan::debug;

pub fn create_instance(entry: &Entry, window: &Window) -> Result<Instance> {
    let app_name = CString::new("Vulkan Editor")?;
    let engine_name = CString::new("No Engine")?;

    let app_info = vk::ApplicationInfo {
        s_type: vk::StructureType::APPLICATION_INFO,
        p_next: std::ptr::null(),
        p_application_name: app_name.as_ptr(),
        application_version: vk::make_api_version(0, 1, 0, 0),
        p_engine_name: engine_name.as_ptr(),
        engine_version: vk::make_api_version(0, 1, 0, 0),
        api_version: vk::API_VERSION_1_0,
        _marker: std::marker::PhantomData,
    };

    // Get required instance extensions for windowing
    let display_handle = window.display_handle()?.as_raw();
    let mut extension_names: Vec<*const c_char> = ash_window::enumerate_required_extensions(display_handle)?.to_vec();

    // Add debug utils extension in debug mode
    #[cfg(debug_assertions)]
    {
        let debug_utils_name = ext::DebugUtils::name().as_ptr();
        extension_names.push(debug_utils_name);
    }

    // Add validation layers in debug mode
    #[cfg(debug_assertions)]
    let (layer_names, layer_names_raw) = debug::setup_validation_layers(entry)?;

    #[cfg(debug_assertions)]
    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT {
        s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        p_next: std::ptr::null(),
        flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
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
        p_user_data: std::ptr::null_mut(),
        _marker: std::marker::PhantomData,
    };

    // Create the instance (non-debug mode)
    #[cfg(not(debug_assertions))]
    let create_info = vk::InstanceCreateInfo {
        s_type: vk::StructureType::INSTANCE_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: vk::InstanceCreateFlags::empty(),
        p_application_info: &app_info,
        pp_enabled_extension_names: extension_names.as_ptr(),
        enabled_extension_count: extension_names.len() as u32,
        pp_enabled_layer_names: std::ptr::null(),
        enabled_layer_count: 0,
        _marker: std::marker::PhantomData,
    };

    // Create the instance with debug utilities (debug mode)
    #[cfg(debug_assertions)]
    let create_info = vk::InstanceCreateInfo {
        s_type: vk::StructureType::INSTANCE_CREATE_INFO,
        p_next: &debug_info as *const _ as *const std::ffi::c_void,
        flags: vk::InstanceCreateFlags::empty(),
        p_application_info: &app_info,
        pp_enabled_extension_names: extension_names.as_ptr(),
        enabled_extension_count: extension_names.len() as u32,
        pp_enabled_layer_names: layer_names_raw.as_ptr(),
        enabled_layer_count: layer_names_raw.len() as u32,
        _marker: std::marker::PhantomData,
    };

    let instance = unsafe { entry.create_instance(&create_info, None)? };
    Ok(instance)
}