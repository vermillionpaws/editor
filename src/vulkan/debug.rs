use ash::{vk, Entry, Instance};
use log;
use std::{
    ffi::{c_void, CStr, CString},
    os::raw::c_char,
};

use crate::vulkan::error::VulkanError;
use anyhow::{anyhow, Result};

#[cfg(debug_assertions)]
pub fn setup_validation_layers(entry: &Entry) -> Result<(Vec<CString>, Vec<*const c_char>)> {
    let validation_layer_name = CString::new("VK_LAYER_KHRONOS_validation")?;

    let available_layers = unsafe { entry.enumerate_instance_layer_properties()? };

    let layer_found = available_layers.iter().any(|layer| {
        let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
        name == validation_layer_name.as_c_str()
    });

    if !layer_found {
        return Err(anyhow!(VulkanError::ValidationLayerNotAvailable(
            validation_layer_name.to_string_lossy().into_owned()
        )));
    }

    let layer_names = vec![validation_layer_name];
    let layer_names_raw: Vec<*const c_char> = layer_names
        .iter()
        .map(|name| name.as_ptr())
        .collect();

    Ok((layer_names, layer_names_raw))
}

#[cfg(debug_assertions)]
pub unsafe fn create_debug_messenger(
    entry: &Entry,
    instance: &Instance,
) -> Result<(vk::DebugUtilsMessengerEXT, ash::ext::DebugUtils)> {
    let debug_utils_loader = ash::ext::DebugUtils::new(entry, instance);

    let messenger_ci = vk::DebugUtilsMessengerCreateInfoEXT {
        message_severity:
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE |
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO |
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING |
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        message_type:
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL |
            vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE |
            vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        pfn_user_callback: Some(vulkan_debug_callback),
        ..Default::default()
    };

    let debug_messenger = debug_utils_loader.create_debug_utils_messenger(&messenger_ci, None)?;
    Ok((debug_messenger, debug_utils_loader))
}

#[cfg(debug_assertions)]
pub extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe {
        CStr::from_ptr((*p_callback_data).p_message)
            .to_str()
            .unwrap()
    };

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
            log::debug!("{:?} - {:?}: {}", message_type, message_severity, message)
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            log::info!("{:?} - {:?}: {}", message_type, message_severity, message)
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            log::warn!("{:?} - {:?}: {}", message_type, message_severity, message)
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            log::error!("{:?} - {:?}: {}", message_type, message_severity, message)
        }
        _ => {
            log::info!("{:?} - {:?}: {}", message_type, message_severity, message)
        }
    }

    vk::FALSE
}