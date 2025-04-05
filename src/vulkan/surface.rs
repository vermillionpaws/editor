use ash::{vk, Entry, Instance};
use anyhow::Result;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

pub fn create_surface(
    entry: &Entry,
    instance: &Instance,
    window: &Window,
) -> Result<(vk::SurfaceKHR, ash::khr::surface::Instance)> {
    let surface_loader = ash::khr::surface::Instance::new(entry, instance);

    let surface = unsafe {
        ash_window::create_surface(
            entry,
            instance,
            window.display_handle()?.as_raw(),
            window.window_handle()?.as_raw(),
            None,
        )?
    };

    Ok((surface, surface_loader))
}