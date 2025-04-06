pub mod app;
pub mod command;
pub mod debug;
pub mod device;
pub mod error;
pub mod framebuffer;
pub mod instance;
pub mod render_pass;
pub mod surface;
pub mod swapchain;
pub mod shader;

pub use app::VulkanApp;
use std::sync::atomic::{AtomicU64, Ordering};

// Add this implementation for the VulkanApp struct
impl Clone for VulkanApp {
    fn clone(&self) -> Self {
        // WARNING: Proper Vulkan resource cloning may require additional handling
        // Some resources might need to be recreated rather than simply cloned
        VulkanApp {
            _entry: self._entry.clone(),
            instance: self.instance.clone(),
            device: self.device.clone(),
            physical_device: self.physical_device,
            surface: self.surface,
            surface_loader: self.surface_loader.clone(),
            queue_family_index: self.queue_family_index,
            graphics_queue: self.graphics_queue,
            swapchain: self.swapchain,
            swapchain_loader: self.swapchain_loader.clone(),
            swapchain_format: self.swapchain_format,
            swapchain_extent: self.swapchain_extent,
            swapchain_images: self.swapchain_images.clone(),
            swapchain_image_views: self.swapchain_image_views.clone(),
            render_pass: self.render_pass,
            framebuffers: self.framebuffers.clone(),
            command_pool: self.command_pool,
            command_buffers: self.command_buffers.clone(),
            image_available_semaphores: self.image_available_semaphores.clone(),
            render_finished_semaphores: self.render_finished_semaphores.clone(),
            in_flight_fences: self.in_flight_fences.clone(),
            images_in_flight: self.images_in_flight.clone(),
            current_frame: self.current_frame,
            frame_counter: AtomicU64::new(self.frame_counter.load(Ordering::Relaxed)),
            #[cfg(debug_assertions)]
            debug_messenger: self.debug_messenger,
            #[cfg(debug_assertions)]
            debug_utils: self.debug_utils.clone(),
            _marker: std::marker::PhantomData,
        }
    }
}