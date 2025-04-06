use anyhow::Result;
use ash::{vk, Entry, Instance};
use winit::window::Window;
use std::sync::atomic::AtomicU64;

use crate::vulkan::{
    command,
    device,
    framebuffer,
    instance,
    render_pass,
    surface,
    swapchain,
};

pub struct VulkanApp {
    pub(crate) _entry: Entry,
    pub(crate) instance: Instance,
    pub(crate) device: ash::Device,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) surface: vk::SurfaceKHR,
    pub(crate) surface_loader: ash::khr::surface::Instance,
    pub(crate) queue_family_index: u32,
    pub(crate) graphics_queue: vk::Queue,
    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) swapchain_loader: ash::khr::swapchain::Device,
    pub(crate) swapchain_format: vk::Format,
    pub(crate) swapchain_extent: vk::Extent2D,
    pub(crate) swapchain_images: Vec<vk::Image>,
    pub(crate) swapchain_image_views: Vec<vk::ImageView>,
    pub(crate) render_pass: vk::RenderPass,
    pub(crate) framebuffers: Vec<vk::Framebuffer>,
    pub(crate) command_pool: vk::CommandPool,
    pub(crate) command_buffers: Vec<vk::CommandBuffer>,
    pub(crate) image_available_semaphores: Vec<vk::Semaphore>,
    pub(crate) render_finished_semaphores: Vec<vk::Semaphore>,
    pub(crate) in_flight_fences: Vec<vk::Fence>,
    pub(crate) images_in_flight: Vec<vk::Fence>,
    pub(crate) current_frame: usize,
    pub(crate) frame_counter: AtomicU64,
    #[cfg(debug_assertions)]
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
    #[cfg(debug_assertions)]
    pub(crate) debug_utils: ash::ext::debug_utils::Instance,
    pub(crate) _marker: std::marker::PhantomData<()>,
}

const MAX_FRAMES_IN_FLIGHT: usize = 2;

impl VulkanApp {
    pub fn new(window: &Window) -> Result<Self> {
        // Load Vulkan Library
        let entry = unsafe { Entry::load()? };

        // Create instance with required extensions for windowing
        let instance = instance::create_instance(&entry, window)?;

        // Set up debug messenger if in debug mode
        #[cfg(debug_assertions)]
        let (debug_messenger, debug_utils) = unsafe {
            use crate::vulkan::debug;
            debug::create_debug_messenger(&entry, &instance)?
        };

        // Create surface
        let (surface, surface_loader) = surface::create_surface(&entry, &instance, window)?;

        // Select physical device
        let (physical_device, queue_family_index) =
            device::select_physical_device(&instance, &surface_loader, surface)?;

        // Create logical device and get graphics queue
        let (device, graphics_queue) =
            device::create_logical_device(&instance, physical_device, queue_family_index)?;

        // Create swapchain
        let (swapchain_loader, swapchain, swapchain_images, swapchain_format, swapchain_extent) =
            swapchain::create_swapchain(
                &instance,
                physical_device,
                &device,
                surface,
                queue_family_index,
                window,
            )?;

        // Create image views
        let swapchain_image_views =
            swapchain::create_image_views(&device, &swapchain_images, swapchain_format)?;

        // Create render pass
        let render_pass = render_pass::create_render_pass(&device, swapchain_format)?;

        // Create framebuffers
        let framebuffers = framebuffer::create_framebuffers(
            &device,
            render_pass,
            &swapchain_image_views,
            swapchain_extent,
        )?;

        // Create command pool and buffers
        let (command_pool, command_buffers) = command::create_command_pool_and_buffers(
            &device,
            queue_family_index,
            swapchain_image_views.len(),
        )?;

        // Record command buffers for each framebuffer
        for (i, &command_buffer) in command_buffers.iter().enumerate() {
            command::record_command_buffer(
                &device,
                command_buffer,
                framebuffers[i],
                render_pass,
                swapchain_extent,
                None,  // No UI renderer initially
                i,     // framebuffer index
            )?;
        }

        // Create synchronization objects
        let mut image_available_semaphores = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut render_finished_semaphores = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut in_flight_fences = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let images_in_flight = vec![vk::Fence::null(); swapchain_images.len()];

        let semaphore_create_info = vk::SemaphoreCreateInfo::default();
        let fence_create_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                let image_available_semaphore = device.create_semaphore(&semaphore_create_info, None)?;
                let render_finished_semaphore = device.create_semaphore(&semaphore_create_info, None)?;
                let in_flight_fence = device.create_fence(&fence_create_info, None)?;

                image_available_semaphores.push(image_available_semaphore);
                render_finished_semaphores.push(render_finished_semaphore);
                in_flight_fences.push(in_flight_fence);
            }
        }

        Ok(VulkanApp {
            _entry: entry,
            instance,
            device,
            physical_device,
            surface,
            surface_loader,
            queue_family_index,
            graphics_queue,
            swapchain,
            swapchain_loader,
            swapchain_format,
            swapchain_extent,
            swapchain_images,
            swapchain_image_views,
            render_pass,
            framebuffers,
            command_pool,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            images_in_flight,
            current_frame: 0,
            frame_counter: AtomicU64::new(0),
            #[cfg(debug_assertions)]
            debug_utils,
            #[cfg(debug_assertions)]
            debug_messenger,
            _marker: std::marker::PhantomData::default(),
        })
    }

    pub fn update_command_buffer(&mut self, image_index: usize, ui_renderer: &crate::ui::UiRenderer) -> Result<()> {
        // Reset the command buffer to begin recording
        unsafe {
            self.device.reset_command_buffer(
                self.command_buffers[image_index],
                vk::CommandBufferResetFlags::empty(),
            )?;
        }

        // Re-record the command buffer with the latest UI elements
        command::record_command_buffer(
            &self.device,
            self.command_buffers[image_index],
            self.framebuffers[image_index],
            self.render_pass,
            self.swapchain_extent,
            Some(ui_renderer),
            image_index,
        )?;

        Ok(())
    }

    pub fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        // Wait for device to be idle before recreating swapchain
        unsafe {
            self.device.device_wait_idle()?;
        }

        // Clean up old resources
        // Destroy old framebuffers
        for framebuffer in &self.framebuffers {
            unsafe {
                self.device.destroy_framebuffer(*framebuffer, None);
            }
        }

        // Destroy old image views
        for image_view in &self.swapchain_image_views {
            unsafe {
                self.device.destroy_image_view(*image_view, None);
            }
        }

        // Get window size for new swapchain
        let window_size = window.inner_size();

        // Create new swapchain
        let (swapchain_loader, swapchain, swapchain_images, swapchain_format, swapchain_extent) =
            swapchain::create_swapchain(
                &self.instance,
                self.physical_device,
                &self.device,
                self.surface,
                self.queue_family_index,
                window,
            )?;

        // Create new image views
        let swapchain_image_views =
            swapchain::create_image_views(&self.device, &swapchain_images, swapchain_format)?;

        // Create new framebuffers
        let framebuffers = framebuffer::create_framebuffers(
            &self.device,
            self.render_pass,
            &swapchain_image_views,
            swapchain_extent,
        )?;

        // Update struct fields
        self.swapchain_loader = swapchain_loader;
        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;
        self.swapchain_format = swapchain_format;
        self.swapchain_extent = swapchain_extent;
        self.swapchain_image_views = swapchain_image_views;
        self.framebuffers = framebuffers;

        Ok(())
    }

    pub fn draw_frame(&mut self, ui_renderer: &crate::ui::UiRenderer, window: &Window) -> anyhow::Result<()> {
        let wait_fences = [self.in_flight_fences[self.current_frame]];
        unsafe {
            self.device.wait_for_fences(&wait_fences, true, std::u64::MAX)?;
        }

        let result = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                std::u64::MAX,
                self.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            )
        };

        let image_index = match result {
            Ok((image_index, _)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain(window)?;
                return Ok(());
            }
            Err(error) => return Err(anyhow::anyhow!("Failed to acquire next image: {}", error)),
        };

        // Important: Reset fence only AFTER acquiring the image
        unsafe {
            self.device.reset_fences(&wait_fences)?;
        }

        self.update_command_buffer(image_index as usize, ui_renderer)?;

        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [self.command_buffers[image_index as usize]];

        let submit_info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: std::ptr::null(),
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: command_buffers.len() as u32,
            p_command_buffers: command_buffers.as_ptr(),
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            _marker: std::marker::PhantomData,
        };

        unsafe {
            self.device.queue_submit(
                self.graphics_queue,
                &[submit_info],
                self.in_flight_fences[self.current_frame],
            )?;
        }

        let swapchains = [self.swapchain];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR {
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            p_next: std::ptr::null(),
            wait_semaphore_count: signal_semaphores.len() as u32,
            p_wait_semaphores: signal_semaphores.as_ptr(),
            swapchain_count: swapchains.len() as u32,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: image_indices.as_ptr(),
            p_results: std::ptr::null_mut(),
            _marker: std::marker::PhantomData,
        };

        let result = unsafe {
            self.swapchain_loader.queue_present(self.graphics_queue, &present_info)
        };

        match result {
            Ok(_) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                self.recreate_swapchain(window)?;
                return Ok(());
            }
            Err(error) => return Err(anyhow::anyhow!("Failed to present queue: {}", error)),
        }

        // Update current frame index for the next frame
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    pub fn get_physical_device_memory_properties(&self) -> vk::PhysicalDeviceMemoryProperties {
        unsafe {
            self.instance.get_physical_device_memory_properties(self.physical_device)
        }
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            // Clean up synchronization objects
            for semaphore in &self.image_available_semaphores {
                self.device.destroy_semaphore(*semaphore, None);
            }
            for semaphore in &self.render_finished_semaphores {
                self.device.destroy_semaphore(*semaphore, None);
            }
            for fence in &self.in_flight_fences {
                self.device.destroy_fence(*fence, None);
            }

            // Clean up command pool and buffers
            self.device.destroy_command_pool(self.command_pool, None);

            // Clean up framebuffers
            for framebuffer in &self.framebuffers {
                self.device.destroy_framebuffer(*framebuffer, None);
            }

            // Clean up render pass
            self.device.destroy_render_pass(self.render_pass, None);

            // Clean up image views
            for image_view in &self.swapchain_image_views {
                self.device.destroy_image_view(*image_view, None);
            }

            // Clean up swapchain
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);

            // Clean up surface
            self.surface_loader.destroy_surface(self.surface, None);

            #[cfg(debug_assertions)]
            self.debug_utils.destroy_debug_utils_messenger(self.debug_messenger, None);

            // Clean up device
            self.device.destroy_device(None);

            // Clean up instance
            self.instance.destroy_instance(None);
        }
    }
}