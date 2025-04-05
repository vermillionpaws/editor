use ash::{vk, Device};
use anyhow::Result;

pub fn create_framebuffers(
    device: &Device,
    render_pass: vk::RenderPass,
    image_views: &[vk::ImageView],
    swapchain_extent: vk::Extent2D,
) -> Result<Vec<vk::Framebuffer>> {
    let mut framebuffers = Vec::with_capacity(image_views.len());

    for &image_view in image_views {
        let attachments = [image_view];

        let framebuffer_info = vk::FramebufferCreateInfo {
            render_pass,
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            width: swapchain_extent.width,
            height: swapchain_extent.height,
            layers: 1,
            ..Default::default()
        };

        let framebuffer = unsafe {
            device.create_framebuffer(&framebuffer_info, None)?
        };

        framebuffers.push(framebuffer);
    }

    Ok(framebuffers)
}