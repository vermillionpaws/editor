use ash::{vk, Device};
use anyhow::Result;

pub fn create_command_pool_and_buffers(
    device: &Device,
    queue_family_index: u32,
    buffer_count: usize,
) -> Result<(vk::CommandPool, Vec<vk::CommandBuffer>)> {
    // Create command pool
    let pool_info = vk::CommandPoolCreateInfo {
        queue_family_index,
        flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        ..Default::default()
    };

    let command_pool = unsafe {
        device.create_command_pool(&pool_info, None)?
    };

    // Allocate command buffers
    let alloc_info = vk::CommandBufferAllocateInfo {
        command_pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: buffer_count as u32,
        ..Default::default()
    };

    let command_buffers = unsafe {
        device.allocate_command_buffers(&alloc_info)?
    };

    Ok((command_pool, command_buffers))
}

pub fn record_command_buffer(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    framebuffer: vk::Framebuffer,
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
) -> Result<()> {
    // Begin command buffer
    let begin_info = vk::CommandBufferBeginInfo {
        flags: vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
        ..Default::default()
    };

    unsafe {
        device.begin_command_buffer(command_buffer, &begin_info)?;
    }

    // Begin render pass
    let clear_color = vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0], // Black color
        },
    };

    let render_pass_begin_info = vk::RenderPassBeginInfo {
        render_pass,
        framebuffer,
        render_area: vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        },
        clear_value_count: 1,
        p_clear_values: &clear_color,
        ..Default::default()
    };

    unsafe {
        device.cmd_begin_render_pass(
            command_buffer,
            &render_pass_begin_info,
            vk::SubpassContents::INLINE,
        );

        // End render pass
        device.cmd_end_render_pass(command_buffer);

        // End command buffer
        device.end_command_buffer(command_buffer)?;
    }

    Ok(())
}