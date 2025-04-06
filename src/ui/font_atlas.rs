use ash::vk;
use anyhow::Result;
use ab_glyph::{Font as AbFont, FontRef, ScaleFont, PxScale};
use std::collections::HashMap;
use log::trace;
use std::path::Path;

use crate::vulkan::VulkanApp;
use super::font::Font;

// Character data for texture atlas
#[derive(Clone, Copy, Debug)]
pub struct CharData {
    pub size: [f32; 2],      // Width and height of the character
    pub offset: [f32; 2],    // Offset from baseline
    pub uv_rect: [f32; 4],   // Texture coordinates [u0, v0, u1, v1]
    pub advance: f32,        // How much to advance cursor after this character
}

pub struct FontAtlas {
    // Font texture
    pub texture: vk::Image,
    pub texture_memory: vk::DeviceMemory,
    pub texture_view: vk::ImageView,
    pub texture_sampler: vk::Sampler,

    // Atlas dimensions
    width: u32,
    height: u32,

    // Character data mapping
    char_data: HashMap<char, CharData>,
}

impl FontAtlas {
    pub fn new(app: &VulkanApp, font: &Font) -> Result<Self> {
        // Load font
        let font_path = Path::new("assets/fonts/JetBrainsMono-Regular.ttf");
        let font_data = std::fs::read(font_path)?;
        let ab_font = FontRef::try_from_slice(&font_data)?;

        // Set up parameters for the atlas
        let font_size = PxScale::from(font.size as f32);
        let scaled_font = ab_font.as_scaled(font_size);

        // Find atlas dimensions by measuring all characters
        let chars_to_include: Vec<char> = (32..127).map(|c| c as u8 as char).collect();

        // First pass: determine atlas size
        let mut max_height = 0.0f32;
        let mut total_width = 0.0f32;

        for c in &chars_to_include {
            if let Some(glyph) = scaled_font.outline_glyph(scaled_font.scaled_glyph(*c)) {
                let bounds = glyph.px_bounds();
                max_height = max_height.max(bounds.height());
                total_width += bounds.width() + 2.0; // Add some padding
            }
        }

        // Add buffer for height
        max_height += 4.0;

        // Atlas dimensions (power of 2 for compatibility)
        let width = next_power_of_two_u32(total_width as u32);
        let height = next_power_of_two_u32(max_height as u32);

        trace!("Creating font atlas with dimensions {}x{}", width, height);

        // Create texture
        let (texture, texture_memory) = create_texture(
            app,
            width,
            height,
            vk::Format::R8_UNORM,
        )?;

        // Create image view
        let view_create_info = vk::ImageViewCreateInfo {
            s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::ImageViewCreateFlags::empty(),
            image: texture,
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::R8_UNORM,
            components: vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::R,
                b: vk::ComponentSwizzle::R,
                a: vk::ComponentSwizzle::R,
            },
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            _marker: std::marker::PhantomData,
        };

        let texture_view = unsafe {
            app.device.create_image_view(&view_create_info, None)?
        };

        // Create sampler
        let sampler_info = vk::SamplerCreateInfo {
            s_type: vk::StructureType::SAMPLER_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::SamplerCreateFlags::empty(),
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            mip_lod_bias: 0.0,
            anisotropy_enable: vk::FALSE,
            max_anisotropy: 1.0,
            compare_enable: vk::FALSE,
            compare_op: vk::CompareOp::ALWAYS,
            min_lod: 0.0,
            max_lod: 0.0,
            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
            unnormalized_coordinates: vk::FALSE,
            _marker: std::marker::PhantomData,
        };

        let texture_sampler = unsafe {
            app.device.create_sampler(&sampler_info, None)?
        };

        // Create atlas buffer
        let mut atlas_data = vec![0u8; (width * height) as usize];

        // Second pass: render glyphs to atlas
        let mut char_data = HashMap::new();
        let mut x_pos = 0.0;

        for c in chars_to_include {
            if let Some(glyph) = scaled_font.outline_glyph(scaled_font.scaled_glyph(c)) {
                // Get the bounds of the glyph
                let bounds = glyph.px_bounds();

                // Skip zero-width glyphs
                if bounds.width() < 1.0 {
                    continue;
                }

                // Get glyph dimensions and offset
                let glyph_width = bounds.width().ceil() as usize;
                let glyph_height = bounds.height().ceil() as usize;
                let advance = scaled_font.h_advance(ab_font.glyph_id(c));

                // Skip if no pixels to draw
                if glyph_width == 0 || glyph_height == 0 {
                    continue;
                }

                // Calculate UV coordinates
                let u0 = x_pos / width as f32;
                let v0 = 0.0;
                let u1 = (x_pos + bounds.width()) / width as f32;
                let v1 = bounds.height() / height as f32;

                // Add character data to the map
                char_data.insert(c, CharData {
                    size: [bounds.width(), bounds.height()],
                    offset: [bounds.min.x, bounds.min.y],
                    uv_rect: [u0, v0, u1, v1],
                    advance,
                });

                // Render glyph to the atlas
                let mut pixel_buffer = vec![0u8; glyph_width * glyph_height];
                glyph.draw(|x, y, v| {
                    let x = x as usize;
                    let y = y as usize;
                    if x < glyph_width && y < glyph_height {
                        pixel_buffer[y * glyph_width + x] = (v * 255.0) as u8;
                    }
                });

                // Copy to atlas
                let x_start = x_pos as usize;
                for y in 0..glyph_height {
                    for x in 0..glyph_width {
                        let atlas_idx = y * width as usize + (x_start + x);
                        if atlas_idx < atlas_data.len() {
                            atlas_data[atlas_idx] = pixel_buffer[y * glyph_width + x];
                        }
                    }
                }

                x_pos += bounds.width() + 2.0; // Add padding
            }
        }

        // Upload atlas data to GPU
        upload_texture_data(app, texture, width, height, &atlas_data)?;

        Ok(Self {
            texture,
            texture_memory,
            texture_view,
            texture_sampler,
            width,
            height,
            char_data,
        })
    }

    // Get character data for a specific character
    pub fn get_char(&self, c: char) -> Option<&CharData> {
        self.char_data.get(&c)
    }

    // Clean up Vulkan resources
    pub fn cleanup(&self, app: &VulkanApp) {
        unsafe {
            app.device.destroy_sampler(self.texture_sampler, None);
            app.device.destroy_image_view(self.texture_view, None);
            app.device.destroy_image(self.texture, None);
            app.device.free_memory(self.texture_memory, None);
        }
    }
}

// Helper to create a texture
fn create_texture(
    app: &VulkanApp,
    width: u32,
    height: u32,
    format: vk::Format,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let image_create_info = vk::ImageCreateInfo {
        s_type: vk::StructureType::IMAGE_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: vk::ImageCreateFlags::empty(),
        image_type: vk::ImageType::TYPE_2D,
        format,
        extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        queue_family_index_count: 0,
        p_queue_family_indices: std::ptr::null(),
        initial_layout: vk::ImageLayout::UNDEFINED,
        _marker: std::marker::PhantomData,
    };

    let image = unsafe {
        app.device.create_image(&image_create_info, None)?
    };

    // Allocate memory for the image
    let memory_requirements = unsafe {
        app.device.get_image_memory_requirements(image)
    };

    let memory_type_index = find_memory_type(
        app,
        memory_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    let alloc_info = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        p_next: std::ptr::null(),
        allocation_size: memory_requirements.size,
        memory_type_index: memory_type_index,
        _marker: std::marker::PhantomData,
    };

    let memory = unsafe {
        app.device.allocate_memory(&alloc_info, None)?
    };

    unsafe {
        app.device.bind_image_memory(image, memory, 0)?;
    }

    Ok((image, memory))
}

// Helper to upload data to texture
fn upload_texture_data(
    app: &VulkanApp,
    image: vk::Image,
    width: u32,
    height: u32,
    data: &[u8],
) -> Result<()> {
    // Create staging buffer
    let buffer_size = data.len() as u64;

    let (staging_buffer, staging_memory) = create_buffer(
        app,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // Copy data to staging buffer
    unsafe {
        let ptr = app.device.map_memory(
            staging_memory,
            0,
            buffer_size,
            vk::MemoryMapFlags::empty(),
        )? as *mut u8;

        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());

        app.device.unmap_memory(staging_memory);
    }

    // Transition image layout for transfer
    transition_image_layout(
        app,
        image,
        vk::Format::R8_UNORM,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    )?;

    // Copy buffer to image
    copy_buffer_to_image(
        app,
        staging_buffer,
        image,
        width,
        height,
    )?;

    // Transition image layout for reading in shader
    transition_image_layout(
        app,
        image,
        vk::Format::R8_UNORM,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    )?;

    // Clean up staging resources
    unsafe {
        app.device.destroy_buffer(staging_buffer, None);
        app.device.free_memory(staging_memory, None);
    }

    Ok(())
}

// Helper to create buffer
fn create_buffer(
    app: &VulkanApp,
    size: u64,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_info = vk::BufferCreateInfo {
        s_type: vk::StructureType::BUFFER_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: vk::BufferCreateFlags::empty(),
        size,
        usage,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        queue_family_index_count: 0,
        p_queue_family_indices: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };

    let buffer = unsafe {
        app.device.create_buffer(&buffer_info, None)?
    };

    let mem_requirements = unsafe {
        app.device.get_buffer_memory_requirements(buffer)
    };

    let memory_type_index = find_memory_type(
        app,
        mem_requirements.memory_type_bits,
        properties,
    )?;

    let alloc_info = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        p_next: std::ptr::null(),
        allocation_size: mem_requirements.size,
        memory_type_index: memory_type_index,
        _marker: std::marker::PhantomData,
    };

    let memory = unsafe {
        app.device.allocate_memory(&alloc_info, None)?
    };

    unsafe {
        app.device.bind_buffer_memory(buffer, memory, 0)?;
    }

    Ok((buffer, memory))
}

// Helper to find memory type
fn find_memory_type(
    app: &VulkanApp,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Result<u32> {
    let memory_properties = app.get_physical_device_memory_properties();

    for i in 0..memory_properties.memory_type_count {
        if (type_filter & (1 << i)) != 0 &&
           (memory_properties.memory_types[i as usize].property_flags & properties) == properties {
            return Ok(i);
        }
    }

    Err(anyhow::anyhow!("Failed to find suitable memory type"))
}

// Helper to transition image layout
fn transition_image_layout(
    app: &VulkanApp,
    image: vk::Image,
    _format: vk::Format,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(app)?;

    let (src_access_mask, dst_access_mask, src_stage, dst_stage) = match (old_layout, new_layout) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        _ => return Err(anyhow::anyhow!("Unsupported layout transition")),
    };

    let barrier = vk::ImageMemoryBarrier {
        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
        p_next: std::ptr::null(),
        src_access_mask,
        dst_access_mask,
        old_layout,
        new_layout,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
        _marker: std::marker::PhantomData,
    };

    unsafe {
        app.device.cmd_pipeline_barrier(
            command_buffer,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }

    end_single_time_commands(app, command_buffer)?;

    Ok(())
}

// Helper to copy buffer to image
fn copy_buffer_to_image(
    app: &VulkanApp,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(app)?;

    let region = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
    };

    unsafe {
        app.device.cmd_copy_buffer_to_image(
            command_buffer,
            buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }

    end_single_time_commands(app, command_buffer)?;

    Ok(())
}

// Begin single time command buffer
fn begin_single_time_commands(app: &VulkanApp) -> Result<vk::CommandBuffer> {
    let alloc_info = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        p_next: std::ptr::null(),
        command_pool: app.command_pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
        ..Default::default()
    };

    let command_buffer = unsafe {
        app.device.allocate_command_buffers(&alloc_info)?[0]
    };

    let begin_info = vk::CommandBufferBeginInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
        p_next: std::ptr::null(),
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        p_inheritance_info: std::ptr::null(),
        ..Default::default()
    };

    unsafe {
        app.device.begin_command_buffer(command_buffer, &begin_info)?;
    }

    Ok(command_buffer)
}

// End single time command buffer
fn end_single_time_commands(app: &VulkanApp, command_buffer: vk::CommandBuffer) -> Result<()> {
    unsafe {
        app.device.end_command_buffer(command_buffer)?;

        let submit_info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: std::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: std::ptr::null(),
            p_wait_dst_stage_mask: std::ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            signal_semaphore_count: 0,
            p_signal_semaphores: std::ptr::null(),
            ..Default::default()
        };

        app.device.queue_submit(app.graphics_queue, &[submit_info], vk::Fence::null())?;
        app.device.queue_wait_idle(app.graphics_queue)?;

        app.device.free_command_buffers(app.command_pool, &[command_buffer]);
    }

    Ok(())
}

// Helper function to find the next power of two for a u32
fn next_power_of_two_u32(n: u32) -> u32 {
    let mut power = 1;
    while power < n {
        power *= 2;
    }
    power
}
