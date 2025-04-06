use anyhow::Result;
use ash::version::DeviceV1_0;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use log::{error, trace};
use std::mem::size_of;
use std::sync::Arc;

use super::font::Font;
use super::font_atlas::{CharData, FontAtlas};
use super::state::{EditorMode, UiState};
use super::theme::Theme;
use crate::vulkan::{VulkanApp, shader};

// Basic component trait
pub trait Component {
    fn render(&self, state: &UiState) -> String;
}

// Text editor component
pub struct TextEditor {
    pub content: String,
    pub cursor_position: (usize, usize),         // (line, column)
    pub selection_start: Option<(usize, usize)>, // Starting point of selection (line, column)
}

impl Component for TextEditor {
    fn render(&self, state: &UiState) -> String {
        format!("Text Editor with {} theme", state.theme.background_color)
    }
}

impl TextEditor {
    pub fn new() -> Self {
        Self {
            content: String::new(),
            cursor_position: (0, 0),
            selection_start: None,
        }
    }

    pub fn has_selection(&self) -> bool {
        self.selection_start.is_some()
    }

    pub fn start_selection(&mut self) {
        self.selection_start = Some(self.cursor_position);
    }

    pub fn clear_selection(&mut self) {
        self.selection_start = None;
    }

    pub fn get_selection_range(&self) -> Option<(usize, usize, usize, usize)> {
        self.selection_start.map(|start| {
            let (start_line, start_col) = start;
            let (end_line, end_col) = self.cursor_position;

            // Normalize selection to ensure start is before end
            if start_line < end_line || (start_line == end_line && start_col < end_col) {
                (start_line, start_col, end_line, end_col)
            } else {
                (end_line, end_col, start_line, start_col)
            }
        })
    }

    pub fn get_selected_text(&self) -> String {
        if let Some((start_line, start_col, end_line, end_col)) = self.get_selection_range() {
            let lines: Vec<&str> = self.content.lines().collect();

            if lines.is_empty() {
                return String::new();
            }

            if start_line == end_line {
                // Selection on a single line
                if start_line < lines.len() {
                    let line = lines[start_line];
                    let end_col = end_col.min(line.len());
                    if start_col <= end_col && start_col < line.len() {
                        return line[start_col..end_col].to_string();
                    }
                }
            } else {
                // Multi-line selection
                let mut selected_text = String::new();

                // First line from start_col to end
                if start_line < lines.len() {
                    let first_line = lines[start_line];
                    if start_col < first_line.len() {
                        selected_text.push_str(&first_line[start_col..]);
                    }
                    selected_text.push('\n');
                }

                // Middle lines completely
                for line_idx in start_line + 1..end_line {
                    if line_idx < lines.len() {
                        selected_text.push_str(lines[line_idx]);
                        selected_text.push('\n');
                    }
                }

                // Last line from start to end_col
                if end_line < lines.len() {
                    let last_line = lines[end_line];
                    let end_col = end_col.min(last_line.len());
                    selected_text.push_str(&last_line[..end_col]);
                }

                return selected_text;
            }
        }

        String::new()
    }
}

// Status bar component
pub struct StatusBar {
    pub left_text: String,
    pub right_text: String,
}

impl Component for StatusBar {
    fn render(&self, state: &UiState) -> String {
        let mode_text = match state.mode {
            EditorMode::Normal => "NORMAL",
            EditorMode::Insert => "INSERT",
            EditorMode::Visual => "VISUAL",
            EditorMode::Command => "COMMAND",
        };
        format!("{} | {} | {}", self.left_text, mode_text, self.right_text)
    }
}

// Line numbers component
pub struct LineNumbers {
    pub line_count: usize,
}

impl Component for LineNumbers {
    fn render(&self, state: &UiState) -> String {
        if !state.show_line_numbers {
            return String::new();
        }

        (1..=self.line_count)
            .map(|n| format!("{}", n))
            .collect::<Vec<String>>()
            .join("\n")
    }
}

// Command line component
pub struct CommandLine {
    pub text: String,
}

impl Component for CommandLine {
    fn render(&self, _state: &UiState) -> String {
        format!(":{}", self.text)
    }
}

// Vertex data structure for UI elements
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
    pub tex_coord: [f32; 2],
}

// Push constant layout (data passed to shader)
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PushConstants {
    pub transform: [f32; 16],
    pub color: [f32; 4],
    pub use_texture: u32,
    pub _padding: [u32; 3], // Align to 16 bytes
}

// UI Element types for rendering
#[derive(Debug, Clone)]
pub enum UiElement {
    Box {
        position: [f32; 2],
        size: [f32; 2],
        color: [f32; 4],
        border_radius: f32,
        border_width: f32,
        border_color: [f32; 4],
    },
    Text {
        position: [f32; 2],
        text: String,
        color: [f32; 4],
        scale: f32,
    },
    Image {
        position: [f32; 2],
        size: [f32; 2],
        uv_rect: [f32; 4],
        color: [f32; 4],
    },
}

// A renderer that handles the drawing of UI components using Vulkan
pub struct UiRenderer {
    // Theme colors
    pub bg_color: [f32; 4],
    pub fg_color: [f32; 4],

    // Vulkan resources
    app: Option<Arc<VulkanApp>>,
    font_atlas: Option<FontAtlas>,
    pipeline: Option<vk::Pipeline>,
    pipeline_layout: Option<vk::PipelineLayout>,
    descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    descriptor_pool: Option<vk::DescriptorPool>,
    descriptor_sets: Vec<vk::DescriptorSet>,
    vertex_buffer: Option<vk::Buffer>,
    vertex_buffer_memory: Option<vk::DeviceMemory>,
    index_buffer: Option<vk::Buffer>,
    index_buffer_memory: Option<vk::DeviceMemory>,

    // UI elements to render
    pending_elements: Vec<UiElement>,

    // Common vertices for boxes and text rendering
    quad_vertices: Vec<Vertex>,
    quad_indices: Vec<u32>,
}

impl UiRenderer {
    pub fn new(theme: &Theme) -> Self {
        // Convert hex colors to RGBA float arrays
        let bg_color = hex_to_rgba(theme.background_color);
        let fg_color = hex_to_rgba(theme.foreground_color);

        // Create a basic quad (two triangles) for rendering UI elements
        let quad_vertices = vec![
            Vertex {
                position: [0.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
                tex_coord: [0.0, 0.0],
            },
            Vertex {
                position: [1.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
                tex_coord: [1.0, 0.0],
            },
            Vertex {
                position: [1.0, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
                tex_coord: [1.0, 1.0],
            },
            Vertex {
                position: [0.0, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
                tex_coord: [0.0, 1.0],
            },
        ];

        // Indices for the quad (two triangles)
        let quad_indices = vec![0, 1, 2, 2, 3, 0];

        Self {
            bg_color,
            fg_color,
            app: None,
            font_atlas: None,
            pipeline: None,
            pipeline_layout: None,
            descriptor_set_layout: None,
            descriptor_pool: None,
            descriptor_sets: Vec::new(),
            vertex_buffer: None,
            vertex_buffer_memory: None,
            index_buffer: None,
            index_buffer_memory: None,
            pending_elements: Vec::new(),
            quad_vertices,
            quad_indices,
        }
    }

    pub fn initialize(&mut self, app: Arc<VulkanApp>, font: &Font) -> Result<()> {
        self.app = Some(app.clone());

        // Compile shaders if they don't exist
        if let Err(e) = shader::compile_shaders() {
            error!("Failed to compile shaders: {}", e);
        }

        // Create font atlas
        self.font_atlas = Some(FontAtlas::new(&app, font)?);

        // Create descriptor set layout
        self.create_descriptor_set_layout(&app)?;

        // Create pipeline layout
        self.create_pipeline_layout(&app)?;

        // Create graphics pipeline
        self.create_graphics_pipeline(&app)?;

        // Create descriptor pool and sets
        self.create_descriptor_sets(&app)?;

        // Create vertex and index buffers
        self.create_vertex_buffer(&app)?;
        self.create_index_buffer(&app)?;

        Ok(())
    }

    fn create_descriptor_set_layout(&mut self, app: &VulkanApp) -> Result<()> {
        let binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        let bindings = [binding];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings)
            .build();

        self.descriptor_set_layout = Some(unsafe {
            app.device
                .create_descriptor_set_layout(&layout_info, None)?
        });

        Ok(())
    }

    fn create_pipeline_layout(&mut self, app: &VulkanApp) -> Result<()> {
        // Define push constant range for passing transformation matrix and other data
        let push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32)
            .build();

        let layouts = [self.descriptor_set_layout.unwrap()];
        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&layouts)
            .push_constant_ranges(&[push_constant_range])
            .build();

        self.pipeline_layout =
            Some(unsafe { app.device.create_pipeline_layout(&layout_info, None)? });

        Ok(())
    }

    fn create_graphics_pipeline(&mut self, app: &VulkanApp) -> Result<()> {
        // Read compiled shaders
        let vert_shader_code = std::fs::read("shaders/ui.vert.spv")?;
        let frag_shader_code = std::fs::read("shaders/ui.frag.spv")?;

        // Create shader modules
        let vert_shader_module = self.create_shader_module(app, &vert_shader_code)?;
        let frag_shader_module = self.create_shader_module(app, &frag_shader_code)?;

        // Shader stage creation
        let vert_shader_stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_module)
            .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap())
            .build();

        let frag_shader_stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module)
            .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap())
            .build();

        let shader_stages = [vert_shader_stage_info, frag_shader_stage_info];

        // Vertex input (position, color, texcoord)
        let binding_description = vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build();

        let position_attr = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0)
            .build();

        let color_attr = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(8) // 2 * sizeof(f32)
            .build();

        let texcoord_attr = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(24) // (2 + 4) * sizeof(f32)
            .build();

        let attribute_descriptions = [position_attr, color_attr, texcoord_attr];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&[binding_description])
            .vertex_attribute_descriptions(&attribute_descriptions)
            .build();

        // Input assembly (triangle list)
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false)
            .build();

        // Viewport and scissor
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: app.swapchain_extent.width as f32,
            height: app.swapchain_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: app.swapchain_extent,
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&[viewport])
            .scissors(&[scissor])
            .build();

        // Rasterization
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE) // No culling so we can see from either side
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .depth_bias_constant_factor(0.0)
            .depth_bias_clamp(0.0)
            .depth_bias_slope_factor(0.0)
            .line_width(1.0)
            .build();

        // Multisampling (disabled)
        let multisampling = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.0)
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false)
            .build();

        // Color blending (alpha blending)
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::TRUE,
            src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        };

        let color_blending = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&[color_blend_attachment])
            .blend_constants([0.0, 0.0, 0.0, 0.0])
            .build();

        // Finally, create the pipeline
        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .layout(self.pipeline_layout.unwrap())
            .render_pass(app.render_pass)
            .subpass(0)
            .base_pipeline_handle(vk::Pipeline::null())
            .base_pipeline_index(-1)
            .build();

        let pipelines = unsafe {
            app.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        }
        .map_err(|(_pipelines, err)| {
            anyhow::anyhow!("Failed to create graphics pipeline: {:?}", err)
        })?;

        self.pipeline = Some(pipelines[0]);

        // Cleanup shader modules
        unsafe {
            app.device.destroy_shader_module(vert_shader_module, None);
            app.device.destroy_shader_module(frag_shader_module, None);
        }

        Ok(())
    }

    fn create_shader_module(&self, app: &VulkanApp, code: &[u8]) -> Result<vk::ShaderModule> {
        let create_info = vk::ShaderModuleCreateInfo::builder()
            .code(bytemuck::cast_slice(code))
            .build();

        let shader_module = unsafe { app.device.create_shader_module(&create_info, None)? };

        Ok(shader_module)
    }

    fn create_descriptor_sets(&mut self, app: &VulkanApp) -> Result<()> {
        // Create descriptor pool
        let pool_size = vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .build();

        let pool_sizes = [pool_size];
        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(1)
            .pool_sizes(&pool_sizes)
            .build();

        self.descriptor_pool =
            Some(unsafe { app.device.create_descriptor_pool(&pool_info, None)? });

        // Allocate descriptor sets
        let layouts = [self.descriptor_set_layout.unwrap()];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool.unwrap())
            .set_layouts(&layouts)
            .build();

        self.descriptor_sets = unsafe { app.device.allocate_descriptor_sets(&alloc_info)? };

        // Update the descriptor sets with the font texture
        if let Some(font_atlas) = &self.font_atlas {
            let image_info = vk::DescriptorImageInfo::builder()
                .sampler(font_atlas.texture_sampler)
                .image_view(font_atlas.texture_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .build();

            let descriptor_write = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[0])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&[image_info])
                .build();

            unsafe {
                app.device.update_descriptor_sets(&[descriptor_write], &[]);
            }
        }

        Ok(())
    }

    fn create_vertex_buffer(&mut self, app: &VulkanApp) -> Result<()> {
        let buffer_size = (size_of::<Vertex>() * self.quad_vertices.len()) as u64;

        // Create staging buffer
        let (staging_buffer, staging_buffer_memory) = create_buffer(
            app,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // Copy vertex data to staging buffer
        unsafe {
            let data_ptr = app.device.map_memory(
                staging_buffer_memory,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )? as *mut Vertex;

            std::ptr::copy_nonoverlapping(
                self.quad_vertices.as_ptr(),
                data_ptr,
                self.quad_vertices.len(),
            );

            app.device.unmap_memory(staging_buffer_memory);
        }

        // Create the vertex buffer
        let (vertex_buffer, vertex_buffer_memory) = create_buffer(
            app,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Copy from staging buffer to vertex buffer
        copy_buffer(app, staging_buffer, vertex_buffer, buffer_size)?;

        // Clean up staging buffer
        unsafe {
            app.device.destroy_buffer(staging_buffer, None);
            app.device.free_memory(staging_buffer_memory, None);
        }

        self.vertex_buffer = Some(vertex_buffer);
        self.vertex_buffer_memory = Some(vertex_buffer_memory);

        Ok(())
    }

    fn create_index_buffer(&mut self, app: &VulkanApp) -> Result<()> {
        let buffer_size = (size_of::<u32>() * self.quad_indices.len()) as u64;

        // Create staging buffer
        let (staging_buffer, staging_buffer_memory) = create_buffer(
            app,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // Copy index data to staging buffer
        unsafe {
            let data_ptr = app.device.map_memory(
                staging_buffer_memory,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )? as *mut u32;

            std::ptr::copy_nonoverlapping(
                self.quad_indices.as_ptr(),
                data_ptr,
                self.quad_indices.len(),
            );

            app.device.unmap_memory(staging_buffer_memory);
        }

        // Create the index buffer
        let (index_buffer, index_buffer_memory) = create_buffer(
            app,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Copy from staging buffer to index buffer
        copy_buffer(app, staging_buffer, index_buffer, buffer_size)?;

        // Clean up staging buffer
        unsafe {
            app.device.destroy_buffer(staging_buffer, None);
            app.device.free_memory(staging_buffer_memory, None);
        }

        self.index_buffer = Some(index_buffer);
        self.index_buffer_memory = Some(index_buffer_memory);

        Ok(())
    }

    // Add UI elements for rendering
    pub fn add_ui_element(&mut self, element: UiElement) {
        self.pending_elements.push(element);
    }

    // Clear pending UI elements
    pub fn clear_elements(&mut self) {
        self.pending_elements.clear();
    }

    // Record UI rendering commands into the command buffer
    pub fn record_ui_commands(
        &self,
        command_buffer: vk::CommandBuffer,
        _framebuffer_idx: usize,
    ) -> Result<()> {
        if let Some(app) = &self.app {
            if let (Some(pipeline), Some(vertex_buffer), Some(index_buffer)) =
                (self.pipeline, self.vertex_buffer, self.index_buffer)
            {
                unsafe {
                    // Bind the graphics pipeline
                    app.device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline,
                    );

                    // Bind descriptor sets for font texture
                    app.device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout.unwrap(),
                        0,
                        &self.descriptor_sets,
                        &[],
                    );

                    // Bind vertex and index buffers
                    app.device
                        .cmd_bind_vertex_buffers(command_buffer, 0, &[vertex_buffer], &[0]);

                    app.device.cmd_bind_index_buffer(
                        command_buffer,
                        index_buffer,
                        0,
                        vk::IndexType::UINT32,
                    );

                    // Draw UI elements
                    self.draw_background(app, command_buffer);

                    for element in &self.pending_elements {
                        match element {
                            UiElement::Box {
                                position,
                                size,
                                color,
                                border_radius: _,
                                border_width,
                                border_color,
                            } => {
                                self.draw_box(
                                    app,
                                    command_buffer,
                                    *position,
                                    *size,
                                    *color,
                                    0.0, // Using a default value for border_radius
                                    *border_width,
                                    *border_color,
                                );
                            }
                            UiElement::Text {
                                position,
                                text,
                                color,
                                scale,
                            } => {
                                self.draw_text(
                                    app,
                                    command_buffer,
                                    *position,
                                    text,
                                    *color,
                                    *scale,
                                );
                            }
                            UiElement::Image {
                                position,
                                size,
                                uv_rect: _,
                                color,
                            } => {
                                self.draw_image(
                                    app,
                                    command_buffer,
                                    *position,
                                    *size,
                                    [0.0, 0.0, 1.0, 1.0], // Using default UV rect
                                    *color,
                                );
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    // Draw the background
    fn draw_background(&self, app: &VulkanApp, command_buffer: vk::CommandBuffer) {
        let width = app.swapchain_extent.width as f32;
        let height = app.swapchain_extent.height as f32;

        let transform = Mat4::orthographic_lh(0.0, width, height, 0.0, -1.0, 1.0);

        let push_constants = PushConstants {
            transform: transform.to_cols_array(),
            color: self.bg_color,
            use_texture: 0,
            _padding: [0; 3],
        };

        unsafe {
            app.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout.unwrap(),
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::cast_slice(&[push_constants]),
            );

            app.device
                .cmd_draw_indexed(command_buffer, self.quad_indices.len() as u32, 1, 0, 0, 0);
        }
    }

    // Draw a box with optional border
    fn draw_box(
        &self,
        app: &VulkanApp,
        command_buffer: vk::CommandBuffer,
        position: [f32; 2],
        size: [f32; 2],
        color: [f32; 4],
        _border_radius: f32,
        border_width: f32,
        border_color: [f32; 4],
    ) {
        let width = app.swapchain_extent.width as f32;
        let height = app.swapchain_extent.height as f32;

        // Create orthographic projection
        let projection = Mat4::orthographic_lh(0.0, width, height, 0.0, -1.0, 1.0);

        // Create translation matrix
        let translation = Mat4::from_translation(Vec3::new(position[0], position[1], 0.0));

        // Create scale matrix
        let scale = Mat4::from_scale(Vec3::new(size[0], size[1], 1.0));

        // Combine transformations
        let transform = projection * translation * scale;

        // Create push constants
        let push_constants = PushConstants {
            transform: transform.to_cols_array(),
            color,
            use_texture: 0,
            _padding: [0; 3],
        };

        unsafe {
            // Push constants to the shader
            app.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout.unwrap(),
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::cast_slice(&[push_constants]),
            );

            // Draw the box
            app.device
                .cmd_draw_indexed(command_buffer, self.quad_indices.len() as u32, 1, 0, 0, 0);

            // Draw border if needed
            if border_width > 0.0 {
                // Draw the border (simplified, just scaling the box slightly)
                let border_push_constants = PushConstants {
                    transform: transform.to_cols_array(),
                    color: border_color,
                    use_texture: 0,
                    _padding: [0; 3],
                };

                app.device.cmd_push_constants(
                    command_buffer,
                    self.pipeline_layout.unwrap(),
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::cast_slice(&[border_push_constants]),
                );

                // Draw border (using line loop would be better but we're using triangles)
                // This is simplified, ideally would use a proper border implementation
                app.device.cmd_draw_indexed(
                    command_buffer,
                    self.quad_indices.len() as u32,
                    1,
                    0,
                    0,
                    0,
                );
            }
        }
    }

    // Draw text
    fn draw_text(
        &self,
        app: &VulkanApp,
        command_buffer: vk::CommandBuffer,
        position: [f32; 2],
        text: &str,
        color: [f32; 4],
        scale: f32,
    ) {
        if let Some(font_atlas) = &self.font_atlas {
            let width = app.swapchain_extent.width as f32;
            let height = app.swapchain_extent.height as f32;

            // Create orthographic projection
            let projection = Mat4::orthographic_lh(0.0, width, height, 0.0, -1.0, 1.0);

            let mut cursor_x = position[0];
            let cursor_y = position[1];

            for c in text.chars() {
                if let Some(char_data) = font_atlas.get_char(c) {
                    // Skip space with just advancing cursor
                    if c == ' ' {
                        cursor_x += char_data.advance * scale;
                        continue;
                    }

                    // Create glyph transform
                    let glyph_pos = [
                        cursor_x + char_data.offset[0] * scale,
                        cursor_y + char_data.offset[1] * scale,
                    ];
                    let glyph_size = [char_data.size[0] * scale, char_data.size[1] * scale];

                    // Create translation matrix
                    let translation =
                        Mat4::from_translation(Vec3::new(glyph_pos[0], glyph_pos[1], 0.0));

                    // Create scale matrix
                    let scale_mat = Mat4::from_scale(Vec3::new(glyph_size[0], glyph_size[1], 1.0));

                    // Combine transformations
                    let transform = projection * translation * scale_mat;

                    // Create push constants
                    let push_constants = PushConstants {
                        transform: transform.to_cols_array(),
                        color,
                        use_texture: 1, // Use texture for text
                        _padding: [0; 3],
                    };

                    unsafe {
                        // Push constants to the shader
                        app.device.cmd_push_constants(
                            command_buffer,
                            self.pipeline_layout.unwrap(),
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            0,
                            bytemuck::cast_slice(&[push_constants]),
                        );

                        // Draw the glyph
                        app.device.cmd_draw_indexed(
                            command_buffer,
                            self.quad_indices.len() as u32,
                            1,
                            0,
                            0,
                            0,
                        );
                    }

                    // Advance cursor for next glyph
                    cursor_x += char_data.advance * scale;
                }
            }
        }
    }

    // Draw an image
    fn draw_image(
        &self,
        app: &VulkanApp,
        command_buffer: vk::CommandBuffer,
        position: [f32; 2],
        size: [f32; 2],
        _uv_rect: [f32; 4],
        color: [f32; 4],
    ) {
        let width = app.swapchain_extent.width as f32;
        let height = app.swapchain_extent.height as f32;

        // Create orthographic projection
        let projection = Mat4::orthographic_lh(0.0, width, height, 0.0, -1.0, 1.0);

        // Create translation matrix
        let translation = Mat4::from_translation(Vec3::new(position[0], position[1], 0.0));

        // Create scale matrix
        let scale = Mat4::from_scale(Vec3::new(size[0], size[1], 1.0));

        // Combine transformations
        let transform = projection * translation * scale;

        // Create push constants
        let push_constants = PushConstants {
            transform: transform.to_cols_array(),
            color,
            use_texture: 1,
            _padding: [0; 3],
        };

        unsafe {
            // Push constants to the shader
            app.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout.unwrap(),
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::cast_slice(&[push_constants]),
            );

            // Draw the image
            app.device
                .cmd_draw_indexed(command_buffer, self.quad_indices.len() as u32, 1, 0, 0, 0);
        }
    }

    // Update colors when theme changes
    pub fn update_colors(&mut self, theme: &Theme) {
        self.bg_color = hex_to_rgba(theme.background_color);
        self.fg_color = hex_to_rgba(theme.foreground_color);
    }

    // Render editor text content with selection support
    pub fn render_editor(&mut self, editor: &TextEditor, state: &UiState) {
        trace!(
            "Rendering editor with background color: {:?}",
            self.bg_color
        );

        // Add a background box for the editor area
        let width = 800.0; // This should be dynamically determined
        let height = 500.0;

        // Add editor background
        self.add_ui_element(UiElement::Box {
            position: [10.0, 30.0], // Allow space for status bar
            size: [width - 20.0, height - 40.0],
            color: self.bg_color,
            border_radius: 0.0,
            border_width: 0.0,
            border_color: [0.0, 0.0, 0.0, 0.0],
        });

        // Get selection range if any
        let selection_range = editor.get_selection_range();
        let content_lines = editor.content.lines().collect::<Vec<_>>();

        let line_height = 20.0;
        let char_width = 9.0; // Approximate character width
        let font_size = 1.0; // Scale factor

        for (i, line) in content_lines.iter().enumerate() {
            let y_pos = 40.0 + i as f32 * line_height;

            // Draw selection highlight if this line is in selection range
            if let Some((start_line, start_col, end_line, end_col)) = selection_range {
                if i >= start_line && i <= end_line {
                    let highlight_start_x = if i == start_line {
                        40.0 + start_col as f32 * char_width
                    } else {
                        40.0 // Start of line
                    };

                    let highlight_end_x = if i == end_line {
                        40.0 + end_col as f32 * char_width
                    } else {
                        40.0 + line.len() as f32 * char_width // End of line
                    };

                    // Add selection highlight
                    self.add_ui_element(UiElement::Box {
                        position: [highlight_start_x, y_pos],
                        size: [highlight_end_x - highlight_start_x, line_height],
                        color: hex_to_rgba(state.theme.selection_color),
                        border_radius: 0.0,
                        border_width: 0.0,
                        border_color: [0.0, 0.0, 0.0, 0.0],
                    });
                }
            }

            // Add text element for each line
            self.add_ui_element(UiElement::Text {
                position: [40.0, y_pos], // Offset for line numbers
                text: line.to_string(),
                color: self.fg_color,
                scale: font_size,
            });
        }

        // Render cursor (blinking effect could be added based on time)
        let cursor_x = 40.0 + editor.cursor_position.1 as f32 * char_width;
        let cursor_y = 40.0 + editor.cursor_position.0 as f32 * line_height;

        // If cursor is beyond the content we've entered, show it at the correct position
        let cursor_line = editor
            .cursor_position
            .0
            .min(content_lines.len().saturating_sub(1));
        let line_content = if cursor_line < content_lines.len() {
            content_lines[cursor_line]
        } else {
            ""
        };

        // Make sure cursor doesn't go too far beyond the last character
        let max_visible_column = line_content.len() + 1;
        let cursor_col = editor.cursor_position.1.min(max_visible_column);
        let cursor_x = 40.0 + cursor_col as f32 * char_width;

        self.add_ui_element(UiElement::Box {
            position: [cursor_x, cursor_y],
            size: [2.0, line_height],
            color: hex_to_rgba(state.theme.cursor_color),
            border_radius: 0.0,
            border_width: 0.0,
            border_color: [0.0, 0.0, 0.0, 0.0],
        });
    }

    // Render status bar
    pub fn render_status_bar(&mut self, status_bar: &StatusBar, state: &UiState) {
        trace!("Rendering status bar: {}", status_bar.render(state));

        if state.show_status_bar {
            let width = 800.0; // Should be dynamically determined

            // Add status bar background
            self.add_ui_element(UiElement::Box {
                position: [0.0, 0.0],
                size: [width, 30.0],
                color: hex_to_rgba("#1E1E1E"),
                border_radius: 0.0,
                border_width: 0.0,
                border_color: [0.0, 0.0, 0.0, 0.0],
            });

            // Add mode indicator
            let mode_text = match state.mode {
                EditorMode::Normal => "NORMAL",
                EditorMode::Insert => "INSERT",
                EditorMode::Visual => "VISUAL",
                EditorMode::Command => "COMMAND",
            };

            // Mode box with distinctive color
            let mode_color = match state.mode {
                EditorMode::Normal => [0.2, 0.6, 0.2, 1.0],  // Green
                EditorMode::Insert => [0.2, 0.2, 0.8, 1.0],  // Blue
                EditorMode::Visual => [0.8, 0.4, 0.1, 1.0],  // Orange
                EditorMode::Command => [0.7, 0.1, 0.1, 1.0], // Red
            };

            self.add_ui_element(UiElement::Box {
                position: [10.0, 5.0],
                size: [70.0, 20.0],
                color: mode_color,
                border_radius: 3.0,
                border_width: 0.0,
                border_color: [0.0, 0.0, 0.0, 0.0],
            });

            self.add_ui_element(UiElement::Text {
                position: [15.0, 7.0],
                text: mode_text.to_string(),
                color: [1.0, 1.0, 1.0, 1.0],
                scale: 0.9,
            });

            // Add left status text
            self.add_ui_element(UiElement::Text {
                position: [90.0, 7.0],
                text: status_bar.left_text.clone(),
                color: [0.8, 0.8, 0.8, 1.0],
                scale: 0.9,
            });

            // Add right status text (position depends on text length)
            self.add_ui_element(UiElement::Text {
                position: [width - 150.0, 7.0],
                text: status_bar.right_text.clone(),
                color: [0.7, 0.7, 0.7, 1.0],
                scale: 0.9,
            });
        }
    }

    // Render line numbers
    pub fn render_line_numbers(&mut self, line_numbers: &LineNumbers, state: &UiState) {
        if state.show_line_numbers {
            trace!(
                "Rendering line numbers for {} lines",
                line_numbers.line_count
            );

            // Add line numbers background
            self.add_ui_element(UiElement::Box {
                position: [0.0, 30.0],
                size: [30.0, 570.0],
                color: hex_to_rgba("#252525"),
                border_radius: 0.0,
                border_width: 0.0,
                border_color: [0.0, 0.0, 0.0, 0.0],
            });

            // Add line numbers
            for i in 0..line_numbers.line_count {
                self.add_ui_element(UiElement::Text {
                    position: [5.0, 40.0 + i as f32 * 20.0],
                    text: format!("{}", i + 1),
                    color: [0.5, 0.5, 0.5, 1.0],
                    scale: 0.9,
                });
            }
        }
    }

    // Render command line
    pub fn render_command_line(&mut self, command_line: &CommandLine, state: &UiState) {
        if let EditorMode::Command = state.mode {
            trace!("Rendering command line: {}", command_line.render(state));

            let width = 800.0; // Should be dynamically determined
            let height = 600.0;

            // Add command line background
            self.add_ui_element(UiElement::Box {
                position: [0.0, height - 30.0],
                size: [width, 30.0],
                color: hex_to_rgba("#1E1E1E"),
                border_radius: 0.0,
                border_width: 0.0,
                border_color: [0.0, 0.0, 0.0, 0.0],
            });

            // Add command prefix and text
            self.add_ui_element(UiElement::Text {
                position: [10.0, height - 23.0],
                text: format!(":{}", command_line.text),
                color: [1.0, 1.0, 1.0, 1.0],
                scale: 1.0,
            });

            // Add command cursor
            self.add_ui_element(UiElement::Box {
                position: [20.0 + command_line.text.len() as f32 * 9.0, height - 23.0],
                size: [2.0, 16.0],
                color: [1.0, 1.0, 1.0, 0.8],
                border_radius: 0.0,
                border_width: 0.0,
                border_color: [0.0, 0.0, 0.0, 0.0],
            });
        }
    }

    // Clean up Vulkan resources
    pub fn cleanup(&mut self) {
        if let Some(app) = &self.app {
            unsafe {
                if let Some(vertex_buffer) = self.vertex_buffer {
                    app.device.destroy_buffer(vertex_buffer, None);
                    self.vertex_buffer = None;
                }

                if let Some(vertex_buffer_memory) = self.vertex_buffer_memory {
                    app.device.free_memory(vertex_buffer_memory, None);
                    self.vertex_buffer_memory = None;
                }

                if let Some(index_buffer) = self.index_buffer {
                    app.device.destroy_buffer(index_buffer, None);
                    self.index_buffer = None;
                }

                if let Some(index_buffer_memory) = self.index_buffer_memory {
                    app.device.free_memory(index_buffer_memory, None);
                    self.index_buffer_memory = None;
                }

                if let Some(pipeline) = self.pipeline {
                    app.device.destroy_pipeline(pipeline, None);
                    self.pipeline = None;
                }

                if let Some(pipeline_layout) = self.pipeline_layout {
                    app.device.destroy_pipeline_layout(pipeline_layout, None);
                    self.pipeline_layout = None;
                }

                if let Some(descriptor_set_layout) = self.descriptor_set_layout {
                    app.device
                        .destroy_descriptor_set_layout(descriptor_set_layout, None);
                    self.descriptor_set_layout = None;
                }

                if let Some(descriptor_pool) = self.descriptor_pool {
                    app.device.destroy_descriptor_pool(descriptor_pool, None);
                    self.descriptor_pool = None;
                }

                if let Some(font_atlas) = &self.font_atlas {
                    font_atlas.cleanup(app);
                }

                self.font_atlas = None;
                self.descriptor_sets.clear();
            }
        }
    }
}

// Helper function to create buffer
fn create_buffer(
    app: &VulkanApp,
    size: u64,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .build();

    let buffer = unsafe { app.device.create_buffer(&buffer_info, None)? };

    let mem_requirements = unsafe { app.device.get_buffer_memory_requirements(buffer) };

    // Find suitable memory type
    let mem_type_index = find_memory_type(app, mem_requirements.memory_type_bits, properties)?;

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(mem_requirements.size)
        .memory_type_index(mem_type_index)
        .build();

    let memory = unsafe { app.device.allocate_memory(&alloc_info, None)? };

    unsafe {
        app.device.bind_buffer_memory(buffer, memory, 0)?;
    }

    Ok((buffer, memory))
}

// Helper function to find suitable memory type
fn find_memory_type(
    app: &VulkanApp,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Result<u32> {
    let mem_properties = app.get_physical_device_memory_properties();

    for i in 0..mem_properties.memory_type_count {
        if (type_filter & (1 << i)) != 0
            && (mem_properties.memory_types[i as usize].property_flags & properties) == properties
        {
            return Ok(i);
        }
    }

    Err(anyhow::anyhow!("Failed to find suitable memory type"))
}

// Helper function to copy between buffers
fn copy_buffer(
    app: &VulkanApp,
    src_buffer: vk::Buffer,
    dst_buffer: vk::Buffer,
    size: u64,
) -> Result<()> {
    // Create a command buffer for transfer
    let command_buffer = begin_single_time_commands(app)?;

    // Record copy command
    let copy_region = vk::BufferCopy::builder()
        .src_offset(0)
        .dst_offset(0)
        .size(size)
        .build();

    unsafe {
        app.device
            .cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, &[copy_region]);
    }

    // Submit and clean up
    end_single_time_commands(app, command_buffer)?;

    Ok(())
}

// Helper function for single-time command buffer operations
fn begin_single_time_commands(app: &VulkanApp) -> Result<vk::CommandBuffer> {
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(app.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1)
        .build();

    let command_buffer = unsafe { app.device.allocate_command_buffers(&alloc_info)?[0] };

    let begin_info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
        .build();

    unsafe {
        app.device
            .begin_command_buffer(command_buffer, &begin_info)?;
    }

    Ok(command_buffer)
}

fn end_single_time_commands(app: &VulkanApp, command_buffer: vk::CommandBuffer) -> Result<()> {
    unsafe {
        app.device.end_command_buffer(command_buffer)?;

        let submit_info = vk::SubmitInfo::builder()
            .command_buffers(&[command_buffer])
            .build();

        app.device
            .queue_submit(app.graphics_queue, &[submit_info], vk::Fence::null())?;
        app.device.queue_wait_idle(app.graphics_queue)?;

        app.device
            .free_command_buffers(app.command_pool, &[command_buffer]);
    }

    Ok(())
}

// Helper function to convert hex color strings to RGBA float arrays
fn hex_to_rgba(hex: &str) -> [f32; 4] {
    let hex = hex.trim_start_matches('#');

    let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(0) as f32 / 255.0;
    let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0) as f32 / 255.0;
    let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(0) as f32 / 255.0;

    [r, g, b, 1.0] // Full opacity by default
}
