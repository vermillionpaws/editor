use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::process::Command;
use anyhow::{Result, anyhow};
use log::info;

use ash::{vk, Device};
use log::{error, trace};
use std::ffi::CString;
use std::collections::HashMap;
use std::sync::Arc;

// Cache to avoid recompiling the same shaders
static mut SHADER_MODULE_CACHE: Option<HashMap<String, vk::ShaderModule>> = None;

pub fn init_shader_cache() {
    unsafe {
        SHADER_MODULE_CACHE = Some(HashMap::new());
    }
}

pub fn cleanup_shader_modules(device: &Device) {
    unsafe {
        if let Some(cache) = SHADER_MODULE_CACHE.take() {
            for (path, module) in cache {
                device.destroy_shader_module(module, None);
                trace!("Destroyed shader module for {}", path);
            }
        }
    }
}

// Compile GLSL shader to SPIR-V and create a shader module
pub fn create_shader_module(
    device: &Device,
    shader_path: &str,
    shader_type: ShaderType
) -> Result<vk::ShaderModule> {
    // Check if shader module is already in cache
    unsafe {
        if let Some(cache) = &SHADER_MODULE_CACHE {
            if let Some(module) = cache.get(shader_path) {
                trace!("Using cached shader module for {}", shader_path);
                return Ok(*module);
            }
        }
    }

    // Compile shader to SPIR-V if needed
    let spv_path = compile_shader_if_needed(shader_path, shader_type)?;

    // Read the SPIR-V binary
    let spv_data = fs::read(spv_path)?;

    // Create the shader module
    let create_info = vk::ShaderModuleCreateInfo {
        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: vk::ShaderModuleCreateFlags::empty(),
        code_size: spv_data.len(),
        p_code: spv_data.as_ptr() as *const u32,
        ..Default::default()
    };

    let shader_module = unsafe {
        device.create_shader_module(&create_info, None)?
    };

    // Cache the shader module
    unsafe {
        if let Some(cache) = &mut SHADER_MODULE_CACHE {
            cache.insert(shader_path.to_string(), shader_module);
        }
    }

    trace!("Created shader module for {}", shader_path);
    Ok(shader_module)
}

// Helper to create shader entry point CString
pub fn create_entry_point(name: &str) -> CString {
    CString::new(name).unwrap()
}

// Determine if shader needs recompilation and compile if needed
fn compile_shader_if_needed(
    shader_path: &str,
    shader_type: ShaderType
) -> Result<String> {
    let shader_file = Path::new(shader_path);
    let spv_path = format!("{}.spv", shader_path);
    let spv_file = Path::new(&spv_path);

    // Check if SPIR-V file exists and is newer than shader source
    let should_compile = if !spv_file.exists() {
        true
    } else {
        let shader_metadata = fs::metadata(shader_file)?;
        let spv_metadata = fs::metadata(spv_file)?;

        let shader_modified = shader_metadata.modified()?;
        let spv_modified = spv_metadata.modified()?;

        shader_modified > spv_modified
    };

    if should_compile {
        trace!("Compiling shader: {}", shader_path);
        compile_shader(shader_path, &spv_path, shader_type)?;
    } else {
        trace!("Using existing compiled shader: {}", spv_path);
    }

    Ok(spv_path)
}

// Compile shader using glslangValidator or similar tool
fn compile_shader(
    shader_path: &str,
    output_path: &str,
    shader_type: ShaderType
) -> Result<()> {
    // Use glslangValidator with appropriate flags
    let stage_flag = match shader_type {
        ShaderType::Vertex => "-S vert",
        ShaderType::Fragment => "-S frag",
        ShaderType::Compute => "-S comp",
    };

    let output = Command::new("glslangValidator")
        .args(&[stage_flag, "-V", shader_path, "-o", output_path])
        .output()?;

    if !output.status.success() {
        let error_message = String::from_utf8_lossy(&output.stderr);
        error!("Failed to compile shader: {}", error_message);
        return Err(anyhow::anyhow!("Shader compilation failed: {}", error_message));
    }

    trace!("Compiled shader {} to {}", shader_path, output_path);
    Ok(())
}

// Shader types
pub enum ShaderType {
    Vertex,
    Fragment,
    Compute,
}

// Create common descriptors for UI rendering
pub fn create_descriptor_set_layout(
    device: &Device,
) -> Result<vk::DescriptorSetLayout> {
    // Binding for font texture
    let bindings = [
        vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: std::ptr::null(),
        },
    ];

    let layout_info = vk::DescriptorSetLayoutCreateInfo {
        s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: vk::DescriptorSetLayoutCreateFlags::empty(),
        binding_count: bindings.len() as u32,
        p_bindings: bindings.as_ptr(),
        ..Default::default()
    };

    let layout = unsafe {
        device.create_descriptor_set_layout(&layout_info, None)?
    };

    trace!("Created descriptor set layout for UI rendering");
    Ok(layout)
}

// Create descriptor pool for UI rendering
pub fn create_descriptor_pool(
    device: &Device,
    max_sets: u32,
) -> Result<vk::DescriptorPool> {
    let pool_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: max_sets,
        },
    ];

    let pool_info = vk::DescriptorPoolCreateInfo {
        s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: vk::DescriptorPoolCreateFlags::empty(),
        max_sets,
        pool_size_count: pool_sizes.len() as u32,
        p_pool_sizes: pool_sizes.as_ptr(),
        ..Default::default()
    };

    let pool = unsafe {
        device.create_descriptor_pool(&pool_info, None)?
    };

    trace!("Created descriptor pool with {} max sets", max_sets);
    Ok(pool)
}

// Allocate descriptor sets for UI rendering
pub fn allocate_descriptor_sets(
    device: &Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    count: u32,
) -> Result<Vec<vk::DescriptorSet>> {
    let layouts = vec![layout; count as usize];

    let alloc_info = vk::DescriptorSetAllocateInfo {
        s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
        p_next: std::ptr::null(),
        descriptor_pool: pool,
        descriptor_count: count,
        p_set_layouts: layouts.as_ptr(),
        ..Default::default()
    };

    let sets = unsafe {
        device.allocate_descriptor_sets(&alloc_info)?
    };

    trace!("Allocated {} descriptor sets", count);
    Ok(sets)
}

// Update descriptor set with font texture
pub fn update_font_descriptor_set(
    device: &Device,
    descriptor_set: vk::DescriptorSet,
    image_view: vk::ImageView,
    sampler: vk::Sampler,
) {
    let image_info = vk::DescriptorImageInfo {
        sampler,
        image_view,
        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    };

    let write_descriptor_set = vk::WriteDescriptorSet {
        s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
        p_next: std::ptr::null(),
        dst_set: descriptor_set,
        dst_binding: 0,
        dst_array_element: 0,
        descriptor_count: 1,
        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        p_image_info: &image_info,
        p_buffer_info: std::ptr::null(),
        p_texel_buffer_view: std::ptr::null(),
        ..Default::default()
    };

    unsafe {
        device.update_descriptor_sets(&[write_descriptor_set], &[]);
    }

    trace!("Updated descriptor set with font texture");
}

pub fn compile_shaders() -> Result<()> {
    // Make sure the shaders directory exists
    let shaders_dir = Path::new("shaders");
    if !shaders_dir.exists() {
        return Err(anyhow!("Shaders directory not found"));
    }

    // Compile vertex shader
    compile_shader("shaders/ui.vert", "shaders/ui.vert.spv", "vert")?;

    // Compile fragment shader
    compile_shader("shaders/ui.frag", "shaders/ui.frag.spv", "frag")?;

    info!("Shaders compiled successfully");
    Ok(())
}

fn compile_shader(source_path: &str, output_path: &str, shader_type: &str) -> Result<()> {
    // Check if the source file exists
    let source_path = Path::new(source_path);
    if !source_path.exists() {
        // Create basic shader if it doesn't exist
        create_default_shader(source_path, shader_type)?;
    }

    // Check if output already exists and is newer than source
    let output_path = Path::new(output_path);
    if output_path.exists() {
        if let (Ok(source_meta), Ok(output_meta)) = (source_path.metadata(), output_path.metadata()) {
            if let (Ok(source_modified), Ok(output_modified)) = (source_meta.modified(), output_meta.modified()) {
                if output_modified > source_modified {
                    // Output is newer than source, no need to recompile
                    return Ok(());
                }
            }
        }
    }

    // Call glslc to compile the shader
    let output = Command::new("glslc")
        .arg(source_path)
        .arg("-o")
        .arg(output_path)
        .output();

    match output {
        Ok(output) => {
            if !output.status.success() {
                let error = String::from_utf8_lossy(&output.stderr);
                return Err(anyhow!("Failed to compile shader: {}", error));
            }
            Ok(())
        },
        Err(e) => {
            // If glslc is not available, provide a fallback for development
            if cfg!(debug_assertions) {
                info!("glslc not found, using pre-compiled shaders if available");
                if !output_path.exists() {
                    return Err(anyhow!("Shader compilation failed and no pre-compiled shader available: {}", e));
                }
                Ok(())
            } else {
                Err(anyhow!("Failed to execute glslc: {}", e))
            }
        }
    }
}

fn create_default_shader(path: &Path, shader_type: &str) -> Result<()> {
    let shader_code = match shader_type {
        "vert" => r#"#version 450
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec2 fragTexCoord;

layout(push_constant) uniform PushConstants {
    mat4 transform;
    vec4 color;
    uint useTexture;
    uint _padding1;
    uint _padding2;
    uint _padding3;
} pushConstants;

void main() {
    gl_Position = pushConstants.transform * vec4(inPosition, 0.0, 1.0);
    fragColor = inColor * pushConstants.color;
    fragTexCoord = inTexCoord;
}"#,
        "frag" => r#"#version 450
layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D texSampler;

layout(push_constant) uniform PushConstants {
    mat4 transform;
    vec4 color;
    uint useTexture;
    uint _padding1;
    uint _padding2;
    uint _padding3;
} pushConstants;

void main() {
    if (pushConstants.useTexture > 0) {
        outColor = fragColor * texture(texSampler, fragTexCoord);
    } else {
        outColor = fragColor;
    }
}"#,
        _ => return Err(anyhow!("Unsupported shader type: {}", shader_type)),
    };

    let mut file = File::create(path)?;
    file.write_all(shader_code.as_bytes())?;

    Ok(())
}