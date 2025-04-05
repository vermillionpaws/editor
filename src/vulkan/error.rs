use thiserror::Error;

#[derive(Debug, Error)]
pub enum VulkanError {
    #[error("Failed to find suitable Vulkan physical device")]
    NoSuitableDevice,
    #[error("Required extension not available: {0}")]
    ExtensionNotAvailable(String),
    #[error("Required validation layer not available: {0}")]
    ValidationLayerNotAvailable(String),
}