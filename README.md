# Vulkan Editor

A modern, high-performance editor built with Rust and the Vulkan graphics API.

## Overview

This project is a graphical editor that leverages the power of Vulkan for rendering. It's built using Rust with a focus on performance, safety, and modern graphics techniques.

## Architecture

### Main Components

- **Main Application**: Handles the window creation and event loop using winit
- **Vulkan Backend**: Manages all GPU-related operations through the Vulkan API

### Core Modules

- **Instance Management**: Creates and configures the Vulkan instance with appropriate extensions
- **Device Management**: Handles physical and logical device selection and setup
- **Surface Handling**: Creates and manages the rendering surface for the window
- **Swapchain**: Manages the image presentation queue
- **Rendering**: Handles command buffers, render passes, and framebuffers
- **Synchronization**: Coordinates rendering operations with semaphores and fences

## Technical Details

### VulkanApp Structure

The primary `VulkanApp` struct manages all Vulkan resources, including:

- Vulkan instance and device
- Surface and swapchain
- Command buffers and pools
- Synchronization primitives
- Render passes and framebuffers

The app uses a double-buffered rendering approach with a maximum of 2 frames in flight.

### Rendering Pipeline

The current implementation provides a basic rendering loop with:

1. Waiting for previous frames to finish
2. Acquiring the next swapchain image
3. Submitting command buffers for rendering
4. Presenting the rendered image

### Debug Support

In debug builds, the application enables Vulkan validation layers and sets up a debug messenger for error reporting and validation.

## Requirements

- Rust 2024 edition
- Vulkan-compatible GPU
- Linux system with Vulkan drivers

## Dependencies

- **ash**: Rust bindings for Vulkan
- **winit**: Window creation and event handling
- **nalgebra**: Linear algebra for graphics calculations
- **anyhow/thiserror**: Error handling
- **env_logger**: Logging capabilities

## Building and Running

```bash
# Build the project
cargo build

# Run in debug mode
cargo run

# Build for release
cargo build --release

# Run in release mode
./target/release/editor
```

## Current Status

The project currently implements a basic rendering loop with Vulkan initialization. It provides the foundation for a more sophisticated editor with future enhancements planned.

## Future Development

Potential areas for enhancement:

- Shader compilation pipeline
- UI framework integration
- Text rendering capabilities
- Asset importing and management

## License

See the LICENSE file for details.
