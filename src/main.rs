mod vulkan;

use anyhow::Result;
use log::error;
use vulkan::VulkanApp;
use winit::{
    application::ApplicationHandler,
    event::{Event, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

struct AppState {
    vulkan_app: VulkanApp,
}

impl ApplicationHandler for AppState {
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { .. } => {
                // Keyboard input handling would go here
            }
            WindowEvent::Resized(_) => {
                // Window resize handling would go here
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Render a frame
        if let Err(e) = self.vulkan_app.draw_frame() {
            error!("Failed to draw frame: {}", e);
            event_loop.exit();
        }
    }

    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {
        // Required by the ApplicationHandler trait
    }
}

fn main() -> Result<()> {
    // Initialize logger
    env_logger::init();

    // Create window and event loop
    let event_loop = EventLoop::new()?;
    let window = event_loop.create_window(
        Window::default_attributes()
            .with_title("Vulkan Editor")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
    )?;

    // Create Vulkan application
    let vulkan_app = VulkanApp::new(&window)?;

    // Create application state
    let mut app_state = AppState { vulkan_app };

    // Run the event loop
    event_loop.run_app(&mut app_state)?;

    Ok(())
}
