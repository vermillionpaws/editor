mod ui;
mod vulkan;

use anyhow::Result;
use ash::vk;
use log::{error, info, trace};
use std::sync::Arc;
use vulkan::VulkanApp;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent, Modifiers},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey, ModifiersState},
    window::{WindowId, WindowAttributes},
};
use ui::{
    font::FontStyle,
    state::EditorMode,
    UiState,
    TextEditor,
    StatusBar,
    LineNumbers,
    CommandLine,
    UiRenderer,
    Theme,
    Font,
};

struct AppState {
    vulkan_app: VulkanApp,
    ui_state: UiState,
    text_editor: TextEditor,
    status_bar: StatusBar,
    line_numbers: LineNumbers,
    command_line: CommandLine,
    current_command: String,
    ui_renderer: UiRenderer,
    modifiers: ModifiersState,
    last_update_time: std::time::Instant,
    cursor_blink: bool,
    window: Arc<winit::window::Window>, // Add window reference
}

impl AppState {
    fn new(vulkan_app: VulkanApp, window: Arc<winit::window::Window>) -> Self {
        let ui_state = UiState::default(); // Update constructor
        let ui_renderer = UiRenderer::new(&ui_state.theme);
        Self {
            vulkan_app,
            ui_renderer,
            text_editor: TextEditor::new(),
            status_bar: StatusBar {
                left_text: String::from("editor"),
                right_text: String::from("Ln 1, Col 1"),
            },
            line_numbers: LineNumbers {
                line_count: 1,
            },
            command_line: CommandLine {
                text: String::new(),
            },
            current_command: String::new(),
            ui_state,
            modifiers: ModifiersState::empty(),
            last_update_time: std::time::Instant::now(),
            cursor_blink: true,
            window, // Store window reference
        }
    }

    fn initialize_renderer(&mut self) -> Result<()> {
        // Clone the VulkanApp and wrap it in an Arc
        let app_arc = Arc::new(self.vulkan_app.clone());
        // Initialize with default font
        let font = self.ui_state.font.clone();
        self.ui_renderer.initialize(app_arc, &font)?;
        Ok(())
    }

    fn handle_key_event(&mut self, key_event: &KeyEvent) {
        // Track modifier keys
        if let Key::Named(name) = &key_event.logical_key {
            match name {
                NamedKey::Shift => {
                    if key_event.state == ElementState::Pressed {
                        self.modifiers = self.modifiers | ModifiersState::SHIFT;
                    } else {
                        self.modifiers = self.modifiers & !ModifiersState::SHIFT;
                    }
                }
                NamedKey::Control => {
                    if key_event.state == ElementState::Pressed {
                        self.modifiers = self.modifiers | ModifiersState::CONTROL;
                    } else {
                        self.modifiers = self.modifiers & !ModifiersState::CONTROL;
                    }
                }
                NamedKey::Alt => {
                    if key_event.state == ElementState::Pressed {
                        self.modifiers = self.modifiers | ModifiersState::ALT;
                    } else {
                        self.modifiers = self.modifiers & !ModifiersState::ALT;
                    }
                }
                _ => {}
            }
        }
        if key_event.state != ElementState::Pressed {
            return;
        }
        match self.ui_state.mode {
            EditorMode::Normal => self.handle_normal_mode_key(key_event),
            EditorMode::Insert => self.handle_insert_mode_key(key_event),
            EditorMode::Visual => self.handle_visual_mode_key(key_event),
            EditorMode::Command => self.handle_command_mode_key(key_event),
        }
    }

    fn handle_normal_mode_key(&mut self, key_event: &KeyEvent) {
        match &key_event.logical_key {
            // Handle mode transitions
            Key::Character(c) if c == "i" => {
                self.ui_state.mode = EditorMode::Insert;
                self.text_editor.clear_selection();
                info!("Switched to INSERT mode");
            }
            Key::Character(c) if c == ":" => {
                self.ui_state.mode = EditorMode::Command;
                info!("Switched to COMMAND mode");
                self.current_command.clear();
            }
            Key::Character(c) if c == "v" => {
                self.ui_state.mode = EditorMode::Visual;
                self.text_editor.start_selection();
                info!("Switched to VISUAL mode");
            }
            // Handle navigation
            Key::Character(c) if c == "h" => {
                if self.text_editor.cursor_position.1 > 0 {
                    self.text_editor.cursor_position.1 -= 1;
                    self.update_status_bar();
                }
            }
            Key::Character(c) if c == "j" => {
                // Move cursor down if possible
                self.text_editor.cursor_position.0 += 1;
                self.update_status_bar();
            }
            Key::Character(c) if c == "k" => {
                // Move cursor up if possible
                if self.text_editor.cursor_position.0 > 0 {
                    self.text_editor.cursor_position.0 -= 1;
                    self.update_status_bar();
                }
            }
            Key::Character(c) if c == "l" => {
                // Move cursor right if possible
                self.text_editor.cursor_position.1 += 1;
                self.update_status_bar();
            }
            // Toggle UI elements
            Key::Character(c) if c == "\\" => {
                if self.modifiers.contains(ModifiersState::SHIFT) {
                    // Toggle line numbers with \N
                    self.ui_state.show_line_numbers = !self.ui_state.show_line_numbers;
                    info!(
                        "Line numbers: {}",
                        if self.ui_state.show_line_numbers {
                            "on"
                        } else {
                            "off"
                        }
                    );
                } else if self.modifiers.contains(ModifiersState::CONTROL) {
                    // Switch font styles with Ctrl+\
                    self.cycle_font_style();
                } else {
                    // Switch theme with \
                    self.cycle_theme();
                }
            }
            // Font size adjustments
            Key::Character(c) if c == "+" && self.modifiers.contains(ModifiersState::CONTROL) => {
                self.change_font_size(1.0);
            }
            Key::Character(c) if c == "-" && self.modifiers.contains(ModifiersState::CONTROL) => {
                self.change_font_size(-1.0);
            }
            // Copy/paste functionality
            Key::Character(c) if c == "y" && self.text_editor.has_selection() => {
                // Yank selected text (copy)
                info!("Copied: {}", self.text_editor.get_selected_text());
                self.text_editor.clear_selection();
            }
            Key::Named(NamedKey::F5) => {
                // Reload font atlas (for testing purposes)
                info!("Reloading font atlas");
                if let Err(e) = self.reload_font_atlas() {
                    error!("Failed to reload font atlas: {}", e);
                }
            }
            _ => {}
        }
    }

    fn handle_insert_mode_key(&mut self, key_event: &KeyEvent) {
        match &key_event.logical_key {
            Key::Named(NamedKey::Escape) => {
                self.ui_state.mode = EditorMode::Normal;
                self.text_editor.clear_selection();
                info!("Switched to NORMAL mode");
            }
            Key::Character(c) => {
                // Insert character at cursor position
                let mut content_lines = self.ensure_content_lines();
                // Make sure we have enough rows
                while self.text_editor.cursor_position.0 >= content_lines.len() {
                    content_lines.push(String::new());
                }
                // Insert the character at the current position
                let row = &mut content_lines[self.text_editor.cursor_position.0];
                if self.text_editor.cursor_position.1 >= row.len() {
                    row.push_str(&" ".repeat(self.text_editor.cursor_position.1 - row.len()));
                    row.push_str(c);
                } else {
                    let mut new_row = row[..self.text_editor.cursor_position.1].to_string();
                    new_row.push_str(c);
                    new_row.push_str(&row[self.text_editor.cursor_position.1..]);
                    *row = new_row;
                }
                // Update content
                self.text_editor.content = content_lines.join("\n");
                // Move cursor forward
                self.text_editor.cursor_position.1 += 1;
                // Update line count for line numbers
                self.line_numbers.line_count = content_lines.len();
                self.update_status_bar();
                trace!("Inserted character: {}", c);
            }
            Key::Named(NamedKey::Enter) => {
                // Insert a new line
                let mut content_lines = self.ensure_content_lines();
                // Make sure we have enough rows
                while self.text_editor.cursor_position.0 >= content_lines.len() {
                    content_lines.push(String::new());
                }
                // Get the current line
                let row = &mut content_lines[self.text_editor.cursor_position.0];
                let new_line_content: String;
                if self.text_editor.cursor_position.1 >= row.len() {
                    // Cursor is at or beyond the end of the line
                    new_line_content = String::new();
                } else {
                    // Split the line at the cursor
                    new_line_content = row[self.text_editor.cursor_position.1..].to_string();
                    *row = row[..self.text_editor.cursor_position.1].to_string();
                }
                // Insert the new line
                content_lines.insert(self.text_editor.cursor_position.0 + 1, new_line_content);
                // Update content
                self.text_editor.content = content_lines.join("\n");
                // Move cursor to the beginning of the next line
                self.text_editor.cursor_position.0 += 1;
                self.text_editor.cursor_position.1 = 0;
                // Update line count for line numbers
                self.line_numbers.line_count = content_lines.len();
                self.update_status_bar();
                trace!("Inserted new line");
            }
            Key::Named(NamedKey::Backspace) => {
                // Delete character before cursor
                let mut content_lines = self.ensure_content_lines();
                if self.text_editor.cursor_position.1 > 0 {
                    // Delete character before cursor on the current line
                    if self.text_editor.cursor_position.0 < content_lines.len() {
                        let row = &mut content_lines[self.text_editor.cursor_position.0];
                        if self.text_editor.cursor_position.1 <= row.len() {
                            let mut new_row =
                                row[..self.text_editor.cursor_position.1 - 1].to_string();
                            new_row.push_str(&row[self.text_editor.cursor_position.1..]);
                            *row = new_row;
                            // Move cursor back
                            self.text_editor.cursor_position.1 -= 1;
                        }
                    }
                } else if self.text_editor.cursor_position.0 > 0 {
                    // At the beginning of a line, merge with previous line
                    let current_line = content_lines.remove(self.text_editor.cursor_position.0);
                    let previous_line_idx = self.text_editor.cursor_position.0 - 1;
                    // Set cursor to end of previous line
                    self.text_editor.cursor_position.0 = previous_line_idx;
                    self.text_editor.cursor_position.1 = content_lines[previous_line_idx].len();
                    // Append current line to previous line
                    content_lines[previous_line_idx].push_str(&current_line);
                }
                // Update content
                self.text_editor.content = content_lines.join("\n");
                // Update line count for line numbers
                self.line_numbers.line_count = content_lines.len();
                self.update_status_bar();
                trace!("Backspace pressed");
            }
            _ => {}
        }
    }

    fn handle_visual_mode_key(&mut self, key_event: &KeyEvent) {
        match &key_event.logical_key {
            Key::Named(NamedKey::Escape) => {
                self.ui_state.mode = EditorMode::Normal;
                self.text_editor.clear_selection();
                info!("Switched to NORMAL mode");
            }
            // Navigation keys to extend selection
            Key::Character(c) if c == "h" => {
                if self.text_editor.cursor_position.1 > 0 {
                    self.text_editor.cursor_position.1 -= 1;
                    self.update_status_bar();
                }
            }
            Key::Character(c) if c == "j" => {
                // Move cursor down if possible while extending selection
                self.text_editor.cursor_position.0 += 1;
                self.update_status_bar();
            }
            Key::Character(c) if c == "k" => {
                // Move cursor up if possible while extending selection
                if self.text_editor.cursor_position.0 > 0 {
                    self.text_editor.cursor_position.0 -= 1;
                    self.update_status_bar();
                }
            }
            Key::Character(c) if c == "l" => {
                // Move cursor right if possible while extending selection
                self.text_editor.cursor_position.1 += 1;
                self.update_status_bar();
            }
            // Select operations
            Key::Character(c) if c == "y" => {
                // Yank (copy) selected text
                info!("Copied: {}", self.text_editor.get_selected_text());
                self.text_editor.clear_selection();
                self.ui_state.mode = EditorMode::Normal;
            }
            Key::Character(c) if c == "d" => {
                // Delete selected text
                if self.text_editor.has_selection() {
                    // Implementation of delete would go here
                    info!("Delete selected text");
                    // For now just clear selection
                    self.text_editor.clear_selection();
                    self.ui_state.mode = EditorMode::Normal;
                }
            }
            _ => {}
        }
    }

    fn handle_command_mode_key(&mut self, key_event: &KeyEvent) {
        match &key_event.logical_key {
            Key::Named(NamedKey::Escape) => {
                self.ui_state.mode = EditorMode::Normal;
                info!("Switched to NORMAL mode");
                self.current_command.clear();
                self.command_line.text = String::new();
            }
            Key::Named(NamedKey::Enter) => {
                self.execute_command();
                self.ui_state.mode = EditorMode::Normal;
                info!("Switched to NORMAL mode");
                self.current_command.clear();
                self.command_line.text = String::new();
            }
            Key::Character(c) => {
                self.current_command.push_str(c);
                self.command_line.text = self.current_command.clone();
            }
            Key::Named(NamedKey::Backspace) => {
                if !self.current_command.is_empty() {
                    self.current_command.pop();
                    self.command_line.text = self.current_command.clone();
                }
            }
            _ => {}
        }
    }

    fn execute_command(&mut self) {
        info!("Executing command: {}", self.current_command);
        match self.current_command.as_str() {
            "w" => {
                info!("Save command");
                // Save file logic would go here
            }
            "q" => {
                info!("Quit command");
                // This would typically exit the app
            }
            "wq" => {
                info!("Save and quit command");
                // Save and quit logic
            }
            "theme" => {
                // Switch theme
                self.cycle_theme();
            }
            _ => {
                info!("Unknown command: {}", self.current_command);
            }
        }
    }

    fn update_status_bar(&mut self) {
        self.status_bar.right_text = format!(
            "Ln {}, Col {}",
            self.text_editor.cursor_position.0 + 1,
            self.text_editor.cursor_position.1 + 1
        );
    }

    fn ensure_content_lines(&mut self) -> Vec<String> {
        if self.text_editor.content.is_empty() {
            vec![String::new()]
        } else {
            self.text_editor.content.lines().map(String::from).collect()
        }
    }

    fn cycle_theme(&mut self) {
        // Switch theme between minimalistic and high contrast
        self.ui_state.theme = if self.ui_state.theme.background_color == "#2E2E2E" {
            Theme::high_contrast()
        } else {
            Theme::minimalistic()
        };
        self.ui_renderer.update_colors(&self.ui_state.theme);
        info!(
            "Switched theme to {}",
            if self.ui_state.theme.background_color == "#2E2E2E" {
                "minimalistic"
            } else {
                "high contrast"
            }
        );
    }

    fn cycle_font_style(&mut self) {
        // Cycle through available font styles
        self.ui_state.font = match self.ui_state.font.style {
            FontStyle::Regular => self.ui_state.font.clone().with_style(FontStyle::Bold),
            FontStyle::Bold => self.ui_state.font.clone().with_style(FontStyle::Italic),
            FontStyle::Italic => self.ui_state.font.clone().with_style(FontStyle::BoldItalic),
            FontStyle::BoldItalic => self.ui_state.font.clone().with_style(FontStyle::Regular),
        };
        // Reload font atlas with new style
        self.reload_font_atlas().unwrap_or_else(|e| {
            error!("Failed to change font style: {}", e);
            // Fallback to regular style
            self.ui_state.font = Font::default();
            self.reload_font_atlas().unwrap_or_else(|e| {
                error!("Failed to reload default font: {}", e);
            });
        });
        info!("Changed font style to {:?}", self.ui_state.font.style);
    }

    fn change_font_size(&mut self, delta: f32) {
        let new_size = (self.ui_state.font.size + delta).max(8.0).min(32.0);
        self.ui_state.font = self.ui_state.font.clone().with_size(new_size);
        // Reload font atlas with new size
        self.reload_font_atlas().unwrap_or_else(|e| {
            error!("Failed to change font size: {}", e);
            // Fallback to default size
            self.ui_state.font = Font::default();
            self.reload_font_atlas().unwrap_or_else(|e| {
                error!("Failed to reload default font: {}", e);
            });
        });
        info!("Changed font size to {}", self.ui_state.font.size);
    }

    fn reload_font_atlas(&mut self) -> Result<()> {
        // Clone the VulkanApp and wrap it in an Arc
        let app_arc = Arc::new(self.vulkan_app.clone());
        // Clean up old resources first
        self.ui_renderer.cleanup();
        // Reinitialize with new font
        self.ui_renderer.initialize(app_arc, &self.ui_state.font)?;
        Ok(())
    }

    fn render_ui(&mut self) {
        // Clear previous frame's elements
        self.ui_renderer.clear_elements();
        // Check if cursor should blink
        let now = std::time::Instant::now();
        if now.duration_since(self.last_update_time).as_millis() > 500 {
            self.cursor_blink = !self.cursor_blink;
            self.last_update_time = now;
        }
        // Render UI components
        self.ui_renderer.render_editor(&self.text_editor, &self.ui_state);
        self.ui_renderer.render_status_bar(&self.status_bar, &self.ui_state);
        self.ui_renderer.render_line_numbers(&self.line_numbers, &self.ui_state);
        self.ui_renderer.render_command_line(&self.command_line, &self.ui_state);
    }
}

impl ApplicationHandler for AppState {
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event: key_event,
                is_synthetic: false,
                ..
            } => {
                self.handle_key_event(&key_event);
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers.state();
            }
            WindowEvent::Resized(_) => {
                // Window resize handling would go here
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Render UI elements
        self.render_ui();
        // Advance the frame counter and get the right semaphore
        let frame_index = self.ui_renderer.advance_frame();
        let semaphore = self
            .ui_renderer
            .get_current_semaphore()
            .expect("No semaphore available");
        // Get the next swapchain image index using the current frame's semaphore
        let next_image_index = match unsafe {
            self.vulkan_app.swapchain_loader.acquire_next_image(
                self.vulkan_app.swapchain,
                u64::MAX,
                // Use the semaphore from UI renderer's current frame tracking
                semaphore,
                vk::Fence::null(),
            )
        } {
            Ok((index, _)) => index as usize,
            Err(_) => {
                // Handle errors (just log for now)
                error!("Failed to acquire next swapchain image");
                event_loop.exit();
                return;
            }
        };
        // Update the command buffer with the latest UI elements
        if let Err(e) = self
            .vulkan_app
            .update_command_buffer(next_image_index, &self.ui_renderer)
        {
            error!("Failed to update command buffer: {}", e);
            event_loop.exit();
            return;
        }
        // Render a frame with Vulkan using our acquired image index
        if let Err(e) = self.vulkan_app.draw_frame(&self.ui_renderer, &self.window) {
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
    // Create window attributes
    let window_attr = WindowAttributes::default()
        .with_title("Vulkan Editor")
        .with_inner_size(winit::dpi::LogicalSize::new(800, 600));
    // Create window directly - we'll accept the deprecation warning
    // since we can't find the correct non-deprecated API in this winit version
    let window = Arc::new(event_loop.create_window(window_attr)?);
    // Create Vulkan application
    let vulkan_app = VulkanApp::new(&window)?;
    // Create application state with UI
    let mut app_state = AppState::new(vulkan_app, window.clone());
    // Initialize the UI renderer
    if let Err(e) = app_state.initialize_renderer() {
        error!("Failed to initialize UI renderer: {}", e);
        return Err(anyhow::anyhow!("Failed to initialize UI renderer"));
    }
    info!(
        "Initialized editor with {} UI",
        if app_state.ui_state.theme.background_color == "#2E2E2E" {
            "minimalistic"
        } else {
            "high contrast"
        }
    );
    info!(
        "Using font: {} at {}pt",
        app_state.ui_state.font.family, app_state.ui_state.font.size
    );
    // Run the event loop
    event_loop.run_app(&mut app_state)?;
    Ok(())
}
