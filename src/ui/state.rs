use super::{font::Font, keybinds::KeyBindings, theme::Theme};

// Current editor mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EditorMode {
    Normal,
    Insert,
    Visual,
    Command,
}

// Overall UI state
#[derive(Debug, Clone)]
pub struct UiState {
    pub mode: EditorMode,
    pub theme: Theme,
    pub font: Font,
    pub show_line_numbers: bool,
    pub show_status_bar: bool,
    pub line_spacing: f32,
    #[allow(dead_code)]
    pub keybinds: KeyBindings,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            mode: EditorMode::Normal,
            theme: Theme::minimalistic(),
            font: Font::default(),
            show_line_numbers: true,
            show_status_bar: true,
            line_spacing: 1.2,
            keybinds: KeyBindings::default(),
        }
    }
}