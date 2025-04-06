use super::theme::Theme;
use super::font::Font;
use super::keybinds::Keybinds;

#[derive(Clone)]
pub enum EditorMode {
    Normal,
    Insert,
    Visual,
    Command,
}

pub struct UiState {
    pub theme: Theme,
    pub font: Font,
    pub keybinds: Keybinds,
    pub mode: EditorMode,
    pub show_line_numbers: bool,
    pub show_status_bar: bool,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            theme: Theme::minimalistic(),
            font: Font::default(),
            keybinds: Keybinds::vim_style(),
            mode: EditorMode::Normal,
            show_line_numbers: true,
            show_status_bar: true,
        }
    }
}