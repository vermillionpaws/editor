use std::collections::HashMap;
use super::state::EditorMode;

#[derive(Debug, Clone)]
pub struct KeyBindings {
    #[allow(dead_code)]
    pub bindings: HashMap<String, Action>,
    #[allow(dead_code)]
    pub mode_transitions: HashMap<String, EditorMode>,
}

#[derive(Debug, Clone)]
pub enum Action {
    MoveCursorUp,
    MoveCursorDown,
    MoveCursorLeft,
    MoveCursorRight,

    DeleteChar,
    DeleteLine,

    Undo,
    Redo,

    Copy,
    Cut,
    Paste,

    Save,
    SaveAs,
    Open,
    Quit,

    FindNext,
    FindPrevious,
    Replace,

    ToggleLineNumbers,
    ToggleStatusBar,
    ToggleTheme,
}

impl Default for KeyBindings {
    fn default() -> Self {
        let mut bindings = HashMap::new();
        let mut mode_transitions = HashMap::new();

        // Normal mode key bindings
        bindings.insert("k".to_string(), Action::MoveCursorUp);
        bindings.insert("j".to_string(), Action::MoveCursorDown);
        bindings.insert("h".to_string(), Action::MoveCursorLeft);
        bindings.insert("l".to_string(), Action::MoveCursorRight);

        bindings.insert("dd".to_string(), Action::DeleteLine);

        bindings.insert("y".to_string(), Action::Copy);
        bindings.insert("p".to_string(), Action::Paste);

        bindings.insert("\\n".to_string(), Action::ToggleLineNumbers);
        bindings.insert("\\s".to_string(), Action::ToggleStatusBar);
        bindings.insert("\\t".to_string(), Action::ToggleTheme);

        // Mode transitions
        mode_transitions.insert("i".to_string(), EditorMode::Insert);
        mode_transitions.insert("v".to_string(), EditorMode::Visual);
        mode_transitions.insert(":".to_string(), EditorMode::Command);

        Self {
            bindings,
            mode_transitions,
        }
    }

    #[allow(dead_code)]
    pub fn get_action(&self, key: &str) -> Option<&Action> {
        self.bindings.get(key)
    }

    #[allow(dead_code)]
    pub fn get_mode_transition(&self, key: &str) -> Option<&EditorMode> {
        self.mode_transitions.get(key)
    }
}