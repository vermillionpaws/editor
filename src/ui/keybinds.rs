use std::collections::HashMap;
use super::state::EditorMode;

pub struct Keybinds {
    pub bindings: HashMap<&'static str, &'static str>,
    pub mode_transitions: HashMap<&'static str, EditorMode>,
}

impl Keybinds {
    pub fn vim_style() -> Self {
        let mut bindings = HashMap::new();
        // Normal mode movement
        bindings.insert("h", "move_left");
        bindings.insert("j", "move_down");
        bindings.insert("k", "move_up");
        bindings.insert("l", "move_right");
        bindings.insert("0", "move_line_start");
        bindings.insert("$", "move_line_end");
        bindings.insert("gg", "move_document_start");
        bindings.insert("G", "move_document_end");
        bindings.insert("w", "move_word_forward");
        bindings.insert("b", "move_word_backward");

        // Editing
        bindings.insert("dd", "delete_line");
        bindings.insert("yy", "yank_line");
        bindings.insert("p", "paste_after");
        bindings.insert("P", "paste_before");
        bindings.insert("u", "undo");
        bindings.insert("ctrl+r", "redo");
        bindings.insert("x", "delete_char");
        bindings.insert("r", "replace_char");
        bindings.insert(">>", "indent");
        bindings.insert("<<", "unindent");

        // File operations
        bindings.insert(":w", "save");
        bindings.insert(":q", "quit");
        bindings.insert(":wq", "save_and_quit");
        bindings.insert(":q!", "force_quit");

        // Search
        bindings.insert("/", "search_forward");
        bindings.insert("?", "search_backward");
        bindings.insert("n", "search_next");
        bindings.insert("N", "search_previous");

        // Visual mode
        bindings.insert("v", "enter_visual");
        bindings.insert("V", "enter_visual_line");

        // Mode transitions
        let mut mode_transitions = HashMap::new();
        mode_transitions.insert("i", EditorMode::Insert);
        mode_transitions.insert("a", EditorMode::Insert);
        mode_transitions.insert("o", EditorMode::Insert);
        mode_transitions.insert("O", EditorMode::Insert);
        mode_transitions.insert("v", EditorMode::Visual);
        mode_transitions.insert("V", EditorMode::Visual);
        mode_transitions.insert(":", EditorMode::Command);
        mode_transitions.insert("Escape", EditorMode::Normal);

        Self {
            bindings,
            mode_transitions,
        }
    }

    pub fn get_action(&self, key: &str) -> Option<&'static str> {
        self.bindings.get(key).copied()
    }

    pub fn get_mode_transition(&self, key: &str) -> Option<EditorMode> {
        self.mode_transitions.get(key).cloned()
    }
}