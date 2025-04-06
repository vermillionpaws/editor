pub struct Theme {
    pub background_color: &'static str,
    pub foreground_color: &'static str,
    // Additional colors for UI elements
    pub selection_color: &'static str,
    pub cursor_color: &'static str,
    pub line_number_color: &'static str,
    pub status_bar_bg_color: &'static str,
    pub status_bar_fg_color: &'static str,
    pub error_color: &'static str,
    pub warning_color: &'static str,
    pub comment_color: &'static str,
    pub keyword_color: &'static str,
    pub string_color: &'static str,
}

impl Theme {
    pub fn minimalistic() -> Self {
        Self {
            background_color: "#2E2E2E", // Dark gray
            foreground_color: "#D3D3D3", // Light gray
            selection_color: "#4A4A4A",  // Medium gray
            cursor_color: "#FFFFFF",     // White
            line_number_color: "#858585", // Medium light gray
            status_bar_bg_color: "#1E1E1E", // Darker gray for status bar
            status_bar_fg_color: "#E0E0E0", // Lighter gray for status text
            error_color: "#FF6B6B",      // Light red
            warning_color: "#FFD166",    // Light yellow/orange
            comment_color: "#6C7A89",    // Steel blue gray
            keyword_color: "#A5A5FF",    // Light purple
            string_color: "#9ACD68",     // Light green
        }
    }

    pub fn high_contrast() -> Self {
        Self {
            background_color: "#202020", // Very dark gray
            foreground_color: "#EEEEEE", // Off-white
            selection_color: "#444444",  // Medium dark gray
            cursor_color: "#FFFFFF",     // White
            line_number_color: "#999999", // Lighter gray
            status_bar_bg_color: "#181818", // Almost black
            status_bar_fg_color: "#FFFFFF", // White
            error_color: "#FF5252",      // Brighter red
            warning_color: "#FFD740",    // Bright yellow
            comment_color: "#757575",    // Medium gray
            keyword_color: "#B4B4FF",    // Brighter light purple
            string_color: "#AADD77",     // Brighter light green
        }
    }
}