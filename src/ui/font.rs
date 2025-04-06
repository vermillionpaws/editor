use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FontStyle {
    Regular,
    Bold,
    Italic,
    BoldItalic,
}

#[derive(Clone, Debug)]
pub struct Font {
    pub family: String,
    pub size: f32,
    pub line_height: f32,
    pub style: FontStyle,
    pub path: Option<PathBuf>,
}

impl Font {
    pub fn monospace(size: f32) -> Self {
        Self {
            family: "JetBrains Mono".to_string(),
            size,
            line_height: size * 1.5,
            style: FontStyle::Regular,
            path: Some(PathBuf::from("assets/fonts/JetBrainsMono-Regular.ttf")),
        }
    }

    pub fn default() -> Self {
        Self::monospace(14.0)
    }

    pub fn with_style(mut self, style: FontStyle) -> Self {
        self.style = style;

        // Update path based on style
        if let Some(path) = &self.path {
            let file_stem = path.file_stem().unwrap_or_default().to_str().unwrap_or("");
            let extension = path.extension().unwrap_or_default().to_str().unwrap_or("ttf");

            // Extract base font name without style suffix
            let base_name = if file_stem.ends_with("-Regular") ||
                              file_stem.ends_with("-Bold") ||
                              file_stem.ends_with("-Italic") ||
                              file_stem.ends_with("-BoldItalic") {
                let parts: Vec<&str> = file_stem.split('-').collect();
                parts[0]
            } else {
                file_stem
            };

            // Create new path with appropriate style suffix
            let style_suffix = match style {
                FontStyle::Regular => "Regular",
                FontStyle::Bold => "Bold",
                FontStyle::Italic => "Italic",
                FontStyle::BoldItalic => "BoldItalic",
            };

            let new_filename = format!("{}-{}.{}", base_name, style_suffix, extension);
            self.path = Some(PathBuf::from("assets/fonts").join(new_filename));
        }

        self
    }

    pub fn with_size(mut self, size: f32) -> Self {
        self.size = size;
        self.line_height = size * 1.5;
        self
    }

    pub fn exists(&self) -> bool {
        self.path.as_ref().map_or(false, |p| p.exists())
    }

    pub fn fallback_path(&self) -> PathBuf {
        PathBuf::from("assets/fonts/JetBrainsMono-Regular.ttf")
    }
}