//! Tiny terminal styling helpers. No dependencies on purpose — swap in
//! `anstream`/`owo-colors` if you want fancier output.

use std::io::{IsTerminal, Write};

pub fn color_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal())
}

fn paint(code: &str, s: &str) -> String {
    if color_enabled() {
        format!("\x1b[{code}m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

pub fn bold(s: &str) -> String {
    paint("1", s)
}
pub fn dim(s: &str) -> String {
    paint("2", s)
}
pub fn red(s: &str) -> String {
    paint("31", s)
}
pub fn green(s: &str) -> String {
    paint("32", s)
}
pub fn yellow(s: &str) -> String {
    paint("33", s)
}
pub fn blue(s: &str) -> String {
    paint("34", s)
}
pub fn magenta(s: &str) -> String {
    paint("35", s)
}
pub fn cyan(s: &str) -> String {
    paint("36", s)
}

/// Clear the current line before we start streaming model output over it.
pub fn clear_line() {
    if std::io::stdout().is_terminal() {
        print!("\r\x1b[2K");
    }
}

pub fn flush() {
    let _ = std::io::stdout().flush();
}

/// Hard-wrap nothing, just pad/truncate a string to a fixed width for tables.
pub fn truncate_chars(s: &str, max: usize) -> (String, bool) {
    if s.chars().count() <= max {
        return (s.to_string(), false);
    }
    let cut: String = s.chars().take(max).collect();
    (cut, true)
}
