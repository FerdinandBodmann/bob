//! Minimal HTML → readable text. No dependencies: strip the noisy elements,
//! turn block boundaries into newlines, drop tags, decode the common entities.
//!
//! Good enough for docs pages, READMEs, news articles and API JSON. Swap in
//! `html2text` or `scraper` if you need truly robust extraction.

const NOISY_ELEMENTS: [&str; 7] = ["script", "style", "noscript", "svg", "iframe", "template", "canvas"];

/// Tags that should produce a line break when they open or close.
const BLOCK_TAGS: [&str; 16] = [
    "p", "div", "br", "li", "ul", "ol", "tr", "td", "th", "table", "section", "article", "header",
    "footer", "pre", "blockquote",
];

pub fn title(html: &str) -> Option<String> {
    let lower = html.to_lowercase();
    let start = lower.find("<title")?;
    let open_end = lower[start..].find('>')? + start + 1;
    let end = lower[open_end..].find("</title>")? + open_end;
    let text = normalize_ws(&decode_entities(&html[open_end..end]));
    (!text.is_empty()).then_some(text)
}

pub fn to_text(html: &str) -> String {
    let mut s = html.to_string();

    // Comments.
    s = strip_ranges(&s, "<!--", "-->");

    // Whole elements that never contain readable prose.
    for tag in NOISY_ELEMENTS {
        s = strip_element(&s, tag);
    }

    // Head holds metadata we do not want in the body text.
    s = strip_element(&s, "head");

    // Block boundaries -> newlines, so paragraphs survive the tag stripping.
    let lower = s.to_lowercase();
    let mut out = String::with_capacity(s.len());
    let mut cursor = 0usize;
    while cursor < s.len() {
        if lower[cursor..].starts_with('<') {
            if let Some(close) = s[cursor..].find('>') {
                let tag = &lower[cursor..cursor + close + 1];
                if BLOCK_TAGS.iter().any(|b| tag_is(tag, b)) {
                    out.push('\n');
                }
                cursor += close + 1;
                continue;
            }
        }
        let ch_len = s[cursor..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
        out.push_str(&s[cursor..cursor + ch_len]);
        cursor += ch_len;
    }
    let mut s = out;

    // Any remaining tags.
    s = strip_ranges(&s, "<", ">");

    s = decode_entities(&s);

    let mut lines = Vec::new();
    for line in s.lines() {
        let line = normalize_ws(line);
        if !line.is_empty() {
            lines.push(line);
        }
    }
    lines.join("\n")
}

/// `<script …>…</script>` (or self-closing `<script/>`) removed entirely.
fn strip_element(s: &str, tag: &str) -> String {
    let lower = s.to_lowercase();
    let mut out = String::with_capacity(s.len());
    let mut cursor = 0usize;

    while let Some(found) = lower[cursor..].find(&format!("<{tag}")) {
        let start = cursor + found;
        // Make sure it is really that tag (`<script>` vs `<scripting>`).
        let after_start = (start + tag.len() + 1).min(lower.len());
        let after = &lower[after_start..];
        if !after.starts_with('>') && !after.starts_with(' ') && !after.starts_with('/') && !after.starts_with('\n') {
            out.push_str(&s[cursor..start + tag.len() + 1]);
            cursor = start + tag.len() + 1;
            continue;
        }
        out.push_str(&s[cursor..start]);

        let close = format!("</{tag}>");
        let search_from = start + tag.len() + 1;
        match lower[search_from..].find(&close) {
            Some(pos) => cursor = search_from + pos + close.len(),
            None => cursor = search_from, // malformed: keep going
        }
    }
    out.push_str(&s[cursor..]);
    out
}

/// Remove everything between `open` and `close`, markers included.
fn strip_ranges(s: &str, open: &str, close: &str) -> String {
    let lower = s.to_lowercase();
    let open_l = open.to_lowercase();
    let close_l = close.to_lowercase();
    let mut out = String::with_capacity(s.len());
    let mut cursor = 0usize;

    while let Some(start) = lower[cursor..].find(&open_l) {
        let start = cursor + start;
        out.push_str(&s[cursor..start]);
        let search_from = start + open_l.len();
        match lower[search_from..].find(&close_l) {
            Some(pos) => cursor = search_from + pos + close_l.len(),
            None => cursor = search_from,
        }
    }
    out.push_str(&s[cursor..]);
    out
}

/// `tag` is like `<br>` or `<br/>`; `name` is `br`.
fn tag_is(tag: &str, name: &str) -> bool {
    let inner = tag.trim_start_matches('<').trim_end_matches('>');
    let inner = inner.trim_start_matches('/');
    inner
        .split(|c: char| c.is_whitespace() || c == '/')
        .next()
        .is_some_and(|head| head == name)
}

pub fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(pos) = rest.find('&') {
        out.push_str(&rest[..pos]);
        let tail = &rest[pos..];
        let end = tail.find(';').filter(|e| *e <= 10).unwrap_or(1);
        let entity = &tail[1..end];
        let decoded: Option<String> = match entity {
            "amp" => Some("&".into()),
            "lt" => Some("<".into()),
            "gt" => Some(">".into()),
            "quot" => Some("\"".into()),
            "apos" | "#39" | "#x27" => Some("'".into()),
            "nbsp" | "#160" | "ensp" | "emsp" => Some(" ".into()),
            "mdash" => Some("—".into()),
            "ndash" => Some("–".into()),
            "hellip" => Some("…".into()),
            "rsquo" | "lsquo" => Some("'".into()),
            "rdquo" | "ldquo" => Some("\"".into()),
            _ if entity.starts_with('#') => {
                let digits = entity.trim_start_matches('#');
                let code = if digits.starts_with('x') || digits.starts_with('X') {
                    u32::from_str_radix(&digits[1..], 16).ok()
                } else {
                    digits.parse().ok()
                };
                code.and_then(char::from_u32).map(|c| c.to_string())
            }
            _ => None,
        };
        match decoded {
            Some(text) => {
                out.push_str(&text);
                rest = &tail[end + 1..];
            }
            None => {
                out.push('&');
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

fn normalize_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_scripts_and_styles() {
        let html = r#"<html><head><title>Hi</title><style>b{color:red}</style></head>
            <body><script>alert(1)</script><p>Hello <b>world</b></p></body></html>"#;
        let text = to_text(html);
        assert!(!text.contains("alert"), "{text}");
        assert!(!text.contains("color:red"), "{text}");
        assert!(text.contains("Hello world"), "{text}");
    }

    #[test]
    fn keeps_paragraph_breaks() {
        let text = to_text("<p>one</p><p>two</p>");
        assert_eq!(text, "one\ntwo");
    }

    #[test]
    fn decodes_entities() {
        assert_eq!(decode_entities("a &amp; b &#65; &nbsp;c"), "a & b A  c");
    }

    #[test]
    fn extracts_title() {
        assert_eq!(title("<html><title> Page </title></html>").as_deref(), Some("Page"));
    }
}
