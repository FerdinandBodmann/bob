//! `search` tool: DuckDuckGo Lite (no API key, plain HTML).
//!
//! If you want a proper search API, this module is the only thing to replace:
//! implement `run()` against Brave / Serper / Tavily / Exa and keep the
//! `Title\n   URL\n   Snippet` output format so the model reads it the same way.

use anyhow::{bail, Context, Result};
use reqwest::header::{ACCEPT, USER_AGENT as UA_HEADER};
use serde_json::Value;

use super::{htmltext, Ctx, USER_AGENT};

const ENDPOINT: &str = "https://lite.duckduckgo.com/lite/";

#[derive(Debug)]
pub struct Hit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

pub async fn run(args: &Value, ctx: &Ctx) -> Result<String> {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .context(r#"missing required argument "query""#)?
        .trim();
    if query.is_empty() {
        bail!("query is empty");
    }
    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(8)
        .clamp(1, 20);

    let url = format!("{ENDPOINT}?q={}", percent_encode(query));
    let resp = ctx
        .http
        .get(&url)
        .header(UA_HEADER, USER_AGENT)
        .header(ACCEPT, "text/html")
        .send()
        .await
        .context("could not reach DuckDuckGo Lite")?;

    let status = resp.status();
    if !status.is_success() {
        bail!("search endpoint returned {status} — likely rate limited, try again shortly");
    }
    let body = resp.text().await.context("could not read search response")?;
    let hits = parse(&body, limit);

    if hits.is_empty() {
        bail!(
            "no results parsed from the search page (blocked or markup changed). \
             Try fetching a search URL directly, or wire a search API into src/tools/search.rs"
        );
    }

    let mut out = String::new();
    for (i, hit) in hits.iter().enumerate() {
        out.push_str(&format!("{}. {}\n   {}\n   {}\n\n", i + 1, hit.title, hit.url, hit.snippet));
    }
    Ok(out.trim_end().to_string())
}

/// Parse DDG Lite results: `<a rel="nofollow" href="…">title</a>` followed by
/// `<td class="result-snippet">…</td>` blocks.
pub fn parse(html: &str, limit: usize) -> Vec<Hit> {
    let mut hits = Vec::new();
    let mut pos = 0usize;

    while hits.len() < limit {
        let Some((title, url, next)) = next_link(html, pos) else {
            break;
        };
        let (snippet, after_snippet) = next_snippet(html, next);
        pos = after_snippet.max(next);
        hits.push(Hit {
            title,
            url,
            snippet,
        });
    }
    hits
}

/// Find the next result link at or after `from`, returning the index just past
/// its `</a>`.
/// Find the next result link at or after `from`, returning the index just past
/// its `</a>`. Only opening tags (`<a …`) count — a bare `find("<a")` would also
/// match the `<a` inside every `</a>` close tag.
fn next_link(html: &str, from: usize) -> Option<(String, String, usize)> {
    let mut cursor = from;
    while let Some(found) = find_open_tag(html, cursor, "a") {
        let start = cursor + found;
        let tag_end = start + html[start..].find('>')?;
        let tag = &html[start..=tag_end];
        let attrs = tag.to_lowercase();
        let href = attr(tag, "href").unwrap_or_default();

        let inner_start = tag_end + 1;
        let inner_end = html[inner_start..].find("</a>").map(|p| inner_start + p)?;
        let title = htmltext::to_text(&html[inner_start..inner_end]);
        let title = htmltext::decode_entities(&title);
        let next = (inner_end + 4).min(html.len());

        let is_result =
            attrs.contains("rel=\"nofollow\"") || attrs.contains("class=\"result-link\"");

        cursor = next;
        if is_result && href.starts_with("http") && !title.is_empty() {
            return Some((title, resolve(&href), next));
        }
    }
    None
}

/// Byte offset of the next opening tag `<name` (attributes or `>` following)
/// at or after `from`. A close tag `</name>` never matches, because of the `/`.
fn find_open_tag(html: &str, from: usize, name: &str) -> Option<usize> {
    let bytes = html.as_bytes();
    let needle = format!("<{name}");
    let mut cursor = from;

    while let Some(found) = html[cursor..].find(&needle) {
        let start = cursor + found;
        let next_byte = bytes.get(start + needle.len()).copied();
        let is_open = match next_byte {
            // `<a>`, `<a …>`, `<a/>`
            None | Some(b'>') | Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r') | Some(b'/') => true,
            // `<area>`, `<abbr>` — a different tag entirely
            _ => false,
        };
        if is_open {
            return Some(found);
        }
        cursor = start + needle.len();
    }
    None
}

fn next_snippet(html: &str, from: usize) -> (String, usize) {
    let Some(start) = html[from..].find("result-snippet") else {
        return (String::new(), from);
    };
    let start = start + from;
    let after_marker = start + "result-snippet".len();
    let Some(open_end) = html[after_marker..].find('>') else {
        return (String::new(), from);
    };
    let inner_start = after_marker + open_end + 1;
    let Some(close) = html[inner_start..].find("</td>") else {
        return (String::new(), from);
    };
    let inner_end = inner_start + close;
    let text = htmltext::decode_entities(&htmltext::to_text(&html[inner_start..inner_end]));
    (text, (inner_end + 5).min(html.len()))
}

fn attr(tag: &str, name: &str) -> Option<String> {
    let lower = tag.to_lowercase();
    let pos = lower.find(&format!("{name}=\""))? + name.len() + 1;
    let end = pos + tag[pos..].find('"')?;
    Some(tag[pos..end].to_string())
}

/// DuckDuckGo wraps outbound links in a redirect; unwrap `uddg=`.
fn resolve(href: &str) -> String {
    if let Some(pos) = href.find("uddg=") {
        let encoded = &href[pos + 5..];
        let cut = encoded.find('&').unwrap_or(encoded.len());
        if let Some(decoded) = percent_decode(&encoded[..cut]) {
            return decoded;
        }
    }
    href.to_string()
}

fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for byte in s.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*byte as char)
            }
            b' ' => out.push('+'),
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

fn percent_decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hex = s.get(i + 1..i + 3)?;
                out.push(u8::from_str_radix(hex, 16).ok()?);
                i += 3;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8(out).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
    <html><body><form>
    <a rel="nofollow" href="https://docs.rs/serde/" class="result-link">serde - <b>Rust</b></a>
    <td class="result-snippet">A <b>serialization</b> framework for Rust.</td>
    <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fx">Example</a>
    <td class="result-snippet">Second hit.</td>
    </form></body></html>"#;

    #[test]
    fn parses_links_and_snippets() {
        let hits = parse(SAMPLE, 10);
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert_eq!(hits[0].title, "serde - Rust");
        assert_eq!(hits[0].url, "https://docs.rs/serde/");
        assert_eq!(hits[0].snippet, "A serialization framework for Rust.");
        assert_eq!(hits[1].url, "https://example.com/x");
    }

    #[test]
    fn respects_limit() {
        assert_eq!(parse(SAMPLE, 1).len(), 1);
    }

    #[test]
    fn encodes_query() {
        assert_eq!(percent_encode("rust ownership"), "rust+ownership");
    }
}
