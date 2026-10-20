//! `fetch` tool: GET a URL, return readable text.

use anyhow::{bail, Context, Result};
use reqwest::header::{ACCEPT, CONTENT_TYPE, USER_AGENT as UA_HEADER};
use serde_json::Value;

use super::{htmltext, Ctx, USER_AGENT};

/// Refuse to buffer absurd bodies into memory.
const MAX_BODY_BYTES: u64 = 8 * 1024 * 1024;

pub async fn run(args: &Value, ctx: &Ctx) -> Result<String> {
    let url = args
        .get("url")
        .and_then(|v| v.as_str())
        .context(r#"missing required argument "url""#)?
        .trim()
        .to_string();

    let parsed = reqwest::Url::parse(&url).context("invalid URL — must be absolute, e.g. https://example.com/path")?;
    if !matches!(parsed.scheme(), "http" | "https") {
        bail!("only http/https URLs can be fetched, got '{}'", parsed.scheme());
    }

    let limit = args
        .get("max_chars")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(ctx.fetch_limit)
        .clamp(200, 200_000);

    let resp = ctx
        .http
        .get(parsed.as_str())
        .header(UA_HEADER, USER_AGENT)
        .header(ACCEPT, "text/html,application/xhtml+xml,application/json,text/plain;q=0.9,*/*;q=0.8")
        .send()
        .await
        .with_context(|| format!("request to {parsed} failed"))?;

    let status = resp.status();
    let final_url = resp.url().to_string();
    let content_type = resp
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();

    if let Some(len) = resp.content_length() {
        if len > MAX_BODY_BYTES {
            bail!("body is {len} bytes, refusing to buffer more than {MAX_BODY_BYTES}");
        }
    }

    let body = resp.text().await.context("could not read response body")?;

    let is_html = content_type.contains("text/html") || content_type.contains("application/xhtml");
    let (text, kind) = if is_html {
        let mut text = htmltext::to_text(&body);
        if let Some(title) = htmltext::title(&body) {
            text = format!("# {title}\n\n{text}");
        }
        (text, "html")
    } else {
        (body, if content_type.contains("json") { "json" } else { "text" })
    };

    let total = text.chars().count();
    let (text, truncated) = if total > limit {
        (text.chars().take(limit).collect::<String>(), true)
    } else {
        (text, false)
    };

    let mut out = format!("[{status} {kind}] {final_url}\n\n{text}");
    if truncated {
        out.push_str(&format!("\n\n…[truncated at {limit} of {total} chars — refetch with max_chars]"));
    }
    Ok(out)
}
