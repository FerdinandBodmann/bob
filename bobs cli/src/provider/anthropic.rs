//! Anthropic Messages API, streamed, with native tool use.
//! Docs: https://docs.anthropic.com/en/api/messages-streaming

use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use serde_json::{json, Value};

use super::{api_error, sse_data, Client, Event, Kind, LineBuffer, Msg, Sink, Step, ToolCall, ToolDef};

pub async fn complete(
    c: &Client,
    msgs: &[Msg],
    tools: &[ToolDef],
    sink: Sink<'_>,
) -> Result<Step> {
    debug_assert_eq!(c.kind, Kind::Anthropic);

    let mut system = String::new();
    let mut wire: Vec<Value> = Vec::new();

    for m in msgs {
        match m {
            Msg::System(s) => {
                if !system.is_empty() {
                    system.push_str("\n\n");
                }
                system.push_str(s);
            }
            Msg::User(s) => wire.push(json!({"role": "user", "content": [text_block(s)]})),
            Msg::Assistant { text, calls } => {
                let mut content: Vec<Value> = Vec::new();
                if let Some(t) = text.as_ref().filter(|t| !t.is_empty()) {
                    content.push(text_block(t));
                }
                for call in calls {
                    content.push(json!({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.args,
                    }));
                }
                if content.is_empty() {
                    content.push(text_block(""));
                }
                wire.push(json!({"role": "assistant", "content": content}));
            }
            Msg::Tool { id, output, .. } => wire.push(json!({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": id, "content": output}],
            })),
        }
    }

    let mut body = json!({
        "model": c.model,
        "max_tokens": c.max_tokens,
        "stream": true,
        "messages": wire,
    });
    if !system.is_empty() {
        body["system"] = json!(system);
    }
    if let Some(t) = c.temperature {
        body["temperature"] = json!(t);
    }
    if !tools.is_empty() {
        body["tools"] = json!(
            tools
                .iter()
                .map(|t| json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }))
                .collect::<Vec<_>>()
        );
    }

    let resp = c
        .http
        .post(format!("{}/messages", c.base_url.trim_end_matches('/')))
        .header("x-api-key", &c.api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .context("could not reach the Anthropic API")?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        bail!(api_error(status.as_u16(), &text, "Anthropic"));
    }

    // index -> in-flight content block
    #[derive(Default)]
    enum Block {
        #[default]
        Text,
        Tool { id: String, name: String, json: String },
    }

    let mut step = Step::default();
    let mut blocks: std::collections::HashMap<usize, Block> = Default::default();
    let mut lines = LineBuffer::new();
    let mut stream = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Anthropic stream was interrupted")?;
        for line in lines.push(&chunk) {
            let Some(data) = sse_data(&line) else { continue };
            let Ok(v) = serde_json::from_str::<Value>(data) else {
                continue;
            };

            match v.get("type").and_then(|t| t.as_str()) {
                Some("content_block_start") => {
                    let idx = v.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                    let block = v.get("content_block").unwrap_or(&Value::Null);
                    let kind = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                    blocks.insert(
                        idx,
                        match kind {
                            "tool_use" => Block::Tool {
                                id: block.get("id").and_then(|i| i.as_str()).unwrap_or("").to_string(),
                                name: block.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string(),
                                json: String::new(),
                            },
                            _ => Block::Text,
                        },
                    );
                }
                Some("content_block_delta") => {
                    let idx = v.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                    let delta = v.get("delta").unwrap_or(&Value::Null);
                    match delta.get("type").and_then(|t| t.as_str()) {
                        Some("text_delta") => {
                            if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                                step.text.push_str(text);
                                sink(Event::Text(text.to_string()));
                            }
                        }
                        Some("input_json_delta") => {
                            if let Some(fragment) =
                                delta.get("partial_json").and_then(|p| p.as_str())
                            {
                                if let Some(Block::Tool { json, .. }) = blocks.get_mut(&idx) {
                                    json.push_str(fragment);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Some("content_block_stop") => {
                    let idx = v.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                    if let Some(Block::Tool { id, name, json }) = blocks.remove(&idx) {
                        let args = serde_json::from_str(&json)
                            .unwrap_or_else(|_| serde_json::json!({}));
                        sink(Event::ToolStart {
                            name: name.clone(),
                            args: args.clone(),
                        });
                        step.calls.push(ToolCall { id, name, args });
                    }
                }
                Some("message_start") => {
                    if let Some(input) = v
                        .pointer("/message/usage/input_tokens")
                        .and_then(|t| t.as_u64())
                    {
                        step.usage.input_tokens = Some(input);
                    }
                }
                Some("message_delta") => {
                    if let Some(output) = v.pointer("/usage/output_tokens").and_then(|t| t.as_u64())
                    {
                        step.usage.output_tokens = Some(output);
                    }
                }
                Some("error") => {
                    let message = v
                        .pointer("/error/message")
                        .and_then(|m| m.as_str())
                        .unwrap_or("unknown streaming error");
                    bail!("Anthropic stream error: {message}");
                }
                _ => {}
            }
        }
    }

    Ok(step)
}

fn text_block(s: &str) -> Value {
    json!({"type": "text", "text": s})
}
