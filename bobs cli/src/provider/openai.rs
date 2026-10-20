//! OpenAI-compatible `/chat/completions`, streamed, with `function` tools.
//! Works against OpenAI, OpenRouter, xAI, Gemini (OpenAI surface), vLLM,
//! llama.cpp server, LM Studio, Ollama, …

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
    debug_assert_eq!(c.kind, Kind::OpenAi);

    let mut wire: Vec<Value> = Vec::new();
    for m in msgs {
        match m {
            Msg::System(s) => wire.push(json!({"role": "system", "content": s})),
            Msg::User(s) => wire.push(json!({"role": "user", "content": s})),
            Msg::Assistant { text, calls } => {
                let mut msg = json!({"role": "assistant"});
                msg["content"] = match text {
                    Some(t) if !t.is_empty() => json!(t),
                    _ => Value::Null,
                };
                if !calls.is_empty() {
                    msg["tool_calls"] = Value::Array(
                        calls
                            .iter()
                            .map(|call| {
                                json!({
                                    "id": call.id,
                                    "type": "function",
                                    "function": {
                                        "name": call.name,
                                        "arguments": call.args.to_string(),
                                    }
                                })
                            })
                            .collect(),
                    );
                }
                wire.push(msg);
            }
            Msg::Tool { id, output, .. } => {
                wire.push(json!({"role": "tool", "tool_call_id": id, "content": output}));
            }
        }
    }

    let mut body = json!({
        "model": c.model,
        "messages": wire,
        "stream": true,
    });
    if let Some(t) = c.temperature {
        body["temperature"] = json!(t);
    }
    if c.max_tokens > 0 {
        body["max_tokens"] = json!(c.max_tokens);
    }
    if !tools.is_empty() {
        body["tools"] = Value::Array(
            tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect(),
        );
    }

    let resp = c
        .http
        .post(format!("{}/chat/completions", c.base_url.trim_end_matches('/')))
        .header("authorization", format!("Bearer {}", c.api_key))
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .context("could not reach the API endpoint")?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        bail!(api_error(status.as_u16(), &text, &c.base_url));
    }

    // index -> (id, name, arguments-so-far)
    let mut calls: std::collections::HashMap<usize, (String, String, String)> = Default::default();
    let mut step = Step::default();
    let mut lines = LineBuffer::new();
    let mut stream = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("stream was interrupted")?;
        for line in lines.push(&chunk) {
            let Some(data) = sse_data(&line) else { continue };
            if data == "[DONE]" {
                break;
            }
            let Ok(v) = serde_json::from_str::<Value>(data) else {
                continue;
            };

            if let Some(usage) = v.get("usage") {
                step.usage.input_tokens = usage.get("prompt_tokens").and_then(|t| t.as_u64());
                step.usage.output_tokens = usage.get("completion_tokens").and_then(|t| t.as_u64());
            }

            let Some(choice) = v.pointer("/choices/0") else {
                continue;
            };
            let Some(delta) = choice.get("delta") else {
                continue;
            };

            if let Some(text) = delta.get("content").and_then(|t| t.as_str()) {
                if !text.is_empty() {
                    step.text.push_str(text);
                    sink(Event::Text(text.to_string()));
                }
            }

            for call in delta.get("tool_calls").and_then(|c| c.as_array()).into_iter().flatten() {
                let idx = call.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                let entry = calls.entry(idx).or_default();
                if let Some(id) = call.get("id").and_then(|i| i.as_str()) {
                    entry.0 = id.to_string();
                }
                if let Some(name) = call.pointer("/function/name").and_then(|n| n.as_str()) {
                    entry.1 = name.to_string();
                }
                if let Some(args) = call.pointer("/function/arguments").and_then(|a| a.as_str()) {
                    entry.2.push_str(args);
                }
            }
        }
    }

    let mut finished: Vec<(usize, (String, String, String))> = calls.into_iter().collect();
    finished.sort_by_key(|(idx, _)| *idx);
    for (position, (idx, (id, name, args))) in finished.into_iter().enumerate() {
        let id = if id.is_empty() {
            format!("call_{position}_{idx}")
        } else {
            id
        };
        let parsed = serde_json::from_str(&args).unwrap_or_else(|_| {
            // Some servers stream arguments that never quite close; keep the raw
            // string so the model still sees what it asked for.
            json!({"_raw": args})
        });
        sink(Event::ToolStart {
            name: name.clone(),
            args: parsed.clone(),
        });
        step.calls.push(ToolCall {
            id,
            name,
            args: parsed,
        });
    }

    Ok(step)
}
