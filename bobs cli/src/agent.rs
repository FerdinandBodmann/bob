//! The agent loop: model -> tool calls -> tool results -> model, until the
//! model answers without asking for anything.

use anyhow::{anyhow, Result};

use crate::provider::{Client, Event, Msg, Sink, ToolDef};
use crate::tools::{self, Ctx};

pub struct Agent {
    pub client: Client,
    pub ctx: Ctx,
    pub tools: Vec<ToolDef>,
    pub system: String,
    pub max_steps: usize,
}

impl Agent {
    pub fn new(client: Client, ctx: Ctx, system: String, no_tools: bool, max_steps: usize) -> Self {
        Self {
            client,
            ctx,
            system,
            tools: if no_tools { Vec::new() } else { tools::defs() },
            max_steps,
        }
    }

    /// Build an agent (HTTP client + tools) from resolved configuration.
    pub fn from_config(cfg: &crate::config::Config) -> Result<Self> {
        let http = reqwest::Client::builder()
            .user_agent(tools::USER_AGENT)
            .timeout(std::time::Duration::from_secs(60))
            .build()?;
        let client = Client {
            kind: cfg.kind,
            api_key: cfg.api_key.clone(),
            base_url: cfg.base_url.clone(),
            model: cfg.model.clone(),
            max_tokens: cfg.max_tokens,
            temperature: cfg.temperature,
            http: http.clone(),
        };
        let ctx = Ctx {
            http,
            fetch_limit: cfg.fetch_limit,
        };
        Ok(Self::new(
            client,
            ctx,
            cfg.system.clone(),
            cfg.no_tools,
            cfg.max_steps,
        ))
    }

    /// Start a new turn with `prompt`.
    pub async fn run_turn(
        &self,
        msgs: &mut Vec<Msg>,
        prompt: &str,
        sink: Sink<'_>,
    ) -> Result<String> {
        ensure_system(msgs, &self.system);
        msgs.push(Msg::User(prompt.to_string()));
        self.drive(msgs, sink).await
    }

    /// Continue an existing turn (used after an interruption).
    pub async fn drive(&self, msgs: &mut Vec<Msg>, sink: Sink<'_>) -> Result<String> {
        for _ in 0..self.max_steps {
            let step = self
                .client
                .complete(msgs, &self.tools, &mut *sink)
                .await?;

            let text = step.text.trim().to_string();
            if step.calls.is_empty() {
                if !text.is_empty() {
                    msgs.push(Msg::Assistant {
                        text: Some(text.clone()),
                        calls: Vec::new(),
                    });
                }
                return Ok(text);
            }

            msgs.push(Msg::Assistant {
                text: (!text.is_empty()).then(|| text.clone()),
                calls: step.calls.clone(),
            });

            for call in &step.calls {
                sink(Event::ToolStart {
                    name: call.name.clone(),
                    args: call.args.clone(),
                });
                let output = tools::run(&call.name, &call.args, &self.ctx).await;
                msgs.push(Msg::Tool {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    output,
                });
            }
        }

        Err(anyhow!(
            "stopped after {} tool rounds without a final answer — raise --max-steps",
            self.max_steps
        ))
    }
}

fn ensure_system(msgs: &mut Vec<Msg>, system: &str) {
    match msgs.first() {
        Some(Msg::System(_)) => {}
        _ => msgs.insert(0, Msg::System(system.to_string())),
    }
}
