mod agent;
mod cli;
mod config;
mod provider;
mod repl;
mod tools;
mod ui;

use std::io::IsTerminal;

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::Value;

use crate::agent::Agent;
use crate::cli::{Cli, Command};
use crate::config::Config;
use crate::provider::Msg;

fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = Config::resolve(&cli.global)?;
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("could not start the async runtime")?;

    match cli.command.unwrap_or(Command::Chat { prompt: Vec::new() }) {
        Command::Chat { prompt } | Command::Ask { prompt } => {
            let agent = Agent::from_config(&cfg)?;
            let prompt = if prompt.is_empty() {
                read_stdin_prompt()?
            } else {
                prompt.join(" ")
            };
            if prompt.trim().is_empty() {
                // Nothing piped and nothing typed: this is a REPL session.
                return repl::run(&rt, &cfg);
            }
            let mut msgs = Vec::new();
            oneshot(&rt, &agent, &mut msgs, prompt.trim())
        }

        Command::Fetch { url, limit } => {
            let agent = Agent::from_config(&cfg)?;
            let mut ctx = agent.ctx.clone();
            if let Some(limit) = limit {
                ctx.fetch_limit = limit;
            }
            let args = serde_json::json!({ "url": url });
            let out = rt.block_on(tools::run("fetch", &args, &ctx));
            println!("{out}");
            Ok(())
        }

        Command::Search { query, limit } => {
            let agent = Agent::from_config(&cfg)?;
            let args = serde_json::json!({ "query": query.join(" "), "limit": limit });
            let out = rt.block_on(tools::run("search", &args, &agent.ctx));
            println!("{out}");
            Ok(())
        }

        Command::Tools => {
            println!("{}", ui::bold("tools the model can call\n"));
            for def in tools::defs() {
                println!("  {} {}", ui::cyan(&def.name), def.description);
                println!(
                    "  {} {}\n",
                    ui::dim("schema"),
                    serde_json::to_string_pretty(&def.parameters).unwrap_or_default()
                );
            }
            Ok(())
        }

        Command::Doctor => doctor(&rt, &cfg),
    }
}

fn oneshot(
    rt: &tokio::runtime::Runtime,
    agent: &Agent,
    msgs: &mut Vec<Msg>,
    prompt: &str,
) -> Result<()> {
    let mut sink = |event: provider::Event| match event {
        provider::Event::Text(chunk) => {
            print!("{chunk}");
            ui::flush();
        }
        provider::Event::ToolStart { name, args } => {
            println!();
            println!("  {} {}", ui::dim("→"), ui::yellow(&tools::summarize(&name, &args)));
        }
    };
    let answer = rt.block_on(agent.run_turn(msgs, prompt, &mut sink))?;
    if !answer.is_empty() {
        println!();
    }
    Ok(())
}

fn read_stdin_prompt() -> Result<String> {
    use std::io::Read;
    if std::io::stdin().is_terminal() {
        return Ok(String::new());
    }
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf)?;
    Ok(buf)
}

fn doctor(rt: &tokio::runtime::Runtime, cfg: &Config) -> Result<()> {
    let key = &cfg.api_key;
    let masked = if key.is_empty() {
        ui::yellow("(none)").to_string()
    } else {
        let head: String = key.chars().take(6).collect();
        let tail: String = key.chars().rev().take(4).collect::<String>().chars().rev().collect();
        format!("{head}…{tail} ({} chars)", key.chars().count())
    };

    println!("{}", ui::bold("bobs doctor\n"));
    println!("  provider     {}", ui::green(&cfg.provider));
    println!("  base url     {}", cfg.base_url);
    println!("  model        {}", cfg.model);
    println!("  api key      {}", masked);
    println!(
        "  tools        {}",
        if cfg.no_tools { "disabled" } else { "fetch, search" }
    );
    println!("  max tokens   {}", cfg.max_tokens);
    println!("  max steps    {}", cfg.max_steps);
    println!("  fetch limit  {} chars", cfg.fetch_limit);
    println!(
        "  history      {}",
        if std::io::stdin().is_terminal() {
            "~/.local/share/bobs/history.txt"
        } else {
            "(non-interactive)"
        }
    );

    println!("\n{}", ui::bold("connectivity"));
    for (label, url) in [
        ("internet", "https://lite.duckduckgo.com/lite/"),
        ("api      ", cfg.base_url.as_str()),
    ] {
        let probe = rt.block_on(async {
            let client = reqwest::Client::builder()
                .user_agent(tools::USER_AGENT)
                .timeout(std::time::Duration::from_secs(10))
                .build()?;
            let resp = client.get(url).send().await?;
            Ok::<_, anyhow::Error>(resp.status().as_u16())
        });
        match probe {
            Ok(status) if status < 500 => {
                println!("  {label} {url} → {}", ui::green(&format!("{status}")))
            }
            Ok(status) => println!("  {label} {url} → {}", ui::yellow(&format!("{status}"))),
            Err(e) => println!("  {label} {url} → {} {e:#}", ui::red("unreachable")),
        }
    }

    // A local model endpoint answers 404/401 without a key; that still proves
    // the process is there. Anything else worth saying:
    if !std::env::var_os("ANTHROPIC_API_KEY").is_some() && cfg.provider == "anthropic" && opencode_auth_has("anthropic") {
        println!(
            "\n  {}",
            ui::dim("note: ANTHROPIC_API_KEY not exported; bobs is reusing the key stored by OpenCode")
        );
    }
    let _: Option<Value> = None;
    Ok(())
}

fn opencode_auth_has(provider: &str) -> bool {
    let Some(home) = std::env::var_os("HOME") else {
        return false;
    };
    let mut p = std::path::PathBuf::from(home);
    p.push(".local/share/opencode/auth.json");
    std::fs::read_to_string(p)
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| v.get(provider).cloned())
        .is_some()
}
