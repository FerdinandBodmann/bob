//! The REPL. Deliberately synchronous (rustyline) with the async runtime used
//! per turn via `block_on` — that keeps Ctrl-C, history and line editing simple.

use std::io::{IsTerminal, Read};

use anyhow::{Context, Result};
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use serde_json::Value;

use crate::agent::Agent;
use crate::config::Config;
use crate::provider::{Event, Msg};
use crate::tools::{self, Ctx};
use crate::ui;

const HELP: &str = "\
commands
  /help              this list
  /tools             show model tools
  /sys <text>        replace the system prompt
  /reset             forget the conversation
  /fetch <url>       fetch a page directly
  /search <query>    web search directly
  /exit, /quit, ^D   leave
anything else       is sent to the model";

pub fn run(rt: &tokio::runtime::Runtime, cfg: &Config) -> Result<()> {
    let agent = Agent::from_config(cfg)?;
    let mut msgs: Vec<Msg> = Vec::new();

    let mut rl = DefaultEditor::new().context("could not start line editor")?;
    let history = history_path();
    if let Some(parent) = history.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = rl.load_history(&history);

    println!(
        "{} {} · {} · tools: {}",
        ui::bold("bobs"),
        ui::dim(&cfg.provider),
        ui::dim(&cfg.model),
        if cfg.no_tools {
            ui::yellow("off").to_string()
        } else {
            ui::green("fetch, search").to_string()
        }
    );
    println!("{}", ui::dim("type /help for commands\n"));

    // Non-interactive use: `echo "question" | bobs chat`
    if !std::io::stdin().is_terminal() {
        let mut piped = String::new();
        std::io::stdin().read_to_string(&mut piped)?;
        if piped.trim().is_empty() {
            return Ok(());
        }
        return ask(rt, &agent, &mut msgs, piped.trim());
    }

    loop {
        let prompt = format!("{} ", ui::cyan("›"));
        match rl.readline(&prompt) {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let _ = rl.add_history_entry(line);

                if let Some(rest) = line.strip_prefix('/') {
                    if handle_slash(rt, &agent, &mut msgs, rest, cfg)? {
                        break;
                    }
                    continue;
                }
                ask(rt, &agent, &mut msgs, line)?;
            }
            Err(ReadlineError::Interrupted) => {
                println!("{}", ui::dim("^C"));
            }
            Err(ReadlineError::Eof) => {
                println!();
                break;
            }
            Err(e) => return Err(e.into()),
        }
    }

    let _ = rl.save_history(&history);
    Ok(())
}

fn ask(rt: &tokio::runtime::Runtime, agent: &Agent, msgs: &mut Vec<Msg>, prompt: &str) -> Result<()> {
    let started = std::time::Instant::now();
    let mut saw_text = false;

    let result = rt.block_on(async {
        let mut sink = |event: Event| match event {
            Event::Text(chunk) => {
                if !saw_text {
                    ui::clear_line();
                    saw_text = true;
                }
                print!("{chunk}");
                ui::flush();
            }
            Event::ToolStart { name, args } => {
                if saw_text {
                    println!();
                    saw_text = false;
                }
                let summary = tools::summarize(&name, &args);
                println!("  {} {}", ui::dim("→"), ui::yellow(&summary));
            }
        };
        agent.run_turn(msgs, prompt, &mut sink).await
    });

    match result {
        Ok(text) => {
            if saw_text {
                println!();
            } else if !text.is_empty() {
                println!("{text}");
            } else {
                println!("{}", ui::dim("(no answer — try rephrasing)"));
            }
            if !text.is_empty() {
                println!(
                    "{}",
                    ui::dim(&format!(
                        "· {}s · {}",
                        started.elapsed().as_secs(),
                        agent.client.model
                    ))
                );
            }
            Ok(())
        }
        Err(e) => {
            if saw_text {
                println!();
            }
            eprintln!("{} {e:#}", ui::red("error:"));
            Ok(()) // keep the session alive; the transcript is still usable
        }
    }
}

/// Returns `true` when the session should end.
fn handle_slash(
    rt: &tokio::runtime::Runtime,
    agent: &Agent,
    msgs: &mut Vec<Msg>,
    rest: &str,
    cfg: &Config,
) -> Result<bool> {
    let (cmd, arg) = match rest.split_once(' ') {
        Some((c, a)) => (c, a.trim()),
        None => (rest, ""),
    };

    match cmd {
        "help" | "h" | "?" => println!("{HELP}"),
        "exit" | "quit" | "q" => return Ok(true),
        "reset" | "clear" => {
            msgs.clear();
            println!("{}", ui::dim("conversation cleared"));
        }
        "tools" => {
            println!("{}", ui::bold("tools"));
            for def in &agent.tools {
                println!("  {} {}", ui::cyan(&def.name), ui::dim(&def.description));
            }
            if agent.tools.is_empty() {
                println!("  {}", ui::yellow("none (started with --no-tools)"));
            }
        }
        "sys" | "system" => {
            if arg.is_empty() {
                println!("{}", ui::dim("usage: /sys <text>"));
            } else {
                msgs.insert(0, Msg::System(arg.to_string()));
                println!("{}", ui::dim("system prompt replaced"));
            }
        }
        "fetch" => {
            if arg.is_empty() {
                println!("{}", ui::dim("usage: /fetch <url>"));
            } else {
                let ctx = Ctx {
                    http: agent.ctx.http.clone(),
                    fetch_limit: cfg.fetch_limit,
                };
                let out = rt.block_on(tools::run("fetch", &serde_json::json!({"url": arg}), &ctx));
                println!("{out}");
            }
        }
        "search" => {
            if arg.is_empty() {
                println!("{}", ui::dim("usage: /search <query>"));
            } else {
                let args: Value = serde_json::json!({"query": arg, "limit": 8});
                let out = rt.block_on(tools::run("search", &args, &agent.ctx));
                println!("{out}");
            }
        }
        other => println!("{} {other} — try /help", ui::yellow("unknown command:")),
    }
    Ok(false)
}

fn history_path() -> std::path::PathBuf {
    let home = std::env::var_os("HOME").unwrap_or_else(|| "/tmp".into());
    let mut p = std::path::PathBuf::from(home);
    p.push(".local/share/bobs/history.txt");
    p
}
