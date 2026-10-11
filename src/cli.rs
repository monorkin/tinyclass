//! The command line: what `tinyclass` understands and what each command runs.

use anyhow::{Result, bail};
use llama_cpp_2::LlamaBackendDeviceType;
use std::io::{self, BufRead, Write};
use usage::{Cli, Subcommands};

use crate::completions;
use crate::config::{Config, Backend};
use crate::decision::{self, Decision};
use crate::device::{self, Choice};
use crate::model::{self, Loaded, Model};
use crate::paths;
use crate::upgrade;

/// A local decision model out of Qwen3: pick one of N choices with a probability for each
#[derive(Cli)]
#[usage(bin = "tinyclass", version, unknown_flags = "error", completion)]
pub struct Cli {
    #[usage(subcommand)]
    pub command: Command,
}

#[derive(Subcommands)]
pub enum Command {
    /// Choose, pull, and list models
    Model {
        #[usage(subcommand)]
        command: ModelCommand,
    },
    /// Configure the ollama server URL and list/resolve remote models
    Host {
        #[usage(subcommand)]
        command: HostCommand,
    },
    /// Switch between llama-cpp and ollama backends
    Backend {
        #[usage(subcommand)]
        command: BackendCommand,
    },
    /// Choose and list the CPU and GPUs the model can run on
    Device {
        #[usage(subcommand)]
        command: DeviceCommand,
    },
    /// Decide once: classify an input into one of the choices
    Decide {
        /// The text to decide about
        input: String,
        /// The choices, at least two
        choices: Vec<String>,
        /// What the model is told to do with the input
        #[usage(long, default = "Choose one option.")]
        instruction: String,
        /// Where to run: auto, cpu, gpu, or gpu:N; defaults to the set device
        #[usage(long)]
        device: Option<String>,
        /// Print the decision as JSON
        #[usage(long)]
        json: bool,
        /// Per-choice criteria as `name:description` pairs (ollama only)
        #[usage(long)]
        criteria: Vec<String>,
    },
    /// Ask a yes/no question: how likely the statement is true, from 0 to 1
    Noul {
        /// The statement to judge
        statement: String,
        /// What the model is told to do with the statement
        #[usage(long, default = "Is the statement true?")]
        instruction: String,
        /// Where to run: auto, cpu, gpu, or gpu:N; defaults to the set device
        #[usage(long)]
        device: Option<String>,
        /// Print the probability as JSON
        #[usage(long)]
        json: bool,
    },
    /// Rate a statement on a numeric scale defined by labeled criteria
    Score {
        /// The statement or context to rate
        statement: String,
        /// What the model is told to do with the statement
        #[usage(long, default = "How would you rate this statement?")]
        instruction: String,
        /// Print the rating and breakdown as JSON
        #[usage(long)]
        json: bool,
    },
    /// Send multiple question types in a single request
    Batch {
        /// The context shared by all questions
        state: String,
        /// Pipe-separated choices, e.g. `billing|technical`
        #[usage(long)]
        choice: Option<String>,
        /// Pipe-separated criteria pairs, `key:desc|key:desc`
        #[usage(long)]
        criteria: Option<String>,
        /// A noul (yes/no judgment) question
        #[usage(long)]
        noul: Option<String>,
        /// A score (numeric rating) question
        #[usage(long)]
        score: Option<String>,
        /// Print answers as JSON
        #[usage(long)]
        json: bool,
    },
    /// Keep the model loaded and decide about every line typed on stdin
    Play {
        /// The choices, at least two
        choices: Vec<String>,
        /// What the model is told to do with each line
        #[usage(long, default = "Choose one option.")]
        instruction: String,
        /// Where to run: auto, cpu, gpu, or gpu:N; defaults to the set device
        #[usage(long)]
        device: Option<String>,
    },
    /// Upgrade a mise install to the latest release
    Upgrade {
        /// Target a specific release instead of the latest
        version: Option<String>,
    },
    /// Print or install the shell completion script
    ShellCompletion {
        #[usage(subcommand)]
        command: ShellCompletionCommand,
    },
}

#[derive(Subcommands)]
pub enum ModelCommand {
    /// List the models tinyclass knows, marking the set one and the pulled ones
    List,
    /// Make a model the one that answers
    Set {
        /// A name from `tinyclass model list`
        name: String,
    },
    /// Fetch the set model, or a named one, from Hugging Face
    Pull {
        /// A name from `tinyclass model list`; defaults to the set model
        name: Option<String>,
    },
    /// Switch between llama-cpp and ollama backends
    Backend {
        #[usage(subcommand)]
        command: BackendCommand,
    },
}

#[derive(Subcommands)]
pub enum HostCommand {
    /// Configure the Ollama API URL
    Set {
        /// the Ollama API URL
        url: String,
    },
    /// Remove the configured URL, reverting to the default
    Clear,
    /// Query the server and list available models
    Models,
    /// Pull a model onto the server
    Pull {
        /// Name of the model to pull
        name: Option<String>,
    },
}

#[derive(Subcommands)]
pub enum BackendCommand {
    /// Set the current backend
    Set {
        /// Which backend to use: llama-cpp or ollama
        #[usage(choices("llama-cpp", "ollama"))]
        backend: String,
    },
    /// Show the current backend
    Get,
}

#[derive(Subcommands)]
pub enum DeviceCommand {
    /// List the devices llama-cpp sees, with the index `gpu:N` refers to
    List,
    /// Make a device the one the model runs on: auto, cpu, gpu, or gpu:N
    Set {
        /// auto, cpu, gpu, or gpu:N from `tinyclass device list`
        device: String,
    },
}

#[derive(Subcommands)]
pub enum ShellCompletionCommand {
    /// Write the completion script to stdout
    Print {
        /// The shell to generate for
        #[usage(choices("bash", "elvish", "zsh", "fish", "nu", "powershell"))]
        shell: String,
    },
    /// Write the completion script where the shell looks for it
    Install {
        /// The shell to install for
        #[usage(choices("bash", "elvish", "zsh", "fish", "nu", "powershell"))]
        shell: String,
    },
}

pub fn main() {
    if let Err(error) = run(Cli::parse()) {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Model { command } => match command {
            ModelCommand::List => list_models(),
            ModelCommand::Set { name } => set_model(&name),
            ModelCommand::Pull { name } => pull_model(name.as_deref()),
            ModelCommand::Backend { command } => match command {
                BackendCommand::Set { backend } => set_backend(&backend),
                BackendCommand::Get => get_backend(),
            },
        },
        Command::Host { command } => match command {
            HostCommand::Set { url } => set_host(&url),
            HostCommand::Clear => clear_host(),
            HostCommand::Models => host_models(),
            HostCommand::Pull { name } => host_pull(name.as_deref()),
        },
        Command::Backend { command } => match command {
            BackendCommand::Set { backend } => set_backend(&backend),
            BackendCommand::Get => get_backend(),
        },
        Command::Device { command } => match command {
            DeviceCommand::List => list_devices(),
            DeviceCommand::Set { device } => set_device(&device),
        },
        Command::Decide { input, choices, instruction, device, json, criteria } => {
            decide(&input, &choices, &instruction, device.as_deref(), json, &criteria)
        }
        Command::Noul { statement, instruction, device, json } => {
            noul(&statement, &instruction, device.as_deref(), json)
        }
        Command::Score { statement, instruction, json } => {
            score(&statement, &instruction, json)
        }
        Command::Batch { state, choice, criteria, noul, score, json } => {
            batch(&state, choice.as_deref(), criteria.as_deref(), noul.as_deref(), score.as_deref(), json)
        }
        Command::Play { choices, instruction, device } => {
            play(&choices, &instruction, device.as_deref())
        }
        Command::Upgrade { version } => upgrade::run(version.as_deref()),
        Command::ShellCompletion { command } => match command {
            ShellCompletionCommand::Print { shell } => completions::print(&shell),
            ShellCompletionCommand::Install { shell } => completions::install(&shell),
        },
    }
}

fn list_models() -> Result<()> {
    let config = Config::load().ok();

    // Local GGUF models
    let current: Option<&str> = config.as_ref().and_then(|c| c.model.as_deref());
    println!("llama-cpp backend");
    for model in model::CATALOG {
        let mut marker = " ";
        if current == Some(model.name) {
            marker = "*";
        }
        let state = if model.available() { "pulled" } else { "not pulled" };
        println!("{marker} {:<12} {:>7}  {state}", model.name, model.size);
    }

    // Remote ollama models
    if let Some(url) = config.as_ref().and_then(|c| c.host.as_deref()) {
        device::check_server(url).ok();
        println!("ollama backend");
        let remote = model::list_remote_models(url).ok();
        if let Some(models) = remote {
            for model in &models {
                let mut marker = " ";
                if current == Some(model.name.as_str()) {
                    marker = "*";
                }
                println!("{marker} {}", model.name);
            }
        }
    }

    Ok(())
}

fn set_model(name: &str) -> Result<()> {
    let mut config = Config::load()?;

    // Check if it's a local catalog model
    if let Ok(model) = Model::find(name) {
        config.model = Some(model.name.to_string());
        config.save()?;
        if model.available() {
            println!("{} is set.", model.name);
        } else {
            println!("{} is set — fetch it with `{} model pull`.", model.name, paths::invoked_as());
        }
        return Ok(());
    }

    // Check if it's a remote model
    if let Some(url) = Config::load().ok().and_then(|c| c.host).as_ref() {
        let url = url.as_str();
        device::check_server(url)?;
        let remote = model::list_remote_models(url).ok();
        if let Some(models) = remote
            && models.iter().any(|m| m.name == name)
        {
            config.model = Some(name.to_string());
            config.save()?;
            println!("{} is set.", name);
            return Ok(());
        }
    }

    bail!("no model named '{}' — try `{} model list` to see what's available", name, paths::invoked_as());
}

fn pull_model(name: Option<&str>) -> Result<()> {
    let given = name.unwrap_or("");

    // Check if it's a local CATALOG model
    if let Some(model) = Model::find(given).ok().or_else(|| Model::current().ok().filter(|_| name.is_none())) {
        if model.available() {
            println!("{} is already pulled.", model.name);
        } else {
            model.pull()?;
            println!("{} is ready.", model.name);
        }
        return Ok(());
    }

    // Fall through to remote pull
    if given.is_empty() {
        bail!("no model name given and none configured — run `{} model list` first", paths::invoked_as());
    }
    host_pull(Some(given))
}

fn set_backend(target: &str) -> Result<()> {
    let mut config = Config::load()?;
    let value = if target == "ollama" { "ollama" } else { "llama-cpp" };
    config.backend = Some(value.to_string());
    config.save()?;
    println!("Backend set to {}.", value);
    Ok(())
}

fn get_backend() -> Result<()> {
    let config = Config::load()?;
    let backend = Backend::current(&config);
    match backend {
        Backend::Local => println!("llama-cpp"),
        Backend::Remote => println!("ollama"),
    }
    Ok(())
}

fn set_host(url: &str) -> Result<()> {
    let parsed = url.parse::<url::Url>()
        .map_err(|_| anyhow::anyhow!("invalid URL: {}", url))?;

    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        bail!("URL scheme must be http or https");
    }

    let mut config = Config::load()?;
    config.host = Some(url.to_string());
    config.save()?;
    println!("Ollama server set to {}.", url);
    Ok(())
}

fn clear_host() -> Result<()> {
    let mut config = Config::load()?;
    config.host = None;
    config.save()?;
    println!("Ollama server URL cleared — will use http://localhost:11434.");
    Ok(())
}

fn host_models() -> Result<()> {
    let url = model::host();
    eprintln!("Checking server at {}…", url);
    device::check_server(&url)?;
    let models = model::list_remote_models(&url)?;
    if models.is_empty() {
        eprintln!("No models on the server. Use `{} host pull <name>` to add one.", paths::invoked_as());
    } else {
        for model in &models {
            println!("  {}", model.name);
        }
    }
    Ok(())
}

fn host_pull(name: Option<&str>) -> Result<()> {
    let url = model::host();
    device::check_server(&url)?;
    let model_name = match name {
        Some(n) => n.to_string(),
        None => Config::load().ok().and_then(|it| it.model).unwrap_or_default(),
    };
    if model_name.is_empty() {
        bail!("no model name given and none configured — run `host models` first");
    }
    if model::host_has_model(&url, &model_name) {
        println!("{} is already on the server.", model_name);
        return Ok(());
    }
    let response = ureq::post(&format!("{url}/api/pull"))
        .header("Content-Type", "application/json")
        .send_json(serde_json::json!({
            "name": model_name,
            "stream": false,
        }))
        .map_err(|e| anyhow::anyhow!("could not pull {}: {e}", model_name))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.into_body().read_to_string().unwrap_or_default();
        bail!("could not pull {}: status {status}: {}", model_name, body);
    }
    println!("{} is ready on the server.", model_name);
    Ok(())
}

fn list_devices() -> Result<()> {
    model::backend()?;
    let choice = Choice::current(None)?;
    let chosen = choice.place()?.device.map(|it| it.index);
    println!("Set: {choice}");
    for device in device::all() {
        let mut marker = " ";
        if chosen == Some(device.index) || (chosen.is_none() && device.device_type == LlamaBackendDeviceType::Cpu) {
            marker = "*";
        }
        let mut memory = String::new();
        if device.memory_total > 0 {
            memory = format!("{:.1} GB", device.memory_total as f64 / 1e9);
        }
        println!("{marker} {:<3} {:<8} {:<40} {memory}", device.index, device.backend, device.description);
    }
    Ok(())
}

fn set_device(text: &str) -> Result<()> {
    let choice = Choice::parse(text)?;
    model::backend()?;
    let placement = choice.place()?;

    let mut config = Config::load()?;
    config.device = Some(choice.to_string());
    config.save()?;
    println!("{choice} is set — the model runs on {placement}.");
    Ok(())
}

/// Load a Loaded handle according to the current backend.
fn load_model(device: Option<&str>) -> Result<Loaded> {
    let config = Config::load()?;
    match Backend::current(&config) {
        Backend::Remote => {
            let (url, model_name) = remote_target(&config)?;
            Ok(Loaded::Remote { model_name, server_url: url })
        }
        Backend::Local => {
            let m = Model::current()?;
            let local = m.load(Choice::current(device)?)?;
            Ok(Loaded::Local {
                backend: local.backend,
                model: local.model,
                placement: local.placement,
                name: m.name,
            })
        }
    }
}

fn remote_target(config: &Config) -> Result<(String, String)> {
    let model_name = config.model.clone().ok_or_else(|| {
        anyhow::anyhow!("no model set — run `tinyclass model set <name>`")
    })?;
    let url = model::host();
    device::check_server(&url)?;
    Ok((url, model_name))
}

fn decide(input: &str, choices: &[String], instruction: &str, device: Option<&str>, json: bool, criteria: &[String]) -> Result<()> {
    let loaded = load_model(device)?;
    let decision = decision::decide(&loaded, instruction, input, choices, criteria)?;
    if json {
        print_json(&decision)?;
    } else {
        print_decision(&decision);
    }
    Ok(())
}

fn noul(statement: &str, instruction: &str, device: Option<&str>, json: bool) -> Result<()> {
    let loaded = load_model(device)?;
    let probability = decision::noul(&loaded, instruction, statement)?;
    if json {
        println!("{}", serde_json::json!({ "probability": probability }));
    } else {
        println!("{probability:.3}");
    }
    Ok(())
}

/// Rate a statement on a numeric scale defined by labeled criteria.
fn score(statement: &str, instruction: &str, json: bool) -> Result<()> {
    let loaded = load_model(None)?;
    let score_result = decision::score(&loaded, instruction, statement)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&score_result.as_json())?);
    } else {
        print_score_plain(&score_result);
    }
    Ok(())
}

/// Send multiple question types in a single /v1/systemone request.
fn batch(
    state: &str,
    choice: Option<&str>,
    criteria: Option<&str>,
    noul: Option<&str>,
    score: Option<&str>,
    json: bool,
) -> Result<()> {
    if choice.is_none() && noul.is_none() && score.is_none() {
        bail!("at least one of --choice, --noul, or --score is required");
    }

    let choices: Option<Vec<String>> = choice.map(|choice_str| {
        choice_str.split('|').map(|s| s.trim().to_string()).collect()
    });
    if let Some(ref choices_vec) = choices
        && choices_vec.len() < 2
    {
        bail!("--choice needs at least 2 pipe-separated choices, got: {}", choice.unwrap_or(""));
    }

    let loaded = load_model(None)?;
    match loaded {
        Loaded::Local { .. } => {
            bail!("batch requires ollama backend — it sends multiple question types in a single /v1/systemone request");
        }
        Loaded::Remote { ref server_url, ref model_name } => {
            let questions = decision::build_questions(
                choices.as_deref(),
                criteria,
                noul,
                score,
            );
            let (parsed, _text) = decision::batch(server_url, model_name, state, &questions)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&parsed)?);
            } else {
                print_batch(&parsed);
            }
            Ok(())
        }
    }
}

/// Print the rating and breakdown as a plain string.
fn print_score_plain(score_result: &decision::ScoreResult) {
    println!("{:.3}", score_result.rating);
    if !score_result.legend.is_empty() {
        for (k, v) in &score_result.legend {
            if let Some(prob) = score_result.probabilities.iter().find(|(pk, _)| pk == k) {
                println!("  {v:<16}  {:>5.1}%", prob.1 * 100.0);
            }
        }
    }
}

fn print_batch(parsed: &serde_json::Value) {
    if let Some(answers) = parsed.get("answers").and_then(|v| v.as_object()) {
        for (key, answer) in answers {
            if let Some(ans_type) = answer.get("type").and_then(|v| v.as_str()) {
                match ans_type {
                    "score" => {
                        let score_val = answer.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        println!("  {}: {:.3}", key, score_val);
                    }
                    "noul" => {
                        let noul_val = answer.get("noul").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        println!("  {}: {:.3}", key, noul_val);
                    }
                    "choice" => {
                        let choice = answer.get("choice").and_then(|v| v.as_str()).unwrap_or("?");
                        println!("  {}: {}", key, choice);
                    }
                    _ => {
                        println!("  {}: (type={})", key, ans_type);
                    }
                }
                continue;
            }
            println!("  {}: {}", key, serde_json::to_string(answer).unwrap_or_else(|_| "?".to_string()));
        }
    } else {
        println!("(no answers)");
    }
}

fn play(choices: &[String], instruction: &str, device: Option<&str>) -> Result<()> {
    eprintln!("Loading model…");
    let loaded = load_model(device)?;
    eprintln!("Running on {}.", loaded.model_name());
    eprintln!("Deciding between {} for every line; Ctrl-D ends.", choices.join(", "));

    let stdin = io::stdin();
    loop {
        eprint!("> ");
        io::stderr().flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            eprintln!();
            break;
        }
        let input = line.trim();
        if !input.is_empty() {
            print_decision(&decision::decide(&loaded, instruction, input, choices, &[])?);
        }
    }
    Ok(())
}

fn print_decision(decision: &Decision) {
    let width = decision.scores.iter().map(|it| it.choice.len()).max().unwrap_or(0);
    for score in &decision.scores {
        let bar = "█".repeat((score.probability * 30.0).round() as usize);
        println!(
            "  {:<width$}  {:>5.1}%  {bar}",
            score.choice,
            score.probability * 100.0,
            width = width
        );
    }
    println!("→ {}", decision.chosen().choice);
}

fn print_json(decision: &Decision) -> Result<()> {
    let scores: Vec<serde_json::Value> = decision
        .scores
        .iter()
        .map(|it| {
            serde_json::json!({
                "choice": it.choice,
                "logit": it.logit,
                "logprob": it.logprob,
                "probability": it.probability,
            })
        })
        .collect();
    let value = serde_json::json!({
        "chosen": decision.chosen().choice,
        "scores": scores,
    });
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}
