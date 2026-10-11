//! The settings: which model answers, and where it runs.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;

use crate::fsutil;
use crate::paths;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Backend {
    #[default]
    Local,
    Remote,
}

impl Backend {
    pub fn current(config: &Config) -> Backend {
        match config.backend.as_deref() {
            Some("ollama") => Backend::Remote,
            Some("llama-cpp") => Backend::Local,
            _ => Backend::Local,
        }
    }
}

#[derive(Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Config {
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
}

impl Config {
    pub fn load() -> Result<Config> {
        let path = paths::config_path();
        if path.exists() {
            let contents = fs::read_to_string(&path)?;
            if contents.is_empty() {
                Ok(Config::default())
            } else {
                serde_json::from_str(&contents)
                    .with_context(|| format!("could not parse {}", path.display()))
            }
        } else {
            Ok(Config::default())
        }
    }

    pub fn save(&self) -> Result<()> {
        fsutil::write_json_atomically(&paths::config_path(), self)
    }
}
