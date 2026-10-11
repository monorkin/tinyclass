//! A decision model out of a chat model: ask it a multiple-choice question,
//! stop after the prompt, and read the logits of the answer letters instead
//! of sampling. The softmax over just those letters is the decision.

use anyhow::{Context, Result, bail};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::llama_backend::LlamaBackend;
use std::num::NonZeroU32;

use crate::device::Placement;
use crate::model::{self, Loaded};

const LABELS: [char; 26] = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
];

fn criteria_map<'a, I>(pairs: I) -> std::collections::HashMap<String, String>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut map = std::collections::HashMap::new();
    for pair in pairs {
        if let Some((key, value)) = pair.split_once(':') {
            map.insert(key.trim().to_string(), value.trim().to_string());
        }
    }
    map
}

pub struct Decision {
    pub scores: Vec<Score>,
}

pub struct Score {
    pub choice: String,
    pub logit: f32,
    pub logprob: f32,
    pub probability: f32,
}

impl Decision {
    pub fn chosen(&self) -> &Score {
        self.scores
            .iter()
            .max_by(|a, b| a.probability.total_cmp(&b.probability))
            .expect("a decision has at least one choice")
    }
}

/// A numeric rating (0–1) with per-option breakdown when asked against
/// labeled criteria, e.g. "Routine / Soon / Urgent".
pub struct ScoreResult {
    pub rating: f32,
    pub legend: Vec<(String, String)>,
    pub probabilities: Vec<(String, f64)>,
    pub confidence: f64,
}

impl ScoreResult {
    pub fn as_json(&self) -> serde_json::Value {
        let probabilities: serde_json::Map<String, serde_json::Value> =
            self.probabilities.iter().map(|(k, v)| (k.clone(), serde_json::json!(v))).collect();
        let legend: serde_json::Map<String, serde_json::Value> =
            self.legend.iter().map(|(k, v)| (k.clone(), serde_json::json!(v))).collect();
        serde_json::json!({
            "score": self.rating,
            "confidence": self.confidence,
            "probabilities": probabilities,
            "legend": legend,
        })
    }
}

/// Jev's yes/no question, named after the Bernoulli distribution: how likely
/// the statement is true, as one number between 0 and 1.
pub fn noul(model: &Loaded, instruction: &str, statement: &str) -> Result<f32> {
    match model {
        Loaded::Remote { server_url, model_name } => {
            let body = serde_json::json!({
                "model": model_name,
                "state": statement,
                "questions": {
                    "noul": {
                        "type": "noul",
                        "instructions": instruction,
                    }
                }
            });
            let parsed = remote_req(server_url, body)?;
            parsed
                .pointer("/answers/noul/noul")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .ok_or_else(|| anyhow::anyhow!("systemone response missing 'answers.noul.noul'"))
        }
        Loaded::Local { .. } => {
            let decision = decide(model, instruction, statement, &["Yes".into(), "No".into()], &[])?;
            Ok(decision.scores[0].probability)
        }
    }
}

/// Load a Loaded handle for the current provider.
pub fn score(loaded: &Loaded, instruction: &str, state: &str) -> Result<ScoreResult> {
    match loaded {
        Loaded::Remote { server_url, model_name } =>
            score_remote(server_url, model_name, instruction, state),
        Loaded::Local { .. } => score_local(loaded, instruction, state),
    }
}

/// Score a statement using an ollama server and return the rating + legend.
fn score_remote(server_url: &str, model_name: &str, instruction: &str, state: &str) -> Result<ScoreResult> {
    let body = serde_json::json!({
        "model": model_name,
        "state": state,
        "questions": {
            "rating": {
                "type": "score",
                "instructions": instruction,
                "criteria": ["Low", "Medium", "High"],
            }
        }
    });
    let parsed = remote_req(server_url, body)?;

    let rating = parsed
        .pointer("/answers/rating/score")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| anyhow::anyhow!("score response missing 'answers.rating.score'"))? as f32;

    let confidence = parsed
        .pointer("/answers/rating/confidence")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    let probabilities_map = parsed
        .pointer("/answers/rating/probabilities")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    let probabilities: Vec<(String, f64)> = probabilities_map
        .iter()
        .map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(0.0)))
        .collect();

    let legend_map = parsed
        .pointer("/answers/rating/legend")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    let legend: Vec<(String, String)> = legend_map
        .iter()
        .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
        .collect();

    Ok(ScoreResult { rating, legend, probabilities, confidence })
}

/// Local score: encode bins as choices and let the model pick a probability distribution.
pub fn score_local(model: &Loaded, instruction: &str, state: &str) -> Result<ScoreResult> {
    let bins = ["0", "500", "1000"];
    let choices: Vec<String> = bins.iter().map(|s| s.to_string()).collect();

    let decision = decide(model, instruction, state, &choices, &[])?;

    let values: Vec<f32> = bins.iter().map(|s| s.parse().unwrap_or(0.0)).collect();
    let mut rating = 0.0f32;
    for (score, &value) in decision.scores.iter().zip(&values) {
        rating += score.probability * value;
    }

    let mean_prob: f32 = decision.scores.iter().map(|s| s.probability).sum::<f32>() / decision.scores.len() as f32;
    let deviation: f32 = decision.scores.iter().map(|s| (s.probability - mean_prob).abs()).sum();
    let confidence: f64 = (1.0 - deviation) as f64;

    let legend = vec![
        ("0".to_string(), "Routine".to_string()),
        ("500".to_string(), "Urgent".to_string()),
        ("1000".to_string(), "Critical".to_string()),
    ];

    let probabilities: Vec<(String, f64)> = decision
        .scores
        .iter()
        .map(|s| (s.choice.clone(), s.probability as f64))
        .collect();

    Ok(ScoreResult { rating, legend, probabilities, confidence })
}

/// POST to ollama /v1/systemone; return the parsed JSON and the raw body text.
fn systemone(server_url: &str, body: serde_json::Value) -> Result<(serde_json::Value, String)> {
    let response = ureq::post(&format!("{server_url}/v1/systemone"))
        .header("Content-Type", "application/json")
        .send_json(body)
        .context("ollama /v1/systemone")?;
    let status = response.status();
    let text = response.into_body().read_to_string()?;
    if !status.is_success() {
        bail!("ollama /v1/systemone status {status}: {text}");
    }
    Ok((serde_json::from_str(&text).context("parse ollama response")?, text))
}

fn remote_req(server_url: &str, body: serde_json::Value) -> Result<serde_json::Value> {
    let (value, _) = systemone(server_url, body)?;
    Ok(value)
}

pub fn batch(
    server_url: &str,
    model_name: &str,
    state: &str,
    questions: &serde_json::Map<String, serde_json::Value>,
) -> Result<(serde_json::Value, String)> {
    let body = serde_json::json!({
        "model": model_name,
        "state": state,
        "questions": questions,
    });
    systemone(server_url, body)
}

pub fn build_questions(
    choices: Option<&[String]>,
    criteria: Option<&str>,
    noul: Option<&str>,
    score: Option<&str>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut questions = serde_json::Map::new();
    if let Some(choice) = choices {
        let mut choice_criteria: serde_json::Map<String, serde_json::Value> =
            choice.iter().map(|c| (c.clone(), serde_json::Value::Null)).collect();
        if let Some(criteria_str) = criteria {
            for (k, v) in criteria_map(criteria_str.split('|')) {
                if choice_criteria.contains_key(&k) {
                    choice_criteria.insert(k, serde_json::Value::String(v));
                }
            }
        }
        questions.insert("label".into(), serde_json::json!({
            "type": "choice",
            "instructions": "Choose one option based on the context.",
            "criteria": choice_criteria,
        }));
    }
    if let Some(noul) = noul {
        questions.insert("harm".into(), serde_json::json!({
            "type": "noul",
            "instructions": noul,
        }));
    }
    if let Some(score) = score {
        questions.insert("urgency".into(), serde_json::json!({
            "type": "score",
            "instructions": score,
            "criteria": ["Low", "Medium", "High"],
        }));
    }
    questions
}

pub fn decide(model: &Loaded, instruction: &str, input: &str, choices: &[String], criteria: &[String]) -> Result<Decision> {
    if choices.len() < 2 {
        bail!("a decision needs at least two choices");
    }
    if choices.len() > LABELS.len() {
        bail!("a decision can have at most {} choices", LABELS.len());
    }

    match model {
        Loaded::Local { backend, model, placement, .. } => {
            let prompt = prompt(instruction, input, choices, criteria);
            let logits = last_token_logits(model, backend, placement, &prompt)?;
            let logits = resolve_label_logits(model, &logits, choices)?;
            let normalizer = log_sum_exp(&logits);
            Ok(scores_from_logits(choices, &logits, normalizer))
        }
        Loaded::Remote { server_url, model_name } => {
            let user = criteria_map(criteria.iter().map(String::as_str));
            let criteria: serde_json::Map<String, serde_json::Value> = user
                .into_iter()
                .map(|(k, v)| (k, serde_json::Value::String(v)))
                .chain(choices.iter().map(|c| (c.clone(), serde_json::Value::Null)))
                .collect();
            let body = serde_json::json!({
                "model": model_name,
                "state": input,
                "questions": {
                    "choice": {
                        "type": "choice",
                        "instructions": instruction,
                        "criteria": criteria
                    }
                }
            });
            let parsed = remote_req(server_url, body)?;
            let probs_map = parsed
                .pointer("/answers/choice/probabilities")
                .and_then(|v| v.as_object())
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("ollama response missing 'probabilities'"))?;
            Ok(scores_from_probs(choices, &probs_map))
        }
    }
}

fn prompt(instruction: &str, input: &str, choices: &[String], criteria: &[String]) -> String {
    let options: Vec<String> = choices
        .iter()
        .zip(LABELS)
        .map(|(choice, label)| format!("{label}. {choice}"))
        .collect();

    let mut user_content = options.join("\n");

    if !criteria.is_empty() {
        let table = criteria_map(criteria.iter().map(String::as_str));
        let mut criteria_lines = String::new();
        for (i, choice) in choices.iter().enumerate() {
            let label = LABELS[i];
            let desc = table
                .get(choice)
                .cloned()
                .unwrap_or_else(|| choice.clone());
            criteria_lines.push_str(&format!("- {label}: {desc}\n"));
        }
        user_content.push('\n');
        user_content.push_str("Criteria:\n");
        user_content.push_str(&criteria_lines);
    }

    format!(
        "<|im_start|>system\n{instruction}<|im_end|>\n\
         <|im_start|>user\n{input}\n\n{}<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\n",
        user_content
    )
}

/// One context per question, sized to the prompt: the whole prompt goes in
/// as a single batch and only its last position keeps logits.
fn last_token_logits(model: &LlamaModel, backend: &'static LlamaBackend, placement: &Placement, prompt: &str) -> Result<Vec<f32>> {
    let tokens = model
        .str_to_token(prompt, AddBos::Never)
        .context("could not tokenize the prompt")?;
    let length = u32::try_from(tokens.len())?;
    let threads = i32::try_from(model::physical_cores())?;

    let on_cpu = placement.device.is_none();
    let params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(length))
        .with_n_batch(length)
        .with_n_ubatch(length)
        .with_n_threads(threads)
        .with_n_threads_batch(threads)
        .with_op_offload(!on_cpu)
        .with_offload_kqv(!on_cpu)
        .with_no_perf(true);
    let mut context = model
        .new_context(backend, params)
        .context("could not create a context for the prompt")?;

    let mut batch = LlamaBatch::new(tokens.len(), 1);
    let last = tokens.len() - 1;
    for (position, token) in tokens.iter().enumerate() {
        batch.add(*token, i32::try_from(position)?, &[0], position == last)?;
    }
    context.decode(&mut batch).context("could not run the prompt")?;

    Ok(context.get_logits_ith(i32::try_from(last)?).to_vec())
}

/// Scoring helpers used by both Local and Remote paths.
fn scores_from_logits(choices: &[String], logits: &[f32], normalizer: f32) -> Decision {
    let scores = choices
        .iter()
        .zip(logits)
        .map(|(choice, &logit)| Score {
            choice: choice.clone(),
            logit,
            logprob: logit - normalizer,
            probability: (logit - normalizer).exp(),
        })
        .collect();
    Decision { scores }
}

fn scores_from_probs(choices: &[String], probs: &serde_json::Map<String, serde_json::Value>) -> Decision {
    let scores = choices
        .iter()
        .map(|c| {
            let p = probs.get(c).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            Score {
                choice: c.clone(),
                logit: (p / (1.0 - p)).ln(),
                logprob: p.ln(),
                probability: p,
            }
        })
        .collect();
    Decision { scores }
}

fn log_sum_exp(values: &[f32]) -> f32 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    max + values.iter().map(|it| (it - max).exp()).sum::<f32>().ln()
}

/// Extract logits at the token positions for each choice label (A, B, C…).
fn resolve_label_logits(model: &LlamaModel, logits: &[f32], choices: &[String]) -> Result<Vec<f32>> {
    let mut result = Vec::with_capacity(choices.len());
    for (i, _) in choices.iter().enumerate() {
        let label = LABELS[i];
        let token_buf = model.str_to_token(&label.to_string(), AddBos::Never)
            .context("could not tokenize label")?;
        let target = token_buf.first().ok_or_else(|| anyhow::anyhow!("empty token for label '{label}'"))?;

        // Find the vocab index for this token by iterating with model.tokens()
        for (vocab_idx, (vocab_token, _)) in model.tokens(false).enumerate() {
            if vocab_token.0 == target.0 {
                result.push(logits[vocab_idx]);
                break;
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chosen_is_the_most_probable_choice() {
        let decision = Decision {
            scores: vec![
                Score { choice: "Spam".into(), logit: 1.0, logprob: -2.0, probability: 0.1 },
                Score { choice: "Phishing".into(), logit: 3.0, logprob: -0.1, probability: 0.9 },
            ],
        };
        assert_eq!(decision.chosen().choice, "Phishing");
    }

    #[test]
    fn prompt_labels_choices_in_order() {
        let choices = vec!["Legitimate".to_string(), "Spam".to_string()];
        let prompt = prompt("Choose one option.", "Hello", &choices, &[]);
        assert!(prompt.contains("Hello\n\nA. Legitimate\nB. Spam<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn build_questions_preserves_noul_value() {
        let q = build_questions(None, None, Some("Is this harmful?"), None);
        let harm = q.get("harm").expect("harm key must be present");
        assert_eq!(harm["instructions"], "Is this harmful?");
        assert_eq!(harm["type"], "noul");
    }

    #[test]
    fn build_questions_choice_criteria_overrides_nulls_only_for_matching_keys() {
        let choices = vec!["a".to_string(), "b".to_string()];
        let q = build_questions(Some(&choices), Some("a:good|c:extra"), None, None);
        let label = q.get("label").expect("label key must be present");
        let crit = &label["criteria"];
        assert_eq!(crit["a"], "good");
        assert!(crit["b"].is_null());
        assert!(crit.get("c").is_none());
    }

    #[test]
    fn score_result_as_json_uses_objects_not_arrays() {
        let result = ScoreResult {
            rating: 0.5,
            confidence: 0.9,
            legend: vec![("0".into(), "Low".into())],
            probabilities: vec![("0".into(), 0.5f64)],
        };
        let json = result.as_json();
        assert!(json["probabilities"]["0"].is_number());
        assert!(json["legend"]["0"].is_string());
    }
}
