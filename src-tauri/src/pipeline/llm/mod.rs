//! The swappable local-LLM layer.
//!
//! Everything above this module talks to [`LlmProvider`], so the model, the
//! quantization and even the runtime can change after benchmarking without the
//! MoM code or the UI knowing (forbids committing to a model before those
//! benchmarks exist). [`LlamaCppProvider`] is the shipped implementation;
//! [`MockLlm`] is what the tests use.

pub mod server;

use crate::models_manager;
// The loopback client lives in `pipeline::http`: whisper-server joined
// llama-server in being driven over a local socket, so it is no longer the
// LLM layer's alone.
use crate::pipeline::http;
use crate::pipeline::paths::ModelPaths;
use crate::settings::Settings;
use crate::system::{self, HostInfo};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

/// Provenance recorded with every generated MoM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmDescriptor {
    pub model: String,
    pub quantization: String,
    pub context_tokens: usize,
    pub runtime: String,
}

pub trait LlmProvider {
    fn describe(&self) -> LlmDescriptor;

    /// Runs one completion that is expected to return a single JSON object.
    /// Implementations return the raw text; JSON extraction and schema
    /// validation are the caller's job (invalid JSON must be repaired
    /// and retried, never silently accepted).
    fn complete_json(&self, system: &str, user: &str, max_tokens: usize) -> Result<String, String>;
}

/// One generation on a local CPU can legitimately take minutes on the smaller
/// machines this app targets.
const GENERATION_TIMEOUT: Duration = Duration::from_secs(900);

pub struct LlamaCppProvider {
    server: server::LlamaServer,
    descriptor: LlmDescriptor,
}

impl LlamaCppProvider {
    /// Starts a local llama.cpp server for Cadence's one language model.
    ///
    /// Refuses to start when the model plainly will not fit in the machine's
    /// free RAM, and says what to do about it — on a 16 GB laptop, an
    /// out-of-memory kill mid-generation would lose the MoM stage, and
    /// failures must be legible and recoverable, not mysterious.
    pub fn start(
        paths: &ModelPaths,
        settings: &Settings,
        host: &HostInfo,
        data_dir: &Path,
    ) -> Result<Self, String> {
        let model_id = settings.effective_llm_model();
        Self::start_with_model(
            paths,
            &model_id,
            models_manager::entry(&model_id),
            settings.effective_llm_context(),
            settings.effective_llm_threads(host),
            host,
            data_dir,
        )
    }

    /// Starts the server on `paths.llm_model` with an explicit model id,
    /// context and thread count, instead of reading them through `Settings`.
    ///
    /// [`Self::start`] is the app's path and delegates here. The other caller
    /// is `bench_cli compare`, which runs candidate models that are
    /// deliberately not in the app's catalog: it points `paths.llm_model` at
    /// the candidate and passes the candidate's own entry for the memory
    /// pre-flight. `catalog_entry` of `None` skips that pre-flight.
    pub fn start_with_model(
        paths: &ModelPaths,
        model_id: &str,
        catalog_entry: Option<&models_manager::CatalogEntry>,
        context_tokens: usize,
        threads: usize,
        host: &HostInfo,
        data_dir: &Path,
    ) -> Result<Self, String> {
        let model_id = model_id.to_string();
        if let Some(entry) = catalog_entry {
            preflight_memory(entry, context_tokens, host)?;
        }

        let log_path = data_dir.join("logs").join("llama-server.log");
        let server = server::LlamaServer::start(
            &paths.llama_server(),
            &paths.llm_model,
            context_tokens,
            threads.max(1),
            &log_path,
        )?;

        let file_name = paths
            .llm_model
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| model_id.clone());

        Ok(Self {
            server,
            descriptor: LlmDescriptor {
                model: model_id,
                quantization: quantization_from_filename(&file_name),
                context_tokens,
                runtime: "llama.cpp (local)".to_string(),
            },
        })
    }
}

impl LlamaCppProvider {
    /// Measured peak RAM of the inference process. `None` before the
    /// process has done any work, or on a non-Windows build.
    pub fn peak_ram_mb(&self) -> Option<u64> {
        self.server.peak_ram_mb()
    }
}

impl LlmProvider for LlamaCppProvider {
    fn describe(&self) -> LlmDescriptor {
        self.descriptor.clone()
    }

    fn complete_json(&self, system: &str, user: &str, max_tokens: usize) -> Result<String, String> {
        // Low temperature: minutes are an extraction task, and sampling
        // variety here shows up as invented detail.
        let mut body = serde_json::json!({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": max_tokens,
            "stream": false,
            "cache_prompt": true,
            "response_format": {"type": "json_object"},
            // Qwen3 reasons out loud by default, which wastes most of the
            // token budget on text nobody reads.
            "chat_template_kwargs": {"enable_thinking": false}
        });

        let mut last_error = String::new();
        for attempt in 0..2 {
            let payload = serde_json::to_string(&body).map_err(|e| e.to_string())?;
            let response = http::request(
                self.server.port(),
                "POST",
                "/v1/chat/completions",
                Some(&payload),
                GENERATION_TIMEOUT,
            )?;

            if response.is_success() {
                return parse_chat_response(&response.body);
            }

            last_error = server_error_message(&response.body, response.status);
            if attempt == 0 {
                // Older llama-server builds reject the constrained-output and
                // template options. Drop them and try once more on the plain
                // request — the JSON schema is still enforced by validation
                // and the repair pass downstream.
                if let Some(map) = body.as_object_mut() {
                    map.remove("response_format");
                    map.remove("chat_template_kwargs");
                }
            }
        }

        Err(format!(
            "the local model server rejected the request: {last_error}. {}",
            self.server.log_tail()
        ))
    }
}

fn preflight_memory(
    entry: &models_manager::CatalogEntry,
    context_tokens: usize,
    host: &HostInfo,
) -> Result<(), String> {
    let estimated = entry.estimated_llm_ram_mb(context_tokens);
    // "Available" RAM understates what the model can get. The weights are
    // memory-mapped, so Windows can leave part of the file on disk and read
    // it back on demand, and it trims idle applications' working sets to
    // make room. Measured 2026-11-20 on a 16 GB i7-1260P with ordinary apps
    // open: Qwen3-8B at 8K context ran to valid minutes from 6,287 MB free,
    // and again from 4,822 MB free (about 20% slower, working set held near
    // 4.8 GB, private memory ~4.1 GB). The old rule — estimate plus 1.5 GB of
    // headroom, 7,550 MB — refused every one of those runs. So the check asks
    // for the estimate minus the share of the weights Windows was seen to
    // keep on disk without the run thrashing (a quarter), and no headroom:
    // a quarter of this model is ~1.2 GB, giving ~4.85 GB, just above the
    // lowest point measured to work. Below that is untested, so refused.
    let paged_weights_mb = entry.approx_size_mb / 4;
    // The free RAM the check actually asks for. The message quotes this
    // number, not the bare estimate: a user who frees exactly the quoted
    // amount must pass.
    let required_free_mb = estimated.saturating_sub(paged_weights_mb);
    if required_free_mb > host.available_ram_mb {
        // There is no smaller model to fall back to — every machine runs the
        // same one — so the advice is about freeing memory. The transcript is
        // already saved by the time this runs; only the minutes wait.
        let minimum_note = if host.meets_minimum() {
            ""
        } else {
            " — this machine has less"
        };
        let shortfall_mb = required_free_mb - host.available_ram_mb;
        return Err(format!(
            "Writing the minutes with {} needs about {} MB of free RAM ({} MB for the model at a {} token context, less about {} MB of it that Windows can leave on disk while it runs), but only {} MB is free on this machine ({} MB total). Close other applications (browsers especially) to free at least {} MB more, then regenerate the minutes. Cadence needs a machine with at least 16 GB of RAM{minimum_note}.",
            entry.label,
            required_free_mb,
            estimated,
            context_tokens,
            paged_weights_mb,
            host.available_ram_mb,
            host.total_ram_mb,
            shortfall_mb
        ));
    }
    if estimated > system::LLM_RAM_BUDGET_MB {
        // Over the budget but the RAM is physically there: proceed, and say
        // so. Only reachable if the catalog's size or the context drifts.
        eprintln!(
            "[llm] {} is estimated at {} MB, above the {} MB budget",
            entry.label,
            estimated,
            system::LLM_RAM_BUDGET_MB
        );
    }
    Ok(())
}

fn parse_chat_response(body: &str) -> Result<String, String> {
    let value: serde_json::Value = serde_json::from_str(body)
        .map_err(|e| format!("could not parse the model server's response: {e}"))?;

    let content = value
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        // `/completion`-shaped responses, in case the endpoint changes.
        .or_else(|| value.get("content").and_then(|c| c.as_str()))
        .ok_or_else(|| {
            format!(
                "the model server returned no completion: {}",
                truncate(body, 300)
            )
        })?;

    Ok(strip_reasoning(content).to_string())
}

fn server_error_message(body: &str, status: u16) -> String {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| {
            v.get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| format!("HTTP {status}: {}", truncate(body, 200)))
}

/// Removes a reasoning block that a thinking model emitted anyway, despite
/// being asked not to.
fn strip_reasoning(text: &str) -> &str {
    match text.split_once("</think>") {
        Some((_, after)) => after.trim_start(),
        None => text,
    }
}

/// Pulls the first balanced JSON object out of a model response, tolerating
/// stray prose or markdown fences around it. Brace counting skips over braces
/// inside strings, so a `{` in the transcript text cannot end the object early.
pub fn extract_json(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (offset, &byte) in bytes[start..].iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start..start + offset + 1].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

fn quantization_from_filename(file_name: &str) -> String {
    let upper = file_name.to_ascii_uppercase();
    for tag in [
        "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_K_S",
        "Q5_K_M", "Q6_K", "Q8_0", "F16", "BF16",
    ] {
        if upper.contains(tag) {
            return tag.to_string();
        }
    }
    "unknown".to_string()
}

fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    text.chars().take(max).collect::<String>() + "…"
}

/// Scripted provider for tests ("mocked AI providers") and for the
/// `--dry-run` mode of the CLI runners.
pub struct MockLlm {
    responses: std::sync::Mutex<Vec<String>>,
    fallback: String,
}

impl MockLlm {
    pub fn new(responses: Vec<String>, fallback: &str) -> Self {
        Self {
            responses: std::sync::Mutex::new(responses),
            fallback: fallback.to_string(),
        }
    }
}

impl LlmProvider for MockLlm {
    fn describe(&self) -> LlmDescriptor {
        LlmDescriptor {
            model: "mock".to_string(),
            quantization: "none".to_string(),
            context_tokens: 4096,
            runtime: "mock".to_string(),
        }
    }

    fn complete_json(&self, _system: &str, _user: &str, _max_tokens: usize) -> Result<String, String> {
        let mut queue = self.responses.lock().map_err(|e| e.to_string())?;
        if queue.is_empty() {
            return Ok(self.fallback.clone());
        }
        Ok(queue.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_a_plain_object() {
        assert_eq!(extract_json("{\"a\":1}").unwrap(), "{\"a\":1}");
    }

    #[test]
    fn extracts_from_around_prose_and_fences() {
        let text = "Sure! Here you go:\n```json\n{\"a\": {\"b\": 2}}\n```\nHope that helps.";
        assert_eq!(extract_json(text).unwrap(), "{\"a\": {\"b\": 2}}");
    }

    #[test]
    fn a_brace_inside_a_string_does_not_end_the_object() {
        let text = "{\"text\": \"a } brace\", \"n\": 1}";
        assert_eq!(extract_json(text).unwrap(), text);
    }

    #[test]
    fn an_escaped_quote_does_not_end_the_string() {
        let text = r#"{"text": "she said \"} \" here", "n": 1}"#;
        assert_eq!(extract_json(text).unwrap(), text);
    }

    #[test]
    fn returns_none_when_there_is_no_object() {
        assert!(extract_json("no json here").is_none());
        assert!(extract_json("{\"unterminated\": 1").is_none());
    }

    #[test]
    fn parses_an_openai_shaped_chat_response() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"{\"summary\":\"ok\"}"}}]}"#;
        assert_eq!(parse_chat_response(body).unwrap(), "{\"summary\":\"ok\"}");
    }

    #[test]
    fn drops_a_reasoning_block() {
        let body = r#"{"choices":[{"message":{"content":"<think>hmm</think>\n{\"a\":1}"}}]}"#;
        assert_eq!(parse_chat_response(body).unwrap(), "{\"a\":1}");
    }

    #[test]
    fn reports_a_response_with_no_completion() {
        assert!(parse_chat_response(r#"{"choices":[]}"#).is_err());
    }

    #[test]
    fn reads_the_quantization_off_the_file_name() {
        assert_eq!(quantization_from_filename("Qwen3-8B-Q4_K_M.gguf"), "Q4_K_M");
        assert_eq!(quantization_from_filename("model-q6_k.gguf"), "Q6_K");
        assert_eq!(quantization_from_filename("something.gguf"), "unknown");
    }

    #[test]
    fn surfaces_a_server_error_message() {
        let body = r#"{"error":{"message":"context is full"}}"#;
        assert_eq!(server_error_message(body, 400), "context is full");
        assert!(server_error_message("not json", 500).contains("HTTP 500"));
    }

    #[test]
    fn a_model_too_big_for_free_ram_is_refused_with_advice() {
        let entry = models_manager::entry(system::LLM_MODEL_ID).unwrap();
        let host = HostInfo {
            total_ram_mb: 15_700,
            available_ram_mb: 3_000,
            logical_cores: 8,
        };
        let err = preflight_memory(entry, system::LLM_CONTEXT_TOKENS, &host).unwrap_err();
        assert!(err.contains("Close other applications"), "{err}");
        assert!(err.contains("16 GB"), "{err}");
        // There is no smaller model any more, so it must not point at one.
        assert!(!err.contains("smaller model"), "{err}");
        assert!(!err.contains("has less"), "a 16 GB machine meets the minimum");
    }

    #[test]
    fn the_free_ram_the_message_asks_for_is_the_free_ram_the_check_needs() {
        let entry = models_manager::entry(system::LLM_MODEL_ID).unwrap();
        let context = system::LLM_CONTEXT_TOKENS;
        let host = |available_ram_mb| HostInfo {
            total_ram_mb: 15_700,
            available_ram_mb,
            logical_cores: 8,
        };
        // One megabyte short: refused, quoting the threshold and the gap.
        let err = preflight_memory(entry, context, &host(3_000)).unwrap_err();
        let quoted: u64 = err
            .split("needs about ")
            .nth(1)
            .and_then(|rest| rest.split(" MB").next())
            .and_then(|n| n.parse().ok())
            .unwrap_or_else(|| panic!("no threshold in: {err}"));
        assert!(err.contains(&format!("{} MB for the model", entry.estimated_llm_ram_mb(context))), "{err}");
        let err = preflight_memory(entry, context, &host(quoted - 1)).unwrap_err();
        assert!(err.contains(&format!("needs about {quoted} MB of free RAM")), "{err}");
        assert!(err.contains("at least 1 MB more"), "{err}");
        // Freeing exactly what the message asked for is enough.
        assert!(preflight_memory(entry, context, &host(quoted)).is_ok());
    }

    #[test]
    fn a_machine_below_the_minimum_is_told_so() {
        let entry = models_manager::entry(system::LLM_MODEL_ID).unwrap();
        let host = HostInfo {
            total_ram_mb: 8_100,
            available_ram_mb: 3_000,
            logical_cores: 4,
        };
        let err = preflight_memory(entry, system::LLM_CONTEXT_TOKENS, &host).unwrap_err();
        assert!(err.contains("this machine has less"), "{err}");
    }

    #[test]
    fn the_free_ram_the_8b_was_measured_to_run_in_is_not_refused() {
        // 2026-11-20, 16 GB i7-1260P: valid minutes from 6,287 MB free and
        // from 4,900 MB free (4,822 at the moment the run started; the
        // threshold sits just above that, so use the round figure the
        // pressure test targeted).
        let entry = models_manager::entry(system::LLM_MODEL_ID).unwrap();
        for available_ram_mb in [6_287, 4_900] {
            let host = HostInfo { total_ram_mb: 16_008, available_ram_mb, logical_cores: 16 };
            assert!(
                preflight_memory(entry, system::LLM_CONTEXT_TOKENS, &host).is_ok(),
                "refused at {available_ram_mb} MB free"
            );
        }
    }

    #[test]
    fn the_model_passes_preflight_on_a_typical_16gb_machine() {
        let entry = models_manager::entry(system::LLM_MODEL_ID).unwrap();
        let host = HostInfo {
            total_ram_mb: 15_700,
            available_ram_mb: 9_000,
            logical_cores: 8,
        };
        assert!(preflight_memory(entry, system::LLM_CONTEXT_TOKENS, &host).is_ok());
    }

    #[test]
    fn the_mock_provider_replays_then_falls_back() {
        let llm = MockLlm::new(vec!["first".to_string()], "fallback");
        assert_eq!(llm.complete_json("s", "u", 10).unwrap(), "first");
        assert_eq!(llm.complete_json("s", "u", 10).unwrap(), "fallback");
    }
}
