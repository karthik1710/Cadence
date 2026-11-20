//! Comparison candidates for `bench_cli compare` — **bench-only**.
//!
//! None of these is part of the app: they are not in
//! `models_manager::catalog()`, so they never appear in Settings, the first-run
//! setup screen or `required_ids()`, and `system`'s one shipped model set is
//! untouched. They install through the app's own installer
//! (`models_manager::install_entry`, which shells out to Windows' bundled
//! downloader exactly as the Model Manager does) but into a separate tree,
//! `<app_data>/models/bench/`, with its own `manifest.json`.
//!
//! Sizes are the real `content-length` of each URL, checked by hand on
//! 2026-09-27 (every URL returned HTTP 200 after its redirect). Re-check them
//! if a download starts failing — upstream repos do get reorganised.
//!
//! `tests/offline_guarantee.rs` allows download URLs in exactly two files:
//! `models_manager.rs` and this one, and holds both to https on a sanctioned
//! publisher.

use app_lib::models_manager::{CatalogEntry, ModelKind};
use std::path::{Path, PathBuf};

/// One candidate, plus whether it only installs with `--with-large`.
pub struct Candidate {
    pub entry: CatalogEntry,
    /// Over the 16 GB machine's budget by design — a quality ceiling to
    /// compare against, not something that could ship.
    pub large: bool,
}

/// `<app_data>/models/bench` — the root `install_entry` is given, so every
/// candidate lands in `models/bench/<subdir>/`.
pub fn models_dir(data_dir: &Path) -> PathBuf {
    app_lib::pipeline::paths::models_dir(data_dir).join("bench")
}

pub fn candidates() -> Vec<Candidate> {
    vec![
        // ---------------------------------------------------------------- STT
        Candidate {
            entry: CatalogEntry {
                id: "whisper-medium",
                kind: ModelKind::Stt,
                label: "Whisper medium (multilingual)",
                subdir: "stt",
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
                download_name: "ggml-medium.bin",
                archive: None,
                expect_relative: "ggml-medium.bin",
                // 1,533,763,059 bytes.
                approx_size_mb: 1463,
                notes: "Bench candidate: 3x whisper-small's parameters.",
            },
            large: false,
        },
        Candidate {
            entry: CatalogEntry {
                id: "whisper-large-v3-turbo",
                kind: ModelKind::Stt,
                label: "Whisper large-v3-turbo (q5_0)",
                subdir: "stt",
                // The q5_0 quantisation rather than the 1.6 GB f16 file: at
                // ~550 MB it is the one large-v3 variant that could plausibly
                // replace whisper-small on a 16 GB machine, which is the
                // question this comparison is for.
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
                download_name: "ggml-large-v3-turbo-q5_0.bin",
                archive: None,
                expect_relative: "ggml-large-v3-turbo-q5_0.bin",
                // 574,041,195 bytes.
                approx_size_mb: 548,
                notes: "Bench candidate: large-v3 encoder, 4-layer decoder, q5_0.",
            },
            large: false,
        },
        // ---------------------------------------------------------------- LLM
        //
        // Smallest first, so `setup` finishes the cheap downloads before the
        // big ones; the two `large` quality ceilings come last.
        //
        // Gemma 4 (arch `gemma4`) and Qwen3.5 (arch `qwen35`) need a llama.cpp
        // that knows them. The pinned b10717 runtime does: both are in its
        // `src/llama-arch.cpp`, and both strings are in the installed
        // `llama.dll`. Their embedded chat templates honour the
        // `enable_thinking: false` the provider already sends, and both take
        // the shipped `--flash-attn on` + q8_0 KV cache unchanged: the CPU
        // flash-attention kernel is generic over head size, and Gemma 4's
        // 256/512 head dims divide q8_0's 32-element block. See
        // `docs/benchmarks.md`.
        Candidate {
            entry: CatalogEntry {
                id: "gemma3-4b-q4",
                kind: ModelKind::Llm,
                label: "Gemma 3 4B instruct (Q4_K_M)",
                subdir: "llm",
                url: "https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf",
                download_name: "gemma-3-4b-it-Q4_K_M.gguf",
                archive: None,
                expect_relative: "gemma-3-4b-it-Q4_K_M.gguf",
                // 2,489,757,856 bytes.
                approx_size_mb: 2375,
                notes: "Bench candidate: half the shipped model's RAM.",
            },
            large: false,
        },
        Candidate {
            entry: CatalogEntry {
                id: "qwen35-4b-q4",
                kind: ModelKind::Llm,
                label: "Qwen3.5 4B (Q4_K_M)",
                subdir: "llm",
                // Qwen publishes no official Qwen3.5 GGUF; unsloth's is the
                // widely used conversion.
                url: "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
                download_name: "Qwen3.5-4B-Q4_K_M.gguf",
                archive: None,
                expect_relative: "Qwen3.5-4B-Q4_K_M.gguf",
                // 2,740,937,888 bytes.
                approx_size_mb: 2614,
                notes: "Bench candidate: speed option from the Qwen3.5 family (hybrid Gated DeltaNet + attention).",
            },
            large: false,
        },
        Candidate {
            entry: CatalogEntry {
                id: "gemma4-e2b-q4",
                kind: ModelKind::Llm,
                label: "Gemma 4 E2B instruct (Q4_K_M)",
                subdir: "llm",
                url: "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf",
                download_name: "gemma-4-E2B-it-Q4_K_M.gguf",
                archive: None,
                expect_relative: "gemma-4-E2B-it-Q4_K_M.gguf",
                // 3,106,738,272 bytes.
                approx_size_mb: 2963,
                notes: "Bench candidate: fast floor / reference for the Gemma 4 family.",
            },
            large: false,
        },
        Candidate {
            entry: CatalogEntry {
                id: "llama3.1-8b-q4",
                kind: ModelKind::Llm,
                label: "Llama 3.1 8B instruct (Q4_K_M)",
                subdir: "llm",
                url: "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                download_name: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                archive: None,
                expect_relative: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                // 4,920,739,232 bytes.
                approx_size_mb: 4693,
                notes: "Bench candidate: same size class as the shipped model, different family.",
            },
            large: false,
        },
        Candidate {
            entry: CatalogEntry {
                id: "gemma4-e4b-qat-q4",
                kind: ModelKind::Llm,
                label: "Gemma 4 E4B instruct (QAT Q4_0)",
                subdir: "llm",
                // Google's own quantisation-aware-trained Q4_0.
                url: "https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/main/gemma-4-E4B_q4_0-it.gguf",
                download_name: "gemma-4-E4B_q4_0-it.gguf",
                archive: None,
                expect_relative: "gemma-4-E4B_q4_0-it.gguf",
                // 5,154,941,280 bytes.
                approx_size_mb: 4916,
                notes: "Bench candidate: ~4.5B effective (8B with per-layer embeddings), small sliding-window KV, ~5x fewer tokens than Qwen3 on Tamil script.",
            },
            large: false,
        },
        Candidate {
            entry: CatalogEntry {
                id: "qwen35-9b-q4",
                kind: ModelKind::Llm,
                label: "Qwen3.5 9B (Q4_K_M)",
                subdir: "llm",
                // Qwen publishes no official Qwen3.5 GGUF; this is LM
                // Studio's community conversion.
                url: "https://huggingface.co/lmstudio-community/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf",
                download_name: "Qwen3.5-9B-Q4_K_M.gguf",
                archive: None,
                expect_relative: "Qwen3.5-9B-Q4_K_M.gguf",
                // 5,627,044,256 bytes.
                approx_size_mb: 5366,
                notes: "Bench candidate: the shipped model's successor family; thinks by default, turned off per request.",
            },
            large: false,
        },
        Candidate {
            entry: CatalogEntry {
                id: "qwen3-8b-q5",
                kind: ModelKind::Llm,
                label: "Qwen3 8B (Q5_K_M)",
                subdir: "llm",
                url: "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q5_K_M.gguf",
                download_name: "Qwen3-8B-Q5_K_M.gguf",
                archive: None,
                expect_relative: "Qwen3-8B-Q5_K_M.gguf",
                // 5,851,112,224 bytes.
                approx_size_mb: 5580,
                notes: "Bench candidate: the shipped model at a higher-precision quantisation.",
            },
            large: false,
        },
        Candidate {
            entry: CatalogEntry {
                id: "gemma4-12b-qat-q4",
                kind: ModelKind::Llm,
                label: "Gemma 4 12B instruct (QAT Q4_0)",
                subdir: "llm",
                url: "https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf/resolve/main/gemma-4-12b-it-qat-q4_0.gguf",
                download_name: "gemma-4-12b-it-qat-q4_0.gguf",
                archive: None,
                expect_relative: "gemma-4-12b-it-qat-q4_0.gguf",
                // 6,975,879,296 bytes.
                approx_size_mb: 6653,
                notes: "Bench reference only: over the RAM budget once its KV cache and runtime are added — a Gemma 4 quality ceiling.",
            },
            large: true,
        },
        Candidate {
            entry: CatalogEntry {
                id: "gemma3-12b-q4",
                kind: ModelKind::Llm,
                label: "Gemma 3 12B instruct (Q4_K_M)",
                subdir: "llm",
                url: "https://huggingface.co/ggml-org/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-Q4_K_M.gguf",
                download_name: "gemma-3-12b-it-Q4_K_M.gguf",
                archive: None,
                expect_relative: "gemma-3-12b-it-Q4_K_M.gguf",
                // 7,300,574,976 bytes.
                approx_size_mb: 6963,
                notes: "Bench reference only: over the 7,000 MB budget by design — a quality ceiling.",
            },
            large: true,
        },
        // ------------------------------------------------------------ denoise
        Candidate {
            entry: CatalogEntry {
                id: "denoise-dpdfnet4",
                kind: ModelKind::Denoise,
                label: "DPDFNet-4 speech enhancement (ONNX, 16 kHz)",
                subdir: "denoise",
                url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet4.onnx",
                download_name: "dpdfnet4.onnx",
                archive: None,
                expect_relative: "dpdfnet4.onnx",
                // 11,707,436 bytes.
                approx_size_mb: 12,
                notes: "Bench candidate: runs on the installed sherpa-onnx-offline-denoiser.exe (v1.13.6) via --speech-denoiser-dpdfnet-model.",
            },
            large: false,
        },
    ]
}

pub fn find(id: &str) -> Option<Candidate> {
    candidates().into_iter().find(|c| c.entry.id == id)
}

/// Where a candidate's primary file lands (whether or not it is there yet).
pub fn path_of(data_dir: &Path, id: &str) -> Option<PathBuf> {
    find(id).map(|c| c.entry.path_in(&models_dir(data_dir)))
}

pub fn ids_of_kind(kind: ModelKind) -> Vec<&'static str> {
    candidates()
        .into_iter()
        .filter(|c| c.entry.kind == kind)
        .map(|c| c.entry.id)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_candidate_leaks_into_the_apps_catalog() {
        for candidate in candidates() {
            assert!(
                app_lib::models_manager::entry(candidate.entry.id).is_none(),
                "{} is in the app catalog — bench candidates must stay out of it",
                candidate.entry.id
            );
            assert!(!app_lib::models_manager::required_ids().contains(&candidate.entry.id));
        }
    }

    #[test]
    fn candidates_install_under_the_bench_tree() {
        let data = Path::new("C:\\data");
        for candidate in candidates() {
            let path = candidate.entry.path_in(&models_dir(data));
            assert!(
                path.starts_with(data.join("models").join("bench")),
                "{} would install at {}",
                candidate.entry.id,
                path.display()
            );
        }
    }

    #[test]
    fn llm_candidates_install_smallest_first_with_the_ceilings_last() {
        let llms: Vec<Candidate> = candidates()
            .into_iter()
            .filter(|c| c.entry.kind == ModelKind::Llm)
            .collect();
        for pair in llms.windows(2) {
            assert!(
                pair[0].entry.approx_size_mb <= pair[1].entry.approx_size_mb,
                "{} ({} MB) is listed before the smaller {} ({} MB)",
                pair[0].entry.id,
                pair[0].entry.approx_size_mb,
                pair[1].entry.id,
                pair[1].entry.approx_size_mb
            );
            assert!(
                !pair[0].large || pair[1].large,
                "default-set {} comes after the large {}",
                pair[1].entry.id,
                pair[0].entry.id
            );
        }
    }

    #[test]
    fn ids_are_unique_and_only_the_ceilings_are_large() {
        let all = candidates();
        let mut ids: Vec<&str> = all.iter().map(|c| c.entry.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), all.len());
        let large: Vec<&str> = all.iter().filter(|c| c.large).map(|c| c.entry.id).collect();
        assert_eq!(large, vec!["gemma4-12b-qat-q4", "gemma3-12b-q4"]);
    }
}
