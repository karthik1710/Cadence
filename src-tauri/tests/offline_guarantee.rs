//! Structural tests for Cadence's central promise: **nothing here talks to the
//! network except the one-time model download.**
//!
//! # What a green run here does and does not prove
//!
//! These tests are *static and structural*. They read the dependency graph, the
//! crate's own source, the frontend, and the Tauri configuration, and assert
//! that no code path exists which could reach off the machine. One test
//! ([`the_local_model_client_really_does_speak_only_to_loopback`]) is
//! behavioural, but it exercises a loopback socket, which is exactly the thing
//! that keeps working with the NIC switched off.
//!
//! So a green run proves: *no outbound code path is present*. Concretely — no
//! HTTP client crate is linked, the only absolute URLs in the crate are the
//! model catalog's, the inference-server client is hard-wired to `127.0.0.1`
//! and takes no host, the servers themselves are launched bound to loopback,
//! the frontend issues no requests of its own, and no Python/PyTorch runtime
//! is pulled in anywhere.
//!
//! A green run does **not** prove that the app *runs correctly* with the
//! network disconnected. That is a different claim — it covers whether Windows,
//! WebView2, WASAPI, the sqlite VFS, whisper.cpp, llama.cpp and sherpa-onnx all
//! behave when name resolution fails and the default route is gone, plus
//! whether every model is present locally. Nothing in a test process can
//! establish that; it needs a human running a full record → process → MoM cycle
//! on a machine with the adapter disabled. `docs/roadmap.md` risk 6 should
//! therefore be *narrowed* to that manual end-to-end check, not deleted.
//!
//! # Why these tests must not need a network themselves
//!
//! An offline-guarantee test that only passes on a connected machine would be
//! self-defeating. Everything below reads files inside the repo; the single
//! subprocess (`cargo tree`) is invoked with `--offline`, which means a cargo
//! that wants to hit the network fails loudly instead of quietly succeeding.
//!
//! # Why `cargo tree` and not `Cargo.lock`
//!
//! Parsing `Cargo.lock` would be more hermetic, and it was tried first — it is
//! wrong for this job. `Cargo.lock` is the union of every resolvable package
//! across all targets and *all optional features*, whether or not they are
//! enabled. This repo's lockfile contains `reqwest`, `hyper`, `tower-http` and
//! friends right now, none of which are linked: they are optional dependencies
//! somewhere under Tauri that this feature set never turns on. A lockfile-based
//! test would fail today for no real reason, and — worse — teach whoever fixes
//! it to add exceptions until the check means nothing. `cargo tree` resolves
//! features and the host target, so it answers the question actually being
//! asked: *what gets linked into the shipped binary?*
//!
//! # Extending this file
//!
//! Every scan below asserts it actually found something before asserting what
//! it found is clean — a scanner that silently matches zero files would pass
//! forever. [`the_scanners_in_this_file_are_not_vacuous`] goes further and
//! feeds the URL extractor and the crate matcher known-bad fixtures, so the
//! detectors themselves are covered. Keep both habits if you add a check.

use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::net::{IpAddr, Ipv4Addr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Locations. Everything is derived from CARGO_MANIFEST_DIR so the suite never
// depends on the working directory or on any path outside the repository.
// ---------------------------------------------------------------------------

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The repository root — `src-tauri/`'s parent.
fn repo_root() -> PathBuf {
    manifest_dir()
        .parent()
        .expect("src-tauri must have a parent directory")
        .to_path_buf()
}

/// A repo-root-relative, forward-slashed path, so failure messages read the
/// same on every machine.
fn rel(path: &Path) -> String {
    path.strip_prefix(repo_root())
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Recursively collects files under `dir` whose extension is in `extensions`,
/// skipping build output and vendored trees.
fn collect(dir: &Path, extensions: &[&str], out: &mut Vec<PathBuf>) {
    const SKIP: &[&str] = &[".git", "target", "node_modules", "dist", "gen"];
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.filter_map(Result::ok).collect();
    // Sorted so failure messages are stable run to run.
    entries.sort_by_key(std::fs::DirEntry::path);
    for entry in entries {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if path.is_dir() {
            if !SKIP.contains(&name.as_str()) {
                collect(&path, extensions, out);
            }
        } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if extensions.contains(&ext) {
                out.push(path);
            }
        }
    }
}

/// Every `.rs` file in the crate, as `(path, contents)`.
fn crate_rust_sources() -> Vec<(PathBuf, String)> {
    let mut paths = Vec::new();
    collect(&manifest_dir().join("src"), &["rs"], &mut paths);
    paths
        .into_iter()
        .map(|path| {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("could not read {}: {e}", rel(&path)));
            (path, text)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// URL handling. Deliberately hand-rolled: pulling in a URL crate to police the
// dependency graph would be its own small joke.
// ---------------------------------------------------------------------------

/// Pulls every `http://` / `https://` literal out of one line.
fn urls_in(line: &str) -> Vec<String> {
    // Terminators for a URL as it appears in source: quotes, brackets, the
    // usual sentence punctuation, and whitespace.
    const STOP: &[char] = &[
        '"', '\'', '`', ' ', '\t', '<', '>', ')', ']', '}', ',', ';', '\\',
    ];
    let mut found = Vec::new();
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let rest = &line[i..];
        if rest.starts_with("http://") || rest.starts_with("https://") {
            let end = rest.find(STOP).unwrap_or(rest.len());
            let url = rest[..end].trim_end_matches('.').to_string();
            i += end.max(1);
            found.push(url);
        } else {
            i += 1;
            // Stay on a char boundary in files that contain Tamil or box
            // drawing characters.
            while i < bytes.len() && !line.is_char_boundary(i) {
                i += 1;
            }
        }
    }
    found
}

/// The host component of a URL: strips the scheme, any userinfo, the path, and
/// the port. Keeps IPv6 brackets so `[::1]` stays recognisable.
fn host_of(url: &str) -> String {
    let after_scheme = url.split_once("://").map(|(_, rest)| rest).unwrap_or(url);
    let authority = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(after_scheme);
    let authority = authority
        .rsplit_once('@')
        .map(|(_, host)| host)
        .unwrap_or(authority);
    if let Some(close) = authority.find(']') {
        return authority[..=close].to_ascii_lowercase();
    }
    authority
        .split_once(':')
        .map(|(host, _)| host)
        .unwrap_or(authority)
        .to_ascii_lowercase()
}

/// True only for the loopback interface.
///
/// The address is *parsed*, never prefix-matched. A `starts_with("127.")` check
/// was the first version of this and it accepted `127.0.0.1.evil.example` — a
/// perfectly registrable public hostname that would have walked straight
/// through the URL scan below. `localhost` is accepted as an exact string
/// because it is not an address; anything else must parse as an IP and answer
/// yes to [`IpAddr::is_loopback`].
fn is_loopback_host(host: &str) -> bool {
    if host == "localhost" {
        return true;
    }
    let bare = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host);
    bare.parse::<IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(false)
}

/// True for a line that is purely a comment. A URL in a comment cannot be
/// dialled, and forbidding them would push contributors to strip the citations
/// that make this codebase's decisions auditable — so the URL scan targets
/// code lines. A URL trailing live code on the same line is still treated as
/// code, which is the conservative reading.
fn is_comment_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with('*')
}

// ---------------------------------------------------------------------------
// 1. No HTTP / networking client crate is linked.
// ---------------------------------------------------------------------------

/// Crates that must never appear in the linked dependency graph, each with the
/// reason a future contributor needs to read.
///
/// Note what is deliberately *absent* from this list: `http`, `url` and `tokio`
/// are all in the graph today, pulled in by Tauri. They are type definitions
/// and an async runtime, not clients — nothing in them opens a connection on
/// its own. The list names transports and clients, because those are what
/// would actually create an outbound path.
const FORBIDDEN_CRATES: &[(&str, &str)] = &[
    ("reqwest", "a general-purpose HTTP client — the exact thing Cadence must not link"),
    ("hyper", "an HTTP client/server stack; if it is here, something is speaking HTTP off-loopback"),
    ("hyper-util", "only pulled in alongside a hyper-based client"),
    ("h2", "HTTP/2 transport — only reached via a real HTTP client"),
    ("isahc", "an HTTP client (libcurl-backed)"),
    ("ureq", "a blocking HTTP client"),
    ("attohttpc", "a small HTTP client"),
    ("minreq", "a minimal HTTP client"),
    ("surf", "an HTTP client"),
    ("awc", "the actix HTTP client"),
    ("actix-web", "an HTTP server framework"),
    ("axum", "an HTTP server framework — Cadence has no server surface"),
    ("warp", "an HTTP server framework"),
    ("tiny_http", "an HTTP server"),
    ("curl", "libcurl bindings; model downloads shell out to Windows' curl.exe on purpose, precisely so this crate is not linked"),
    ("curl-sys", "libcurl bindings (see curl)"),
    ("tungstenite", "a WebSocket client"),
    ("tokio-tungstenite", "a WebSocket client"),
    ("websocket", "a WebSocket client"),
    ("native-tls", "a TLS stack — only needed by something making outbound TLS connections"),
    ("openssl", "a TLS stack (see native-tls)"),
    ("openssl-sys", "a TLS stack (see native-tls)"),
    ("rustls", "a TLS stack (see native-tls)"),
    ("tokio-rustls", "a TLS stack (see native-tls)"),
    ("hickory-resolver", "a DNS resolver — Cadence resolves no names"),
    ("trust-dns-resolver", "a DNS resolver (see hickory-resolver)"),
    ("sentry", "crash/error telemetry — Cadence sends no telemetry"),
    ("opentelemetry", "telemetry export (see sentry)"),
    ("tracing-opentelemetry", "telemetry export (see sentry)"),
    ("posthog-rs", "product analytics (see sentry)"),
    ("aws-config", "a cloud SDK — no cloud services, ever"),
    ("azure_core", "a cloud SDK (see aws-config)"),
    ("google-cloud-storage", "a cloud SDK (see aws-config)"),
    ("async-openai", "a cloud LLM client — the hardest constraint in the project"),
    ("openai-api-rs", "a cloud LLM client (see async-openai)"),
    ("anthropic-sdk", "a cloud LLM client (see async-openai)"),
];

/// Runs `cargo tree` and returns the set of package names actually linked.
///
/// `--edges normal,build` covers both what ships in the binary and what runs at
/// build time; dev-dependencies are excluded because they never reach a user's
/// machine. `--offline` keeps this test honest — a cargo that would need the
/// network errors out instead of silently going and getting something.
fn linked_crate_names() -> BTreeSet<String> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let output = Command::new(&cargo)
        .arg("tree")
        .arg("--offline")
        .arg("--edges")
        .arg("normal,build")
        .arg("--prefix")
        .arg("none")
        .arg("--format")
        .arg("{p}")
        .arg("--manifest-path")
        .arg(manifest_dir().join("Cargo.toml"))
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "could not run `{cargo} tree`: {e}\n\
                 This test resolves the real dependency graph rather than reading \
                 Cargo.lock (see this file's module docs for why). It needs a working \
                 cargo and a populated registry cache; run `cargo fetch` once on a \
                 connected machine if this is a fresh checkout."
            )
        });
    assert!(
        output.status.success(),
        "`{cargo} tree --offline` failed ({}):\n{}\n\
         If this says the registry index is missing, run `cargo fetch` once while \
         connected. The test itself never touches the network.",
        output.status,
        String::from_utf8_lossy(&output.stderr).trim()
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let names: BTreeSet<String> = stdout
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .collect();

    // Non-vacuity: this crate at minimum depends on tauri, cpal, hound,
    // rusqlite, serde and sha2. A parse that produced a handful of names would
    // mean the output format changed under us.
    assert!(
        names.len() > 50,
        "`cargo tree` produced only {} package names — the output format probably \
         changed and this test is no longer looking at the real graph. Output was:\n{stdout}",
        names.len()
    );
    for expected in ["app", "tauri", "cpal", "hound", "rusqlite", "serde", "sha2"] {
        assert!(
            names.contains(expected),
            "`cargo tree` output is missing `{expected}`, so it is not the graph for this crate"
        );
    }
    names
}

#[test]
fn no_http_or_networking_client_crate_is_linked_into_the_app() {
    let linked = linked_crate_names();
    let mut offenders = Vec::new();
    for (name, reason) in FORBIDDEN_CRATES {
        if linked.contains(*name) {
            offenders.push(format!("  - {name}: {reason}"));
        }
    }
    assert!(
        offenders.is_empty(),
        "Forbidden networking crate(s) are now linked into Cadence:\n{}\n\n\
         Cadence's hard constraint is no cloud STT, no cloud LLM, no external AI \
         APIs, no telemetry and no networked backend (CLAUDE.md). An empty HTTP \
         stack in `cargo tree` is the cheap structural proof of that, and this \
         test is what keeps it true.\n\n\
         There are exactly two sanctioned exceptions, and neither adds a crate:\n\
           * model downloads shell out to Windows' bundled curl.exe/tar.exe \
             (src-tauri/src/models_manager.rs);\n\
           * the local inference servers are driven by a hand-written, \
             loopback-only HTTP/1.1 client (src-tauri/src/pipeline/http.rs).\n\n\
         If you need to talk to a local server, extend pipeline/http.rs. If you \
         genuinely believe a network crate belongs here, that is an architecture \
         decision: change docs/architecture.md and CLAUDE.md first, then this test.",
        offenders.join("\n")
    );
}

// ---------------------------------------------------------------------------
// 2. The only outbound URLs in the crate are the model catalog's.
// ---------------------------------------------------------------------------

/// The upstream projects Cadence's runtimes and weights actually come from.
/// Adding a host here means adding a party that can serve code to users'
/// machines, so it is a deliberate decision, not a config tweak.
const ALLOWED_MODEL_HOSTS: &[&str] = &["github.com", "huggingface.co"];

/// `bench_cli`'s candidate catalog, the one other file allowed to carry
/// download URLs. It lives inside the dev-only bench binary on purpose, so the
/// candidates never enter the app's catalog, Settings or setup screen.
fn is_bench_catalog(path: &Path) -> bool {
    path.to_string_lossy()
        .replace('\\', "/")
        .ends_with("src/bin/bench_cli/bench/catalog.rs")
}

#[test]
fn the_only_outbound_urls_in_the_crate_source_are_the_model_download_catalogs() {
    let sources = crate_rust_sources();
    assert!(
        sources.len() > 25,
        "only {} Rust files found under src-tauri/src — the scan is not seeing the crate",
        sources.len()
    );

    let mut offenders = Vec::new();
    let mut catalog_urls = 0usize;
    let mut bench_urls = 0usize;
    let mut loopback_urls = 0usize;

    for (path, text) in &sources {
        let is_catalog = path.file_name().and_then(|n| n.to_str()) == Some("models_manager.rs");
        let is_bench = is_bench_catalog(path);
        for (index, line) in text.lines().enumerate() {
            if is_comment_line(line) {
                continue;
            }
            for url in urls_in(line) {
                let host = host_of(&url);
                if is_loopback_host(&host) {
                    loopback_urls += 1;
                } else if is_bench {
                    // The bench candidates feed a dev-only binary that never
                    // ships, but they still put weights on a developer's
                    // machine, so they meet the app catalog's bar: TLS, and a
                    // sanctioned publisher.
                    if url.starts_with("https://") && ALLOWED_MODEL_HOSTS.contains(&host.as_str()) {
                        bench_urls += 1;
                    } else {
                        offenders.push(format!(
                            "  {}:{} -> {url} (bench catalog URLs must be https on {ALLOWED_MODEL_HOSTS:?})",
                            rel(path),
                            index + 1
                        ));
                    }
                } else if is_catalog {
                    catalog_urls += 1;
                } else {
                    offenders.push(format!("  {}:{} -> {url}", rel(path), index + 1));
                }
            }
        }
    }

    // Non-vacuity: the catalog really does carry a pile of https URLs. If this
    // ever drops to zero, the scanner has stopped matching and every other
    // assertion in this test is worthless.
    assert!(
        catalog_urls >= 8,
        "expected the model catalog in models_manager.rs to contain at least 8 download \
         URLs, found {catalog_urls} — the URL scanner is no longer matching anything, \
         so this test would pass no matter what was added elsewhere"
    );

    assert!(
        offenders.is_empty(),
        "Absolute URL(s) found in crate source outside the model download catalog:\n{}\n\n\
         Cadence may reach the network in exactly one place: the one-time model \
         install in src-tauri/src/models_manager.rs, which shells out to curl.exe. \
         Everywhere else, a hard-coded remote URL is the regression this test \
         exists to catch — the app must run with the Internet fully disconnected \
         once models are installed.\n\n\
         Loopback URLs (127.0.0.0/8, localhost, [::1]) are allowed anywhere: that \
         is how the local whisper.cpp and llama.cpp servers are addressed. \
         ({loopback_urls} loopback, {catalog_urls} catalog and {bench_urls} bench-catalog \
         URLs were accepted. The bench catalog, src/bin/bench_cli/bench/catalog.rs, holds \
         comparison candidates for the dev-only bench_cli, installed through the \
         same curl.exe-based installer.)\n\n\
         A URL on its own comment line is also allowed — it cannot be dialled. If \
         one of the hits above is a citation, move it onto its own `//` line.",
        offenders.join("\n")
    );
}

#[test]
fn every_model_download_url_is_https_and_points_at_a_known_publisher() {
    // Asserted against the real catalog rather than its source text, so a URL
    // built at runtime could not slip past a grep.
    let catalog = app_lib::models_manager::catalog();
    assert!(
        catalog.len() >= 8,
        "the model catalog has only {} entries — is this reading the real catalog?",
        catalog.len()
    );

    const ALLOWED_HOSTS: &[&str] = ALLOWED_MODEL_HOSTS;

    for entry in catalog {
        assert!(
            entry.url.starts_with("https://"),
            "catalog entry `{}` downloads over a non-TLS URL: {}. \
             Model downloads are trust-on-first-install (no pinned digests — see \
             models_manager.rs), so TLS is the only thing standing between a user \
             and a substituted binary.",
            entry.id,
            entry.url
        );
        let host = host_of(entry.url);
        assert!(
            ALLOWED_HOSTS.contains(&host.as_str()),
            "catalog entry `{}` downloads from `{host}` ({}), which is not one of \
             the sanctioned publishers {ALLOWED_HOSTS:?}. Every host here can hand a \
             native binary or model weights to a user's machine; adding one is an \
             architecture decision that belongs in docs/architecture.md.",
            entry.id,
            entry.url
        );
    }
}

// ---------------------------------------------------------------------------
// 3. The local model client is loopback-only.
// ---------------------------------------------------------------------------

/// Everything in `std::net` that can create or accept a connection.
const SOCKET_CONSTRUCTORS: &[&str] = &[
    "TcpStream::connect",
    "TcpListener::bind",
    "UdpSocket::bind",
    "UdpSocket::connect",
    "connect_timeout",
    "ToSocketAddrs",
];

#[test]
fn the_only_module_that_opens_a_socket_is_the_loopback_model_client() {
    let sources = crate_rust_sources();
    let mut sites = Vec::new();

    for (path, text) in &sources {
        for (index, line) in text.lines().enumerate() {
            if is_comment_line(line) {
                continue;
            }
            if SOCKET_CONSTRUCTORS.iter().any(|c| line.contains(c)) {
                sites.push((rel(path), index + 1, line.trim().to_string()));
            }
        }
    }

    assert!(
        sites.len() >= 2,
        "no socket construction found anywhere in the crate. pipeline/http.rs is \
         supposed to connect to the local inference servers and reserve a free \
         loopback port, so finding fewer than two sites means this scan is broken \
         and would not catch a new outbound socket either. Found: {sites:?}"
    );

    let strays: Vec<_> = sites
        .iter()
        .filter(|(file, _, _)| file != "src-tauri/src/pipeline/http.rs")
        .map(|(file, line, code)| format!("  {file}:{line}  {code}"))
        .collect();

    assert!(
        strays.is_empty(),
        "Socket(s) opened outside src-tauri/src/pipeline/http.rs:\n{}\n\n\
         All local IPC with whisper-server / llama-server goes through \
         pipeline/http.rs, which hard-codes Ipv4Addr::LOCALHOST and takes a port \
         rather than a host — that single choke point is what makes \
         \"loopback only\" checkable at all. A socket opened anywhere else is \
         unreviewed network surface. Route it through pipeline/http.rs, or if it \
         is genuinely a new kind of local IPC, extend that module and this test \
         together.",
        strays.join("\n")
    );
}

#[test]
fn the_loopback_model_client_hard_codes_the_loopback_address_and_takes_no_host() {
    let path = manifest_dir().join("src").join("pipeline").join("http.rs");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read {}: {e}", rel(&path)));

    // Every socket site in this module must sit within a few lines of an
    // explicit loopback address, so the address cannot come from elsewhere.
    let lines: Vec<&str> = text.lines().collect();
    let mut checked = 0usize;
    for (index, line) in lines.iter().enumerate() {
        if is_comment_line(line) {
            continue;
        }
        if !SOCKET_CONSTRUCTORS.iter().any(|c| line.contains(c)) {
            continue;
        }
        let start = index.saturating_sub(3);
        let end = (index + 4).min(lines.len());
        let window = lines[start..end].join("\n");
        assert!(
            window.contains("Ipv4Addr::LOCALHOST")
                || window.contains("127.0.0.1")
                || window.contains("Ipv6Addr::LOCALHOST"),
            "{}:{} opens a socket without a literal loopback address nearby:\n{window}\n\n\
             pipeline/http.rs is the one module allowed to open a socket, and it is \
             allowed only because the address is a compile-time constant. An address \
             that arrives from a variable is an injection point.",
            rel(&path),
            index + 1
        );
        checked += 1;
    }
    assert!(
        checked >= 2,
        "found {checked} socket sites in pipeline/http.rs; expected at least the \
         connect in request_bytes() and the bind in free_local_port(). This check \
         has stopped matching."
    );

    // No function here may accept a host, URL or address as a string: the public
    // surface takes a `u16` port, full stop.
    const INJECTABLE_PARAMS: &[&str] = &[
        "host: &str",
        "host: String",
        "host: &String",
        "url: &str",
        "url: String",
        "base_url",
        "endpoint: &str",
        "endpoint: String",
        "addr: &str",
        "addr: String",
        "address: &str",
        "address: String",
        "addr: SocketAddr",
        "address: SocketAddr",
        "host: impl",
    ];
    for needle in INJECTABLE_PARAMS {
        assert!(
            !text.contains(needle),
            "pipeline/http.rs now contains `{needle}`, which means a caller can choose \
             what the client connects to. The whole reason this hand-written client is \
             acceptable in a no-network app is that `request`/`request_bytes` take a \
             port and nothing else — the destination is not configurable, so it cannot \
             be pointed off-machine even by accident, by a settings value, or by a \
             malformed model config. Keep the port-only signature."
        );
    }

    assert!(
        text.contains("Ipv4Addr::LOCALHOST"),
        "pipeline/http.rs no longer names Ipv4Addr::LOCALHOST at all"
    );
}

#[test]
fn the_local_model_client_really_does_speak_only_to_loopback() {
    // Behavioural counterpart to the static checks: stand up a listener on
    // 127.0.0.1, ask the client for that port, and confirm the connection
    // arrives from loopback with a loopback Host header. This needs no network
    // — loopback is exactly what still works with the adapter disabled, which
    // is the point.
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind loopback listener");
    let port = listener.local_addr().expect("local_addr").port();

    let server = thread::spawn(move || {
        let (mut socket, peer) = listener.accept().expect("accept");
        let mut request = Vec::new();
        let mut buffer = [0u8; 2048];
        loop {
            let read = socket.read(&mut buffer).expect("read request");
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..read]);
            if request.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }
        let body = "{\"status\":\"ok\"}";
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        socket
            .write_all(response.as_bytes())
            .expect("write response");
        socket.flush().ok();
        drop(socket);
        (peer, String::from_utf8_lossy(&request).to_string())
    });

    let response =
        app_lib::pipeline::http::request(port, "GET", "/health", None, Duration::from_secs(10))
            .expect("the loopback client should reach a loopback listener");

    let (peer, request_head) = server.join().expect("server thread");

    assert!(
        peer.ip().is_loopback(),
        "the client connected from {peer}, which is not a loopback address"
    );
    assert!(
        request_head.contains("Host: 127.0.0.1:"),
        "the client sent a non-loopback Host header:\n{request_head}"
    );
    assert_eq!(response.status, 200);
    assert!(response.is_success());
    assert!(response.body.contains("\"ok\""));
}

#[test]
fn every_local_inference_server_is_launched_bound_to_loopback() {
    // whisper-server and llama-server are ordinary HTTP servers. Launched with
    // their defaults, or with `--host 0.0.0.0`, they would be reachable from
    // any machine on the same network — a meeting's audio and transcript
    // exposed to the LAN. Every spawn site must pin the bind address.
    let sources = crate_rust_sources();
    let mut sites = 0usize;
    let mut offenders = Vec::new();

    for (path, text) in &sources {
        let lines: Vec<&str> = text.lines().collect();
        for (index, line) in lines.iter().enumerate() {
            if is_comment_line(line) || !line.contains("--host") {
                continue;
            }
            sites += 1;
            // The address may be on this line (`.args(["--host", "127.0.0.1"])`)
            // or on one of the next couple (`.arg("--host").arg("127.0.0.1")`
            // split across lines by rustfmt).
            let end = (index + 3).min(lines.len());
            let window = lines[index..end].join(" ");
            if !(window.contains("127.0.0.1") || window.contains("localhost")) {
                offenders.push(format!("  {}:{}  {}", rel(path), index + 1, line.trim()));
            }
        }
    }

    assert!(
        sites >= 2,
        "found {sites} `--host` spawn arguments; expected at least the llama-server \
         and whisper-server launches. If a local server is now started without an \
         explicit --host it is binding to that binary's default, which for \
         llama.cpp has historically been reachable beyond loopback."
    );
    assert!(
        offenders.is_empty(),
        "Local server(s) launched without pinning the bind address to loopback:\n{}\n\n\
         Pass `--host 127.0.0.1`. These servers hold the meeting audio and the \
         transcript; bound to anything else they are readable by every machine on \
         the network, which breaks the local-only promise just as thoroughly as a \
         cloud API would.",
        offenders.join("\n")
    );
}

// ---------------------------------------------------------------------------
// 4. No Python / PyTorch runtime.
// ---------------------------------------------------------------------------

#[test]
fn nothing_pulls_in_a_python_or_pytorch_runtime() {
    // CLAUDE.md: "the shipped app must not depend on a Python/PyTorch runtime".
    // The pull towards one is real — faster-whisper and pyannote-audio are the
    // obvious implementations of two pipeline stages — so this is checked three
    // ways: the linked graph, the crate source, and the repo's own files.
    const FORBIDDEN_CRATES: &[(&str, &str)] = &[
        ("pyo3", "embeds a Python interpreter"),
        ("cpython", "embeds a Python interpreter"),
        ("rust-cpython", "embeds a Python interpreter"),
        ("python3-sys", "links libpython"),
        ("inline-python", "embeds a Python interpreter"),
        ("tch", "libtorch (PyTorch) bindings"),
        ("torch-sys", "libtorch (PyTorch) bindings"),
        ("tensorflow", "the TensorFlow runtime"),
    ];
    let linked = linked_crate_names();
    for (name, reason) in FORBIDDEN_CRATES {
        assert!(
            !linked.contains(*name),
            "`{name}` is linked ({reason}). Cadence's AI runtime is Rust-orchestrated \
             native binaries only — whisper.cpp, llama.cpp and ONNX-based \
             diarization/VAD. A Python or PyTorch runtime would multiply the \
             installer size, break the single-binary install story, and add a large \
             second supply chain to a privacy-critical app."
        );
    }

    // No shelling out to an interpreter, either.
    const FORBIDDEN_SPAWNS: &[&str] = &[
        "\"python\"",
        "\"python3\"",
        "\"python.exe\"",
        "\"py.exe\"",
        "\"pythonw.exe\"",
        "pip install",
        "conda ",
    ];
    for (path, text) in crate_rust_sources() {
        for (index, line) in text.lines().enumerate() {
            if is_comment_line(line) {
                continue;
            }
            for needle in FORBIDDEN_SPAWNS {
                assert!(
                    !line.contains(needle),
                    "{}:{} references a Python runtime (`{needle}`):\n  {}\n\n\
                     Pipeline stages must call native binaries/libraries, not an \
                     interpreter that would have to be installed alongside the app.",
                    rel(&path),
                    index + 1,
                    line.trim()
                );
            }
        }
    }

    // And nothing in the repo declares a Python project.
    let mut python_files = Vec::new();
    collect(&repo_root(), &["py", "pyc", "ipynb"], &mut python_files);
    assert!(
        python_files.is_empty(),
        "Python source file(s) committed to the repo: {:?}. Even a helper script \
         here becomes something a contributor has to have Python installed to run, \
         and the first step towards a Python pipeline stage. Scripts in this repo \
         are PowerShell (see scripts/setup-models.ps1).",
        python_files.iter().map(|p| rel(p)).collect::<Vec<_>>()
    );
    for marker in [
        "requirements.txt",
        "pyproject.toml",
        "Pipfile",
        "environment.yml",
        "setup.py",
    ] {
        let path = repo_root().join(marker);
        assert!(
            !path.exists(),
            "{marker} exists at the repo root — Cadence ships no Python runtime"
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Additional surfaces that could reach the network.
// ---------------------------------------------------------------------------

#[test]
fn the_only_network_capable_program_the_app_shells_out_to_is_the_model_downloader() {
    // The dependency graph proves nothing about `Command::new`. Windows ships
    // several programs that will happily fetch a URL, and any of them would be
    // an outbound path that `cargo tree` cannot see.
    const NETWORK_CAPABLE: &[(&str, &str)] = &[
        ("curl.exe", "fetches URLs"),
        ("wget", "fetches URLs"),
        ("bitsadmin", "fetches URLs via the BITS service"),
        (
            "certutil",
            "`certutil -urlcache` is a well-known download primitive",
        ),
        ("Invoke-WebRequest", "PowerShell's HTTP client"),
        ("Invoke-RestMethod", "PowerShell's HTTP client"),
        ("Start-BitsTransfer", "PowerShell's BITS download cmdlet"),
        ("nslookup", "performs DNS queries"),
        ("ftp.exe", "transfers files over the network"),
        ("ssh.exe", "opens a network connection"),
        ("scp.exe", "transfers files over the network"),
    ];

    let mut curl_sites = 0usize;
    let mut offenders = Vec::new();

    for (path, text) in crate_rust_sources() {
        let is_installer = path.file_name().and_then(|n| n.to_str()) == Some("models_manager.rs");
        for (index, line) in text.lines().enumerate() {
            if is_comment_line(line) {
                continue;
            }
            for (program, reason) in NETWORK_CAPABLE {
                if !line.contains(program) {
                    continue;
                }
                if is_installer && *program == "curl.exe" {
                    curl_sites += 1;
                    continue;
                }
                offenders.push(format!(
                    "  {}:{}  {}  ({program}: {reason})",
                    rel(&path),
                    index + 1,
                    line.trim()
                ));
            }
        }
    }

    assert!(
        curl_sites >= 1,
        "models_manager.rs no longer mentions curl.exe. That shell-out is the \
         sanctioned download path and the reason no HTTP crate is linked; if it \
         moved, this scan needs to move with it, because right now it would \
         accept a downloader anywhere in the crate."
    );
    assert!(
        offenders.is_empty(),
        "Network-capable program(s) invoked outside the model installer:\n{}\n\n\
         Model downloads are the one sanctioned outbound path, they live in \
         src-tauri/src/models_manager.rs, and they run only when the user asks for \
         an install — never during recording, processing, or MoM generation. A \
         download primitive anywhere else is an outbound path that the dependency \
         graph cannot show and a reviewer would not think to look for.",
        offenders.join("\n")
    );
}

#[test]
fn the_frontend_issues_no_requests_of_its_own() {
    // The Rust side is only half the app. A `fetch()` in the WebView would be a
    // cloud call that never touches Rust and that no `cargo tree` check could
    // see. The UI talks to Rust exclusively through Tauri's `invoke` IPC.
    const FORBIDDEN: &[(&str, &str)] = &[
        ("XMLHttpRequest", "an HTTP request from the WebView"),
        ("WebSocket(", "a WebSocket connection from the WebView"),
        (
            "EventSource(",
            "a server-sent-events connection from the WebView",
        ),
        ("sendBeacon", "a fire-and-forget telemetry POST"),
        ("navigator.geolocation", "a location lookup"),
        ("@tauri-apps/plugin-http", "Tauri's HTTP plugin"),
        ("plugin-websocket", "Tauri's WebSocket plugin"),
    ];

    let mut files = Vec::new();
    collect(
        &repo_root().join("src"),
        &["ts", "tsx", "js", "jsx", "css"],
        &mut files,
    );
    files.push(repo_root().join("index.html"));
    assert!(
        files.len() > 10,
        "only {} frontend files found — the scan is not seeing the UI",
        files.len()
    );

    let mut offenders = Vec::new();
    for path in &files {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        for (index, line) in text.lines().enumerate() {
            let trimmed = line.trim_start();
            // Line comments in TS/CSS/HTML, same reasoning as the Rust scan.
            if trimmed.starts_with("//") || trimmed.starts_with("*") || trimmed.starts_with("<!--")
            {
                continue;
            }
            for (needle, reason) in FORBIDDEN {
                if line.contains(needle) {
                    offenders.push(format!(
                        "  {}:{}  {}  ({reason})",
                        rel(path),
                        index + 1,
                        line.trim()
                    ));
                }
            }
            // `fetch(` needs a word boundary so `prefetch(`/`refetch(` do not
            // trip it.
            if let Some(at) = line.find("fetch(") {
                let preceding = line[..at].chars().next_back().unwrap_or(' ');
                if !preceding.is_alphanumeric() && preceding != '_' && preceding != '.' {
                    offenders.push(format!(
                        "  {}:{}  {}  (fetch(): an HTTP request from the WebView)",
                        rel(path),
                        index + 1,
                        line.trim()
                    ));
                }
            }
            for url in urls_in(line) {
                if !is_loopback_host(&host_of(&url)) {
                    offenders.push(format!("  {}:{}  remote URL {url}", rel(path), index + 1));
                }
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "The frontend now reaches the network directly:\n{}\n\n\
         Every remote font, CDN script, analytics snippet or fetch() in the WebView \
         is an outbound call that bypasses Rust entirely — invisible to `cargo \
         tree`, invisible to a review of src-tauri/, and enough on its own to break \
         \"works with the Internet disconnected\". The UI's only channel is Tauri \
         `invoke`; assets are bundled under public/.",
        offenders.join("\n")
    );
}

#[test]
fn the_tauri_configuration_grants_no_network_capability() {
    let config_path = manifest_dir().join("tauri.conf.json");
    let config = std::fs::read_to_string(&config_path).expect("read tauri.conf.json");

    // Textual rather than parsed, to keep this test std-only and to catch the
    // key wherever it is nested.
    for (needle, reason) in [
        (
            "\"updater\"",
            "the auto-updater polls a remote endpoint on every launch",
        ),
        (
            "\"createUpdaterArtifacts\"",
            "updater artifacts imply an update endpoint",
        ),
        (
            "plugin-http",
            "Tauri's HTTP plugin is a general-purpose network client",
        ),
    ] {
        assert!(
            !config.contains(needle),
            "tauri.conf.json now contains `{needle}` — {reason}. Cadence sends no \
             telemetry and must run fully disconnected; updates are a manual \
             installer download."
        );
    }

    // `devUrl` legitimately points at the Vite dev server, and `$schema` at the
    // Tauri JSON schema (an editor hint, never fetched at runtime). Anything
    // else with a scheme is a remote resource the app would load.
    for (index, line) in config.lines().enumerate() {
        for url in urls_in(line) {
            let host = host_of(&url);
            // Tauri intercepts these protocol origins inside the webview;
            // only CSP directives may name them, never a window/dev URL.
            let csp_line = line.trim_start().starts_with("\"csp\":")
                || line.trim_start().starts_with("\"devCsp\":");
            let internal_protocol = csp_line
                && matches!(host.as_str(), "asset.localhost" | "ipc.localhost");
            let allowed = is_loopback_host(&host)
                || (line.contains("$schema") && host == "schema.tauri.app")
                || internal_protocol;
            assert!(
                allowed,
                "tauri.conf.json:{} points at a remote resource: {url}\n\
                 A remote frontendDist, devUrl or window URL would make the app load \
                 code over the network at runtime.",
                index + 1
            );
        }
    }

    // Capabilities are the allowlist of what the WebView may ask Rust to do.
    let capabilities_dir = manifest_dir().join("capabilities");
    let mut capability_files = Vec::new();
    collect(&capabilities_dir, &["json"], &mut capability_files);
    assert!(
        !capability_files.is_empty(),
        "no capability files found under {} — this check would pass vacuously",
        rel(&capabilities_dir)
    );
    for path in capability_files {
        let text = std::fs::read_to_string(&path).expect("read capability file");
        for prefix in ["\"http:", "\"websocket:", "\"upload:"] {
            assert!(
                !text.contains(prefix),
                "{} grants a `{prefix}..` permission, handing the WebView a network \
                 primitive. The frontend needs no network permission at all: it \
                 talks only to Cadence's own commands.",
                rel(&path)
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Self-test: the detectors above actually detect.
// ---------------------------------------------------------------------------

#[test]
fn the_scanners_in_this_file_are_not_vacuous() {
    // Every check above is a scan, and a scan that quietly matches nothing
    // passes forever while the guarantee rots. Each test asserts a minimum hit
    // count against real files; this one runs the primitives against fixtures,
    // so a refactor that breaks extraction fails here loudly instead of turning
    // the whole suite green-but-meaningless.

    // URL extraction, from the shapes these actually appear in.
    let planted = r#"    let endpoint = "https://evil.example/v1/telemetry?id=1";"#;
    let found = urls_in(planted);
    assert_eq!(found, vec!["https://evil.example/v1/telemetry?id=1"]);
    assert_eq!(host_of(&found[0]), "evil.example");
    assert!(!is_loopback_host(&host_of(&found[0])));
    assert!(!is_comment_line(planted));

    // Two URLs on one line, and one with credentials and a port.
    let two = "get(\"http://a.example:8080/x\"); get(\"https://user:pw@b.example/y\");";
    let urls = urls_in(two);
    assert_eq!(urls.len(), 2);
    assert_eq!(host_of(&urls[0]), "a.example");
    assert_eq!(host_of(&urls[1]), "b.example");

    // Loopback in every spelling the codebase might use.
    for url in [
        "http://127.0.0.1:8080/completion",
        "http://localhost:1420",
        "http://[::1]:9000/health",
    ] {
        assert!(
            is_loopback_host(&host_of(url)),
            "{url} should be recognised as loopback"
        );
    }
    // Hosts that only *look* like loopback. The first of these was a real bug
    // in this file's first draft: a `starts_with("127.")` check accepted it,
    // and `127.0.0.1.evil.example` is a name anyone can register. The second
    // is the classic userinfo trick — the authority is `evil.example`, and the
    // `127.0.0.1` before the `@` is a username.
    for spoof in [
        "https://127.0.0.1.evil.example/",
        "https://127.0.0.1@evil.example/",
        "https://localhost.evil.example/",
        "https://0x7f000001.evil.example/",
    ] {
        assert!(
            !is_loopback_host(&host_of(spoof)),
            "{spoof} resolves off-machine but was classified as loopback; the URL \
             scan would wave it through"
        );
    }

    // Comment detection, which is the one deliberate hole in the URL scan.
    assert!(is_comment_line("    // see https://example.com/issue/1"));
    assert!(is_comment_line("//! https://example.com"));
    assert!(is_comment_line("     * https://example.com"));
    assert!(!is_comment_line("let x = 1; // https://example.com"));

    // The crate matcher: a forbidden name must actually be recognised as such.
    let pretend_graph: BTreeSet<String> = ["serde", "reqwest", "tauri"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let caught: Vec<&str> = FORBIDDEN_CRATES
        .iter()
        .filter(|(name, _)| pretend_graph.contains(*name))
        .map(|(name, _)| *name)
        .collect();
    assert_eq!(
        caught,
        vec!["reqwest"],
        "the forbidden-crate matcher failed to flag reqwest in a graph containing it"
    );

    // And the file walker: it must find the crate, and must skip target/.
    let mut rust = Vec::new();
    collect(&manifest_dir().join("src"), &["rs"], &mut rust);
    assert!(
        rust.len() > 25,
        "walker found only {} .rs files",
        rust.len()
    );
    assert!(
        rust.iter().all(|p| !rel(p).contains("/target/")),
        "the walker descended into target/, which would scan vendored source"
    );
}
#[test]
fn production_csp_blocks_remote_content_and_keeps_local_playback() {
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(manifest_dir().join("tauri.conf.json")).unwrap(),
    )
    .unwrap();
    let csp = config["app"]["security"]["csp"]
        .as_str()
        .expect("production CSP must be enabled");
    let directives: std::collections::BTreeMap<_, _> = csp
        .split(';')
        .filter_map(|part| {
            let mut words = part.split_whitespace();
            Some((words.next()?, words.collect::<Vec<_>>()))
        })
        .collect();
    assert_eq!(directives["default-src"], vec!["'self'"]);
    assert_eq!(directives["script-src"], vec!["'self'"]);
    assert_eq!(directives["style-src"], vec!["'self'", "'unsafe-inline'"]);
    assert_eq!(directives["img-src"], vec!["'self'", "data:"]);
    assert_eq!(
        directives["connect-src"],
        vec!["'self'", "ipc:", "http://ipc.localhost"]
    );
    assert_eq!(
        directives["media-src"],
        vec!["'self'", "asset:", "http://asset.localhost"]
    );
    for name in ["object-src", "base-uri", "frame-src", "form-action"] {
        assert_eq!(directives[name], vec!["'none'"]);
    }
}

#[test]
fn development_csp_keeps_hot_reload_permissions_out_of_production() {
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(manifest_dir().join("tauri.conf.json")).unwrap(),
    ).unwrap();
    let security = &config["app"]["security"];
    let production = security["csp"].as_str().unwrap();
    let development = security["devCsp"].as_str().unwrap();
    assert!(development.contains("ws://localhost:1420"));
    assert!(development.contains("script-src 'self' 'unsafe-inline'"));
    assert!(!production.contains("ws:"));
    assert!(!production.contains("'unsafe-eval'"));
    for policy in [production, development] {
        for directive in ["object-src 'none'", "frame-src 'none'", "form-action 'none'"] {
            assert!(policy.contains(directive), "missing {directive}");
        }
        assert!(!policy.split_whitespace().any(|word| word == "*"));
    }
}

#[test]
fn audio_asset_scope_does_not_grant_access_to_other_app_data() {
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(manifest_dir().join("tauri.conf.json")).unwrap(),
    ).unwrap();
    let asset = &config["app"]["security"]["assetProtocol"];
    assert_eq!(asset["enable"], true);
    assert_eq!(asset["scope"], serde_json::json!(["$APPDATA/meetings/**"]));
}
