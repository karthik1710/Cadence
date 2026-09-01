//! A deliberately tiny HTTP/1.1 client that only ever talks to `127.0.0.1`.
//!
//! **This is not a network dependency.** llama.cpp's and whisper.cpp's local
//! inference servers are driven over a loopback socket — that is their IPC
//! mechanism, the same way sherpa-onnx is driven over stdin/stdout. Nothing
//! here can reach off the machine: [`request`] hard-codes the loopback address
//! and takes a port, not a host or a URL, so there is no way to point it at a
//! remote service even by accident.
//!
//! It is written by hand rather than pulled from a crate for the same reason:
//! adding `reqwest` (or any HTTP client) to the dependency graph would destroy
//! the cheap structural check that Cadence has no cloud path in it —
//! `cargo tree` showing no HTTP stack at all. [`Multipart`] exists for the
//! same reason: whisper-server takes audio as a `multipart/form-data` upload,
//! and that format is a dozen lines of string building, not a crate.
//!
//! The alternative — re-running `llama-cli.exe` / `whisper-cli.exe` once per
//! prompt or per audio block — was rejected because MoM generation makes one
//! call per transcript chunk (§14) and code-switched STT one call per ~30 s
//! block (§8), and reloading the weights every time would dominate the §29
//! processing-time budget.

use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::time::Duration;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const WRITE_TIMEOUT: Duration = Duration::from_secs(30);

pub struct Response {
    pub status: u16,
    pub body: String,
}

impl Response {
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }
}

/// Sends one JSON (or bodiless) request to `127.0.0.1:<port>`.
pub fn request(
    port: u16,
    method: &str,
    path: &str,
    body: Option<&str>,
    read_timeout: Duration,
) -> Result<Response, String> {
    let body = body.map(|payload| ("application/json", payload.as_bytes()));
    request_bytes(port, method, path, body, read_timeout)
}

/// Sends one request to `127.0.0.1:<port>` and reads the whole response.
///
/// `Connection: close` is sent so the server ends the body by closing the
/// socket, which keeps the reader simple and correct for both
/// `Content-Length` and chunked responses.
pub fn request_bytes(
    port: u16,
    method: &str,
    path: &str,
    body: Option<(&str, &[u8])>,
    read_timeout: Duration,
) -> Result<Response, String> {
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let mut stream = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT)
        .map_err(|e| format!("could not reach the local model server on port {port}: {e}"))?;
    stream
        .set_read_timeout(Some(read_timeout))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(WRITE_TIMEOUT))
        .map_err(|e| e.to_string())?;
    stream.set_nodelay(true).ok();

    let mut head = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAccept: application/json\r\nConnection: close\r\n"
    );
    if let Some((content_type, payload)) = body {
        head.push_str(&format!("Content-Type: {content_type}\r\n"));
        head.push_str(&format!("Content-Length: {}\r\n", payload.len()));
    }
    head.push_str("\r\n");

    stream
        .write_all(head.as_bytes())
        .map_err(|e| format!("failed writing request to the local model server: {e}"))?;
    if let Some((_, payload)) = body {
        stream
            .write_all(payload)
            .map_err(|e| format!("failed writing request body to the local model server: {e}"))?;
    }
    stream.flush().map_err(|e| e.to_string())?;

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).map_err(|e| {
        format!("the local model server did not answer within {read_timeout:?}: {e}")
    })?;

    parse_response(&raw)
}

fn parse_response(raw: &[u8]) -> Result<Response, String> {
    let split = find_subslice(raw, b"\r\n\r\n")
        .ok_or_else(|| "malformed response from the local model server".to_string())?;
    let head = String::from_utf8_lossy(&raw[..split]).to_string();
    let body_bytes = &raw[split + 4..];

    let mut lines = head.lines();
    let status_line = lines
        .next()
        .ok_or_else(|| "empty response from the local model server".to_string())?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("unparseable status line: {status_line}"))?;

    let mut chunked = false;
    let mut content_length: Option<usize> = None;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim();
        if name == "transfer-encoding" && value.to_ascii_lowercase().contains("chunked") {
            chunked = true;
        } else if name == "content-length" {
            content_length = value.parse().ok();
        }
    }

    let body = if chunked {
        dechunk(body_bytes)?
    } else {
        match content_length {
            Some(len) if len <= body_bytes.len() => body_bytes[..len].to_vec(),
            _ => body_bytes.to_vec(),
        }
    };

    Ok(Response {
        status,
        body: String::from_utf8_lossy(&body).to_string(),
    })
}

fn dechunk(mut input: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    loop {
        let line_end = find_subslice(input, b"\r\n")
            .ok_or_else(|| "truncated chunked response".to_string())?;
        let size_line = String::from_utf8_lossy(&input[..line_end]);
        // A chunk size may carry `;extension` after it.
        let size_text = size_line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_text, 16)
            .map_err(|_| format!("invalid chunk size: {size_text}"))?;
        input = &input[line_end + 2..];
        if size == 0 {
            return Ok(out);
        }
        if input.len() < size {
            return Err("truncated chunk body".to_string());
        }
        out.extend_from_slice(&input[..size]);
        // Skip the CRLF that terminates the chunk.
        input = &input[(size + 2).min(input.len())..];
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// A `multipart/form-data` body, the only shape whisper-server's `/inference`
/// endpoint accepts audio in.
///
/// A boundary is only unsafe if the body contains it. The field values here
/// are literals this crate wrote (a thread count, an ISO language code), and
/// the file part is 16-bit PCM, so the odds of these 27 ASCII bytes appearing
/// are ~2^-216 per offset — which is why the boundary is a constant rather
/// than something this module has to generate and thread around.
pub struct Multipart {
    body: Vec<u8>,
}

const BOUNDARY: &str = "----CadenceLoopbackBoundary";

impl Multipart {
    pub fn new() -> Self {
        Self { body: Vec::new() }
    }

    pub fn field(&mut self, name: &str, value: &str) {
        self.body.extend_from_slice(
            format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n")
                .as_bytes(),
        );
        self.body.extend_from_slice(value.as_bytes());
        self.body.extend_from_slice(b"\r\n");
    }

    pub fn file(&mut self, name: &str, file_name: &str, content_type: &str, bytes: &[u8]) {
        self.body.extend_from_slice(
            format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{file_name}\"\r\nContent-Type: {content_type}\r\n\r\n"
            )
            .as_bytes(),
        );
        self.body.extend_from_slice(bytes);
        self.body.extend_from_slice(b"\r\n");
    }

    /// The content type to send and the finished body.
    pub fn finish(mut self) -> (String, Vec<u8>) {
        self.body
            .extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
        (
            format!("multipart/form-data; boundary={BOUNDARY}"),
            self.body,
        )
    }
}

impl Default for Multipart {
    fn default() -> Self {
        Self::new()
    }
}

/// Reserves a free loopback port by binding and immediately releasing it.
///
/// There is a small race between release and the server binding it; on a
/// desktop running one local server that is acceptable, and a collision
/// surfaces as a clear startup failure rather than as silent misbehaviour.
pub fn free_local_port() -> Result<u16, String> {
    let listener = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .map_err(|e| format!("could not reserve a local port: {e}"))?;
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();
    drop(listener);
    Ok(port)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_content_length_response() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 13\r\n\r\n{\"status\":1}\n";
        let response = parse_response(raw).unwrap();
        assert_eq!(response.status, 200);
        assert!(response.is_success());
        assert!(response.body.starts_with("{\"status\":1}"));
    }

    #[test]
    fn parses_a_chunked_response() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
        let response = parse_response(raw).unwrap();
        assert_eq!(response.body, "hello world");
    }

    #[test]
    fn reports_a_server_error_status() {
        let raw = b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 2\r\n\r\n{}";
        let response = parse_response(raw).unwrap();
        assert_eq!(response.status, 503);
        assert!(!response.is_success());
    }

    #[test]
    fn keeps_a_body_that_arrives_without_a_content_length() {
        let raw = b"HTTP/1.1 200 OK\r\n\r\n{\"a\":1}";
        assert_eq!(parse_response(raw).unwrap().body, "{\"a\":1}");
    }

    #[test]
    fn rejects_a_response_with_no_header_terminator() {
        assert!(parse_response(b"HTTP/1.1 200 OK").is_err());
    }

    #[test]
    fn allocates_a_usable_port() {
        let port = free_local_port().unwrap();
        assert!(port > 0);
    }

    #[test]
    fn builds_a_multipart_body_a_server_can_split() {
        let mut form = Multipart::new();
        form.field("language", "auto");
        form.file("file", "block.wav", "audio/wav", &[0x52, 0x49, 0x46, 0x46]);
        let (content_type, body) = form.finish();

        assert!(content_type.starts_with("multipart/form-data; boundary="));
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("name=\"language\"\r\n\r\nauto\r\n"));
        assert!(text.contains("filename=\"block.wav\""));
        assert!(text.contains("Content-Type: audio/wav"));
        // Two parts and the closing delimiter.
        assert_eq!(text.matches(BOUNDARY).count(), 3);
        assert!(text.ends_with("--\r\n"));
        // The raw bytes survive verbatim — a WAV must not be re-encoded.
        assert!(find_subslice(&body, b"RIFF").is_some());
    }
}
