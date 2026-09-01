//! A deliberately tiny HTTP/1.1 client that only ever talks to `127.0.0.1`.
//!
//! **This is not a network dependency.** llama.cpp's local inference server is
//! driven over a loopback socket — that is its IPC mechanism, the same way
//! whisper.cpp and sherpa-onnx are driven over stdin/stdout. Nothing here can
//! reach off the machine: [`request`] hard-codes the loopback address and
//! takes a port, not a host or a URL, so there is no way to point it at a
//! remote service even by accident.
//!
//! It is written by hand rather than pulled from a crate for the same reason:
//! adding `reqwest` (or any HTTP client) to the dependency graph would destroy
//! the cheap structural check that EchoMind has no cloud path in it —
//! `cargo tree` showing no HTTP stack at all.
//!
//! The alternative — re-running `llama-cli.exe` once per prompt — was
//! rejected because MoM generation makes one call per transcript chunk (§14),
//! and reloading multi-gigabyte weights for every chunk would dominate the
//! §29 processing-time budget.

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

/// Sends one request to `127.0.0.1:<port>` and reads the whole response.
///
/// `Connection: close` is sent so the server ends the body by closing the
/// socket, which keeps the reader simple and correct for both
/// `Content-Length` and chunked responses.
pub fn request(
    port: u16,
    method: &str,
    path: &str,
    body: Option<&str>,
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
    if let Some(payload) = body {
        head.push_str("Content-Type: application/json\r\n");
        head.push_str(&format!("Content-Length: {}\r\n", payload.len()));
    }
    head.push_str("\r\n");

    stream
        .write_all(head.as_bytes())
        .map_err(|e| format!("failed writing request to the local model server: {e}"))?;
    if let Some(payload) = body {
        stream
            .write_all(payload.as_bytes())
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
}
