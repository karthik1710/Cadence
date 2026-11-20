//! The comparison harness behind `bench_cli setup`, `bench_cli compare` and
//! `bench_cli baseline`. Bench-only: nothing here is reachable from the app.

pub mod baseline;
pub mod catalog;
pub mod compare;
pub mod render;
pub mod setup;
pub mod text;
