//! WebAssembly bindings. Compiled only for `wasm32` (see lib.rs).
//!
//! Live-training flow the viewer uses:
//!   session_begin(features, …) → layout JSON; then call session_step() once per
//!   animation frame (each returns one telemetry frame, or "" when done) so the
//!   browser renders the net as it learns. classify() scores new inputs after.

use crate::trainer::{train, Config, Session};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

// wasm is single-threaded; hold the in-progress / trained session here.
thread_local! {
    static SESSION: RefCell<Option<Session>> = const { RefCell::new(None) };
}

fn parse_hidden(s: &str) -> Vec<usize> {
    s.split(',')
        .filter_map(|t| t.trim().parse::<usize>().ok())
        .filter(|&x| x > 0)
        .collect()
}

/// Build a training session over uploaded image features and keep it. `features`
/// is a row-major `n × dim` matrix; `labels` are ±1 (class A = +1, B = -1).
/// Returns `{"layers":[..],"steps":N}` so the viewer can build the cube up front.
#[wasm_bindgen]
pub fn session_begin(
    features: &[f32],
    n: usize,
    dim: usize,
    labels: &[f32],
    hidden: &str,
    steps: usize,
    seed: u32,
    lr: f64,
) -> String {
    let xs: Vec<Vec<f64>> = (0..n)
        .map(|i| features[i * dim..(i + 1) * dim].iter().map(|&v| v as f64).collect())
        .collect();
    let ys: Vec<f64> = labels.iter().map(|&v| if v >= 0.0 { 1.0 } else { -1.0 }).collect();
    let lr0 = if lr > 0.0 { lr } else { 0.2 };
    let mut rng = StdRng::seed_from_u64(seed as u64);

    let s = Session::new(xs, ys, &parse_hidden(hidden), steps.max(1), lr0, true, &mut rng);
    let layout = s.layout_json();
    SESSION.with(|m| *m.borrow_mut() = Some(s));
    layout
}

/// Advance the kept session one training step; returns the frame JSON, or "" when
/// training is complete (or no session exists).
#[wasm_bindgen]
pub fn session_step() -> String {
    SESSION.with(|m| m.borrow_mut().as_mut().and_then(|s| s.step()).unwrap_or_default())
}

/// Score a feature vector against the kept model. `> 0` ⇒ class A, `< 0` ⇒ class B.
/// NaN if no session exists or the dimension doesn't match.
#[wasm_bindgen]
pub fn classify(features: &[f32]) -> f64 {
    SESSION.with(|m| match m.borrow_mut().as_mut() {
        Some(s) if s.dim == features.len() => {
            let x: Vec<f64> = features.iter().map(|&v| v as f64).collect();
            s.classify(&x)
        }
        _ => f64::NAN,
    })
}

/// The built-in moons demo (full recording in one call). Kept for the native
/// example parity; the browser uses the live session API above.
#[wasm_bindgen]
pub fn train_recording(hidden: &str, steps: usize, samples: usize, noise: f64, seed: u32, lr: f64) -> String {
    let cfg = Config {
        hidden: parse_hidden(hidden),
        steps: steps.max(1),
        samples: samples.max(4),
        noise,
        seed: seed as u64,
        lr0: if lr > 0.0 { lr } else { 1.0 },
    };
    train(&cfg)
}
