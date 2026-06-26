pub mod engine;
pub mod nn;
pub mod trainer;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
