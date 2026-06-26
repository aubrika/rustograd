//! Trains the moons MLP and writes `viz/recording.js` for the 3D viewer.
//!
//!     cargo run --release --example record
//!
//! All the real work lives in `rustograd::trainer` (shared with the wasm build);
//! this just runs it natively and wraps the JSON as `window.RECORDING = …`.

use rustograd::trainer::{train, Config};
use std::fs;

fn main() {
    let json = train(&Config::default()); // 2 → 16 → 16 → 1, 100 steps
    fs::create_dir_all("viz").expect("create viz/ dir");
    fs::write("viz/recording.js", format!("window.RECORDING = {json};\n"))
        .expect("write recording.js");
    println!("wrote viz/recording.js");
}
