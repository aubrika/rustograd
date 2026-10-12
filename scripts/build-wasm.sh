#!/usr/bin/env bash
# Rebuilds the WebAssembly bundle for the 3D viewer (macOS/Linux; build-wasm.ps1 is the
# Windows equivalent).
#   1. compile the lib to wasm32
#   2. generate JS bindings (viz/wasm/rustograd.js + _bg.wasm)
#   3. inline the .wasm as base64 so viz/index.html runs from file:// (no server)
#
# Needs:  rustup target add wasm32-unknown-unknown
#         cargo install wasm-bindgen-cli --version 0.2.126   (must match Cargo.toml)
# Run from the repo root:  scripts/build-wasm.sh
set -euo pipefail

# no-modules (classic script, not ES module) so it loads from file:// without CORS
cargo build --release --target wasm32-unknown-unknown --lib
wasm-bindgen target/wasm32-unknown-unknown/release/rustograd.wasm --out-dir viz/wasm --target no-modules

b64=$(base64 < viz/wasm/rustograd_bg.wasm | tr -d '\n')
printf 'window.WASM_BASE64 = "%s";\n' "$b64" > viz/wasm/rustograd_inline.js
echo "ok: viz/wasm rebuilt ($(wc -c < viz/wasm/rustograd_bg.wasm | tr -d ' ') bytes wasm)"
