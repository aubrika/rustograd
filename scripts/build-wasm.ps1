# Rebuilds the WebAssembly bundle for the 3D viewer.
#   1. compile the lib to wasm32
#   2. generate JS bindings (viz/wasm/rustograd.js + _bg.wasm)
#   3. inline the .wasm as base64 so viz/index.html runs from file:// (no server)
#
# Run from the repo root:  pwsh scripts/build-wasm.ps1
$ErrorActionPreference = "Stop"

# no-modules (classic script, not ES module) so it loads from file:// without CORS
cargo build --release --target wasm32-unknown-unknown --lib
wasm-bindgen target/wasm32-unknown-unknown/release/rustograd.wasm --out-dir viz/wasm --target no-modules

$bytes = [IO.File]::ReadAllBytes("viz/wasm/rustograd_bg.wasm")
$b64 = [Convert]::ToBase64String($bytes)
"window.WASM_BASE64 = `"$b64`";" | Out-File -Encoding ascii "viz/wasm/rustograd_inline.js"
Write-Host "ok: viz/wasm rebuilt ($($bytes.Length) bytes wasm)"
