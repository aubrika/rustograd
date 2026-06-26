declare namespace wasm_bindgen {
    /* tslint:disable */
    /* eslint-disable */

    /**
     * Score a feature vector against the kept model. `> 0` ⇒ class A, `< 0` ⇒ class B.
     * NaN if no session exists or the dimension doesn't match.
     */
    export function classify(features: Float32Array): number;

    /**
     * Build a training session over uploaded image features and keep it. `features`
     * is a row-major `n × dim` matrix; `labels` are ±1 (class A = +1, B = -1).
     * Returns `{"layers":[..],"steps":N}` so the viewer can build the cube up front.
     */
    export function session_begin(features: Float32Array, n: number, dim: number, labels: Float32Array, hidden: string, steps: number, seed: number, lr: number): string;

    /**
     * Advance the kept session one training step; returns the frame JSON, or "" when
     * training is complete (or no session exists).
     */
    export function session_step(): string;

    /**
     * The built-in moons demo (full recording in one call). Kept for the native
     * example parity; the browser uses the live session API above.
     */
    export function train_recording(hidden: string, steps: number, samples: number, noise: number, seed: number, lr: number): string;

}
declare type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

declare interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly classify: (a: number, b: number) => number;
    readonly session_begin: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
    readonly session_step: () => [number, number];
    readonly train_recording: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
declare function wasm_bindgen (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
