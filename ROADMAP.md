# rustograd — Feature Parity Roadmap

A checklist for reimplementing [micrograd](https://github.com/karpathy/micrograd)
in Rust from scratch. Work top-to-bottom; each phase builds on the last. Check
items off as you go.

> Reference originals (in `../micrograd`): `engine.py`, `nn.py`,
> `test/test_engine.py`, `demo.ipynb`, `trace_graph.ipynb`.

---

## Phase 0 — Project scaffolding ✅ (done)
- [x] `cargo new rustograd --lib` with `repl` binary
- [x] Module layout: `engine`, `nn`, `bin/repl`, `tests/engine.rs`
- [x] `rand` dependency wired for weight init
- [x] Compiles green (`cargo build`) and runs (`cargo run --bin repl`)

## Phase 1 — Core types (engine.rs)
- [x] **Representation decided: arena / tape.** Single `Node` struct (data, grad,
      op), `usize` handles, `Graph` owns the `Vec<Node>`. No `Rc`/`RefCell`, no
      handle type, no operator overloading. (Skeleton already laid down.)
- [x] Flesh out the `Op` enum with every op you'll support (Leaf/Add/Mul/Pow/Relu —
      the full micrograd set; `#[derive(Clone, Copy)]` so `backward` can match by value)
- [x] `Graph::value(data)` leaf constructor + `data`/`grad` access via `g.nodes[i]`
- [ ] `Display` for a node matching Python's `Value(data=.., grad=..)`

## Phase 2 — Forward ops (methods on `Graph`)
Each op computes its `data` from the inputs' data, then appends a node tagged
with the right `Op` variant and returns the new index:
- [x] `add(a, b)`, `mul(a, b)` — core binary ops
- [x] `powf(a, exponent)` — int/float powers only (as in micrograd)
- [x] `relu(a)`
- [x] Derived ops (mirror `__neg__`/`__sub__`/`__truediv__`):
      `neg = mul(a, value(-1))`, `sub = add(a, neg(b))`, `div = mul(a, powf(b,-1))`
- [ ] Scalar conveniences if you want them, e.g. `add_scalar(a, f64)` —
      since there's no operator overloading, decide how `Value + 2.0` is spelled
      (skipped for now: tests/nn just create a scalar leaf with `g.value(..)`)

## Phase 3 — Backward pass (the heart of it)
- [x] Topological sort — **free**: the append-only arena is already topo-ordered
      (every input index < its node's index), so no DFS/visited set is needed
- [x] Seed output `grad = 1`, walk nodes in reverse, apply each local gradient
- [x] Gradients **accumulate** (`+=`), so a node used twice sums both paths
- [x] Correct local grads for every op: add, mul, pow, relu (and the derived ones)

## Phase 4 — Tests / gradient check (tests/engine.rs)
- [x] Port `test_sanity_check` — bake in PyTorch's expected `data`/`grad`
- [x] Port `test_more_ops` — exercise every op, assert within `1e-6`
- [x] Remove the `#[ignore]` attributes; `cargo test` passes (3/3 green)
- [x] (Bonus) Numerical gradient check: finite-difference vs analytic grad

## Phase 5 — Neural net library (nn.rs)
- [x] ~~`Module` **trait**~~ **dropped** — no polymorphism/`dyn` use and no shared
      default (we have no `zero_grad`), so `parameters()` is an inherent method on
      each type instead. No `zero_grad`: `backward` already zeros every grad on entry.
- [x] `Neuron::new(g, nin, nonlin)` with `rand` uniform(-1, 1) weights, bias 0
- [x] `Neuron` forward: `sum(wᵢ·xᵢ) + b`, then `relu()` if `nonlin`
- [x] `Layer` = `Vec<Neuron>`; forward maps a layer over its neurons
- [x] `MLP` = `Vec<Layer>`; last layer linear, earlier layers ReLU
- [x] `parameters()` flattens the whole network's params
- [ ] `Display`/`Debug` mirroring `MLP of [Layer of [...]]`

## Phase 6 — Training demo (parity with demo.ipynb)
- [x] An `examples/demo.rs` that trains the MLP on a toy 2-D dataset
      (make-moons hand-rolled: two noisy arcs, Box-Muller noise, seeded RNG)
- [x] SVM "max-margin" hinge loss + L2 regularization, like the notebook
- [x] Manual SGD loop: forward → loss → `backward()` → step params by `-lr * grad`
      (no `zero_grad` — `backward` self-zeros; the watermark `truncate` resets the tape)
- [x] Print loss + accuracy per step; confirm it actually learns (50% → 100% by step 40)
- [x] (Bonus) ASCII decision-boundary plot at the end — a down payment on Phase 8

## Phase 7 — Interactive REPL  ⟵ requested feature
- [ ] Grow `src/bin/repl.rs` past the echo skeleton
- [ ] Add `rustyline` for line editing, history, and arrow keys
- [ ] Bind variables (`a = Value(-4)`) and evaluate expressions over them
- [ ] Commands: `backward <var>`, `grad <var>`, `params`, `help`, `quit`
- [ ] **Alternative/parallel track:** use `evcxr` for a real Rust REPL —
      `cargo install evcxr_repl`, then `:dep rustograd = { path = "." }`.
      (`evcxr` also provides a Jupyter kernel, see Phase 8.)

## Phase 8 — Data & graph visualization  ⟵ requested feature
Two kinds, matching the two micrograd notebooks:
- [ ] **Computation graph** (parity with `trace_graph.ipynb`): walk the graph
      and emit Graphviz **DOT** text; render with the `dot` binary
      (already installed) or the `graphviz-rust` crate. Show data + grad per node.
- [ ] **Data/plots** (parity with `demo.ipynb`): use the `plotters` crate to
      draw the decision boundary and/or the loss curve to a PNG/SVG.
- [ ] (Optional) Run the whole thing in a notebook via the `evcxr` Jupyter
      kernel for inline plots — the closest experience to the originals.

## Phase 9 — Polish
- [ ] Crate-level docs + a README with a quickstart
- [ ] `cargo clippy` clean, `cargo fmt`
- [ ] CI (GitHub Actions) running `build` + `test` (optional)

---

### Suggested crates (uncomment in `Cargo.toml` as you reach them)
| Need | Crate |
|------|-------|
| Random weight init | `rand` *(already added)* |
| REPL line editing | `rustyline` |
| Plotting (matplotlib parity) | `plotters` |
| Graph rendering (graphviz parity) | `graphviz-rust`, or shell out to `dot` |
| Float-approx test asserts | `approx` (dev-dep) |
| Live REPL / Jupyter | `evcxr` (installed via `cargo install`, not a dep) |

### The one genuinely hard part — now settled
The interesting Rust challenge was the representation: how to let many graph
parents share mutable child nodes under the borrow checker. **Resolved** by the
arena — share by `usize` index instead of by reference. With that decided, the
rest is mechanical: the real remaining substance is `backward` (Phase 3), which
becomes a reverse loop over the `Vec` with a `match` on each node's `Op`.
