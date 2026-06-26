//! Neural-network library built on top of the autograd `engine`.
//!
//! Parity target: `micrograd/nn.py` — `Neuron`, `Layer`, `MLP`.
//!
//! DESIGN NOTES (consequences of the arena engine):
//!   • A "parameter" is a `usize` index into a `Graph`, not a self-contained
//!     `Value` object. So every constructor and `forward` takes `&mut Graph`
//!     explicitly — there is no hidden global graph.
//!   • All parameters are allocated up front (at `new`), so they occupy the front
//!     of the arena. A training step pushes its forward pass *above* them and
//!     truncates back down afterwards (see the Phase-6 training loop).
//!   • No `zero_grad`: `Graph::backward` already zeros every grad on entry, so
//!     params start clean each step. (Micrograd needs `zero_grad` only because
//!     its `backward` accumulates into pre-existing grads.)

use crate::engine::Graph;
use rand::Rng;

/// A single neuron: `Σ wᵢ·xᵢ + b`, optionally passed through ReLU.
pub struct Neuron {
    w: Vec<usize>, // weights
    b: usize, // bias 
    nonlin: bool, // whether to apply ReLU
}

impl Neuron {
    /// `nin` weights with He/Kaiming-uniform init — `U(-1, 1) · √(6/nin)` — and
    /// bias 0. The `√(6/nin)` scaling keeps activation variance roughly constant
    /// across ReLU layers, which is what lets *deep* nets train; plain `U(-1, 1)`
    /// (like micrograd) only works because micrograd is shallow.
    ///
    /// Takes an explicit RNG so nets are reproducible and the code is wasm-safe
    /// (no `thread_rng`, which would pull in `getrandom`'s JS shim).
    pub fn new(g: &mut Graph, rng: &mut impl Rng, nin: usize, nonlin: bool) -> Self {
        let scale = (6.0 / nin.max(1) as f64).sqrt();
        let w = (0..nin).map(|_| g.value(rng.gen_range(-1.0..1.0) * scale)).collect();
        let b = g.value(0.0);
        Neuron { w, b, nonlin }
    }

    /// Forward pass over an input slice of node indices. Appends the
    /// `Σ wᵢ·xᵢ + b` (and optional relu) subgraph and returns the output index.
    pub fn forward(&self, g: &mut Graph, x: &[usize]) -> usize {

        let mut acc = self.b; // start with the bias

        for (&wi, &xi) in self.w.iter().zip(x) { // for each weight and input
            let prod = g.mul(wi, xi); // compute wᵢ·xᵢ
            acc = g.add(acc, prod); // accumulate into the sum
        }
        if self.nonlin { // optionally apply ReLU
            g.relu(acc)
        } else {
            acc
        }
    }

    /// The neuron's trainable arena indices: its weights followed by its bias.
    pub fn parameters(&self) -> Vec<usize> {
        let mut p = self.w.clone();
        p.push(self.b);
        p
    }
}

/// A fully-connected layer: `nout` neurons, each over the same `nin` inputs.
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(g: &mut Graph, rng: &mut impl Rng, nin: usize, nout: usize, nonlin: bool) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(g, rng, nin, nonlin)).collect();
        Layer { neurons }
    }

    /// Maps each neuron over the input, returning one output index per neuron.
    pub fn forward(&self, g: &mut Graph, x: &[usize]) -> Vec<usize> {
        self.neurons.iter().map(|n| n.forward(g, x)).collect()
    }

    /// Flattens every neuron's parameters into one list.
    pub fn parameters(&self) -> Vec<usize> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }

    /// The layer's neurons, for per-neuron introspection (e.g. visualization).
    pub fn neurons(&self) -> &[Neuron] {
        &self.neurons
    }
}

/// A multi-layer perceptron. Earlier layers use ReLU; the final layer is linear.
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    /// `nin` inputs, then one layer per entry in `nouts`. E.g.
    /// `MLP::new(g, rng, 2, &[16, 16, 1])` is a 2→16→16→1 net.
    pub fn new(g: &mut Graph, rng: &mut impl Rng, nin: usize, nouts: &[usize]) -> Self {
        let sizes: Vec<usize> = std::iter::once(nin).chain(nouts.iter().copied()).collect();
        let last = nouts.len() - 1;
        let layers = (0..nouts.len())
            .map(|i| Layer::new(g, rng, sizes[i], sizes[i + 1], i != last))
            .collect();
        MLP { layers }
    }

    /// Threads the input through every layer; returns the final layer's outputs.
    pub fn forward(&self, g: &mut Graph, x: &[usize]) -> Vec<usize> {
        let mut h = x.to_vec();
        for layer in &self.layers {
            h = layer.forward(g, &h);
        }
        h
    }

    /// Like `forward`, but returns every layer's per-neuron outputs (not just the
    /// last layer's). `result[l][n]` is the output node index of neuron `n` in
    /// layer `l` — used by the visualizer to read per-neuron activations.
    pub fn forward_verbose(&self, g: &mut Graph, x: &[usize]) -> Vec<Vec<usize>> {
        let mut acts = Vec::with_capacity(self.layers.len());
        let mut h = x.to_vec();
        for layer in &self.layers {
            h = layer.forward(g, &h);
            acts.push(h.clone());
        }
        acts
    }

    /// Flattens every layer's parameters into one list — the whole network's.
    pub fn parameters(&self) -> Vec<usize> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    /// The network's layers, for per-layer/per-neuron introspection.
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }
}
