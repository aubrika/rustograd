//! Shared training engine + telemetry, used by the native `record` example and
//! the wasm bindings. A [`Session`] trains a binary classifier (SVM hinge + L2)
//! one step at a time, capturing a per-neuron telemetry frame each step — which
//! lets the browser render the network *as it learns*, not just afterwards.

use crate::engine::Graph;
use crate::nn::MLP;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Hyperparameters for the built-in moons demo.
pub struct Config {
    pub hidden: Vec<usize>,
    pub steps: usize,
    pub samples: usize,
    pub noise: f64,
    pub seed: u64,
    /// Initial learning rate; the schedule decays from `lr0` to `0.1·lr0`.
    pub lr0: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config { hidden: vec![16, 16], steps: 100, samples: 100, noise: 0.1, seed: 1337, lr0: 1.0 }
    }
}

fn randn(rng: &mut impl Rng) -> f64 {
    let u1: f64 = rng.gen_range(0.0..1.0f64).max(1e-12);
    let u2: f64 = rng.gen_range(0.0..1.0f64);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// `sklearn.datasets.make_moons`: two interleaving noisy arcs, labels ±1.
pub fn make_moons(n: usize, noise: f64, rng: &mut impl Rng) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n_out = n / 2;
    let n_in = n - n_out;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n_out {
        let t = std::f64::consts::PI * i as f64 / (n_out as f64 - 1.0);
        xs.push(vec![t.cos() + noise * randn(rng), t.sin() + noise * randn(rng)]);
        ys.push(-1.0);
    }
    for i in 0..n_in {
        let t = std::f64::consts::PI * i as f64 / (n_in as f64 - 1.0);
        xs.push(vec![
            1.0 - t.cos() + noise * randn(rng),
            1.0 - t.sin() - 0.5 + noise * randn(rng),
        ]);
        ys.push(1.0);
    }
    (xs, ys)
}

/// Full-batch SVM hinge + L2 loss over feature vectors. Returns (loss idx, accuracy).
fn build_loss(g: &mut Graph, model: &MLP, xs: &[Vec<f64>], ys: &[f64]) -> (usize, f64) {
    let n = xs.len();
    let mut losses = Vec::with_capacity(n);
    let mut correct = 0usize;
    for (point, &yi) in xs.iter().zip(ys) {
        let leaves: Vec<usize> = point.iter().map(|&v| g.value(v)).collect();
        let score = model.forward(g, &leaves)[0];
        if (yi > 0.0) == (g.nodes[score].data > 0.0) {
            correct += 1;
        }
        let yi_node = g.value(yi);
        let ys_prod = g.mul(yi_node, score);
        let one = g.value(1.0);
        let margin = g.sub(one, ys_prod);
        losses.push(g.relu(margin));
    }
    let mut sum = losses[0];
    for &li in &losses[1..] {
        sum = g.add(sum, li);
    }
    let inv_n = g.value(1.0 / n as f64);
    let data_loss = g.mul(sum, inv_n);

    let params = model.parameters();
    let mut reg_sum = g.mul(params[0], params[0]);
    for &p in &params[1..] {
        let pp = g.mul(p, p);
        reg_sum = g.add(reg_sum, pp);
    }
    let alpha = g.value(1e-4);
    let reg_loss = g.mul(alpha, reg_sum);
    (g.add(data_loss, reg_loss), correct as f64 / n as f64)
}

fn neuron_json(grad: f64, weight: f64, act: f64) -> String {
    format!("{{\"g\":{grad:.5},\"w\":{weight:.5},\"a\":{act:.5}}}")
}

/// A live training run over a fixed dataset. Build it, then call [`step`](Session::step)
/// repeatedly (each returns one frame's JSON) until it returns `None`. The kept
/// model can [`classify`](Session::classify) new inputs at any point.
pub struct Session {
    g: Graph,
    model: MLP,
    xs: Vec<Vec<f64>>,
    ys: Vec<f64>,
    watermark: usize,
    /// Input feature dimension.
    pub dim: usize,
    layers: Vec<usize>,
    k: usize,
    steps: usize,
    lr0: f64,
}

impl Session {
    /// `xs[i]` is a feature vector of length `dim`, `ys[i]` is ±1. `hidden` is the
    /// hidden-layer sizes; a single linear output neuron is appended.
    pub fn new(
        xs: Vec<Vec<f64>>,
        ys: Vec<f64>,
        hidden: &[usize],
        steps: usize,
        lr0: f64,
        rng: &mut impl Rng,
    ) -> Session {
        let dim = xs.first().map(|x| x.len()).unwrap_or(2);
        let nouts: Vec<usize> = hidden.iter().copied().chain(std::iter::once(1)).collect();
        let mut g = Graph::new();
        let model = MLP::new(&mut g, rng, dim, &nouts);
        let watermark = g.nodes.len();
        let layers: Vec<usize> = std::iter::once(dim).chain(nouts).collect();
        Session { g, model, xs, ys, watermark, dim, layers, k: 0, steps: steps.max(1), lr0 }
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    /// `{"layers":[..],"steps":N}` — enough for the viewer to build the cube before
    /// the first frame arrives.
    pub fn layout_json(&self) -> String {
        let l = self.layers.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
        format!("{{\"layers\":[{}],\"steps\":{}}}", l, self.steps)
    }

    /// Advance one training step; returns that step's telemetry frame JSON, or
    /// `None` once all steps are done.
    pub fn step(&mut self) -> Option<String> {
        if self.k >= self.steps {
            return None;
        }
        let k = self.k;
        self.g.nodes.truncate(self.watermark);
        let (loss, acc) = build_loss(&mut self.g, &self.model, &self.xs, &self.ys);
        self.g.backward(loss);
        let loss_val = self.g.nodes[loss].data;

        let lr = self.lr0 * (1.0 - 0.9 * k as f64 / self.steps as f64);
        for &p in &self.model.parameters() {
            self.g.nodes[p].data -= lr * self.g.nodes[p].grad;
        }

        let frame = self.capture_frame(k, loss_val, acc);
        self.k += 1;
        Some(frame)
    }

    fn capture_frame(&mut self, k: usize, loss_val: f64, acc: f64) -> String {
        let n = self.xs.len() as f64;

        // grad/weight norms off the (preserved) parameter nodes
        let mut layer_stats: Vec<Vec<(f64, f64)>> = Vec::new();
        for layer in self.model.layers() {
            let mut ns = Vec::new();
            for neuron in layer.neurons() {
                let ps = neuron.parameters();
                let gnorm = ps.iter().map(|&p| self.g.nodes[p].grad.powi(2)).sum::<f64>().sqrt();
                let wnorm = ps.iter().map(|&p| self.g.nodes[p].data.powi(2)).sum::<f64>().sqrt();
                ns.push((gnorm, wnorm));
            }
            layer_stats.push(ns);
        }

        // activations: mean |output| over the batch (truncate between points)
        let mut act_acc: Vec<Vec<f64>> =
            self.model.layers().iter().map(|l| vec![0.0; l.neurons().len()]).collect();
        let mut input_acc = vec![0.0f64; self.dim];
        for point in &self.xs {
            self.g.nodes.truncate(self.watermark);
            for (d, &v) in point.iter().enumerate() {
                input_acc[d] += v.abs();
            }
            let leaves: Vec<usize> = point.iter().map(|&v| self.g.value(v)).collect();
            let acts = self.model.forward_verbose(&mut self.g, &leaves);
            for (li, la) in acts.iter().enumerate() {
                for (ni, &idx) in la.iter().enumerate() {
                    act_acc[li][ni] += self.g.nodes[idx].data.abs();
                }
            }
        }

        let input_neurons: Vec<String> =
            input_acc.iter().map(|&a| neuron_json(0.0, 0.0, a / n)).collect();
        let mut layers_json: Vec<String> = vec![format!("[{}]", input_neurons.join(","))];
        for (li, stats) in layer_stats.iter().enumerate() {
            let neurons: Vec<String> = stats
                .iter()
                .enumerate()
                .map(|(ni, &(gn, wn))| neuron_json(gn, wn, act_acc[li][ni] / n))
                .collect();
            layers_json.push(format!("[{}]", neurons.join(",")));
        }
        format!(
            "{{\"step\":{k},\"loss\":{loss_val:.5},\"acc\":{acc:.3},\"neurons\":[{}]}}",
            layers_json.join(",")
        )
    }

    /// Forward a single feature vector; `> 0` ⇒ positive class.
    pub fn classify(&mut self, x: &[f64]) -> f64 {
        self.g.nodes.truncate(self.watermark);
        let leaves: Vec<usize> = x.iter().map(|&v| self.g.value(v)).collect();
        let score = self.model.forward(&mut self.g, &leaves)[0];
        self.g.nodes[score].data
    }
}

/// Train a dataset to completion and return the full JSON recording (used by the
/// native `record` example). At most ~60 frames are kept regardless of `steps`.
pub fn train_dataset(
    xs: &[Vec<f64>],
    ys: &[f64],
    hidden: &[usize],
    steps: usize,
    lr0: f64,
    rng: &mut impl Rng,
) -> (String, Session) {
    let mut s = Session::new(xs.to_vec(), ys.to_vec(), hidden, steps, lr0, rng);
    let total = s.steps();
    let record_every = (total / 60).max(1);
    let mut frames = Vec::new();
    let mut i = 0;
    while let Some(frame) = s.step() {
        if i % record_every == 0 || i == total - 1 {
            frames.push(frame);
        }
        i += 1;
    }
    let layers_str = s.layers.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
    let json = format!("{{\"layers\":[{}],\"frames\":[{}]}}", layers_str, frames.join(","));
    (json, s)
}

/// Train the built-in moons demo per `cfg`; returns the JSON recording.
pub fn train(cfg: &Config) -> String {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let (xs, ys) = make_moons(cfg.samples, cfg.noise, &mut rng);
    train_dataset(&xs, &ys, &cfg.hidden, cfg.steps, cfg.lr0, &mut rng).0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trains_custom_architecture() {
        let cfg = Config { hidden: vec![10, 10, 10], steps: 40, ..Config::default() };
        let json = train(&cfg);
        assert!(json.contains("\"layers\":[2,10,10,10,1]"), "got: {}", &json[..40]);
        let losses: Vec<f64> = json
            .split("\"loss\":")
            .skip(1)
            .map(|s| s.split(&[',', '}'][..]).next().unwrap().parse().unwrap())
            .collect();
        assert!(*losses.last().unwrap() < losses[0], "loss didn't fall");
    }

    /// Train a steppable session on a tiny separable 3-D set; the kept model
    /// classifies fresh points on the correct side.
    #[test]
    fn session_steps_and_classifies() {
        let xs = vec![
            vec![1.0, 1.0, 1.0],
            vec![0.9, 1.1, 0.8],
            vec![-1.0, -1.0, -1.0],
            vec![-0.8, -1.2, -0.9],
        ];
        let ys = vec![1.0, 1.0, -1.0, -1.0];
        let mut rng = StdRng::seed_from_u64(7);
        let mut s = Session::new(xs, ys, &[8], 120, 0.2, &mut rng);

        assert_eq!(s.dim, 3);
        assert!(s.layout_json().contains("\"layers\":[3,8,1]"));
        let mut frames = 0;
        while s.step().is_some() {
            frames += 1;
        }
        assert_eq!(frames, 120);
        assert!(s.classify(&[1.2, 1.0, 0.9]) > 0.0, "positive cluster misclassified");
        assert!(s.classify(&[-1.1, -0.9, -1.0]) < 0.0, "negative cluster misclassified");
    }
}
