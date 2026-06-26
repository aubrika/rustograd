//! Shared training engine + telemetry, used by the native `record` example and
//! the wasm bindings. A [`Session`] trains a binary classifier (SVM hinge + L2)
//! one step at a time, capturing a per-neuron telemetry frame each step — which
//! lets the browser render the network *as it learns*.
//!
//! Two modes:
//!   • full-batch (every step uses all images) — the built-in moons demo.
//!   • SGD (every step uses ONE image, in shuffled order) — the image classifier,
//!     so each step's gradient is that single image's influence on the loss.

use crate::engine::Graph;
use crate::nn::MLP;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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

/// Hinge loss for one example: `relu(1 - y·score)`. Returns the loss node index.
fn hinge_one(g: &mut Graph, model: &MLP, x: &[f64], y: f64) -> usize {
    let leaves: Vec<usize> = x.iter().map(|&v| g.value(v)).collect();
    let score = model.forward(g, &leaves)[0];
    let yn = g.value(y);
    let ys = g.mul(yn, score);
    let one = g.value(1.0);
    let margin = g.sub(one, ys);
    g.relu(margin)
}

/// `data_loss + alpha·Σ p²` — appends the L2 term and returns the total loss node.
fn add_reg(g: &mut Graph, model: &MLP, data_loss: usize) -> usize {
    let params = model.parameters();
    let mut reg = g.mul(params[0], params[0]);
    for &p in &params[1..] {
        let pp = g.mul(p, p);
        reg = g.add(reg, pp);
    }
    let alpha = g.value(1e-4);
    let reg_loss = g.mul(alpha, reg);
    g.add(data_loss, reg_loss)
}

fn neuron_json(grad: f64, weight: f64, act: f64) -> String {
    format!("{{\"g\":{grad:.5},\"w\":{weight:.5},\"a\":{act:.5}}}")
}

/// A live training run. Build it, then call [`step`](Session::step) until it
/// returns `None`. The kept model can [`classify`](Session::classify) anytime.
pub struct Session {
    g: Graph,
    model: MLP,
    xs: Vec<Vec<f64>>,
    ys: Vec<f64>,
    order: Vec<usize>, // shuffled visiting order for SGD
    watermark: usize,
    pub dim: usize,
    layers: Vec<usize>,
    k: usize,
    steps: usize,
    lr0: f64,
    sgd: bool,
}

impl Session {
    pub fn new(
        xs: Vec<Vec<f64>>,
        ys: Vec<f64>,
        hidden: &[usize],
        steps: usize,
        lr0: f64,
        sgd: bool,
        rng: &mut impl Rng,
    ) -> Session {
        let dim = xs.first().map(|x| x.len()).unwrap_or(2);
        let nouts: Vec<usize> = hidden.iter().copied().chain(std::iter::once(1)).collect();
        let mut g = Graph::new();
        let model = MLP::new(&mut g, rng, dim, &nouts);
        let watermark = g.nodes.len();
        let layers: Vec<usize> = std::iter::once(dim).chain(nouts).collect();

        let n = xs.len();
        let mut order: Vec<usize> = (0..n).collect(); // Fisher-Yates shuffle (interleaves classes)
        for i in (1..n).rev() {
            order.swap(i, rng.gen_range(0..=i));
        }

        Session { g, model, xs, ys, order, watermark, dim, layers, k: 0, steps: steps.max(1), lr0, sgd }
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

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
        let frame = if self.sgd { self.step_sgd(k) } else { self.step_full(k) };
        self.k += 1;
        Some(frame)
    }

    fn lr(&self, k: usize) -> f64 {
        self.lr0 * (1.0 - 0.9 * k as f64 / self.steps as f64)
    }

    fn update(&mut self, lr: f64) {
        for &p in &self.model.parameters() {
            self.g.nodes[p].data -= lr * self.g.nodes[p].grad;
        }
    }

    fn grad_norms(&self) -> Vec<Vec<f64>> {
        self.model
            .layers()
            .iter()
            .map(|layer| {
                layer
                    .neurons()
                    .iter()
                    .map(|nrn| nrn.parameters().iter().map(|&p| self.g.nodes[p].grad.powi(2)).sum::<f64>().sqrt())
                    .collect()
            })
            .collect()
    }

    fn weight_norms(&self) -> Vec<Vec<f64>> {
        self.model
            .layers()
            .iter()
            .map(|layer| {
                layer
                    .neurons()
                    .iter()
                    .map(|nrn| nrn.parameters().iter().map(|&p| self.g.nodes[p].data.powi(2)).sum::<f64>().sqrt())
                    .collect()
            })
            .collect()
    }

    /// Per-model-layer per-neuron output activations for a single input.
    fn activations_for(&mut self, x: &[f64]) -> Vec<Vec<f64>> {
        self.g.nodes.truncate(self.watermark);
        let leaves: Vec<usize> = x.iter().map(|&v| self.g.value(v)).collect();
        let acts = self.model.forward_verbose(&mut self.g, &leaves);
        acts.iter().map(|la| la.iter().map(|&idx| self.g.nodes[idx].data.abs()).collect()).collect()
    }

    /// Full-set loss (mean hinge + L2) and accuracy — for a stable display curve.
    fn full_metrics(&mut self) -> (f64, f64) {
        let n = self.xs.len();
        let mut total = 0.0;
        let mut correct = 0usize;
        for i in 0..n {
            self.g.nodes.truncate(self.watermark);
            let leaves: Vec<usize> = self.xs[i].iter().map(|&v| self.g.value(v)).collect();
            let si = self.model.forward(&mut self.g, &leaves)[0];
            let s = self.g.nodes[si].data;
            if (self.ys[i] > 0.0) == (s > 0.0) {
                correct += 1;
            }
            total += (1.0 - self.ys[i] * s).max(0.0);
        }
        let reg: f64 = self.model.parameters().iter().map(|&p| self.g.nodes[p].data.powi(2)).sum();
        (total / n as f64 + 1e-4 * reg, correct as f64 / n as f64)
    }

    fn frame_json(
        k: usize,
        img: i64,
        loss: f64,
        acc: f64,
        grads: &[Vec<f64>],
        weights: &[Vec<f64>],
        input_acts: &[f64],
        hidden_acts: &[Vec<f64>],
    ) -> String {
        let input_neurons: Vec<String> = input_acts.iter().map(|&a| neuron_json(0.0, 0.0, a)).collect();
        let mut layers_json = vec![format!("[{}]", input_neurons.join(","))];
        for li in 0..grads.len() {
            let neurons: Vec<String> = (0..grads[li].len())
                .map(|ni| neuron_json(grads[li][ni], weights[li][ni], hidden_acts[li][ni]))
                .collect();
            layers_json.push(format!("[{}]", neurons.join(",")));
        }
        format!(
            "{{\"step\":{k},\"img\":{img},\"loss\":{loss:.5},\"acc\":{acc:.3},\"neurons\":[{}]}}",
            layers_json.join(",")
        )
    }

    // ── one image per step (stochastic) ──────────────────────────────────────
    fn step_sgd(&mut self, k: usize) -> String {
        let img = self.order[k % self.order.len()];

        self.g.nodes.truncate(self.watermark);
        let dl = hinge_one(&mut self.g, &self.model, &self.xs[img], self.ys[img]);
        let loss_node = add_reg(&mut self.g, &self.model, dl);
        self.g.backward(loss_node);

        let grads = self.grad_norms(); // this image's gradient = its influence
        self.update(self.lr(k));
        let weights = self.weight_norms();

        let hidden_acts = self.activations_for(&self.xs[img].clone());
        let input_acts = self.xs[img].clone();
        let (loss, acc) = self.full_metrics();

        Session::frame_json(k, img as i64, loss, acc, &grads, &weights, &input_acts, &hidden_acts)
    }

    // ── full batch (every step uses all images) ──────────────────────────────
    fn step_full(&mut self, k: usize) -> String {
        let n = self.xs.len();
        self.g.nodes.truncate(self.watermark);

        // build the summed/mean hinge loss + reg over the whole batch
        let mut losses = Vec::with_capacity(n);
        let mut correct = 0usize;
        for i in 0..n {
            let leaves: Vec<usize> = self.xs[i].iter().map(|&v| self.g.value(v)).collect();
            let score = self.model.forward(&mut self.g, &leaves)[0];
            if (self.ys[i] > 0.0) == (self.g.nodes[score].data > 0.0) {
                correct += 1;
            }
            let yn = self.g.value(self.ys[i]);
            let ys = self.g.mul(yn, score);
            let one = self.g.value(1.0);
            let margin = self.g.sub(one, ys);
            losses.push(self.g.relu(margin));
        }
        let mut sum = losses[0];
        for &li in &losses[1..] {
            sum = self.g.add(sum, li);
        }
        let inv_n = self.g.value(1.0 / n as f64);
        let data_loss = self.g.mul(sum, inv_n);
        let loss_node = add_reg(&mut self.g, &self.model, data_loss);
        self.g.backward(loss_node);
        let loss = self.g.nodes[loss_node].data;
        let acc = correct as f64 / n as f64;

        let grads = self.grad_norms();
        self.update(self.lr(k));
        let weights = self.weight_norms();

        // mean activations over the batch
        let mut hidden: Vec<Vec<f64>> =
            self.model.layers().iter().map(|l| vec![0.0; l.neurons().len()]).collect();
        let mut input = vec![0.0f64; self.dim];
        for i in 0..n {
            let a = self.activations_for(&self.xs[i].clone());
            for (li, la) in a.iter().enumerate() {
                for (ni, &v) in la.iter().enumerate() {
                    hidden[li][ni] += v;
                }
            }
            for (d, &v) in self.xs[i].iter().enumerate() {
                input[d] += v.abs();
            }
        }
        let nf = n as f64;
        input.iter_mut().for_each(|a| *a /= nf);
        hidden.iter_mut().for_each(|l| l.iter_mut().for_each(|a| *a /= nf));

        Session::frame_json(k, -1, loss, acc, &grads, &weights, &input, &hidden)
    }

    /// Forward a single feature vector; `> 0` ⇒ positive class.
    pub fn classify(&mut self, x: &[f64]) -> f64 {
        self.g.nodes.truncate(self.watermark);
        let leaves: Vec<usize> = x.iter().map(|&v| self.g.value(v)).collect();
        let score = self.model.forward(&mut self.g, &leaves)[0];
        self.g.nodes[score].data
    }
}

/// Train a dataset (full-batch) to completion and return the JSON recording —
/// used by the native `record` example. At most ~60 frames are kept.
pub fn train_dataset(
    xs: &[Vec<f64>],
    ys: &[f64],
    hidden: &[usize],
    steps: usize,
    lr0: f64,
    rng: &mut impl Rng,
) -> (String, Session) {
    let mut s = Session::new(xs.to_vec(), ys.to_vec(), hidden, steps, lr0, false, rng);
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

    /// One-image-per-step SGD trains a separable set and classifies fresh points,
    /// and each frame records which image it used.
    #[test]
    fn sgd_steps_and_classifies() {
        let xs = vec![
            vec![1.0, 1.0, 1.0],
            vec![0.9, 1.1, 0.8],
            vec![-1.0, -1.0, -1.0],
            vec![-0.8, -1.2, -0.9],
        ];
        let ys = vec![1.0, 1.0, -1.0, -1.0];
        let mut rng = StdRng::seed_from_u64(7);
        let mut s = Session::new(xs, ys, &[8], 200, 0.1, true, &mut rng);

        assert_eq!(s.dim, 3);
        let mut frames = 0;
        let mut saw_img = false;
        while let Some(f) = s.step() {
            if f.contains("\"img\":") && !f.contains("\"img\":-1") {
                saw_img = true;
            }
            frames += 1;
        }
        assert_eq!(frames, 200);
        assert!(saw_img, "frames should record the image index used");
        assert!(s.classify(&[1.2, 1.0, 0.9]) > 0.0, "positive cluster misclassified");
        assert!(s.classify(&[-1.1, -0.9, -1.0]) < 0.0, "negative cluster misclassified");
    }
}
