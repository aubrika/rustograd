//! Training demo — parity with micrograd's `demo.ipynb`.
//!
//! Trains a 2→16→16→1 ReLU MLP on a make-moons dataset with the SVM max-margin
//! hinge loss + L2 regularization, then prints an ASCII picture of the learned
//! decision boundary. Run it with:
//!
//!     cargo run --release --example demo
//!
//! (`--release` matters: it's a full-batch pass over a 337-param net, 100 steps.)

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustograd::engine::Graph;
use rustograd::nn::MLP;

/// One sample from a standard normal, via Box-Muller (so we need no extra crate).
fn randn(rng: &mut impl Rng) -> f64 {
    let u1: f64 = rng.gen_range(0.0..1.0f64).max(1e-12);
    let u2: f64 = rng.gen_range(0.0..1.0f64);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Hand-rolled `sklearn.datasets.make_moons`: two interleaving half-circle arcs
/// plus Gaussian noise. Labels are ±1 (outer arc -1, inner arc +1).
fn make_moons(n: usize, noise: f64, rng: &mut impl Rng) -> (Vec<[f64; 2]>, Vec<f64>) {
    let n_out = n / 2;
    let n_in = n - n_out;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);

    for i in 0..n_out {
        let t = std::f64::consts::PI * i as f64 / (n_out as f64 - 1.0);
        xs.push([t.cos() + noise * randn(rng), t.sin() + noise * randn(rng)]);
        ys.push(-1.0);
    }
    for i in 0..n_in {
        let t = std::f64::consts::PI * i as f64 / (n_in as f64 - 1.0);
        xs.push([
            1.0 - t.cos() + noise * randn(rng),
            1.0 - t.sin() - 0.5 + noise * randn(rng),
        ]);
        ys.push(1.0);
    }
    (xs, ys)
}

/// Build the full-batch loss graph above the arena's current tail, returning the
/// loss node index and the batch accuracy.
///
///   data_loss = mean_i relu(1 - yᵢ · scoreᵢ)      (SVM max-margin hinge)
///   reg_loss  = alpha · Σ p²                       (L2 weight decay)
///   total     = data_loss + reg_loss
fn build_loss(g: &mut Graph, model: &MLP, xs: &[[f64; 2]], ys: &[f64]) -> (usize, f64) {
    let n = xs.len();
    let mut losses = Vec::with_capacity(n);
    let mut correct = 0usize;

    for (point, &yi) in xs.iter().zip(ys) {
        let x0 = g.value(point[0]);
        let x1 = g.value(point[1]);
        let score = model.forward(g, &[x0, x1])[0]; // single output neuron

        // accuracy: do the predicted sign and the label agree?
        if (yi > 0.0) == (g.nodes[score].data > 0.0) {
            correct += 1;
        }

        // hinge: relu(1 - yᵢ·score)
        let yi_node = g.value(yi);
        let ys_prod = g.mul(yi_node, score);
        let one = g.value(1.0);
        let margin = g.sub(one, ys_prod);
        losses.push(g.relu(margin));
    }

    // data_loss = mean(losses)
    let mut sum = losses[0];
    for &li in &losses[1..] {
        sum = g.add(sum, li);
    }
    let inv_n = g.value(1.0 / n as f64);
    let data_loss = g.mul(sum, inv_n);

    // reg_loss = alpha · Σ p²  — note each param now feeds BOTH this path and the
    // data-loss path; backward's `+=` accumulation sums them. That's weight decay.
    let params = model.parameters();
    let mut reg_sum = g.mul(params[0], params[0]);
    for &p in &params[1..] {
        let pp = g.mul(p, p);
        reg_sum = g.add(reg_sum, pp);
    }
    let alpha = g.value(1e-4);
    let reg_loss = g.mul(alpha, reg_sum);

    let total = g.add(data_loss, reg_loss);
    (total, correct as f64 / n as f64)
}

/// Draw the learned decision boundary as ASCII: `#` where the model scores > 0
/// (inner-moon side), space otherwise. Data points are overlaid as `O` (+1) and
/// `x` (-1) so you can see the boundary thread between the two moons.
fn plot(g: &mut Graph, watermark: usize, model: &MLP, xs: &[[f64; 2]], ys: &[f64]) {
    let (cols, rows) = (64usize, 28usize);
    let xmin = xs.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min) - 0.5;
    let xmax = xs.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max) + 0.5;
    let ymin = xs.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min) - 0.5;
    let ymax = xs.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max) + 0.5;

    // Region shading from the model.
    let mut grid = vec![vec![' '; cols]; rows];
    for (r, row) in grid.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            let x = xmin + (xmax - xmin) * c as f64 / (cols as f64 - 1.0);
            let y = ymax - (ymax - ymin) * r as f64 / (rows as f64 - 1.0); // row 0 = top
            g.nodes.truncate(watermark);
            let x0 = g.value(x);
            let x1 = g.value(y);
            let score = model.forward(g, &[x0, x1])[0];
            *cell = if g.nodes[score].data > 0.0 { '#' } else { '.' };
        }
    }

    // Overlay the data points.
    for (point, &yi) in xs.iter().zip(ys) {
        let c = ((point[0] - xmin) / (xmax - xmin) * (cols as f64 - 1.0)).round() as usize;
        let r = ((ymax - point[1]) / (ymax - ymin) * (rows as f64 - 1.0)).round() as usize;
        if r < rows && c < cols {
            grid[r][c] = if yi > 0.0 { 'O' } else { 'x' };
        }
    }

    println!("\nDecision boundary (`#` = +1 side, `.` = -1 side; O/x = data):");
    for row in &grid {
        println!("  {}", row.iter().collect::<String>());
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(1337);
    let (xs, ys) = make_moons(100, 0.1, &mut rng);

    let mut g = Graph::new();
    let model = MLP::new(&mut g, &mut rng, 2, &[16, 16, 1]);
    let watermark = g.nodes.len();
    println!("number of parameters: {}", model.parameters().len());

    for k in 0..100 {
        g.nodes.truncate(watermark); // drop last step's tape, keep the params
        let (loss, acc) = build_loss(&mut g, &model, &xs, &ys);
        g.backward(loss);

        let lr = 1.0 - 0.9 * k as f64 / 100.0; // decay 1.0 → 0.1
        for &p in &model.parameters() {
            g.nodes[p].data -= lr * g.nodes[p].grad;
        }

        if k % 10 == 0 || k == 99 {
            println!(
                "step {k:3}  loss {:.4}  acc {:.0}%",
                g.nodes[loss].data,
                acc * 100.0
            );
        }
    }

    plot(&mut g, watermark, &model, &xs, &ys);
}
