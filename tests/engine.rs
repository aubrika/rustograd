//! Integration tests for the autograd engine.
//!
//! Parity target: `micrograd/test/test_engine.py`.

use rustograd::engine::Graph;

const TOL: f64 = 1e-6;

/// Mirrors micrograd's `test_sanity_check`:
///   x = Value(-4.0)
///   z = 2*x + 2 + x
///   q = z.relu() + z*x
///   h = (z*z).relu()
///   y = h + q + q*x
///   y.backward()

#[test]
fn sanity_check() {
    let mut g = Graph::new();
    let x = g.value(-4.0);

    // z = 2*x + 2 + x
    let two = g.value(2.0);
    let z = g.mul(two, x);
    let two2 = g.value(2.0);
    let z = g.add(z, two2);
    let z = g.add(z, x);

    // q = z.relu() + z*x
    let zr = g.relu(z);
    let zx = g.mul(z, x);
    let q = g.add(zr, zx);

    // h = (z*z).relu()
    let zz = g.mul(z, z);
    let h = g.relu(zz);

    // y = h + q + q*x
    let y = g.add(h, q);
    let qx = g.mul(q, x);
    let y = g.add(y, qx);

    g.backward(y);

    assert!((g.nodes[y].data - (-20.0)).abs() < TOL, "y.data = {}", g.nodes[y].data);
    assert!((g.nodes[x].grad - 46.0).abs() < TOL, "x.grad = {}", g.nodes[x].grad);
}

/// Mirrors micrograd's `test_more_ops`: exercises +, *, **, /, -, unary neg and relu, and checks the final output and leaf grads.
#[test]
fn more_ops() {
    let mut g = Graph::new();
    let a = g.value(-4.0);
    let b = g.value(2.0);

    // c = a + b
    let mut c = g.add(a, b);

    // d = a*b + b**3
    let ab = g.mul(a, b);
    let b3 = g.powf(b, 3.0);
    let mut d = g.add(ab, b3);

    // c += c + 1
    let one = g.value(1.0);
    let c1 = g.add(c, one);
    c = g.add(c, c1);

    // c += 1 + c + (-a)
    let one2 = g.value(1.0);
    let oc = g.add(one2, c);
    let na = g.neg(a);
    let t = g.add(oc, na);
    c = g.add(c, t);

    // d += d*2 + (b + a).relu()
    let two = g.value(2.0);
    let d2 = g.mul(d, two);
    let ba = g.add(b, a);
    let rba = g.relu(ba);
    let t = g.add(d2, rba);
    d = g.add(d, t);

    // d += 3*d + (b - a).relu()
    let three = g.value(3.0);
    let d3 = g.mul(three, d);
    let bma = g.sub(b, a);
    let rbma = g.relu(bma);
    let t = g.add(d3, rbma);
    d = g.add(d, t);

    // e = c - d ; f = e**2 ; out = f/2 + 10/f
    let e = g.sub(c, d);
    let f = g.powf(e, 2.0);
    let two_f = g.value(2.0);
    let out = g.div(f, two_f);
    let ten = g.value(10.0);
    let tf = g.div(ten, f);
    let out = g.add(out, tf);

    g.backward(out);

    assert!((g.nodes[out].data - 24.70408163265306).abs() < TOL, "out.data = {}", g.nodes[out].data);
    assert!((g.nodes[a].grad - 138.83381924198252).abs() < TOL, "a.grad = {}", g.nodes[a].grad);
    assert!((g.nodes[b].grad - 645.5772594752186).abs() < TOL, "b.grad = {}", g.nodes[b].grad);
}

/// Build a fixed expression over the leaves `xs`. The leaves are created first,
/// so leaf `i` is node index `i` and its grad is `g.nodes[i].grad` after backward.
/// Returns the graph and the output node's index.
///
/// f(x0,x1,x2) = relu(x0*x1) + x2^3 / x0 - (x1 - x0)
fn build(xs: &[f64]) -> (Graph, usize) {
    let mut g = Graph::new();
    let v: Vec<usize> = xs.iter().map(|&x| g.value(x)).collect();

    let p = g.mul(v[0], v[1]);
    let r = g.relu(p);
    let cube = g.powf(v[2], 3.0);
    let dq = g.div(cube, v[0]);
    let s = g.sub(v[1], v[0]);
    let t = g.add(r, dq);
    let out = g.sub(t, s);
    (g, out)
}

/// Gradient check: compare each leaf's analytic grad (from `backward`) against a
/// central finite difference of the forward pass. This is the test that catches a
/// wrong local gradient if a new op is added later.
#[test]
fn grad_check_finite_difference() {
    let xs = [2.0_f64, 3.0, 1.5]; // chosen so relu's input (x0*x1) is positive
    let h = 1e-6;

    // Analytic grads.
    let (mut g, out) = build(&xs);
    g.backward(out);
    let analytic: Vec<f64> = (0..xs.len()).map(|i| g.nodes[i].grad).collect();

    // Numeric grads via central difference.
    let eval = |xs: &[f64]| -> f64 {
        let (g, out) = build(xs);
        g.nodes[out].data
    };
    for i in 0..xs.len() {
        let mut up = xs.to_vec();
        up[i] += h;
        let mut down = xs.to_vec();
        down[i] -= h;
        let numeric = (eval(&up) - eval(&down)) / (2.0 * h);
        assert!(
            (analytic[i] - numeric).abs() < 1e-4,
            "leaf {i}: analytic {} vs numeric {}",
            analytic[i],
            numeric
        );
    }
}
