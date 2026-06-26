//! The autograd engine.
//!
//! Parity target: `micrograd/engine.py`.

#[derive(Debug, Clone, Copy)]
pub enum Op {
    Leaf,
    Add(usize, usize),
    Mul(usize, usize),
    Pow(usize, f64), // base index, exponent
    Relu(usize),
}

pub struct Node {
    pub data: f64,
    pub grad: f64,
    pub op: Op,
}

//"Arena" of nodes, each with a data value, a gradient, and an op that produced it. The arena is a Vec<Node> and each node is referenced by its index in the Vec. The arena is used to implement reverse-mode autodiff. 
// The arena is a struct called Graph, which has a Vec<Node> and methods to create new nodes and perform forward and backward passes.
pub struct Graph {
    pub nodes: Vec<Node>,
}

impl Graph {
    pub fn new() -> Self {
        Graph { nodes: Vec::new() }
    }

    /// Push a node and return its index
    fn push(&mut self, data: f64, op: Op) -> usize {
        self.nodes.push(Node { data, grad: 0.0, op });
        self.nodes.len() - 1
    }

    /// A leaf input — micrograd's `Value(x)`
    pub fn value(&mut self, data: f64) -> usize {
        self.push(data, Op::Leaf)
    }

    // ── Forward ops ─────────────────────────────────────────────────────────
    // Each computes its data from the inputs' data, then records the Op.

    pub fn add(&mut self, a: usize, b: usize) -> usize {
        self.push(self.nodes[a].data + self.nodes[b].data, Op::Add(a,b))
    }

    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        self.push(self.nodes[a].data * self.nodes[b].data, Op::Mul(a,b))
    }

    pub fn powf(&mut self, a: usize, exponent: f64) -> usize {
        self.push(self.nodes[a].data.powf(exponent), Op::Pow(a,exponent))
    }

    pub fn relu(&mut self, a: usize) -> usize {
        self.push(self.nodes[a].data.max(0.0), Op::Relu(a))     
    }

    pub fn neg(&mut self, a: usize) -> usize {
        let neg_1 = self.value(-1.0);
        self.mul(a, neg_1)
    }

    pub fn sub(&mut self, a: usize, b: usize) -> usize {
        let neg_b = self.neg(b);
        self.add(a, neg_b)
    }

    pub fn div(&mut self, a: usize, b: usize) -> usize {
        let inv_b = self.powf(b, -1.0);
        self.mul(a, inv_b)
    }

    // ── Backward ─────────────────────────────────────────────────────────────
    /// Reverse-mode autodiff from `root`. Zero all grads, seed `nodes[root].grad
    /// = 1.0`, then iterate the arena in REVERSE and `match` each node's op to
    /// accumulate (`+=`) gradient into its inputs.
    
    pub fn backward(&mut self, root: usize) {
        for n in &mut self.nodes { n.grad = 0.0; }
        self.nodes[root].grad = 1.0;

        for i in (0..self.nodes.len()).rev() {
            
            let op = self.nodes[i].op;
            let grad = self.nodes[i].grad;
            let data = self.nodes[i].data;

            match op {
                Op::Leaf => {},
                Op::Add(a, b) => {
                    self.nodes[a].grad += grad;
                    self.nodes[b].grad += grad;
                },
                Op::Mul(a, b) => {
                    self.nodes[a].grad += self.nodes[b].data * grad;
                    self.nodes[b].grad += self.nodes[a].data * grad;
                },
                Op::Pow(a, exponent) => {
                    self.nodes[a].grad += exponent * self.nodes[a].data.powf(exponent - 1.0) * grad;
                },
                Op::Relu(a) => {
                    self.nodes[a].grad += if data > 0.0 { grad } else { 0.0 };
                },
            }
        }
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}
