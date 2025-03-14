use std::{
    collections::HashMap,
    ops::{Add, Sub},
    sync::{Mutex, MutexGuard},
};

use bullet_core::graph::{
    builder::{GraphBuilder, Node},
    operation::Operation,
    Graph,
};

use crate::{Activation, ExecutionContext, Shape};

#[derive(Clone, Copy, Debug)]
pub enum InitSettings {
    Zeroed,
    Normal { mean: f32, stdev: f32 },
    Uniform { mean: f32, stdev: f32 },
}

#[derive(Default)]
pub struct NetworkBuilder {
    graph_builder: Mutex<GraphBuilder>,
    init_data: Mutex<HashMap<String, InitSettings>>,
}

impl NetworkBuilder {
    fn builder(&self) -> MutexGuard<GraphBuilder> {
        self.graph_builder.try_lock().unwrap()
    }

    fn init(&self) -> MutexGuard<HashMap<String, InitSettings>> {
        self.init_data.try_lock().unwrap()
    }

    pub fn new_dense_input<'a>(&'a self, id: &str, shape: Shape) -> NetworkBuilderNode<'a> {
        let node = self.builder().create_dense_input(id, shape).unwrap();
        NetworkBuilderNode { node, builder: self }
    }

    pub fn new_sparse_input<'a>(&'a self, id: &str, shape: Shape, nnz: usize) -> NetworkBuilderNode<'a> {
        let node = self.builder().create_sparse_input(id, shape, nnz).unwrap();
        NetworkBuilderNode { node, builder: self }
    }

    pub fn new_weights<'a>(&'a self, id: &str, shape: Shape, init: InitSettings) -> NetworkBuilderNode<'a> {
        let node = self.builder().create_weights(id, shape).unwrap();
        self.init().insert(id.to_string(), init);
        NetworkBuilderNode { node, builder: self }
    }

    pub fn new_affine(&self, id: &str, input_size: usize, output_size: usize) -> Affine {
        self.new_affine_custom(id, input_size, output_size, 1)
    }

    pub fn new_affine_custom(&self, id: &str, input_size: usize, output_size: usize, bias_cols: usize) -> Affine {
        let wid = format!("{}w", id);
        let init = InitSettings::Normal { mean: 0.0, stdev: 1.0 / (input_size as f32 * bias_cols as f32).sqrt() };
        let weights = self.new_weights(&wid, Shape::new(output_size, input_size), init);
        let bias = self.new_weights(&format!("{}b", id), Shape::new(output_size, bias_cols), InitSettings::Zeroed);

        Affine { weights: weights.node, bias: bias.node }
    }

    pub fn apply(&self, operation: Operation) -> NetworkBuilderNode {
        match self.builder().create_result_of_operation(operation, true) {
            Ok(node) => NetworkBuilderNode { node, builder: self },
            Err(e) => {
                println!("{e:#?}");
                panic!();
            }
        }
    }

    pub fn build(self, execution_context: ExecutionContext) -> Graph<ExecutionContext> {
        let mut builder = self.graph_builder.into_inner().unwrap();
        builder.create_result_of_operation(Operation::ReduceAcrossBatch(builder.root()), true).unwrap();
        let mut graph = builder.build(execution_context).unwrap();

        for (id, init_data) in self.init_data.lock().unwrap().iter() {
            match *init_data {
                InitSettings::Zeroed => {}
                InitSettings::Normal { mean, stdev } => {
                    graph.get_weights_mut(id).seed_random(mean, stdev, true).unwrap()
                }
                InitSettings::Uniform { mean, stdev } => {
                    graph.get_weights_mut(id).seed_random(mean, stdev, false).unwrap()
                }
            };
        }

        graph
    }
}

#[derive(Clone, Copy)]
pub struct NetworkBuilderNode<'a> {
    node: Node,
    builder: &'a NetworkBuilder,
}

impl Add<Self> for NetworkBuilderNode<'_> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.linear_comb(1.0, rhs, 1.0)
    }
}

impl Sub<Self> for NetworkBuilderNode<'_> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.linear_comb(1.0, rhs, -1.0)
    }
}

impl NetworkBuilderNode<'_> {
    pub fn node(self) -> Node {
        self.node
    }

    pub fn reshape(mut self, shape: Shape) -> Self {
        self.node = self.node.reshape(shape).unwrap();
        self
    }

    pub fn activate(self, activation: Activation) -> Self {
        self.builder.apply(Operation::Activate(self.node, activation))
    }

    pub fn select(self, buckets: Self) -> Self {
        self.builder.apply(Operation::Select(self.node, buckets.node))
    }

    pub fn concat(self, rhs: Self) -> Self {
        self.builder.apply(Operation::Concat(self.node, rhs.node))
    }

    pub fn linear_comb(self, alpha: f32, rhs: Self, beta: f32) -> Self {
        self.builder.apply(Operation::LinearCombination(alpha, self.node, beta, rhs.node))
    }

    pub fn matmul(self, rhs: Self) -> Self {
        if rhs.node.is_sparse() {
            self.builder.apply(Operation::SparseAffine(self.node, rhs.node, None))
        } else {
            self.builder.apply(Operation::Matmul(self.node, false, rhs.node, false))
        }
    }

    pub fn gemm(self, transa: bool, rhs: Self, transb: bool) -> Self {
        self.builder.apply(Operation::Matmul(self.node, transa, rhs.node, transb))
    }

    pub fn mpe(self, targets: Self, power: f32) -> Self {
        self.builder.apply(Operation::PowerError(self.node, targets.node, power))
    }

    pub fn mse(self, targets: Self) -> Self {
        self.mpe(targets, 2.0)
    }

    pub fn pairwise_mul(self) -> Self {
        self.builder.apply(Operation::PairwiseMul(self.node, false))
    }

    pub fn pairwise_mul_post_affine_dual(self) -> Self {
        self.builder.apply(Operation::PairwiseMul(self.node, true))
    }

    pub fn mask(self, mask: Self) -> Self {
        self.builder.apply(Operation::Mask(self.node, mask.node))
    }

    pub fn gather(self, indices: Self) -> Self {
        self.builder.apply(Operation::Gather(self.node, indices.node))
    }

    pub fn softmax_crossentropy_loss(self, targets: Self) -> Self {
        self.builder.apply(Operation::SoftmaxCrossEntropyLoss(self.node, targets.node))
    }

    pub fn masked_softmax_crossentropy_loss(self, targets: Self, mask: Self) -> Self {
        self.builder.apply(Operation::MaskedSoftmaxCrossEntropyLoss(mask.node, self.node, targets.node))
    }

    pub fn slice_rows(self, start: usize, end: usize) -> Self {
        self.builder.apply(Operation::Slice(self.node, start, end))
    }

    pub fn to_dense(self) -> Self {
        let node = self.builder.builder().create_result_of_operation(Operation::ToDense(self.node), false).unwrap();
        Self { node, builder: self.builder }
    }
}

#[derive(Clone, Copy)]
pub struct Affine {
    pub weights: Node,
    pub bias: Node,
}

impl Affine {
    pub fn forward(self, input: NetworkBuilderNode<'_>) -> NetworkBuilderNode<'_> {
        if input.node.is_sparse() {
            input.builder.apply(Operation::SparseAffine(self.weights, input.node, Some(self.bias)))
        } else {
            input.builder.apply(Operation::Affine(self.weights, input.node, self.bias))
        }
    }

    pub fn forward_sparse_dual_with_activation<'a>(
        self,
        stm: NetworkBuilderNode<'a>,
        ntm: NetworkBuilderNode<'a>,
        activation: Activation,
    ) -> NetworkBuilderNode<'a> {
        stm.builder.apply(Operation::SparseAffineDualActivate(self.weights, stm.node, ntm.node, self.bias, activation))
    }
}
