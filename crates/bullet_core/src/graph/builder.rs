use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt::Debug,
    num::NonZeroUsize,
    sync::Arc,
};

use super::{
    error::GraphError,
    operation::{GraphBuilderError, GraphBuilderErrorType, Operation},
    Graph,
};
use crate::{
    device::{Device, OperationError},
    shape::Shape,
    tensor::Tensor,
};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Node {
    pub idx: usize,
    pub(crate) shape: Shape,
    pub(crate) sparse: Option<NonZeroUsize>,
}

impl Node {
    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn reshape(mut self, shape: Shape) -> Result<Self, GraphBuilderErrorType> {
        if self.shape.size() == shape.size() {
            self.shape = shape;
            Ok(self)
        } else {
            Err(GraphBuilderErrorType::MismatchedInputShapes(vec![self.shape, shape]))
        }
    }

    pub fn is_sparse(&self) -> bool {
        self.sparse.is_some()
    }
}

pub(crate) struct NodeData {
    id: Option<String>,
    size: usize,
    requires_grad: bool,
    parent_operation: Option<Operation>,
    own: Node,
}

impl NodeData {
    fn new(
        id: Option<String>,
        parent_operation: Option<Operation>,
        size: usize,
        requires_grad: bool,
        sparse: Option<NonZeroUsize>,
    ) -> Self {
        let own = Node { idx: usize::MAX, shape: Shape::new(usize::MAX, usize::MAX), sparse };
        Self { id, size, requires_grad, parent_operation, own }
    }
}

#[derive(Default)]
pub struct GraphBuilder {
    nodes: Vec<NodeData>,
    roots: HashSet<Node>,
    inputs: HashSet<Node>,
    weights: HashSet<Node>,
    ids: HashSet<String>,
}

impl GraphBuilder {
    pub(crate) fn get_node(&self, node: Node) -> &NodeData {
        &self.nodes[node.idx]
    }

    fn create_node(
        &mut self,
        mut data: NodeData,
        shape: Shape,
        sparse: Option<NonZeroUsize>,
    ) -> Result<Node, GraphBuilderError> {
        if let Some(id) = data.id.as_ref() {
            assert!(self.ids.insert(id.to_string()))
        }

        let node = Node { idx: self.nodes.len(), shape, sparse };
        data.own = node;

        if let Some(op) = &data.parent_operation {
            for parent in &op.nodes() {
                self.roots.remove(parent);
            }
        }

        self.nodes.push(data);
        self.roots.insert(node);

        Ok(node)
    }

    pub fn create_dense_input(&mut self, id: &str, shape: Shape) -> Result<Node, GraphBuilderError> {
        let data = NodeData::new(Some(id.to_string()), None, shape.size(), false, None);
        let node = self.create_node(data, shape, None)?;

        self.inputs.insert(node);

        Ok(node)
    }

    pub fn create_sparse_input(&mut self, id: &str, shape: Shape, nnz: usize) -> Result<Node, GraphBuilderError> {
        let data = NodeData::new(Some(id.to_string()), None, shape.size(), false, None);
        let node = self.create_node(data, shape, Some(NonZeroUsize::try_from(nnz).unwrap()))?;

        self.inputs.insert(node);

        Ok(node)
    }

    pub fn create_weights(&mut self, id: &str, shape: Shape) -> Result<Node, GraphBuilderError> {
        let data = NodeData::new(Some(id.to_string()), None, shape.size(), true, None);
        let node = self.create_node(data, shape, None)?;

        self.weights.insert(node);

        Ok(node)
    }

    pub fn create_result_of_operation(
        &mut self,
        operation: Operation,
        requires_grad: bool,
    ) -> Result<Node, GraphBuilderError> {
        match operation.output_shape() {
            Ok(shape) => {
                let data = NodeData::new(None, Some(operation), shape.size(), requires_grad, None);
                self.create_node(data, shape, None)
            }
            Err(s) => panic!("{s:#?}"),
        }
    }

    pub fn root(&self) -> Node {
        assert_eq!(self.roots.len(), 1, "Graph must have a single output!");
        *self.roots.iter().next().unwrap()
    }

    pub fn build<D: Device>(self, device: D) -> Result<Graph<D>, GraphError<D::DeviceError>> {
        assert_eq!(self.roots.len(), 1, "Graph must have a single output!");

        let root = *self.roots.iter().next().unwrap();
        assert!(self.get_node(root).requires_grad, "Output cannot be an input!");
        assert!(!self.weights.contains(&root), "Can't output trainable weights!");
        assert_eq!(root.shape, Shape::new(1, 1), "Graph output must be scalar!");
        assert_eq!(self.get_node(root).size, 1);

        let device = Arc::new(device);

        let mut nodes = Vec::new();
        for node_data in &self.nodes {
            let tensor = Tensor::new(
                device.clone(),
                node_data.size,
                node_data.requires_grad,
                node_data.parent_operation,
                node_data.own,
            );
            let tensor = tensor.map_err(OperationError::from);

            nodes.push(RefCell::new(tensor?));
        }

        let inputs =
            self.inputs.iter().map(|&node| (self.get_node(node).id.clone().unwrap(), node)).collect::<HashMap<_, _>>();

        let weights =
            self.weights.iter().map(|&node| (self.get_node(node).id.clone().unwrap(), node)).collect::<HashMap<_, _>>();

        Ok(Graph { nodes, root, inputs, weights, device })
    }
}
