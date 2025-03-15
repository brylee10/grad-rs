//! Components to build a neural network

use std::sync::atomic::{self, AtomicUsize};

use rand_distr::{Distribution, Normal};
use thiserror::Error;

use crate::values::Value;

/// Errors for the neural network
#[derive(Debug, Error)]
pub enum NNError {
    #[error("Input size mismatch")]
    InputSizeMismatch { expected: usize, got: usize },
}

/// Represents the torch.nn.Module. NNs should implement this trait.
/// <https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/module.py#L402>
pub trait Module {
    fn zero_grad(&mut self) {
        for p in self.parameters().iter_mut() {
            p.zero_grad();
        }
    }

    fn parameters(&self) -> Vec<Value>;
    fn forward(&self, inputs: &[Value]) -> Result<Vec<Value>, NNError>;
}

/// A single neuron in a layer of a NN
pub struct Neuron {
    /// weights of the neuron
    pub weights: Vec<Value>,
    /// bias of the neuron
    pub bias: Value,
}

impl Neuron {
    fn new(n_inputs: usize) -> Self {
        // He initialization to ensure the variance of the output is the same as the input
        // and keep weights relatively small to avoid exploding or vanishing gradients (or even just
        // activation values for that matter, e.g. softmax)
        let std = (2.0 / n_inputs as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        let weights = (0..n_inputs)
            .map(|_| Value::new(normal.sample(&mut rand::rng())))
            .collect();
        Self {
            weights,
            bias: Value::new(normal.sample(&mut rand::rng())),
        }
    }

    // Testing utility for a deterministic and simple neuron
    #[cfg(test)]
    fn new_ones(n_inputs: usize) -> Self {
        Self {
            weights: (0..n_inputs).map(|_| Value::new(1.0)).collect(),
            bias: Value::new(1.0),
        }
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.weights
            .iter()
            .chain(std::iter::once(&self.bias))
            .cloned()
            .collect()
    }

    pub fn forward(&self, inputs: &[Value]) -> Result<Value, NNError> {
        if inputs.len() != self.weights.len() {
            return Err(NNError::InputSizeMismatch {
                expected: self.weights.len(),
                got: inputs.len(),
            });
        }
        let output = self
            .weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, i)| w * i)
            .sum::<Value>();
        let output = &output + &self.bias;
        Ok(output)
    }
}

/// A layer of a neural network
pub struct Layer {
    neurons: Vec<Neuron>,
    n_output_nans: AtomicUsize,
    n_parameters_nans: AtomicUsize,
}

impl Layer {
    /// Creates a new layer with the given number of inputs and outputs
    pub fn new(n_inputs: usize, n_outputs: usize) -> Self {
        let neurons = (0..n_outputs).map(|_| Neuron::new(n_inputs)).collect();
        Self {
            neurons,
            n_output_nans: AtomicUsize::new(0),
            n_parameters_nans: AtomicUsize::new(0),
        }
    }

    #[cfg(test)]
    fn new_ones(n_inputs: usize, n_outputs: usize) -> Self {
        let neurons = (0..n_outputs).map(|_| Neuron::new_ones(n_inputs)).collect();
        Self {
            neurons,
            n_output_nans: AtomicUsize::new(0),
            n_parameters_nans: AtomicUsize::new(0),
        }
    }

    /// Returns all the parameters in the layer
    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }

    /// Computes forward pass for a layer
    pub fn forward(&self, inputs: &[Value]) -> Result<Vec<Value>, NNError> {
        let outputs = self
            .neurons
            .iter()
            .map(|n| n.forward(inputs))
            .collect::<Result<Vec<_>, _>>()?;
        let n_output_nans = outputs.iter().filter(|v| v.data().is_nan()).count();
        self.n_output_nans
            .store(n_output_nans, atomic::Ordering::Relaxed);
        let n_parameters_nans = self
            .parameters()
            .iter()
            .filter(|v| v.data().is_nan())
            .count();
        self.n_parameters_nans
            .store(n_parameters_nans, atomic::Ordering::Relaxed);
        log::debug!(
            "n_output_nans: {}, n_parameters_nans: {}",
            n_output_nans,
            n_parameters_nans
        );
        Ok(outputs)
    }
}

/// Applies ReLU to a set of values, works for arbitrary number of inputs
#[derive(Default)]
pub struct ReLU {
    n_dead_neurons: AtomicUsize,
}

impl ReLU {
    pub fn new() -> Self {
        Self {
            n_dead_neurons: AtomicUsize::new(0),
        }
    }

    /// Takes the element-wise ReLU of the input values
    pub fn forward(&self, inputs: &[Value]) -> Vec<Value> {
        let n_dead_neurons = inputs.iter().filter(|v| v.data() <= 0.0).count();
        self.n_dead_neurons
            .store(n_dead_neurons, atomic::Ordering::Relaxed);
        inputs.iter().map(|v| v.relu()).collect()
    }

    /// Returns the number of dead neurons in the layer (used for debugging)
    pub fn n_dead_neurons(&self) -> usize {
        self.n_dead_neurons.load(atomic::Ordering::Relaxed)
    }
}

/// Applies softmax to a set of values
#[derive(Default)]
pub struct Softmax {}

impl Softmax {
    pub fn new() -> Self {
        Self {}
    }

    pub fn forward(&self, inputs: &[Value]) -> Vec<Value> {
        let exp_sum = inputs.iter().map(|v| v.exp()).sum::<Value>();
        // Note: Large weights can cause overflow in the exponential function, leading to dividing by `inf`, for example
        // which causes the softmax to return NaN, so it is important to initialize the weights properly
        inputs.iter().map(|v| v.exp() / exp_sum.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::assert_eq_float;

    use super::*;

    #[test]
    fn test_layer_forward() {
        let layer = Layer::new_ones(2, 3);
        let inputs = vec![Value::new(1.0), Value::new(2.0)];
        let outputs = layer.forward(&inputs).unwrap();
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].data(), 4.0);
        assert_eq!(outputs[1].data(), 4.0);
        assert_eq!(outputs[2].data(), 4.0);
    }

    #[test]
    fn test_dim_mismatch() {
        let layer = Layer::new_ones(2, 3);
        let inputs = vec![Value::new(1.0)];
        let outputs = layer.forward(&inputs).unwrap_err();
        assert!(matches!(
            outputs,
            NNError::InputSizeMismatch {
                expected: 2,
                got: 1
            }
        ));
    }

    #[test]
    fn test_softmax() {
        let softmax = Softmax::new();
        let inputs = vec![Value::new(1.0), Value::new(2.0)];
        let mut outputs = softmax.forward(&inputs);
        assert_eq!(outputs.len(), 2);
        assert_eq_float!(outputs[0].data(), 0.2689414);
        assert_eq_float!(outputs[1].data(), 0.7310585);

        // Softmax(x1, x2) = (exp(x1) / (exp(x1) + exp(x2)), exp(x2) / (exp(x1) + exp(x2)))
        // Let s1 = exp(x1) / (exp(x1) + exp(x2)) and s2 = exp(x2) / (exp(x1) + exp(x2))
        // d s1 / dx1 = s1 * (1 - s1)
        // d s1 / dx2 = -s1 * s2
        outputs[0].backward();
        let s1 = outputs[0].data();
        let s2 = outputs[1].data();
        assert_eq_float!(inputs[0].grad(), s1 * (1.0 - s1));
        assert_eq_float!(inputs[1].grad(), -s1 * s2);

        // Note that `inputs[i].zero_grad()` (zeroing leaf nodes) is insufficient because there are many intermediate nodes
        // created to compute the output that will not be zeroed out! Printing `outputs[0]` will show these intermediate nodes.
        outputs[0].zero_grad();
        // d s2 / dx1 = -s1 * s2
        // d s2 / dx2 = s2 * (1 - s2)
        outputs[1].backward();
        assert_eq_float!(inputs[0].grad(), -s1 * s2);
        assert_eq_float!(inputs[1].grad(), s2 * (1.0 - s2));
    }
}
