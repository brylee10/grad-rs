//! Optimizer(s)

use crate::values::Value;

/// Common interface for optimizers
/// Analogous to the torch.optim.Optimizer interface
/// <https://pytorch.org/docs/stable/optim.html#base-class>
pub trait Optim {
    /// Performs a single optimization step with accumulated gradients
    fn step(&mut self);
    /// Zeros gradients for all parameters
    fn zero_grad(&mut self);
}

/// SGD with momentum
pub struct SGD {
    params: Vec<Value>,
    // currently does not change the learning rate based on the iteration
    // ideally lr would decay over time
    lr: f32,
    momentum: f32,
    // velocity per parameter
    velocity: Vec<f32>,
}

impl SGD {
    pub fn new(params: Vec<Value>, lr: f32, momentum: f32) -> Self {
        let velocity = vec![0.0; params.len()];
        Self {
            params,
            lr,
            momentum,
            velocity,
        }
    }

    #[cfg(test)]
    fn velocities(&self) -> &[f32] {
        &self.velocity
    }
}

impl Optim for SGD {
    fn step(&mut self) {
        for (idx, param) in self.params.iter_mut().enumerate() {
            // SGD with momentum
            let velocity = self.momentum * self.velocity[idx] - self.lr * param.grad();
            let new_val = param.data() + velocity;
            self.velocity[idx] = velocity;
            param.set_data(new_val);
        }
    }

    fn zero_grad(&mut self) {
        for param in self.params.iter_mut() {
            param.zero_grad();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_no_momentum() {
        let a = Value::new(1.0);
        let b = Value::new(2.0);
        let c = &a + &b;
        c.backward();

        let mut optim = SGD::new(vec![a.clone(), b.clone(), c.clone()], 0.1, 0.0);
        optim.step();
        assert_eq!(a.grad(), 1.0);
        assert_eq!(b.grad(), 1.0);
        assert_eq!(c.grad(), 1.0);
        assert_eq!(a.data(), 0.9);
        assert_eq!(b.data(), 1.9);
        assert_eq!(c.data(), 2.9);
    }

    #[test]
    fn test_sgd_with_momentum() {
        let a = Value::new(1.0);
        let b = Value::new(2.0);
        let c = &a + &b;
        c.backward();

        let mut optim = SGD::new(vec![a.clone(), b.clone(), c.clone()], 0.1, 0.9);
        optim.step();
        assert_eq!(a.grad(), 1.0);
        assert_eq!(b.grad(), 1.0);
        assert_eq!(c.grad(), 1.0);
        assert_eq!(a.data(), 0.9);
        assert_eq!(b.data(), 1.9);
        assert_eq!(c.data(), 2.9);
        assert_eq!(optim.velocities(), &[-0.1, -0.1, -0.1]);
        optim.step();
        assert_eq!(a.data(), 0.71);
        assert_eq!(b.data(), 1.71);
        assert_eq!(c.data(), 2.71);
        assert_eq!(optim.velocities(), &[-0.19, -0.19, -0.19]);
    }
}
