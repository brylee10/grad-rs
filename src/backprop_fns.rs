//! Incremental gradient update functions for backprop
//!
//! Applied depending on the operation which created outputs from inputs. Corresponds to
//! a node in the computation graph.

use crate::values::Value;

/// Represents the function in the computation graph
#[derive(Debug, Clone, Copy)]
pub enum BackpropFunc {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Pow,
    ReLU,
    Exp,
}

impl BackpropFunc {
    pub fn n_operands(&self) -> usize {
        match self {
            BackpropFunc::Add => 2,
            BackpropFunc::Sub => 2,
            BackpropFunc::Mul => 2,
            BackpropFunc::Div => 2,
            BackpropFunc::Neg => 1,
            BackpropFunc::Pow => 2,
            BackpropFunc::ReLU => 1,
            BackpropFunc::Exp => 1,
        }
    }
}

/// Takes two values and updates their gradients
/// Represents backprop for the operation `in1 + in2 = out`
pub fn add(in1: &Value, in2: &Value, out: &Value) {
    in1.0.borrow_mut().grad += out.0.borrow().grad;
    in2.0.borrow_mut().grad += out.0.borrow().grad;
}

/// Represents backprop for the operation `in1 - in2 = out`
pub fn sub(in1: &Value, in2: &Value, out: &Value) {
    in1.0.borrow_mut().grad += out.0.borrow().grad;
    in2.0.borrow_mut().grad += -out.0.borrow().grad;
}

/// Represents backprop for the operation `in1 * in2 = out`
pub fn mul(in1: &Value, in2: &Value, out: &Value) {
    in1.0.borrow_mut().grad += in2.0.borrow().data * out.0.borrow().grad;
    in2.0.borrow_mut().grad += in1.0.borrow().data * out.0.borrow().grad;
}

/// Represents backprop for the operation `in1 / in2 = out`
pub fn div(in1: &Value, in2: &Value, out: &Value) {
    let grad = out.0.borrow().grad;
    let in1_data = in1.0.borrow().data;
    let in2_data = in2.0.borrow().data;
    in1.0.borrow_mut().grad += grad / in2_data;
    in2.0.borrow_mut().grad += -in1_data * grad / (in2_data.powf(2.0));
}

/// Represents backprop for the operation `-in = out`
pub fn neg(in1: &Value, out: &Value) {
    in1.0.borrow_mut().grad += -out.0.borrow().grad;
}

/// Represents backprop for the operation `in1^in2 = out`
/// only support numerical float powers, da^b/db is only defined for positive a, so gradient is
/// not calculated for the exponent
pub fn pow(in1: &Value, in2: &Value, out: &Value) {
    let grad = out.0.borrow().grad;
    let in1_data = in1.0.borrow().data;
    let in2_data = in2.0.borrow().data;
    in1.0.borrow_mut().grad += in2_data * in1_data.powf(in2_data - 1.0) * grad;
    // in2.0.borrow_mut().grad += in1_data.powf(in2_data) * grad * in1_data.ln();
}

/// Represents backprop for the operation `relu(in) = out`
pub fn relu(in1: &Value, out: &Value) {
    let in1_data = in1.0.borrow().data;
    let grad = out.0.borrow().grad;
    in1.0.borrow_mut().grad += if in1_data > 0.0 { grad } else { 0.0 };
}

/// Represents backprop for the operation `exp(in) = out`
pub fn exp(in1: &Value, out: &Value) {
    let grad = out.0.borrow().grad; // accumulated gradient
    let out_data = out.0.borrow().data; // exp(in)
    in1.0.borrow_mut().grad += out_data * grad;
}

/// Applies a backprop function for operators with two operands
pub fn update_gradients_two_operands(in1: &Value, in2: &Value, out: &Value) {
    let backprop_fn = { out.0.borrow().backprop_fn };
    match backprop_fn {
        Some(BackpropFunc::Add) => add(in1, in2, out),
        Some(BackpropFunc::Sub) => sub(in1, in2, out),
        Some(BackpropFunc::Mul) => mul(in1, in2, out),
        Some(BackpropFunc::Div) => div(in1, in2, out),
        Some(BackpropFunc::Pow) => pow(in1, in2, out),
        None => {}
        _ => panic!("Invalid backprop function: {:?}", backprop_fn),
    }
}

/// Applies a backprop function for operators with one operand
pub fn update_gradients_one_operand(in1: &Value, out: &Value) {
    let backprop_fn = { out.0.borrow().backprop_fn };
    match backprop_fn {
        Some(BackpropFunc::Neg) => neg(in1, out),
        Some(BackpropFunc::ReLU) => relu(in1, out),
        Some(BackpropFunc::Exp) => exp(in1, out),
        None => {}
        _ => panic!("Invalid backprop function: {:?}", backprop_fn),
    }
}
