//! Scalar values which form a computation graph
use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::{self, Debug, Display},
    iter::Sum,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

use rand::Rng;

use crate::backprop_fns::{
    BackpropFunc, update_gradients_one_operand, update_gradients_two_operands,
};

type SharedValue = Rc<RefCell<InnerValue>>;

/// Newtype representing a shared value in a computation graph
#[derive(Debug, Clone)]
pub struct Value(pub(crate) SharedValue);

impl Value {
    /// Create a new value, not derived from any other values
    pub fn new(data: f32) -> Self {
        Self(Rc::new(RefCell::new(InnerValue::new(data, None))))
    }

    /// Create a new value derived from an operation between two values (i.e. not a leaf node)
    fn new_derived(data: f32, backprop_fn: BackpropFunc) -> Self {
        Self(Rc::new(RefCell::new(InnerValue::new(
            data,
            Some(backprop_fn),
        ))))
    }

    fn add_child(&self, child: Value) {
        self.0.borrow_mut().children.push(child.0);
    }

    pub fn data(&self) -> f32 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f32 {
        self.0.borrow().grad
    }

    // Strictly, &mut isn't needed since the value is behind a shared mutable type,
    // but it indicates that the value should be excusively mutable
    pub fn set_data(&mut self, data: f32) {
        self.0.borrow_mut().data = data;
    }

    /// Unlike PyTorch which only zeros out the gradients of the leaf nodes, this zeros out
    /// all gradients in the computation graph which are children of this node
    pub fn zero_grad(&mut self) {
        // Zero out all gradients, recursively
        self.0.borrow_mut().grad = 0.0;

        // Traverse all nodes as if backpropagating, but zero out the gradients
        let mut backprop_order = vec![];
        let mut visited: HashSet<u64> = HashSet::new();
        self.backward_inner(&mut backprop_order, &mut visited);
        for value in backprop_order.into_iter().rev() {
            value.0.borrow_mut().grad = 0.0;
        }
    }

    pub fn backward(&self) {
        // d out / d out = 1
        self.0.borrow_mut().grad = 1.0;

        let mut backprop_order = vec![];
        let mut visited: HashSet<u64> = HashSet::new();

        // visit in post order
        self.backward_inner(&mut backprop_order, &mut visited);

        // apply backprop, reversed to start from root first
        for value in backprop_order.into_iter().rev() {
            let n_operands = value.0.borrow().children.len();
            if let Some(f) = value.0.borrow().backprop_fn {
                debug_assert!(f.n_operands() == n_operands);
            }
            match n_operands {
                0 => {}
                1 => {
                    let in1 = Value(value.0.borrow().children[0].clone());
                    update_gradients_one_operand(&in1, &value);
                }
                2 => {
                    let in1 = Value(value.0.borrow().children[0].clone());
                    let in2 = Value(value.0.borrow().children[1].clone());
                    update_gradients_two_operands(&in1, &in2, &value);
                }
                _ => {
                    panic!("Unsupported number of operands: {}", n_operands);
                }
            }
        }
    }

    fn backward_inner(&self, backprop_order: &mut Vec<Value>, visited: &mut HashSet<u64>) {
        for c in self.0.borrow().children.iter() {
            if visited.contains(&c.borrow().id) {
                continue;
            }
            visited.insert(c.borrow().id);
            let value = Value(c.clone());
            value.backward_inner(backprop_order, visited);
        }
        backprop_order.push(self.clone());
    }
}

// Various operations on values
impl Value {
    pub fn pow(&self, other: &Value) -> Value {
        let result = Value::new_derived(
            self.0.borrow().data.powf(other.0.borrow().data),
            BackpropFunc::Pow,
        );
        result.add_child(self.clone());
        result.add_child(other.clone());

        result
    }

    pub fn relu(&self) -> Value {
        let result = Value::new_derived(self.0.borrow().data.max(0.0), BackpropFunc::ReLU);
        result.add_child(self.clone());

        result
    }

    pub fn exp(&self) -> Value {
        let result = Value::new_derived(self.0.borrow().data.exp(), BackpropFunc::Exp);
        result.add_child(self.clone());

        result
    }
}

// pretty print a value and its children recursively in a JSON-like format
impl Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn fmt_value(value: &Value, indent: usize, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let inner = value.0.borrow();
            let indent_str = " ".repeat(indent);
            let indent_inner = " ".repeat(indent + 2);
            writeln!(f, "{}{{", indent_str)?;
            writeln!(f, "{}\"data\": {},", indent_inner, inner.data)?;
            writeln!(f, "{}\"grad\": {},", indent_inner, inner.grad)?;
            writeln!(f, "{}\"id\": {},", indent_inner, inner.id)?;
            writeln!(
                f,
                "{}\"backprop_fn\": {:?},",
                indent_inner, inner.backprop_fn
            )?;
            writeln!(f, "{}\"children\": [", indent_inner)?;
            for (i, child) in inner.children.iter().enumerate() {
                let child_value = Value(child.clone());
                fmt_value(&child_value, indent + 4, f)?;
                if i < inner.children.len() - 1 {
                    writeln!(f, ",")?;
                } else {
                    writeln!(f)?;
                }
            }
            writeln!(f, "{}]", indent_inner)?;
            write!(f, "{}}}", indent_str)
        }
        fmt_value(self, 0, f)
    }
}

impl Add for &Value {
    type Output = Value;

    fn add(self, other: &Value) -> Value {
        // let self_nan = self.0.borrow().data.is_nan();
        // let other_nan = other.0.borrow().data.is_nan();
        // let data = if self_nan && !other_nan {
        //     other.0.borrow().data
        // } else if !self_nan && other_nan {
        //     self.0.borrow().data
        // } else if self_nan && other_nan {
        //     0.0
        // } else {
        //     self.0.borrow().data + other.0.borrow().data
        // };
        let data = self.0.borrow().data + other.0.borrow().data;

        let result = Value::new_derived(data, BackpropFunc::Add);
        result.add_child(self.clone());
        result.add_child(other.clone());

        result
    }
}

impl Sub for &Value {
    type Output = Value;

    fn sub(self, other: &Value) -> Value {
        let result = Value::new_derived(
            self.0.borrow().data - other.0.borrow().data,
            BackpropFunc::Sub,
        );
        result.add_child(self.clone());
        result.add_child(other.clone());

        result
    }
}

impl Mul for &Value {
    type Output = Value;

    fn mul(self, other: &Value) -> Value {
        let result = Value::new_derived(
            self.0.borrow().data * other.0.borrow().data,
            BackpropFunc::Mul,
        );
        result.add_child(self.clone());
        result.add_child(other.clone());

        result
    }
}

impl Div for &Value {
    type Output = Value;

    fn div(self, other: &Value) -> Value {
        let data = if other.0.borrow().data.is_nan() {
            0.0
        } else {
            self.0.borrow().data / other.0.borrow().data
        };
        let result = Value::new_derived(data, BackpropFunc::Div);
        result.add_child(self.clone());
        result.add_child(other.clone());

        result
    }
}

impl Neg for &Value {
    type Output = Value;

    fn neg(self) -> Value {
        let result = Value::new_derived(-self.0.borrow().data, BackpropFunc::Neg);
        result.add_child(self.clone());

        result
    }
}

impl Sum for Value {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Value::new(0.0), |acc, v| acc + v)
    }
}

/// Convenience macro to implement operations on `[Value]` when ownership can be transferred
macro_rules! impl_arithmetic(
    ($trait:ident, $trait_method:ident, $operator:tt, $struct:ident) => {
        impl $trait for $struct {
            type Output = Self;

            fn $trait_method(self, other: Self) -> Self {
                &self $operator &other
            }
        }
    }
);
impl_arithmetic!(Add, add, +, Value);
impl_arithmetic!(Sub, sub, -, Value);
impl_arithmetic!(Mul, mul, *, Value);
impl_arithmetic!(Div, div, /, Value);

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.0.borrow().data == other.0.borrow().data
    }
}

impl Eq for Value {}

// unfortunate that there isn't a blanket impl for AsRef<T> for T
// <https://doc.rust-lang.org/std/convert/trait.AsRef.html#reflexivity>
impl AsRef<Value> for Value {
    fn as_ref(&self) -> &Value {
        self
    }
}

#[derive(Debug)]
pub(crate) struct InnerValue {
    // the network uses 32 bit precision floats (roughly 7 decimal digits of precision)
    pub(crate) data: f32,
    /// gradient of the value with respect to the output
    pub(crate) grad: f32,
    /// List of the node inputs in the forward pass
    /// These nodes are "children" in the backwards pass
    children: Vec<SharedValue>,
    /// Unique identifier for the node
    id: u64,
    /// The function which created this value from its children
    /// `None` when the value is a leaf node
    pub(crate) backprop_fn: Option<BackpropFunc>,
}

impl InnerValue {
    pub fn new(data: f32, backprop_fn: Option<BackpropFunc>) -> Self {
        Self {
            data,
            grad: 0.0,
            children: vec![],
            id: generate_random_id(),
            backprop_fn,
        }
    }
}

fn generate_random_id() -> u64 {
    let mut rng = rand::rng();
    rng.random() // generates a random u64
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[macro_export]
    macro_rules! assert_eq_float {
        ($a:expr, $b:expr) => {
            assert!((($a) - ($b)).abs() < 1e-6);
        };
    }

    #[test]
    fn test_add() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);

        let c = &a + &b;
        assert_eq!(c.data(), 5.0);
        c.backward();

        // dc/da = 1
        // dc/db = 1
        assert_eq!(a.grad(), 1.0);
        assert_eq!(b.grad(), 1.0);
    }

    #[test]
    fn test_mul() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);

        let c = &a * &b;
        assert_eq!(c.data(), 6.0);

        c.backward();

        // dc/da = b
        // dc/db = a
        assert_eq!(a.grad(), 3.0);
        assert_eq!(b.grad(), 2.0);
    }

    #[test]
    fn test_neg() {
        let a = Value::new(2.0);
        let b = -&a;
        assert_eq!(b.data(), -2.0);

        b.backward();

        assert_eq!(a.grad(), -1.0);
    }

    #[test]
    fn test_sub() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);

        let c = &a - &b;
        assert_eq!(c.data(), -1.0);

        c.backward();

        // dc/da = 1
        // dc/db = -1
        assert_eq!(a.grad(), 1.0);
        assert_eq!(b.grad(), -1.0);
    }

    #[test]
    fn test_div() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);

        let c = &a / &b;
        assert_eq_float!(c.data(), 2.0 / 3.0);

        c.backward();

        // dc/da = 1/b
        // dc/db = -a/b^2
        assert_eq_float!(a.grad(), 1.0 / 3.0);
        assert_eq_float!(b.grad(), -2.0 / 9.0);
    }

    #[test]
    fn test_pow() {
        let a = Value::new(2.0);
        let b = Value::new(2.0);
        let c = a.pow(&b);
        assert_eq_float!(c.data(), 4.0);

        c.backward();

        // dc/da = 2a
        // dc/db = a^2 * ln(a)
        assert_eq_float!(a.grad(), 4.0);
        assert_eq_float!(b.grad(), 4.0 * 2.0f32.ln());
    }

    #[test]
    fn test_relu() {
        let a = Value::new(1.0);
        let b = Value::new(2.0);
        let c = &a * &b;
        let z = c.relu();
        assert_eq_float!(z.data(), 2.0);

        z.backward();

        // dz/dc = 1
        // dc/da = b
        // dc/db = a
        assert_eq_float!(a.grad(), 2.0);
        assert_eq_float!(b.grad(), 1.0);
        assert_eq_float!(c.grad(), 1.0);
    }
}
