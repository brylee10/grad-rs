//! Loss function(s)

use crate::values::Value;

/// Mean Squared Error Loss between two vectors of values
pub struct MSELoss;

impl MSELoss {
    pub fn call<T, U>(y_pred: &[T], y_true: &[U]) -> Value
    where
        T: AsRef<Value>,
        U: AsRef<Value>,
    {
        let loss = y_pred
            .iter()
            .zip(y_true.iter())
            .map(|(a, b)| (a.as_ref() - b.as_ref()).pow(&Value::new(2.0)))
            .sum::<Value>();
        loss / Value::new(y_pred.len() as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let y_pred = vec![Value::new(2.0), Value::new(3.0)];
        let y_true = vec![Value::new(1.0), Value::new(5.0)];
        let loss = MSELoss::call(&y_pred, &y_true);
        assert_eq!(loss.data(), 2.5);

        loss.backward();
        // dloss / dy_pred = 1/N * 2 * (y_pred - y_true)
        // dloss / dy_true = -1/N * 2 * (y_pred - y_true)
        assert_eq!(y_pred[0].grad(), 1.0);
        assert_eq!(y_pred[1].grad(), -2.0);
        assert_eq!(y_true[0].grad(), -1.0);
        assert_eq!(y_true[1].grad(), 2.0);
    }
}
