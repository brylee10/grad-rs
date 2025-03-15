//! Data loader

use std::collections::HashSet;

use rand::seq::SliceRandom;
use thiserror::Error;

use crate::values::Value;

/// Errors for the dataloader
#[derive(Debug, Error)]
pub enum DataLoaderError {
    #[error(
        "All input vectors must have the same dimension. Received different sizes: {input_dims:?}"
    )]
    InputDimensionMismatch { input_dims: HashSet<usize> },
    #[error("Labels must have the same length as the data")]
    LabelLengthMismatch { label_len: usize, data_len: usize },
}

/// Data loader, returns batches of data and labels optionally shuffled
/// Takes inspiration from the PyTorch DataLoader
/// <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>
pub struct DataLoader {
    data: Vec<Vec<Value>>,
    // one hot encoded labels
    labels: Vec<Vec<Value>>,
    batch_size: usize,
    shuffle: bool,
}

impl DataLoader {
    pub fn new(
        data: Vec<Vec<f32>>,
        labels: Vec<Vec<u8>>,
        batch_size: usize,
        shuffle: bool,
    ) -> Result<Self, DataLoaderError> {
        if data.len() != labels.len() {
            return Err(DataLoaderError::LabelLengthMismatch {
                label_len: labels.len(),
                data_len: data.len(),
            });
        }
        let input_dims = data.iter().map(|d| d.len()).collect::<HashSet<_>>();
        if input_dims.len() > 1 {
            return Err(DataLoaderError::InputDimensionMismatch { input_dims });
        }
        let data = data
            .iter()
            .map(|d| d.iter().map(|v| Value::new(*v)).collect())
            .collect();
        let labels = labels
            .iter()
            .map(|l| l.iter().map(|v| Value::new(*v as f32)).collect())
            .collect();
        Ok(Self {
            data,
            labels,
            batch_size,
            shuffle,
        })
    }

    #[cfg(test)]
    fn seeded_iter(&self, seed: u64) -> DataLoaderIterator<'_> {
        use rand::SeedableRng;
        use rand_pcg::Pcg64Mcg;

        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let mut indices = (0..self.data.len()).collect::<Vec<_>>();
        indices.shuffle(&mut rng);
        DataLoaderIterator {
            data: &self.data,
            labels: &self.labels,
            batch_size: self.batch_size,
            indices,
            curr_iter: 0,
        }
    }

    pub fn iter(&self) -> DataLoaderIterator<'_> {
        let mut indices = (0..self.data.len()).collect::<Vec<_>>();
        if self.shuffle {
            indices.shuffle(&mut rand::rng());
        }
        DataLoaderIterator {
            data: &self.data,
            labels: &self.labels,
            batch_size: self.batch_size,
            indices,
            curr_iter: 0,
        }
    }
}

/// An iterator which returns mini batches of data and labels until the end of the dataset
pub struct DataLoaderIterator<'a> {
    data: &'a [Vec<Value>],
    labels: &'a [Vec<Value>],
    batch_size: usize,
    // optionally shuffled indices
    indices: Vec<usize>,
    curr_iter: usize,
}

impl<'a> Iterator for DataLoaderIterator<'a> {
    type Item = (Vec<&'a [Value]>, Vec<&'a [Value]>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr_iter >= self.data.len() {
            return None;
        }
        let batch_data = self.indices[self.curr_iter..self.curr_iter + self.batch_size]
            .iter()
            .map(|&i| self.data[i].as_slice())
            .collect::<Vec<_>>();
        let batch_labels = self.indices[self.curr_iter..self.curr_iter + self.batch_size]
            .iter()
            .map(|i| self.labels[*i].as_slice())
            .collect::<Vec<_>>();
        self.curr_iter += self.batch_size;
        Some((batch_data, batch_labels))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataloader() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let labels = vec![vec![1, 0], vec![0, 1]];
        let dataloader = DataLoader::new(data, labels, 2, false).unwrap();
        let mut iter = dataloader.iter();
        assert_eq!(
            iter.next(),
            Some((
                vec![
                    [Value::new(1.0), Value::new(2.0), Value::new(3.0)].as_slice(),
                    [Value::new(4.0), Value::new(5.0), Value::new(6.0)].as_slice(),
                ],
                vec![
                    [Value::new(1.0), Value::new(0.0)].as_slice(),
                    [Value::new(0.0), Value::new(1.0)].as_slice(),
                ],
            ))
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_dataloader_shuffle() {
        let seed = 42;
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let labels = vec![vec![1, 0], vec![0, 1]];
        let dataloader = DataLoader::new(data, labels, 2, true).unwrap();
        let mut iter = dataloader.seeded_iter(seed);
        assert_eq!(
            iter.next(),
            Some((
                vec![
                    [Value::new(4.0), Value::new(5.0), Value::new(6.0)].as_slice(),
                    [Value::new(1.0), Value::new(2.0), Value::new(3.0)].as_slice(),
                ],
                vec![
                    [Value::new(0.0), Value::new(1.0)].as_slice(),
                    [Value::new(1.0), Value::new(0.0)].as_slice(),
                ],
            ))
        );
    }

    #[test]
    fn test_dataloader_errors() {
        // different length data and labels
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let labels = vec![vec![1, 0], vec![0, 1], vec![1, 0]];
        let expected_label_len = labels.len();
        let expected_data_len = data.len();
        let dataloader = DataLoader::new(data, labels, 2, false);
        assert!(matches!(
            dataloader,
            Err(DataLoaderError::LabelLengthMismatch {
                label_len,
                data_len,
            }) if label_len == expected_label_len && data_len == expected_data_len
        ));
    }
}
