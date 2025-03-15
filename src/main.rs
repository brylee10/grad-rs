//! A simple implementation of a neural network for multiclass classification
//! using the library provided by `grad_rs`
//!
//! # Usage
//! Runnable via
//! ```sh
//! cargo run -- -h
//! cargo run
//! ```
//!
//! Supports a few classic datasets out of the box and allows custom learning rate, momentum, batch size, etc.

use grad_rs::{
    dataloader::DataLoader,
    datasets::{Dataset, draw_dot, load_dataset, plot_data, plot_decision_boundary},
    loss::MSELoss,
    nn::{Layer, Module, NNError, ReLU, Softmax},
    optim::{Optim, SGD},
    values::Value,
};

use clap::Parser;

#[derive(Parser)]
struct Args {
    #[clap(short, long, default_value_t = Dataset::XOR)]
    dataset: Dataset,
    #[clap(short, long, default_value_t = 1000)]
    class_size: usize,
    #[clap(short, long, default_value_t = 50)]
    batch_size: usize,
    #[clap(short, long, default_value_t = 50)]
    epochs: usize,
    #[clap(short, long, default_value_t = 0.001)]
    lr: f32,
    #[clap(short, long, default_value_t = 0.9)]
    momentum: f32,
    #[clap(short, long, default_value_t = 5)]
    print_epochs: usize,
    #[clap(short, long, default_value_t = format!("output"))]
    output_dir: String,
    // Note that when increasing the hidden size, activation values may explode if
    // the weights are not initialized properly
    #[clap(long, default_value_t = 10)]
    hidden_units: usize,
    #[clap(long, default_value_t = false)]
    graphviz: bool,
}

// A NN with one hidden layer, output is a vector of two values representing
// the probability of each class
struct Model {
    l1: Layer,
    l1_relu: ReLU,
    l2: Layer,
    softmax: Softmax,
}

impl Model {
    fn new(n_classes: usize, hidden_size: usize) -> Self {
        Self {
            l1: Layer::new(2, hidden_size),
            l1_relu: ReLU::new(),
            l2: Layer::new(hidden_size, n_classes),
            softmax: Softmax::new(),
        }
    }
}

impl Module for Model {
    fn forward(&self, inputs: &[Value]) -> Result<Vec<Value>, NNError> {
        let l1_out = self.l1.forward(inputs)?;
        let l1_relu_out = self.l1_relu.forward(&l1_out);
        let l2_out = self.l2.forward(&l1_relu_out)?;
        let out = self.softmax.forward(&l2_out);
        Ok(out)
    }

    fn parameters(&self) -> Vec<Value> {
        self.l1
            .parameters()
            .into_iter()
            .chain(self.l2.parameters())
            .collect()
    }
}

fn main() {
    env_logger::init();

    let args = Args::parse();
    let class_size = args.class_size;
    let minibatch_size = args.batch_size;
    let (data, labels) = load_dataset(args.dataset, class_size);
    let data_clone = data.clone();
    let labels_clone = labels.clone();
    let n_classes = labels[0].len();

    plot_data(
        &data,
        &labels,
        &format!("{}/dataset_{}.png", args.output_dir, args.dataset),
        args.dataset,
    )
    .unwrap();

    let epochs = args.epochs;
    let model = Model::new(n_classes, args.hidden_units);
    let mut optim = SGD::new(model.parameters(), args.lr, args.momentum);
    let data_loader = DataLoader::new(data, labels, minibatch_size, true).unwrap();
    let print_every = args.print_epochs;

    for epoch in 0..epochs {
        let data_loader_iter = data_loader.iter();
        let mut epoch_loss = Value::new(0.0);
        let mut total_n_dead_neurons = 0;
        let mut loss_cached = Value::new(0.0);
        for (batch_data, batch_labels) in data_loader_iter {
            assert_eq!(batch_data.len(), batch_labels.len());
            assert_eq!(batch_data.len(), minibatch_size);
            for (data, label) in batch_data.into_iter().zip(batch_labels.into_iter()) {
                // note that after y_pred is freed, all the intermediate children (that are not model parameters) are also freed
                // because their reference count drops to 0. This implicitly zeros the gradients of the intermediate children
                let y_pred = model.forward(data).unwrap();
                assert_eq!(y_pred.len(), n_classes);
                let loss = MSELoss::call(&y_pred, label);
                loss.backward();
                total_n_dead_neurons += model.l1_relu.n_dead_neurons();
                epoch_loss = &epoch_loss + &loss;
                loss_cached = loss;
            }
            // take steps in minibatches
            optim.step();
            optim.zero_grad();
        }
        log::debug!(
            "Average n_dead_neurons in epoch {}: {}",
            epoch + 1,
            total_n_dead_neurons / minibatch_size
        );
        if epoch % print_every == 0 || epoch == epochs - 1 {
            if args.graphviz {
                draw_dot(
                    &loss_cached,
                    &format!(
                        "{}/weights_epoch_{}_{}.dot",
                        args.output_dir,
                        epoch + 1,
                        args.dataset
                    ),
                )
                .unwrap();
            }
            log::info!("epoch: {}, epoch_loss: {}", epoch + 1, epoch_loss.data());
            plot_decision_boundary(
                &model,
                &format!(
                    "{}/decision_boundary_epoch_{}_{}.png",
                    args.output_dir,
                    epoch + 1,
                    args.dataset
                ),
                args.dataset,
                &data_clone,
                &labels_clone,
            )
            .unwrap();
        }
    }
}
