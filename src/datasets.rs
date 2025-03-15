//! Generates multiclass classification datasets and utilities for plotting them and decision boundaries
//!
//! By convention class 0 is plotted in red and class 1 is plotted in blue.

use crate::{backprop_fns::BackpropFunc, nn::Module, values::Value};
use std::{
    cmp::Ordering,
    error::Error,
    f32::consts::PI,
    fmt::{self, Display},
};

use clap::ValueEnum;
use plotters::{
    chart::ChartBuilder,
    prelude::{BitMapBackend, Circle, IntoDrawingArea, Rectangle},
    style::{BLUE, Color, GREEN, RED, RGBColor, WHITE},
};
use rand::Rng;

/// Toggles between dataset types
#[derive(Debug, ValueEnum, Clone, Copy)]
pub enum Dataset {
    Line,
    Circle,
    XOR,
    Moon,
}

/// Maps a class index to a color (for data points)
const CLASS_COLORS: [RGBColor; 3] = [RED, BLUE, GREEN];
/// Maps a class index to a color (for decision boundary), slightly transparent
const DECISION_BOUNDARY_COLORS: [RGBColor; 3] = [
    RGBColor(255, 200, 200),
    RGBColor(200, 200, 255),
    RGBColor(200, 255, 200),
];

impl Display for Dataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dataset::Line => write!(f, "line"),
            Dataset::Circle => write!(f, "circle"),
            Dataset::XOR => write!(f, "xor"),
            Dataset::Moon => write!(f, "moon"),
        }
    }
}

/// Loads a dataset based on the dataset type
pub fn load_dataset(dataset: Dataset, class_size: usize) -> (Vec<Vec<f32>>, Vec<Vec<u8>>) {
    match dataset {
        Dataset::Line => gen_linear_data(class_size),
        Dataset::Circle => gen_circle_data(class_size),
        Dataset::XOR => gen_xor_data(class_size),
        Dataset::Moon => gen_moon_data(class_size),
    }
}

/// Generates a simple linearly separable dataset, labels are one hot encoded vectors
pub fn gen_linear_data(class_size: usize) -> (Vec<Vec<f32>>, Vec<Vec<u8>>) {
    let mut rng = rand::rng();
    let mut data = Vec::new();
    let mut labels = Vec::new();

    for _ in 0..class_size {
        let x = rng.random_range(-5.0..5.0);
        let y = rng.random_range(-5.0..5.0);
        data.push(vec![x, y]);
        labels.push(if x > y { vec![1, 0] } else { vec![0, 1] });
    }

    (data, labels)
}

/// Generates a multiclass classification dataset of three concentric circles, labels are one hot encoded vectors
pub fn gen_circle_data(class_size: usize) -> (Vec<Vec<f32>>, Vec<Vec<u8>>) {
    const N_CLASSES: usize = 3;
    let c1_radius = 1.0;
    let c2_radius = 3.0;
    let c3_radius = 5.0;
    let radii = vec![c1_radius, c2_radius, c3_radius];

    let mut rng = rand::rng();
    let mut data = Vec::new();
    let mut labels = Vec::new();

    for (class_idx, radius) in radii.iter().enumerate() {
        let mut gt_label = vec![0; N_CLASSES];
        gt_label[class_idx] = 1;
        for _ in 0..class_size {
            let angle = rng.random_range(0.0..2.0 * std::f32::consts::PI);
            let radius_delta = radius * rng.random_range(-0.25..0.25);
            let x = (radius + radius_delta) * angle.cos();
            let y = (radius + radius_delta) * angle.sin();
            data.push(vec![x, y]);
            labels.push(gt_label.clone());
        }
    }

    (data, labels)
}

pub fn gen_xor_data(class_size: usize) -> (Vec<Vec<f32>>, Vec<Vec<u8>>) {
    let mut rng = rand::rng();
    let mut data = Vec::new();
    let mut labels = Vec::new();

    for _ in 0..class_size {
        let x = rng.random_range(-5.0..5.0);
        let y = rng.random_range(-5.0..5.0);
        if x > 0.0 && y > 0.0 || x < 0.0 && y < 0.0 {
            data.push(vec![x, y]);
            labels.push(vec![1, 0]);
        } else {
            data.push(vec![x, y]);
            labels.push(vec![0, 1]);
        }
    }

    (data, labels)
}

/// Generates a moons dataset (two interleaving partial circles) for binary classification
/// The labels are one-hot encoded vectors
pub fn gen_moon_data(class_size: usize) -> (Vec<Vec<f32>>, Vec<Vec<u8>>) {
    let mut rng = rand::rng();
    let mut data = Vec::new();
    let mut labels = Vec::new();

    // First moon
    for _ in 0..class_size {
        let theta = rng.random_range(PI * -1.0 / 4.0..PI * 5.0 / 4.0);
        let radius = 3.0;
        let x = radius * theta.cos();
        let y = radius * theta.sin();
        let noise_x = rng.random_range(-0.1..0.1);
        let noise_y = rng.random_range(-0.1..0.1);
        data.push(vec![x + noise_x, y + noise_y]);
        labels.push(vec![1, 0]);
    }

    // Second moon
    for _ in 0..class_size {
        let theta = rng.random_range(PI * -1.0 / 4.0..PI * 5.0 / 4.0);
        let radius = 3.0;
        let x = radius * theta.cos() + 2.0;
        let y = -radius * theta.sin() - 0.2;
        let noise_x = rng.random_range(-0.1..0.1);
        let noise_y = rng.random_range(-0.1..0.1);
        data.push(vec![x + noise_x, y + noise_y]);
        labels.push(vec![0, 1]);
    }

    (data, labels)
}

/// Plots the data points and labels for a given dataset
pub fn plot_data(
    data: &[Vec<f32>],
    labels: &[Vec<u8>],
    file_name: &str,
    dataset: Dataset,
) -> Result<(), Box<dyn Error>> {
    let root_area = BitMapBackend::new(file_name, (640, 480)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root_area)
        .caption(format!("Dataset: {}", dataset), ("sans-serif", 50))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-6.0f32..6.0f32, -6.0f32..6.0f32)?;

    chart.configure_mesh().draw()?;

    let n_classes = labels[0].len();

    for class_idx in 0..n_classes {
        let class_data: Vec<&Vec<f32>> = data
            .iter()
            .zip(labels.iter())
            .filter(|&(_, label)| get_class(label) == class_idx)
            .map(|(data, _)| data)
            .collect();
        chart.draw_series(
            class_data
                .iter()
                .map(|data| Circle::new((data[0], data[1]), 3, CLASS_COLORS[class_idx].filled())),
        )?;
    }

    root_area.present()?;
    log::info!("Data plot has been saved to '{}'.", file_name);

    Ok(())
}

/// Plots the decision boundary for a given model on a given dataset by sampling a grid of points and evaluating the model
pub fn plot_decision_boundary(
    model: &dyn Module,
    file_name: &str,
    dataset: Dataset,
    data: &[Vec<f32>],
    labels: &[Vec<u8>],
) -> Result<(), Box<dyn Error>> {
    let root_area = BitMapBackend::new(file_name, (640, 480)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let grid_min = -6.0;
    let grid_max = 6.0;

    let mut chart = ChartBuilder::on(&root_area)
        .caption(
            format!("Decision Boundary for {}", dataset),
            ("sans-serif", 50),
        )
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(grid_min..grid_max, grid_min..grid_max)?;

    chart.configure_mesh().draw()?;

    let step = 0.20;
    let n_steps: f32 = (grid_max - grid_min) / step;
    let n_steps = n_steps.round() as i32;
    let grid_points = (0..n_steps).flat_map(|xi| {
        let x = grid_min + (xi as f32 * step);
        (0..n_steps).map(move |yi| {
            let y = grid_min + (yi as f32 * step);
            (x, y)
        })
    });

    chart.draw_series(grid_points.map(|(x, y)| {
        let input = [Value::new(x), Value::new(y)];
        let output = model.forward(&input).unwrap();
        let pred = output
            .iter()
            .enumerate()
            .max_by(|(_, v), (_, v2)| v.data().partial_cmp(&v2.data()).unwrap_or(Ordering::Equal))
            .unwrap();
        let color = DECISION_BOUNDARY_COLORS[pred.0];

        Rectangle::new(
            [
                (x - step / 2.0, y - step / 2.0),
                (x + step / 2.0, y + step / 2.0),
            ],
            color.filled(),
        )
    }))?;

    // plot the data points
    chart.draw_series(data.iter().zip(labels.iter()).map(|(data, label)| {
        let color = CLASS_COLORS[get_class(label)];
        Circle::new((data[0], data[1]), 3, color.filled())
    }))?;

    root_area.present()?;
    log::info!("Decision boundary plot saved to '{}'.", file_name);
    Ok(())
}

/// Returns the class of a given label by taking the index of the maximum value
pub fn get_class(label: &[u8]) -> usize {
    label
        .iter()
        .enumerate()
        .max_by_key(|(_, v)| **v)
        .map(|(i, _)| i)
        .unwrap()
}

use std::collections::HashSet;
use std::fs::File;
use std::io::Write; // adjust path if needed

/// Returns a fill color for each operation variant.
fn color_for_op(op: &Option<BackpropFunc>) -> &'static str {
    match op {
        Some(BackpropFunc::Add) => "#FFF2B8",  // light yellow
        Some(BackpropFunc::Sub) => "#FAD7D4",  // light red
        Some(BackpropFunc::Mul) => "#D6EAF8",  // light blue
        Some(BackpropFunc::Div) => "#F9E79F",  // a different yellow
        Some(BackpropFunc::Neg) => "#F5EEF8",  // light purple/pink
        Some(BackpropFunc::Pow) => "#FDEDEC",  // very light red
        Some(BackpropFunc::ReLU) => "#D5F5E3", // light green
        Some(BackpropFunc::Exp) => "#F5CBA7",  // orange
        None => "#BBEFFF",                     // default color for leaf nodes
    }
}

/// Generate a Graphviz DOT file that shows each Value node
/// with (data, grad, operation) in a record-style label
/// Apply to the root of the graph (i.e. loss output)
/// View via a command like:
/// ```sh
/// dot -Tpng output.dot -o output.png
/// ```
pub fn draw_dot(root: &Value, filename: &str) -> std::io::Result<()> {
    let mut visited = HashSet::new();
    let mut nodes = String::new();
    let mut edges = String::new();

    fn traverse(v: &Value, visited: &mut HashSet<u64>, nodes: &mut String, edges: &mut String) {
        let inner = v.0.borrow();
        let node_id = inner.id();

        if visited.contains(&node_id) {
            return;
        }
        visited.insert(node_id);

        // Build an operation label (e.g. "Add", "Mul", "ReLU", or "leaf")
        let op_label = match &inner.backprop_fn {
            Some(op) => format!("{:?}", op),
            None => "leaf".to_string(),
        };

        // Create a label with shape=record:  { data=.. | grad=.. | op=.. }
        let label = format!(
            "{{ data={:.4} | grad={:.4} | {} }}",
            inner.data, inner.grad, op_label
        );

        // Add this node to the DOT "nodes" list
        nodes.push_str(&format!(
            "  {} [label=\"{}\", shape=record, style=filled, fillcolor=\"{}\"];\n",
            node_id,
            label,
            color_for_op(&inner.backprop_fn)
        ));

        // For each "child" in the node's `children`, draw an edge child -> current
        for child_rc in inner.children() {
            let child_value = Value(child_rc.clone());
            let child_inner = child_value.0.borrow();
            let child_id = child_inner.id();

            edges.push_str(&format!("  {} -> {};\n", child_id, node_id));

            traverse(&child_value, visited, nodes, edges);
        }
    }

    // Start DFS from the root node
    traverse(root, &mut visited, &mut nodes, &mut edges);

    let dot_content = format!(
        "digraph G {{
  rankdir=LR; // Left-to-right layout
  node [fontsize=12, fontname=\"Verdana\"];
{}
{}
}}",
        nodes, edges
    );

    let mut file = File::create(filename)?;
    file.write_all(dot_content.as_bytes())?;
    log::info!("Model graphviz file saved to '{}'.", filename);
    Ok(())
}
