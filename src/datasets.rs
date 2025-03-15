//! Generates binary classification datasets and utilities for plotting them and decision boundaries
//!
//! By convention class 0 is plotted in red and class 1 is plotted in blue.

use crate::{nn::Module, values::Value};
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
    style::{BLUE, Color, RED, RGBColor, WHITE},
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

/// Generates a binary classification dataset of two concentric circles, labels are one hot encoded vectors
pub fn gen_circle_data(class_size: usize) -> (Vec<Vec<f32>>, Vec<Vec<u8>>) {
    let n_c1 = class_size;
    let n_c2 = class_size;
    let c1_radius = 3.0;
    let c2_radius = 5.0;

    let mut rng = rand::rng();
    let mut data = Vec::new();
    let mut labels = Vec::new();

    // Generate points for class 1
    for _ in 0..n_c1 {
        let angle = rng.random_range(0.0..2.0 * std::f32::consts::PI);
        let radius_delta = c1_radius * rng.random_range(-0.25..0.25);
        let x = (c1_radius + radius_delta) * angle.cos();
        let y = (c1_radius + radius_delta) * angle.sin();
        data.push(vec![x, y]);
        labels.push(vec![1, 0]);
    }

    // Generate points for class 2
    for _ in 0..n_c2 {
        let angle = rng.random_range(0.0..2.0 * std::f32::consts::PI);
        let radius_delta = c2_radius * rng.random_range(-0.25..0.25);
        let x = (c2_radius + radius_delta) * angle.cos();
        let y = (c2_radius + radius_delta) * angle.sin();
        data.push(vec![x, y]);
        labels.push(vec![0, 1]);
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

    let class1: Vec<&Vec<f32>> = data
        .iter()
        .zip(labels.iter())
        .filter(|&(_, label)| get_class(label) == 0)
        .map(|(data, _)| data)
        .collect();
    let class2: Vec<&Vec<f32>> = data
        .iter()
        .zip(labels.iter())
        .filter(|&(_, label)| get_class(label) == 1)
        .map(|(data, _)| data)
        .collect();

    // Plot class 1 points in red
    chart.draw_series(
        class1
            .iter()
            .map(|data| Circle::new((data[0], data[1]), 3, RED.filled())),
    )?;

    // Plot class 2 points in blue
    chart.draw_series(
        class2
            .iter()
            .map(|data| Circle::new((data[0], data[1]), 3, BLUE.filled())),
    )?;

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

    let red_bg = RGBColor(255, 200, 200);
    let blue_bg = RGBColor(200, 200, 255);

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
        let color = if pred.0 == 0 { red_bg } else { blue_bg };

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
        let color = if get_class(label) == 0 { RED } else { BLUE };
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
