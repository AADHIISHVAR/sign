use tract_onnx::prelude::*;
use tract_onnx::tract_core::plan::SimplePlan;
use std::path::Path;
use std::sync::Arc;
use anyhow::Result;
use image::RgbImage;

pub struct ShapeDetector {
    // Made public to allow access from main.rs
    pub model: Option<Arc<SimplePlan<TypedFact, Box<dyn TypedOp>>>>,
    input_size: usize,
}

impl ShapeDetector {
    pub fn new(model_path: &Path) -> Result<Self> {
        let input_size = 64; // Default input size for our micro-CNN

        if !model_path.exists() {
            println!(
                "Warning: ONNX model file {:?} not found. Inference will bypass neural network classification.",
                model_path
            );
            return Ok(Self {
                model: None,
                input_size,
            });
        }

        println!("Loading ONNX model from {:?}", model_path);
        // Load model, set input facts, optimize graph, and prepare plan
        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            .with_input_fact(0, f32::fact(&[1, 3, input_size, input_size]).into())?
            .into_optimized()?
            .into_runnable()?;

        Ok(Self {
            model: Some(model),
            input_size,
        })
    }

    pub fn classify(&self, img: &RgbImage) -> Result<(usize, &'static str, f32)> {
        // If the model is not loaded, return a dummy class index and name
        let Some(ref plan) = self.model else {
            return Ok((18, get_class_name(18), 1.0)); // Default fallback
        };

        // Resize image to expected input dimensions using fast nearest-neighbor scaling
        let resized = image::imageops::resize(
            img,
            self.input_size as u32,
            self.input_size as u32,
            image::imageops::FilterType::Nearest,
        );

        // Formulate NCHW tensor layout using tract's internal ndarray version
        let mut tensor_matrix = tract_ndarray::Array4::<f32>::zeros((1, 3, self.input_size, self.input_size));
        for (x, y, pixel) in resized.enumerate_pixels() {
            tensor_matrix[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0; // R
            tensor_matrix[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0; // G
            tensor_matrix[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0; // B
        }

        // Convert the ndarray matrix to a tract-compatible Tensor
        let input_tensor = tensor_matrix.into_tensor();

        // Execute optimized inference pass
        let results = plan.run(tvec!(input_tensor.into()))?;

        // Extract logits array view using to_plain_array_view
        let output_logits = results[0].to_plain_array_view::<f32>()?;

        // Compute argmax to retrieve predicted class
        let mut max_confidence = -1.0;
        let mut predicted_class = 0;
        for (index, &confidence) in output_logits.iter().enumerate() {
            if confidence > max_confidence {
                max_confidence = confidence;
                predicted_class = index;
            }
        }

        let class_name = get_class_name(predicted_class);
        Ok((predicted_class, class_name, max_confidence))
    }
}

pub fn get_class_name(class_id: usize) -> &'static str {
    match class_id {
        0 => "Speed limit (20km/h)",
        1 => "Speed limit (30km/h)",
        2 => "Speed limit (50km/h)",
        3 => "Speed limit (60km/h)",
        4 => "Speed limit (70km/h)",
        5 => "Speed limit (80km/h)",
        6 => "End of speed limit (80km/h)",
        7 => "Speed limit (100km/h)",
        8 => "Speed limit (120km/h)",
        9 => "No passing",
        10 => "No passing for vehicles over 3.5 metric tons",
        11 => "Right-of-way at the next intersection",
        12 => "Priority road",
        13 => "Yield",
        14 => "Stop",
        15 => "No vehicles",
        16 => "Vehicles over 3.5 metric tons prohibited",
        17 => "No entry",
        18 => "General caution",
        19 => "Dangerous curve to the left",
        20 => "Dangerous curve to the right",
        21 => "Double curve",
        22 => "Bumpy road",
        23 => "Slippery road",
        24 => "Road narrows on the right",
        25 => "Road work",
        26 => "Traffic signals",
        27 => "Pedestrians",
        28 => "Children crossing",
        29 => "Bicycles crossing",
        30 => "Beware of ice/snow",
        31 => "Wild animals crossing",
        32 => "End of all speed and passing limits",
        33 => "Turn right ahead",
        34 => "Turn left ahead",
        35 => "Ahead only",
        36 => "Go straight or right",
        37 => "Go straight or left",
        38 => "Keep right",
        39 => "Keep left",
        40 => "Roundabout mandatory",
        41 => "End of no passing",
        42 => "End of no passing by vehicles over 3.5 metric tons",
        _ => "Unknown sign"
    }
}
