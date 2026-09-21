use sign_board_cnn::detect::{sign, post};
use sign_board_cnn::util::image::resize_128;
use sign_board_cnn::onnx::detector::ShapeDetector;
use image::{RgbImage, ImageBuffer};
use anyhow::{Result, Context};
use std::path::Path;
use std::process::{Command, Stdio};
use std::io::Read;

fn main() -> Result<()> {
    println!("Program started...");
    let args: Vec<String> = std::env::args().collect();

    let mut image_path = None;
    let mut video_path = None;
    let mut clean_crops = false;

    // Default video frame resolution (downscaled for speed)
    let width = 320;
    let height = 240;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-i" | "--image" => {
                if i + 1 < args.len() {
                    image_path = Some(args[i + 1].clone());
                    i += 2;
                } else { break; }
            }
            "-v" | "--video" => {
                if i + 1 < args.len() {
                    video_path = Some(args[i + 1].clone());
                    i += 2;
                } else { break; }
            }
            "-c" | "--clean" => {
                clean_crops = true;
                i += 1;
            }
            path if !path.starts_with('-') => {
                // Positional arguments default to image paths
                image_path = Some(path.to_string());
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    if clean_crops {
        println!("Cleaning up created sign head/post images...");
        let mut count = 0;
        if let Ok(entries) = std::fs::read_dir(".") {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                if (name.starts_with("sign_head") || name.starts_with("sign_post")) && name.ends_with(".png") {
                    if let Ok(()) = std::fs::remove_file(entry.path()) {
                        count += 1;
                    }
                }
            }
        }
        println!("Deleted {} images.", count);
        if image_path.is_none() && video_path.is_none() {
            return Ok(());
        }
    }

    // Initialize the classification engine (tract model)
    let model_path = Path::new("models/sign_classifier_paper.onnx");
    let detector = ShapeDetector::new(model_path)?;

    if let Some(v_path) = video_path {
        println!("Spawning FFmpeg to decode video: {}...", v_path);

        // Crop to top 60% of the frame — signs appear in the upper portion,
        // the bottom 40% is typically road surface and car hood
        let roi_height = (height as f32 * 0.6) as usize; // 144px at 320x240
        let vf_filter = format!("scale={}:{},crop={}:{}:0:0", width, height, width, roi_height);

        let mut ffmpeg_child = Command::new("ffmpeg")
            .args(&[
                "-i", &v_path,
                "-vf", &vf_filter,
                "-f", "image2pipe",
                "-pix_fmt", "rgb24",
                "-vcodec", "rawvideo",
                "-"
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null()) // Suppresses FFmpeg verbose outputs
            .spawn()
            .context("Failed to launch ffmpeg. Make sure ffmpeg is installed on your PATH.")?;

        let mut ffmpeg_stdout = ffmpeg_child.stdout.take()
            .context("Failed to capture FFmpeg standard output pipe.")?;

        // Buffer uses ROI height (not full height) since FFmpeg crops before output
        let frame_size = width * roi_height * 3;
        let mut buffer = vec![0u8; frame_size];
        let mut frame_count: usize = 0;
        let mut detect_count: usize = 0;

        // Sliding window of detections from the last 2 processed frames to implement temporal tracking
        let mut detection_history: Vec<Vec<sign_board_cnn::util::geometry::BBox>> = Vec::new();

        // Only search for high-signal sign colors in video (skip White/Green to avoid
        // false positives from sky, clouds, vegetation, and road markings)
        use sign_board_cnn::util::image::SignColor;
        let video_colors = [SignColor::Red, SignColor::Blue, SignColor::Yellow];

        // Process every Nth frame to save CPU (signs persist across consecutive frames)
        let frame_skip = 3;

        println!("Reading raw frame stream (Resolution: {}x{}, ROI top {}px, processing every {} frame)...", width, height, roi_height, frame_skip);

        loop {
            // Read exactly one raw RGB24 frame from stdout pipe
            match ffmpeg_stdout.read_exact(&mut buffer) {
                Ok(_) => {
                    frame_count += 1;

                    // Skip frames that aren't on the processing cadence
                    if frame_count % frame_skip != 0 {
                        continue;
                    }

                    // Construct image directly from buffer via from_fn to avoid cloning
                    let w = width as u32;
                    let img: RgbImage = ImageBuffer::from_fn(w, roi_height as u32, |x, y| {
                        let idx = ((y * w + x) * 3) as usize;
                        image::Rgb([buffer[idx], buffer[idx + 1], buffer[idx + 2]])
                    });

                    // Run the morphological detection with filtered colors
                    let current_detections = sign::detect_sign_with_colors(&img, &video_colors);
                    
                    // Filter candidates: only confirm if we saw a similar sign in the last 2 processed frames
                    let mut confirmed_box = None;
                    for bbox in &current_detections {
                        let mut confirmed = false;
                        for prev_frame_detections in &detection_history {
                            for prev_bbox in prev_frame_detections {
                                if is_similar(bbox, prev_bbox) {
                                    confirmed = true;
                                    break;
                                }
                            }
                            if confirmed {
                                break;
                            }
                        }
                        if confirmed {
                            confirmed_box = Some(*bbox);
                            break; // Take the highest scoring confirmed candidate
                        }
                    }

                    // Keep only the detections of the last 2 processed frames in the history window
                    detection_history.push(current_detections);
                    if detection_history.len() > 2 {
                        detection_history.remove(0);
                    }

                    if let Some(sign_box) = confirmed_box {
                        detect_count += 1;
                        println!("Frame {}: Sign detected at {:?}", frame_count, sign_box);

                        // Run CNN classification on the cropped sign (no PNG save in video mode)
                        let sign_crop = sign::crop_sign(&img, &sign_box);
                        if let Ok((class_id, class_name, confidence)) = detector.classify(&sign_crop) {
                            if detector.model.is_some() {
                                println!(
                                    " -> Class = {} (ID: {}), Confidence = {:.4}",
                                    class_name, class_id, confidence
                                );
                            }
                        }
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    let processed = frame_count / frame_skip;
                    println!(
                        "Video stream ended. Read {} frames, processed {} (1/{} skip), {} detections.",
                        frame_count, processed, frame_skip, detect_count
                    );
                    break;
                }
                Err(e) => {
                    return Err(e).context("Error reading raw bytes from FFmpeg stream");
                }
            }
        }

        let _ = ffmpeg_child.kill();

    } else if let Some(img_path) = image_path {
        println!("Loading image: {}", img_path);
        let img: RgbImage = image::open(&img_path)?.to_rgb8();
        println!("Image loaded successfully: {}x{}", img.width(), img.height());

        println!("Detecting road sign...");
        if let Some(sign_box) = sign::detect_sign(&img).first().cloned() {
            println!("Sign detected at: {:?}", sign_box);
            let sign_crop = sign::crop_sign(&img, &sign_box);
            let sign_resized = resize_128(&sign_crop);
            sign_resized.save("sign_head.png")?;
            println!("Saved sign_head.png");

            if let Ok((class_id, class_name, confidence)) = detector.classify(&sign_crop) {
                if detector.model.is_some() {
                    println!(
                        "CNN Classification: Class = {} (ID: {}), Confidence = {:.4}",
                        class_name, class_id, confidence
                    );
                }
            }

            if let Some(post_box) = post::detect_post(&img, &sign_box) {
                println!("Post detected at: {:?}", post_box);
                let post_crop = post::crop_post(&img, &post_box);
                let post_resized = resize_128(&post_crop);
                post_resized.save("sign_post.png")?;
            }
        } else {
            println!("No sign detected in {}", img_path);
        }
    } else {
        println!("Usage:\n  cargo run -- -i <path_to_image>\n  cargo run -- -v <path_to_video>\n  cargo run -- -c  (or --clean to delete all created images)");
    }

    Ok(())
}

fn is_similar(a: &sign_board_cnn::util::geometry::BBox, b: &sign_board_cnn::util::geometry::BBox) -> bool {
    let (ax, ay, aw, ah) = *a;
    let (bx, by, bw, bh) = *b;
    
    // Check if IoU is above a very low threshold
    if sign::iou(a, b) > 0.05 {
        return true;
    }
    
    // Check center distance
    let acx = ax as f32 + aw as f32 / 2.0;
    let acy = ay as f32 + ah as f32 / 2.0;
    let bcx = bx as f32 + bw as f32 / 2.0;
    let bcy = by as f32 + bh as f32 / 2.0;
    
    let dist = ((acx - bcx).powi(2) + (acy - bcy).powi(2)).sqrt();
    if dist < 35.0 {
        let size_ratio = (aw * ah) as f32 / (bw * bh) as f32;
        if size_ratio > 0.3 && size_ratio < 3.0 {
            return true;
        }
    }
    
    false
}