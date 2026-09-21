use crate::util::geometry::{bbox, classify_shape_robust, score_candidate, RoadSignShape, BBox, RgbImage};
use crate::util::image::{create_color_masks_single_pass, color_purity, SignColor};
use imageproc::morphology::{dilate, erode};
use imageproc::distance_transform::Norm;
use imageproc::contours::{find_contours, BorderType};
use imageproc::geometry::contour_area;

/// Detect signs using all 5 color channels (full scan — use for single images).
pub fn detect_sign(img: &RgbImage) -> Vec<BBox> {
    let all_colors = [SignColor::Red, SignColor::Blue, SignColor::Yellow, SignColor::White, SignColor::Green];
    detect_sign_core(img, &all_colors, false)
}

/// Detect signs using only the specified color channels (use for video to skip noisy colors).
pub fn detect_sign_with_colors(img: &RgbImage, colors: &[SignColor]) -> Vec<BBox> {
    detect_sign_core(img, colors, false)
}

pub fn detect_sign_with_debug(img: &RgbImage, debug: bool) -> Vec<BBox> {
    let all_colors = [SignColor::Red, SignColor::Blue, SignColor::Yellow, SignColor::White, SignColor::Green];
    detect_sign_core(img, &all_colors, debug)
}

pub fn iou(a: &BBox, b: &BBox) -> f32 {
    let (ax, ay, aw, ah) = *a;
    let (bx, by, bw, bh) = *b;
    
    let x_overlap = ax.max(bx) as f32;
    let y_overlap = ay.max(by) as f32;
    let x2_overlap = (ax + aw).min(bx + bw) as f32;
    let y2_overlap = (ay + ah).min(by + bh) as f32;
    
    let w_overlap = 0.0f32.max(x2_overlap - x_overlap);
    let h_overlap = 0.0f32.max(y2_overlap - y_overlap);
    
    let intersection = w_overlap * h_overlap;
    let area_a = (aw * ah) as f32;
    let area_b = (bw * bh) as f32;
    
    intersection / (area_a + area_b - intersection)
}

fn detect_sign_core(img: &RgbImage, colors: &[SignColor], debug: bool) -> Vec<BBox> {
    let mut candidates = Vec::new();
    let w = img.width();
    let h = img.height();

    // Single-pass: convert every pixel to HSV once and build all color masks simultaneously
    let masks = create_color_masks_single_pass(img, colors);

    for (color_idx, &color) in colors.iter().enumerate() {
        let mask = &masks[color_idx];
        
        // Perform Morphological Closing to fill internal gaps
        let closed_mask = erode(&dilate(mask, Norm::LInf, 1), Norm::LInf, 1);
        
        // Find outer contours on the closed mask
        let contours = find_contours::<i32>(&closed_mask);
        
        for contour in contours {
            if contour.border_type == BorderType::Outer {
                let points = &contour.points;
                if points.len() < 6 {
                    continue; // Noise filter
                }
                
                let area = contour_area(points);
                if !(100.0..=250000.0).contains(&area) {
                    continue; // Too small or too large
                }
                
                let (bx, by, bw, bh) = bbox(points);

                // Reject bounding boxes smaller than 10px in either dimension
                if bw < 10 || bh < 10 {
                    continue;
                }

                // Solidity filter - reject contours with area/(w*h) < 0.5
                let solidity = area / (bw * bh) as f64;
                if solidity < 0.5 {
                    continue;
                }

                let aspect_ratio = bw as f32 / bh as f32;
                
                // Sign aspect ratio is approximately 1.0 (square bounding box)
                if (0.70..=1.40).contains(&aspect_ratio) {
                    let shape = classify_shape_robust(points);
                    let purity = color_purity(img, points, color);
                    
                    if shape != RoadSignShape::Unknown && purity >= 0.40 {
                        let score = score_candidate(points, shape, purity, w, h);
                        if score >= 0.35 {
                            candidates.push((score, (bx, by, bw, bh), color, shape));
                        }
                    }
                }
            }
        }
    }
    
    // Sort candidates by score descending
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    
    let mut final_bboxes = Vec::new();
    let mut final_metadata = Vec::new();
    
    // Non-Maximum Suppression (NMS)
    for (score, bbox, color, shape) in candidates {
        let mut keep = true;
        for existing in &final_bboxes {
            if iou(&bbox, existing) > 0.25 {
                keep = false;
                break;
            }
        }
        
        if keep {
            final_bboxes.push(bbox);
            final_metadata.push((score, color, shape));
        }
    }
    
    if debug {
        println!("Detected {} road signs:", final_bboxes.len());
        for (i, bbox) in final_bboxes.iter().enumerate() {
            let (score, color, shape) = final_metadata[i];
            println!(
                "  Sign #{}: bbox={:?}, color={:?}, shape={:?}, score={:.4}",
                i, bbox, color, shape, score
            );
        }
    }
    
    final_bboxes
}

pub fn crop_sign(img: &RgbImage, b: &BBox) -> RgbImage {
    let (x, y, w, h) = *b;
    // Zero-copy extraction generating an immutable sub-image view
    let cropped_view = image::imageops::crop_imm(img, x, y, w, h);
    cropped_view.to_image()
}