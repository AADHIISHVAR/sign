use imageproc::geometry::{contour_area, arc_length};
use imageproc::point::Point;
pub use image::{RgbImage, ImageBuffer};

pub type BBox = (u32, u32, u32, u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoadSignShape {
    Circle,
    Triangle,
    Rectangle,
    Hexagon,
    Octagon,
    Unknown,
}

pub fn bbox(points: &[Point<i32>]) -> BBox {
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    for p in points {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }
    (
        min_x.max(0) as u32,
        min_y.max(0) as u32,
        (max_x - min_x).max(1) as u32,
        (max_y - min_y).max(1) as u32,
    )
}

pub fn circularity(points: &[Point<i32>]) -> f64 {
    let area = contour_area(points);
    let perim = arc_length(points, true);
    if perim == 0.0 {
        return 0.0;
    }
    4.0 * std::f64::consts::PI * area / (perim * perim)
}

pub fn crop_rgb(img: &RgbImage, b: &BBox) -> RgbImage {
    let (x, y, w, h) = *b;
    let x = x.min(img.width().saturating_sub(1));
    let y = y.min(img.height().saturating_sub(1));
    let w = w.min(img.width() - x);
    let h = h.min(img.height() - y);
    ImageBuffer::from_fn(w, h, |i, j| *img.get_pixel(x + i, y + j))
}

fn distance_to_segment(p: Point<i32>, s1: Point<i32>, s2: Point<i32>) -> f64 {
    let dx = (s2.x - s1.x) as f64;
    let dy = (s2.y - s1.y) as f64;
    
    let l2 = dx * dx + dy * dy;
    if l2 == 0.0 {
        let px = (p.x - s1.x) as f64;
        let py = (p.y - s1.y) as f64;
        return (px * px + py * py).sqrt();
    }
    
    let t = (((p.x - s1.x) as f64 * dx + (p.y - s1.y) as f64 * dy) / l2).clamp(0.0, 1.0);
    let proj_x = s1.x as f64 + t * dx;
    let proj_y = s1.y as f64 + t * dy;
    
    let rx = p.x as f64 - proj_x;
    let ry = p.y as f64 - proj_y;
    (rx * rx + ry * ry).sqrt()
}

pub fn ramer_douglas_peucker(points: &[Point<i32>], epsilon: f64) -> Vec<Point<i32>> {
    if points.len() < 3 {
        return points.to_vec();
    }
    
    let mut max_dist = 0.0;
    let mut index = 0;
    let end = points.len() - 1;
    
    for i in 1..end {
        let dist = distance_to_segment(points[i], points[0], points[end]);
        if dist > max_dist {
            index = i;
            max_dist = dist;
        }
    }
    
    if max_dist > epsilon {
        let mut results1 = ramer_douglas_peucker(&points[0..=index], epsilon);
        let results2 = ramer_douglas_peucker(&points[index..=end], epsilon);
        
        results1.pop(); // Remove duplicate connecting point
        results1.extend(results2);
        results1
    } else {
        vec![points[0], points[end]]
    }
}

pub fn classify_shape(points: &[Point<i32>]) -> RoadSignShape {
    classify_shape_with_epsilon(points, None)
}

pub fn classify_shape_with_epsilon(points: &[Point<i32>], epsilon: Option<f64>) -> RoadSignShape {
    let circ = circularity(points);
    if circ > 0.78 {
        return RoadSignShape::Circle;
    }
    
    // Simplify using RDP algorithm with a dynamic/heuristic epsilon
    let epsilon = epsilon.unwrap_or_else(|| compute_adaptive_epsilon(points));
    let mut vertices = ramer_douglas_peucker(points, epsilon);
    if vertices.len() >= 2 && vertices.first() == vertices.last() {
        vertices.pop();
    }
    
    match vertices.len() {
        3 => RoadSignShape::Triangle,
        4 => RoadSignShape::Rectangle,
        6 => RoadSignShape::Hexagon,
        8 => RoadSignShape::Octagon,
        _ => RoadSignShape::Unknown,
    }
}

fn compute_adaptive_epsilon(points: &[Point<i32>]) -> f64 {
    let (_, _, w, h) = bbox(points);
    let diagonal = ((w * w + h * h) as f64).sqrt();
    // Use ~1% of diagonal as epsilon, clamped to smaller bounds for small signs
    (diagonal * 0.01).clamp(1.0, 5.0)
}

pub fn classify_shape_robust(points: &[Point<i32>]) -> RoadSignShape {
    // Try multiple epsilon values for robustness (reduced from 7 to 3)
    let epsilons = [1.0, 2.0, compute_adaptive_epsilon(points)];
    
    for eps in epsilons {
        let shape = classify_shape_with_epsilon(points, Some(eps));
        if shape != RoadSignShape::Unknown {
            return shape;
        }
    }
    
    // Fallback: use circularity for circles, vertex count for polygons
    let circ = circularity(points);
    if circ > 0.70 {
        return RoadSignShape::Circle;
    }
    
    // For small contours, allow lower circularity threshold due to pixelation jagginess (raised from 0.60 to 0.68)
    if points.len() < 60 {
        if circ > 0.68 {
            return RoadSignShape::Circle;
        }
    }
    
    RoadSignShape::Unknown
}

pub fn score_candidate(
    points: &[Point<i32>],
    shape: RoadSignShape,
    color_purity: f32,
    img_w: u32,
    img_h: u32,
) -> f64 {
    let circ = circularity(points);
    let expected_circ = match shape {
        RoadSignShape::Circle => 1.0,
        RoadSignShape::Octagon => 0.948,
        RoadSignShape::Hexagon => 0.907,
        RoadSignShape::Rectangle => 0.785,
        RoadSignShape::Triangle => 0.604,
        RoadSignShape::Unknown => 0.0,
    };
    
    let shape_confidence = (1.0 - (circ - expected_circ).abs() * 2.0).clamp(0.1, 1.0);
    
    let area = contour_area(points);
    let size_ratio = area / (img_w * img_h) as f64;
    
    let size_score = if size_ratio < 0.0005 {
        size_ratio / 0.0005
    } else if size_ratio > 0.10 {
        (0.10 / size_ratio).min(1.0)
    } else {
        1.0
    };
    
    let bbox_val = bbox(points);
    let (bx, by, bw, bh) = bbox_val;
    let min_dim = bw.min(bh);
    let size_penalty = if min_dim < 15 {
        let ratio = min_dim as f64 / 15.0;
        ratio * ratio // Aggressive penalty for tiny detections (e.g. below 15px)
    } else {
        1.0
    };
    
    let touches = bx == 0 || by == 0 || (bx + bw) >= img_w || (by + bh) >= img_h;
    let position_score = if touches { 0.3 } else { 1.0 };
    
    color_purity as f64 * shape_confidence * size_score * size_penalty * position_score
}