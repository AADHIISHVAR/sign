use image::{RgbImage, GrayImage};
use imageproc::filter::gaussian_blur_f32;
use imageproc::edges::canny;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignColor {
    Red,
    Blue,
    Yellow,
    White,
    Green,
}

pub fn to_gray(img: &RgbImage) -> GrayImage {
    image::imageops::grayscale(img)
}

pub fn edge_map(gray: &GrayImage) -> GrayImage {
    let blurred = gaussian_blur_f32(gray, 1.0);
    let edges = canny(&blurred, 5.0, 20.0);
    let mut binary = edges.clone();
    for p in binary.pixels_mut() {
        if p.0[0] > 0 { p.0[0] = 255; }
    }
    binary
}

pub fn resize_128(img: &RgbImage) -> RgbImage {
    image::imageops::resize(img, 128, 128, image::imageops::FilterType::Triangle)
}

/// Convert an RGB pixel to HSV representation.
/// H is in degrees [0.0, 360.0], S in [0.0, 1.0], V in [0.0, 1.0]
pub fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let r = r as f32 / 255.0;
    let g = g as f32 / 255.0;
    let b = b as f32 / 255.0;
    
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    
    let mut h = 0.0;
    if delta > 0.0 {
        if max == r {
            h = 60.0 * (((g - b) / delta) % 6.0);
        } else if max == g {
            h = 60.0 * (((b - r) / delta) + 2.0);
        } else {
            h = 60.0 * (((r - g) / delta) + 4.0);
        }
    }
    if h < 0.0 {
        h += 360.0;
    }
    
    let s = if max == 0.0 { 0.0 } else { delta / max };
    let v = max;
    
    (h, s, v)
}

/// Check if the given HSV values fall within the range of a standard road sign color.
pub fn is_color(h: f32, s: f32, v: f32, color: SignColor) -> bool {
    match color {
        SignColor::Red => {
            // Red wraps around 360/0 degrees
            (h <= 12.0 || h >= 340.0) && s >= 0.40 && v >= 0.25
        }
        SignColor::Blue => {
            (195.0..=255.0).contains(&h) && s >= 0.35 && v >= 0.25
        }
        SignColor::Yellow => {
            (35.0..=65.0).contains(&h) && s >= 0.35 && v >= 0.35
        }
        SignColor::White => {
            // White has low saturation and high value
            s <= 0.22 && v >= 0.55
        }
        SignColor::Green => {
            (80.0..=165.0).contains(&h) && s >= 0.25 && v >= 0.20
        }
    }
}

/// Generate a binary mask for the specified sign color.
pub fn create_color_mask(img: &RgbImage, color: SignColor) -> GrayImage {
    let mut mask = GrayImage::new(img.width(), img.height());
    for (x, y, pixel) in img.enumerate_pixels() {
        let (h, s, v) = rgb_to_hsv(pixel[0], pixel[1], pixel[2]);
        if is_color(h, s, v, color) {
            mask.put_pixel(x, y, image::Luma([255]));
        } else {
            mask.put_pixel(x, y, image::Luma([0]));
        }
    }
    mask
}

/// Generate binary masks for multiple colors in a single pass over the image.
/// Converts each pixel to HSV only once and checks all colors, avoiding redundant
/// HSV conversions. Returns masks in the same order as the input color slice.
pub fn create_color_masks_single_pass(img: &RgbImage, colors: &[SignColor]) -> Vec<GrayImage> {
    let w = img.width();
    let h = img.height();
    let mut masks: Vec<GrayImage> = colors.iter().map(|_| GrayImage::new(w, h)).collect();

    for (x, y, pixel) in img.enumerate_pixels() {
        // Single HSV conversion per pixel (instead of once per color)
        let (hue, sat, val) = rgb_to_hsv(pixel[0], pixel[1], pixel[2]);
        for (i, &color) in colors.iter().enumerate() {
            if is_color(hue, sat, val, color) {
                masks[i].put_pixel(x, y, image::Luma([255]));
            }
        }
    }

    masks
}

/// Compute the percentage of points in the contour that match the target sign color.
pub fn color_purity(img: &RgbImage, points: &[imageproc::point::Point<i32>], color: SignColor) -> f32 {
    if points.is_empty() {
        return 0.0;
    }
    let mut matching = 0;
    for p in points {
        let x = p.x.clamp(0, img.width() as i32 - 1) as u32;
        let y = p.y.clamp(0, img.height() as i32 - 1) as u32;
        let pixel = img.get_pixel(x, y);
        let (h, s, v) = rgb_to_hsv(pixel[0], pixel[1], pixel[2]);
        if is_color(h, s, v, color) {
            matching += 1;
        }
    }
    matching as f32 / points.len() as f32
}