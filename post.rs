use crate::util::geometry::{BBox, RgbImage};
use crate::util::image::{edge_map, to_gray};

#[derive(Debug, Clone, Copy)]
pub struct PostDetectConfig {
    pub horizontal_margin: u32,
    pub vertical_search_range: u32,
    pub max_post_width: u32,
    pub min_run_length: u32,
    pub max_run_length: u32,
    pub post_width: u32,
}

impl Default for PostDetectConfig {
    fn default() -> Self {
        Self {
            horizontal_margin: 30,
            vertical_search_range: 200,
            max_post_width: 12,
            min_run_length: 1,
            max_run_length: 12,
            post_width: 8,
        }
    }
}

pub fn detect_post(img: &RgbImage, sign_box: &BBox) -> Option<BBox> {
    detect_post_with_config(img, sign_box, PostDetectConfig::default())
}

pub fn detect_post_with_config(img: &RgbImage, sign_box: &BBox, config: PostDetectConfig) -> Option<BBox> {
    let gray = to_gray(img);
    let edges = edge_map(&gray);
    
    let (sx, sy, sw, sh) = *sign_box;
    let sign_cx = sx + sw / 2;
    let sign_bottom = sy + sh;
    
    let search_x0 = sign_cx.saturating_sub(sw / 2 + config.horizontal_margin);
    let search_x1 = (sign_cx + sw / 2 + config.horizontal_margin).min(img.width());
    let search_y0 = sign_bottom;
    let search_y1 = (sign_bottom + config.vertical_search_range).min(img.height());
    
    let mut best: Option<BBox> = None;
    let mut best_len = 0;
    
    for y in search_y0..search_y1 {
        let mut run_start = None;
        for x in search_x0..search_x1 {
            if edges.get_pixel(x, y).0[0] > 0 {
                if run_start.is_none() { run_start = Some(x); }
            } else if let Some(start) = run_start {
                let len = x - start;
                if (config.min_run_length..=config.max_run_length).contains(&len) {
                    let cx = (start + x) / 2;
                    let post_h = search_y1 - y;
                    if post_h > best_len {
                        best_len = post_h;
                        let half_w = config.post_width / 2;
                        best = Some((cx.saturating_sub(half_w), y, config.post_width, post_h));
                    }
                }
                run_start = None;
            }
        }
    }
    best
}

pub fn crop_post(img: &RgbImage, b: &BBox) -> RgbImage {
    let (x, y, w, h) = *b;
    let x = x.min(img.width().saturating_sub(1));
    let y = y.min(img.height().saturating_sub(1));
    let w = w.min(img.width() - x);
    let h = h.min(img.height() - y);
    image::ImageBuffer::from_fn(w, h, |i, j| *img.get_pixel(x + i, y + j))
}