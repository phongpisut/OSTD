//! PP-OCRv6 medium text reader on ONNX Runtime (DirectML GPU, CPU fallback). On the game test scenes, with detection on
//! a frame scaled to 960 px: 0.2% character errors and no junk lines (Windows OCR: 8.4%, 2 junk), ~170 ms and ~0.75 GB
//! VRAM on an RTX 4060 Ti, and it reads Japanese/Chinese without a Windows language pack.
use crate::ocr::Line;
use ort::{ep, session::Session, value::Tensor};
use std::path::Path;

/// (Hugging Face repo, file, saved as) — the official Paddle ONNX exports, Apache-2.0.
pub const FILES: [(&str, &str, &str); 3] = [
    ("PaddlePaddle/PP-OCRv6_medium_det_onnx", "inference.onnx", "ppocr-v6-det.onnx"),
    ("PaddlePaddle/PP-OCRv6_medium_rec_onnx", "inference.onnx", "ppocr-v6-rec.onnx"),
    ("PaddlePaddle/PP-OCRv6_medium_rec_onnx", "inference.yml", "ppocr-v6-rec.yml"),
];

pub struct Paddle {
    _env: ort::environment::Environment,
    det: Session,
    rec: Session,
    /// index 0 is the CTC blank
    chars: Vec<String>,
    pub gpu: bool,
}

fn err(e: impl std::fmt::Display) -> String {
    format!("PaddleOCR: {e}")
}

impl Paddle {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let env = ort::init().build().map_err(err)?;
        let open = |file: &str, gpu: bool| -> Result<Session, String> {
            let mut b = Session::builder(&env).map_err(err)?;
            if gpu {
                // DirectML needs these two off
                b = (b.with_memory_pattern(false).and_then(|b| b.with_parallel_execution(false)))
                    .and_then(|b| b.with_execution_providers([ep::DirectML::default().build().error_on_failure()]))
                    .map_err(err)?;
            }
            b.commit_from_file(dir.join(file)).map_err(err)
        };
        let (det, rec, gpu) = match (open(FILES[0].2, true), open(FILES[1].2, true)) {
            (Ok(d), Ok(r)) => (d, r, true),
            _ => (open(FILES[0].2, false)?, open(FILES[1].2, false)?, false),
        };
        let yml = std::fs::read_to_string(dir.join(FILES[2].2)).map_err(err)?;
        let mut chars = vec![String::new()];
        chars.extend(read_dict(&yml));
        Ok(Self { _env: env, det, rec, chars, gpu })
    }

    pub fn recognize(&mut self, bgra: &[u8], w: i32, h: i32) -> Result<Vec<Line>, String> {
        let (w, h) = (w as usize, h as usize);
        let boxes = self.detect(bgra, w, h)?;
        let crops: Vec<[f32; 4]> = boxes.iter().map(|b| b.0).collect();
        let texts = self.read(bgra, w, &crops)?;
        Ok(boxes
            .into_iter()
            .zip(texts)
            .filter(|(_, (_, score))| *score >= 0.5)
            .filter_map(|((_, tight), (text, _))| {
                let text = crate::ocr::clean_text(&text.split_whitespace().map(String::from).collect::<Vec<_>>())?;
                Some(Line { text, rect: tight, line_h: tight[3] })
            })
            .collect())
    }

    /// DB text detection → per line (padded box for recognition, tight text box for layout), [x, y, w, h] in source pixels.
    /// shortcut: axis-aligned boxes only (no rotated text), fine for game UI; upgrade to min-area rects for tilted text.
    fn detect(&mut self, bgra: &[u8], w: usize, h: usize) -> Result<Vec<([f32; 4], [f32; 4])>, String> {
        // longest side 960: as accurate as full 1080p here (0.2% vs 0.8% CER) at less than half the VRAM; 800 lost small HUD text
        let ratio = if w.max(h) > 960 { 960.0 / w.max(h) as f32 } else { 1.0 };
        let round32 = |v: usize| (((v as f32 * ratio) / 32.0).round() as usize * 32).max(32);
        let (rw, rh) = (round32(w), round32(h));
        let mut data = vec![0f32; 3 * rw * rh];
        to_chw(bgra, w, [0, 0, w, h], rw, rh, &mut data, rw);
        let outputs = self.det.run(ort::inputs![Tensor::from_array(([1usize, 3, rh, rw], data)).map_err(err)?]).map_err(err)?;
        let (_, prob) = outputs[0].try_extract_tensor::<f32>().map_err(err)?;

        // binarize at 0.3, dilate with a 2x2 kernel, then 8-connected components
        let on = |x: usize, y: usize| prob[y * rw + x] > 0.3;
        let mut mask = vec![false; rw * rh];
        for y in 0..rh {
            for x in 0..rw {
                mask[y * rw + x] = on(x, y) || (x > 0 && on(x - 1, y)) || (y > 0 && on(x, y - 1)) || (x > 0 && y > 0 && on(x - 1, y - 1));
            }
        }
        let (sx, sy) = (w as f32 / rw as f32, h as f32 / rh as f32);
        let mut seen = vec![false; rw * rh];
        let mut boxes = vec![];
        let mut stack = vec![];
        for start in 0..rw * rh {
            if !mask[start] || seen[start] {
                continue;
            }
            let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
            seen[start] = true;
            stack.push(start);
            while let Some(i) = stack.pop() {
                let (x, y) = (i % rw, i / rw);
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
                for (dx, dy) in [(-1i64, -1i64), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx >= 0 && ny >= 0 && (nx as usize) < rw && (ny as usize) < rh {
                        let j = ny as usize * rw + nx as usize;
                        if mask[j] && !seen[j] {
                            seen[j] = true;
                            stack.push(j);
                        }
                    }
                }
            }
            let (bw, bh) = (x1 - x0 + 1, y1 - y0 + 1);
            if bw.min(bh) < 3 || boxes.len() >= 1000 {
                continue;
            }
            let score = (y0..=y1).flat_map(|y| (x0..=x1).map(move |x| (x, y))).map(|(x, y)| prob[y * rw + x]).sum::<f32>() / (bw * bh) as f32;
            if score < 0.5 {
                continue;
            }
            // DB predicts shrunk text kernels; grow them back by a polygon offset of area * k / perimeter:
            // k = 1.6 gives Paddle's padded crop for recognition, k = 1 - 0.4² undoes the training shrink (the text itself)
            let grow = |k: f32| {
                let d = (bw * bh) as f32 * k / (2 * (bw + bh)) as f32;
                let (ex0, ey0, ex1, ey1) = (x0 as f32 - d, y0 as f32 - d, x1 as f32 + 1.0 + d, y1 as f32 + 1.0 + d);
                let (bx0, by0) = ((ex0 * sx).max(0.0), (ey0 * sy).max(0.0));
                let (bx1, by1) = ((ex1 * sx).min(w as f32), (ey1 * sy).min(h as f32));
                [bx0, by0, bx1 - bx0, by1 - by0]
            };
            let crop = grow(1.6);
            if crop[2].min(crop[3]) < 5.0 * sx.min(sy) {
                continue;
            }
            boxes.push((crop, grow(0.84)));
        }
        // reading order: top to bottom, left to right within a ~10 px row
        boxes.sort_by(|(a, _), (b, _)| ((a[1] / 10.0) as i32, a[0] as i32).cmp(&((b[1] / 10.0) as i32, b[0] as i32)));
        Ok(boxes)
    }

    /// CTC recognition of each box, batched by aspect ratio like Paddle (height 48, width ≥ 320).
    fn read(&mut self, bgra: &[u8], w: usize, boxes: &[[f32; 4]]) -> Result<Vec<(String, f32)>, String> {
        let mut out = vec![(String::new(), 0.0); boxes.len()];
        let ratio = |b: &[f32; 4]| b[2] / b[3].max(1.0);
        let mut order: Vec<usize> = (0..boxes.len()).collect();
        order.sort_by(|&a, &b| ratio(&boxes[a]).total_cmp(&ratio(&boxes[b])));
        for batch in order.chunks(6) {
            let max_ratio = batch.iter().map(|&i| ratio(&boxes[i])).fold(320.0 / 48.0, f32::max);
            let img_w = (48.0 * max_ratio).ceil() as usize;
            let mut data = vec![0f32; batch.len() * 3 * 48 * img_w];
            for (k, &i) in batch.iter().enumerate() {
                let b = boxes[i];
                let crop = [b[0] as usize, b[1] as usize, (b[2] as usize).max(1), (b[3] as usize).max(1)];
                let dw = ((48.0 * ratio(&b)).ceil() as usize).clamp(1, img_w);
                to_chw(bgra, w, crop, dw, 48, &mut data[k * 3 * 48 * img_w..(k + 1) * 3 * 48 * img_w], img_w);
            }
            let outputs = self.rec.run(ort::inputs![Tensor::from_array(([batch.len(), 3, 48, img_w], data)).map_err(err)?]).map_err(err)?;
            let (shape, probs) = outputs[0].try_extract_tensor::<f32>().map_err(err)?;
            let (steps, classes) = (shape[1] as usize, shape[2] as usize);
            if classes == self.chars.len() + 1 && self.chars.last().is_some_and(|c| c != " ") {
                self.chars.push(" ".into()); // Paddle appends a space class after the dictionary
            }
            for (k, &i) in batch.iter().enumerate() {
                let (mut text, mut sum, mut n, mut last) = (String::new(), 0.0, 0, 0);
                for t in 0..steps {
                    let row = &probs[(k * steps + t) * classes..(k * steps + t + 1) * classes];
                    let (best, p) = row.iter().enumerate().fold((0, f32::MIN), |m, (j, &p)| if p > m.1 { (j, p) } else { m });
                    if best != 0 && best != last {
                        text.push_str(self.chars.get(best).map_or("", |s| s.as_str()));
                        (sum, n) = (sum + p, n + 1);
                    }
                    last = best;
                }
                out[i] = (text, if n > 0 { sum / n as f32 } else { 0.0 });
            }
        }
        Ok(out)
    }
}

/// Bilinear resize of a BGRA crop into a normalized CHW tensor ((v/255 - 0.5) / 0.5), BGR channel order like Paddle's
/// cv2 input. Writes `dw` columns of a `stride`-wide plane; the rest stays 0 (Paddle's padding).
fn to_chw(src: &[u8], src_w: usize, [cx, cy, cw, ch]: [usize; 4], dw: usize, dh: usize, out: &mut [f32], stride: usize) {
    let plane = dh * stride;
    let src_h = src.len() / 4 / src_w;
    for y in 0..dh {
        let fy = ((y as f32 + 0.5) * ch as f32 / dh as f32 - 0.5).max(0.0);
        let (y0, wy) = (fy as usize, fy.fract());
        let (ya, yb) = ((cy + y0).min(src_h - 1), (cy + y0 + 1).min(cy + ch - 1).min(src_h - 1));
        for x in 0..dw {
            let fx = ((x as f32 + 0.5) * cw as f32 / dw as f32 - 0.5).max(0.0);
            let (x0, wx) = (fx as usize, fx.fract());
            let (xa, xb) = ((cx + x0).min(src_w - 1), (cx + x0 + 1).min(cx + cw - 1).min(src_w - 1));
            for c in 0..3 {
                let px = |xx: usize, yy: usize| src[(yy * src_w + xx) * 4 + c] as f32;
                let v = (px(xa, ya) * (1.0 - wx) + px(xb, ya) * wx) * (1.0 - wy) + (px(xa, yb) * (1.0 - wx) + px(xb, yb) * wx) * wy;
                out[c * plane + y * stride + x] = (v / 255.0 - 0.5) / 0.5;
            }
        }
    }
}

/// The `character_dict:` list from Paddle's inference.yml (one YAML scalar per line).
fn read_dict(yml: &str) -> Vec<String> {
    let mut lines = yml.lines().skip_while(|l| l.trim() != "character_dict:").skip(1);
    let mut out = vec![];
    while let Some(v) = lines.next().and_then(|l| l.strip_prefix("  - ")) {
        out.push(match v.as_bytes() {
            [b'\'', .., b'\''] => v[1..v.len() - 1].replace("''", "'"),
            [b'"', .., b'"'] => serde_json::from_str(v).unwrap_or_else(|_| v[1..v.len() - 1].to_string()),
            _ => v.to_string(),
        });
    }
    out
}

#[test]
fn dict_parsing() {
    let yml = "PostProcess:\n  name: CTCLabelDecode\n  character_dict:\n  - '!'\n  - '\"'\n  - ''''\n  - a\n  - \"\\u3042\"\nother: 1\n";
    assert_eq!(read_dict(yml), ["!", "\"", "'", "a", "あ"]);
}

/// `SCENES=<dir> cargo test --release paddle_bench -- --ignored --nocapture` (needs the model files in %LOCALAPPDATA%\PaperTranslate\models)
#[test]
#[ignore]
fn paddle_bench() {
    let models = std::path::PathBuf::from(std::env::var("LOCALAPPDATA").unwrap()).join("PaperTranslate").join("models");
    let mut p = Paddle::load(&models).unwrap();
    println!("gpu: {}", p.gpu);
    let dir = std::path::PathBuf::from(std::env::var("SCENES").unwrap());
    let mut out = serde_json::Map::new();
    for name in ["dialogue", "cards", "hard_fonts", "hard_small", "hard_glow", "japanese"] {
        let px = std::fs::read(dir.join(format!("{name}.bgra"))).unwrap();
        p.recognize(&px, 1920, 1080).unwrap();
        let t = std::time::Instant::now();
        let lines = p.recognize(&px, 1920, 1080).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("{name}: {} lines in {ms:.0} ms", lines.len());
        let b = crate::ocr::blocks(lines.clone());
        let c = crate::ocr::find_choice(&b);
        println!("   choice {:?}", c.as_ref().map(|(o, strong)| (o.iter().map(|o| o.iter().map(|&i| b[i].text.as_str()).collect::<Vec<_>>().join(" | ")).collect::<Vec<_>>(), strong)));
        for r in crate::ocr::regions(&b, c.as_ref().map(|c| c.0.as_slice())) {
            println!("   region {:?}", r.iter().map(|&i| &b[i].text).collect::<Vec<_>>());
        }
        out.insert(name.into(), serde_json::json!({ "ms": ms, "lines": lines.iter().map(|l| l.text.clone()).collect::<Vec<_>>() }));
    }
    std::fs::write(dir.join("paddle_rs.json"), serde_json::to_string_pretty(&out).unwrap()).unwrap();
}
