use windows::Globalization::Language;
use windows::Graphics::Imaging::{BitmapPixelFormat, SoftwareBitmap};
use windows::Media::Ocr::OcrEngine;
use windows::Storage::Streams::DataWriter;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Gdi::*;
use windows::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetSystemMetrics, GetWindowRect, SM_CXSCREEN, SM_CYSCREEN};
use windows::core::HSTRING;

#[derive(Clone, Debug)]
pub struct Line {
    pub text: String,
    /// x, y, w, h in physical screen pixels
    pub rect: [f32; 4],
    pub line_h: f32,
}

pub fn languages() -> Vec<String> {
    OcrEngine::AvailableRecognizerLanguages()
        .map(|v| v.into_iter().filter_map(|l| l.LanguageTag().ok()).map(|s| s.to_string()).collect())
        .unwrap_or_default()
}

pub fn engine(lang: &str) -> Result<OcrEngine, String> {
    let l = Language::CreateLanguage(&HSTRING::from(lang)).map_err(|e| e.to_string())?;
    OcrEngine::TryCreateFromLanguage(&l).map_err(|_| format!("OCR language '{lang}' is not installed in Windows"))
}

/// Captures the box the player dragged, or else the foreground window (the game), clipped to the primary monitor;
/// returns BGRA pixels and the captured rect [x, y, w, h]. Falls back to the whole monitor for tiny or missing windows.
/// shortcut: primary monitor only via GDI, upgrade to Windows.Graphics.Capture for multi-monitor or exclusive-fullscreen games.
pub fn capture_foreground(boundary: Option<[i32; 4]>) -> Result<(Vec<u8>, [i32; 4]), String> {
    unsafe {
        let (sw, sh) = (GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN));
        let mut r = RECT::default();
        let fg = GetForegroundWindow();
        if let Some([bx, by, bw, bh]) = boundary {
            r = RECT { left: bx, top: by, right: bx + bw, bottom: by + bh };
        }
        let [x, y, w, h] = match boundary.is_some() || (!fg.is_invalid() && GetWindowRect(fg, &mut r).is_ok()) {
            true if r.right.min(sw) - r.left.max(0) >= 16 && r.bottom.min(sh) - r.top.max(0) >= 16 => {
                [r.left.max(0), r.top.max(0), r.right.min(sw) - r.left.max(0), r.bottom.min(sh) - r.top.max(0)]
            }
            _ => [0, 0, sw, sh],
        };
        let screen = GetDC(None);
        let mem = CreateCompatibleDC(Some(screen));
        let bmp = CreateCompatibleBitmap(screen, w, h);
        let old = SelectObject(mem, bmp.into());
        let ok = BitBlt(mem, 0, 0, w, h, Some(screen), x, y, SRCCOPY);
        let mut bi = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: w,
                biHeight: -h, // top-down
                biPlanes: 1,
                biBitCount: 32,
                biCompression: BI_RGB.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut buf = vec![0u8; (w * h * 4) as usize];
        let rows = GetDIBits(mem, bmp, 0, h as u32, Some(buf.as_mut_ptr().cast()), &mut bi, DIB_RGB_COLORS);
        SelectObject(mem, old);
        let _ = DeleteObject(bmp.into());
        let _ = DeleteDC(mem);
        ReleaseDC(None, screen);
        ok.map_err(|e| e.to_string())?;
        if rows == 0 {
            return Err("screen capture failed".into());
        }
        Ok((buf, [x, y, w, h]))
    }
}

/// Windows' built-in OCR: ~30 ms for a 1080p frame on CPU, ~20x faster than PP-OCRv5 (ONNX, CPU) with the same
/// text on game scenes, and no GPU time taken from the game or the LLM.
pub fn recognize(engine: &OcrEngine, bgra: &[u8], w: i32, h: i32) -> Result<Vec<Line>, String> {
    let run = || -> windows::core::Result<Vec<Line>> {
        let writer = DataWriter::new()?;
        writer.WriteBytes(bgra)?;
        let bmp = SoftwareBitmap::CreateCopyFromBuffer(&writer.DetachBuffer()?, BitmapPixelFormat::Bgra8, w, h)?;
        let result = engine.RecognizeAsync(&bmp)?.join()?;
        let mut out = vec![];
        for line in result.Lines()? {
            let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            let mut words = vec![];
            for word in line.Words()? {
                let text = word.Text()?.to_string();
                if junk_word(&text) {
                    continue;
                }
                let r = word.BoundingRect()?;
                (x0, y0) = (x0.min(r.X), y0.min(r.Y));
                (x1, y1) = (x1.max(r.X + r.Width), y1.max(r.Y + r.Height));
                words.push(text);
            }
            if let Some(text) = clean_text(&words).filter(|_| x0 < x1) {
                out.push(Line { text, rect: [x0, y0, x1 - x0, y1 - y0], line_h: y1 - y0 });
            }
        }
        Ok(out)
    };
    run().map_err(|e| e.to_string())
}

/// Drops icon-noise words and single glyphs; None when nothing is left. Numbers stay ("1. 5", "500 gold"): quiz and
/// shop choices are often only numbers.
pub fn clean_text(words: &[String]) -> Option<String> {
    let kept: Vec<String> = words.iter().filter(|w| !junk_word(w)).cloned().collect();
    let text = join_words(&kept);
    (text.chars().filter(|c| c.is_alphanumeric()).count() >= 2).then_some(text)
}

/// Lines with words get translated; lines of numbers and symbols are shown as they are.
pub fn has_words(text: &str) -> bool {
    text.chars().filter(|c| c.is_alphabetic()).count() >= 2
}

/// OCR of icons and textures gives tokens like "L8fi0WN" or "lIlI1"; real words don't put digits between
/// letters or flip from lower to upper case more than once ("McDonald" flips once).
fn junk_word(w: &str) -> bool {
    let c: Vec<char> = w.chars().filter(|c| c.is_alphanumeric()).collect();
    let digit_inside = c.windows(3).any(|t| t[0].is_alphabetic() && t[1].is_ascii_digit() && t[2].is_alphabetic());
    let flips = c.windows(2).filter(|p| p[0].is_lowercase() && p[1].is_uppercase()).count();
    digit_inside || flips > 1
}

/// Downscales a BGRA capture to at most `max_w` wide (area average, keeps text legible) and encodes a 24-bit BMP,
/// which llama.cpp's image loader reads, so no image codec is needed.
pub fn bmp_thumbnail(bgra: &[u8], w: i32, h: i32, max_w: i32) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    let ow = w.min(max_w as usize).max(1);
    let oh = (h * ow / w).max(1);
    let row = (ow * 3 + 3) & !3; // BMP rows are padded to 4 bytes
    let size = 54 + row * oh;
    let mut out = Vec::with_capacity(size);
    out.extend_from_slice(b"BM");
    for v in [size as u32, 0, 54, 40, ow as u32, oh as u32] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    out.extend_from_slice(&[0u8; 24]);
    for y in (0..oh).rev() {
        let (y0, y1) = (y * h / oh, ((y + 1) * h / oh).max(y * h / oh + 1));
        for x in 0..ow {
            let (x0, x1) = (x * w / ow, ((x + 1) * w / ow).max(x * w / ow + 1));
            let mut sum = [0u32; 3];
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let p = (sy * w + sx) * 4;
                    for c in 0..3 {
                        sum[c] += bgra[p + c] as u32;
                    }
                }
            }
            let n = ((y1 - y0) * (x1 - x0)) as u32;
            out.extend(sum.map(|s| (s / n) as u8));
        }
        out.resize(out.len() + row - ow * 3, 0);
    }
    out
}

/// Every second pixel in gray, to notice when nothing on screen changed. Finer than a coarse grid on purpose: on a mostly
/// plain page, new text fell between 64x36 samples and the old reading was reused.
pub fn fingerprint(bgra: &[u8], w: i32, h: i32) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    (0..h / 2)
        .flat_map(|y| (0..w / 2).map(move |x| (y * 2 * w + x * 2) * 4))
        .map(|p| ((bgra[p] as u32 + bgra[p + 1] as u32 * 2 + bgra[p + 2] as u32) / 4) as u8)
        .collect()
}

/// Same picture unless more than a few dozen samples clearly changed (a blinking caret or video noise stays "same",
/// a new word does not).
pub fn same_frame(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).filter(|(x, y)| x.abs_diff(**y) > 24).take(40).count() < 40
}

#[test]
fn frame_change_detection() {
    let (w, h) = (200, 100);
    let plain = vec![255u8; w * h * 4];
    let mut caret = plain.clone();
    for y in 10..14 {
        caret[(y * w + 50) * 4..(y * w + 51) * 4].fill(0); // a 1x4 caret
    }
    let mut word = plain.clone();
    for y in 40..56 {
        for x in 60..120 {
            if (x + y) % 3 == 0 {
                word[(y * w + x) * 4..(y * w + x + 1) * 4].fill(0); // strokes of a word
            }
        }
    }
    let f = |px: &[u8]| fingerprint(px, w as i32, h as i32);
    assert!(same_frame(&f(&plain), &f(&caret)));
    assert!(!same_frame(&f(&plain), &f(&word)));
}

/// Japanese and Chinese are written without spaces.
fn cjk(c: char) -> bool {
    matches!(c, '\u{3000}'..='\u{30FF}' | '\u{3400}'..='\u{9FFF}' | '\u{FF00}'..='\u{FFEF}')
}

/// Windows OCR returns words; put spaces back only between non-CJK words ("金貨" "500" "枚" → "金貨500枚").
fn join_words(words: &[String]) -> String {
    let mut out = String::new();
    for w in words {
        if !out.is_empty() && !out.chars().last().is_some_and(cjk) && !w.chars().next().is_some_and(cjk) {
            out.push(' ');
        }
        out.push_str(w);
    }
    out
}

/// Installs Windows' OCR pack for a language tag (e.g. "ja-JP") through an elevated PowerShell; Windows asks for admin.
pub fn install_language(tag: &str) {
    use windows::Win32::UI::Shell::ShellExecuteW;
    use windows::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;
    let args = format!(
        "-NoProfile -Command \"Add-WindowsCapability -Online -Name 'Language.OCR~~~{tag}~0.0.1.0'; Write-Host 'Done, click Refresh in OSTD.'; Start-Sleep 3\""
    );
    unsafe {
        ShellExecuteW(None, &HSTRING::from("runas"), &HSTRING::from("powershell.exe"), &HSTRING::from(args), None, SW_SHOWNORMAL);
    }
}

/// "A. ", "b) ", "1. ", "[2] ", "(3) " style option markers, plus ①–⑨ and ▶/・ bullets used by Japanese games.
pub fn marker(t: &str) -> bool {
    let c: Vec<char> = t.trim_start_matches(['[', '(']).chars().take(3).collect();
    if c.len() >= 2 && matches!(c[0], '\u{2460}'..='\u{2468}' | '▶' | '►' | '▷' | '・' | '◆') {
        return true;
    }
    c.len() == 3 && (matches!(c[0], 'A'..='F' | 'a'..='f' | '1'..='9')) && matches!(c[1], '.' | ')' | ']' | ':') && c[2] == ' '
}

/// The marker letter of "1.5", "A.Venus", "2)Run": OCR often drops the space after a marker. Alone this could be a decimal
/// ("3.2"), so it only counts as an option when the markers run in order (see `find_choice`).
fn loose_marker(t: &str) -> Option<char> {
    let c: Vec<char> = t.trim_start_matches(['[', '(']).chars().take(3).collect();
    (c.len() == 3 && matches!(c[0], 'A'..='F' | 'a'..='f' | '1'..='9') && matches!(c[1], '.' | ')') && c[2] != ' ').then_some(c[0])
}

/// Merge stacked lines of one paragraph (dialogue boxes) so they are translated together. A line joins the block whose
/// last line sits right above it, wherever that block is in the list: readers return lines column-wise (Windows) or
/// row-wise across columns (PaddleOCR).
pub fn blocks(lines: Vec<Line>) -> Vec<Line> {
    let mut out: Vec<(Line, [f32; 4])> = vec![];
    for l in lines {
        let [x, y, w, h] = l.rect;
        let fits = |last: &[f32; 4]| {
            let gap = y - (last[1] + last[3]);
            let left = (x - last[0]).abs() < 2.0 * h;
            let centered = ((x + w / 2.0) - (last[0] + last[2] / 2.0)).abs() < 2.0 * h;
            let same_size = (0.7..1.4).contains(&(h / last[3]));
            // wrapped paragraphs never get much wider; a short speaker name above a long line is its own block
            let continues = w <= last[2] * 1.3;
            gap > -0.3 * h && gap < 0.9 * h && (left || centered) && same_size && continues
        };
        // a marked line starts an option; two lines that both open with a marker ("1.5" over "2.6") are two options
        let fits_after = |b: &Line| !marker(&l.text) && !(loose_marker(&l.text).is_some() && loose_marker(&b.text).is_some());
        match out.iter_mut().rev().find(|(b, last)| fits(last) && fits_after(b)) {
            Some((b, last)) => {
                b.text = join_words(&[b.text.clone(), l.text]);
                let (x0, y0) = (b.rect[0].min(x), b.rect[1].min(y));
                let (x1, y1) = ((b.rect[0] + b.rect[2]).max(x + w), (b.rect[1] + b.rect[3]).max(y + h));
                b.rect = [x0, y0, x1 - x0, y1 - y0];
                *last = l.rect;
            }
            None => {
                let r = l.rect;
                out.push((l, r));
            }
        }
    }
    out.into_iter().map(|(b, _)| b).collect()
}

fn cx(b: &Line) -> f32 {
    b.rect[0] + b.rect[2] / 2.0
}

/// Finds a set of player options; each option is a list of block indices (a card can be title + text + rarity).
/// Evidence in order: explicit "A./1." markers, a row of evenly spaced cards (both strong), then a spaced
/// vertical list (weak: could be a menu or narration, so the decision model must confirm it). Returns (options, strong).
pub fn find_choice(b: &[Line]) -> Option<(Vec<Vec<usize>>, bool)> {
    let marked: Vec<usize> = (0..b.len()).filter(|&i| marker(&b[i].text)).collect();
    if (2..=8).contains(&marked.len()) {
        return Some((marked.into_iter().map(|i| vec![i]).collect(), true));
    }
    // markers without the space, accepted when they count up from 1 / A in reading order
    let mut loose: Vec<(usize, char)> = (0..b.len()).filter_map(|i| loose_marker(&b[i].text).map(|m| (i, m))).collect();
    loose.sort_by(|x, y| b[x.0].rect[1].total_cmp(&b[y.0].rect[1]));
    let in_order = loose.first().is_some_and(|f| matches!(f.1, '1' | 'A' | 'a'))
        && loose.windows(2).all(|w| w[1].1 as u32 == w[0].1 as u32 + 1);
    if in_order && (2..=8).contains(&loose.len()) {
        return Some((loose.into_iter().map(|(i, _)| vec![i]).collect(), true));
    }
    if let Some(cards) = card_row(b) {
        return Some((cards, true));
    }
    let run = option_run(b);
    (!run.is_empty()).then(|| (run.into_iter().map(|i| vec![i]).collect(), false))
}

/// For weak evidence only: a list is worth asking the decision model about when a question is nearby or the
/// options read like replies. Short labels with no question ("Continue / Settings / Quit") are menus.
pub fn worth_asking(options: &[String], scene: &[String]) -> bool {
    let reply = |o: &String| o.split_whitespace().count() >= 3 || o.chars().filter(|&c| cjk(c)).count() >= 5;
    scene.iter().any(|t| t.contains('?') || t.contains('？')) || options.iter().all(reply)
}

/// Roguelike reward screens: 2–5 titles on one line, evenly spaced, each heading a column of text below.
fn card_row(b: &[Line]) -> Option<Vec<Vec<usize>>> {
    let mut best: Option<Vec<usize>> = None;
    for i in 0..b.len() {
        let h = b[i].line_h;
        let mut row: Vec<usize> = (0..b.len())
            .filter(|&j| (b[j].rect[1] - b[i].rect[1]).abs() < 0.5 * h && (0.7..1.4).contains(&(b[j].line_h / h)))
            .collect();
        row.sort_by(|&p, &q| b[p].rect[0].total_cmp(&b[q].rect[0]));
        let spaced = row.windows(2).all(|w| b[w[1]].rect[0] - (b[w[0]].rect[0] + b[w[0]].rect[2]) > 2.0 * h);
        let centers: Vec<f32> = row.iter().map(|&k| cx(&b[k])).collect();
        let step = if centers.len() > 1 { centers[1] - centers[0] } else { 0.0 };
        let even = centers.windows(2).all(|w| ((w[1] - w[0]) - step).abs() < 0.25 * step);
        let better = best.as_ref().is_none_or(|bb| row.len() > bb.len() || (row.len() == bb.len() && b[row[0]].rect[1] < b[bb[0]].rect[1]));
        if (2..=5).contains(&row.len()) && spaced && even && better {
            best = Some(row);
        }
    }
    let row = best?;
    let h = b[row[0]].line_h;
    let half = (cx(&b[row[1]]) - cx(&b[row[0]])) / 2.0;
    let top = b[row[0]].rect[1] - 0.5 * h;
    let columns = row.iter().map(|&r| {
        let mut col: Vec<usize> = (0..b.len()).filter(|&j| (cx(&b[j]) - cx(&b[r])).abs() < 0.8 * half && b[j].rect[1] >= top).collect();
        col.sort_by(|&p, &q| b[p].rect[1].total_cmp(&b[q].rect[1]));
        // the card ends at the first big vertical gap
        let end = col.windows(2).position(|w| b[w[1]].rect[1] - (b[w[0]].rect[1] + b[w[0]].rect[3]) > 5.0 * h);
        col.truncate(end.map_or(col.len(), |p| p + 1));
        col
    });
    Some(columns.collect())
}

/// Indices of the longest run of short, single-line, aligned blocks: dialogue options or a menu.
fn option_run(blocks: &[Line]) -> Vec<usize> {
    let (mut best, mut run): (Vec<usize>, Vec<usize>) = (vec![], vec![]);
    for (i, b) in blocks.iter().enumerate() {
        if b.rect[3] > b.line_h * 1.5 || b.text.chars().count() > 80 {
            run.clear();
            continue;
        }
        let follows = run.last().is_some_and(|&j| {
            let (a, h) = (&blocks[j].rect, b.line_h);
            let gap = b.rect[1] - (a[1] + a[3]);
            let centered = ((b.rect[0] + b.rect[2] / 2.0) - (a[0] + a[2] / 2.0)).abs() < 2.0 * h;
            (0.0..3.0 * h).contains(&gap) && ((b.rect[0] - a[0]).abs() < 2.0 * h || centered)
        });
        if !follows {
            run.clear();
        }
        run.push(i);
        if run.len() >= best.len() {
            best = run.clone(); // ties go to the lower run: choices usually sit below the dialogue
        }
    }
    if (2..=6).contains(&best.len()) { best } else { vec![] }
}

/// Context boundaries: blocks close to each other (a dialogue box, a panel) form one region, and all blocks of a
/// choice set share one. Each region is translated in its own request so the model only sees text that belongs together.
pub fn regions(b: &[Line], choice: Option<&[Vec<usize>]>) -> Vec<Vec<usize>> {
    fn root(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    fn join(p: &mut [usize], i: usize, j: usize) {
        let (ri, rj) = (root(p, i), root(p, j));
        p[ri] = rj;
    }
    let mut parent: Vec<usize> = (0..b.len()).collect();
    for i in 0..b.len() {
        for j in i + 1..b.len() {
            let (a, c, h) = (b[i].rect, b[j].rect, b[i].line_h.max(b[j].line_h));
            let dx = a[0].max(c[0]) - (a[0] + a[2]).min(c[0] + c[2]);
            let dy = a[1].max(c[1]) - (a[1] + a[3]).min(c[1] + c[3]);
            if dx < 3.0 * h && dy < 1.5 * h {
                join(&mut parent, i, j);
            }
        }
    }
    let in_choice: Vec<usize> = choice.into_iter().flatten().flatten().copied().collect();
    for w in in_choice.windows(2) {
        join(&mut parent, w[0], w[1]);
    }
    let mut groups: Vec<(usize, Vec<usize>)> = vec![];
    for i in 0..b.len() {
        let r = root(&mut parent, i);
        match groups.iter_mut().find(|g| g.0 == r) {
            Some(g) => g.1.push(i),
            None => groups.push((r, vec![i])),
        }
    }
    let mut groups: Vec<Vec<usize>> = groups.into_iter().map(|g| g.1).collect();
    for g in &mut groups {
        g.sort_by(|&p, &q| b[p].rect[1].total_cmp(&b[q].rect[1]).then(b[p].rect[0].total_cmp(&b[q].rect[0])));
    }
    groups
}

/// `OCR_BENCH=<file.bgra> cargo test --release ocr_bench -- --ignored --nocapture` (1920x1080 raw BGRA)
#[test]
#[ignore]
fn ocr_bench() {
    let px = std::fs::read(std::env::var("OCR_BENCH").unwrap()).unwrap();
    let e = engine("en-US").unwrap();
    for run in 0..4 {
        let t = std::time::Instant::now();
        let lines = recognize(&e, &px, 1920, 1080).unwrap();
        if run == 3 {
            for l in &lines {
                println!("{:?} {:?}", l.text, l.rect.map(|v| v as i32));
            }
        }
        println!("run {run}: {} lines in {:?}", lines.len(), t.elapsed());
        if run == 3 {
            let b = blocks(lines);
            let c = find_choice(&b);
            println!("choice {c:?}");
            for r in regions(&b, c.as_ref().map(|c| c.0.as_slice())) {
                println!("region {:?}", r.iter().map(|&i| &b[i].text).collect::<Vec<_>>());
            }
        }
    }
}

#[cfg(test)]
fn fixture(v: &[(&str, [f32; 4])]) -> Vec<Line> {
    v.iter().map(|(t, r)| Line { text: t.to_string(), rect: *r, line_h: r[3] }).collect()
}

#[test]
fn merges_paragraph_but_not_far_lines() {
    let l = |t: &str, x, y| Line { text: t.into(), rect: [x, y, 300.0, 20.0], line_h: 20.0 };
    let b = blocks(vec![l("Hello there,", 100.0, 100.0), l("traveler.", 100.0, 125.0), l("Yes", 100.0, 300.0), l("No", 100.0, 360.0)]);
    assert_eq!(b.iter().map(|b| b.text.as_str()).collect::<Vec<_>>(), ["Hello there, traveler.", "Yes", "No"]);
    assert_eq!(b[0].rect, [100.0, 100.0, 300.0, 45.0]);
    assert_eq!(find_choice(&b), Some((vec![vec![1], vec![2]], false)));
}

#[test]
fn thumbnail_is_a_valid_bmp() {
    let px = vec![200u8; 10 * 6 * 4];
    let bmp = bmp_thumbnail(&px, 10, 6, 4);
    assert_eq!(&bmp[..2], b"BM");
    assert_eq!(bmp.len(), 54 + 12 * 2); // 4x2 pixels, 12-byte rows
    assert_eq!(bmp[54], 200);
}

#[test]
fn quiz_options_without_spaces() {
    // PaddleOCR reads "1. 5" as "1.5"; tightly stacked, they used to merge into one paragraph
    let b = blocks(fixture(&[
        ("one + one is", [520.0, 330.0, 200.0, 28.0]),
        ("1.5", [540.0, 400.0, 46.0, 26.0]),
        ("2.6", [540.0, 450.0, 46.0, 26.0]),
        ("3.2", [540.0, 500.0, 46.0, 26.0]),
    ]));
    assert_eq!(b.len(), 4);
    assert_eq!(find_choice(&b), Some((vec![vec![1], vec![2], vec![3]], true)));
    // a lone decimal is not an option list
    assert!(find_choice(&blocks(fixture(&[("Speed 1.5", [0.0, 0.0, 120.0, 20.0]), ("3.2", [0.0, 300.0, 40.0, 20.0])]))).is_none());
}

#[test]
fn numbers_are_kept_but_not_translated() {
    let w = |x: &[&str]| x.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    assert_eq!(clean_text(&w(&["1.", "5"])).as_deref(), Some("1. 5"));
    assert_eq!(clean_text(&w(&["4"])), None);
    assert!(!has_words("1. 5") && has_words("one + one is"));
}

#[test]
fn junk_filter() {
    assert!(junk_word("L8fi0WNuuaicnunnt") && junk_word("lIlIlI"));
    assert!(!junk_word("McDonald") && !junk_word("500") && !junk_word("HP") && !junk_word("3rd") && !junk_word("gold,"));
    let v = |x: &[&str]| x.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    assert!(!worth_asking(&v(&["Continue", "New Game", "Settings", "Quit"]), &v(&["Version 1.0.4"])));
    assert!(worth_asking(&v(&["Sure.", "Not today."]), &v(&["Will you help me?"])));
    assert!(marker("① 今夜、泥棒を追いかける。") && marker("▶はい"));
    let w = |x: &[&str]| x.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    assert_eq!(join_words(&w(&["先に", "金貨", "500", "枚を払え。"])), "先に金貨500枚を払え。");
    assert_eq!(join_words(&w(&["Pay", "me", "500", "gold"])), "Pay me 500 gold");
    assert!(marker("A. Yes") && marker("(2) No") && !marker("e.g. this") && !marker("Ah. Hello"));
}

// rects below are real Windows OCR output for the test scenes (dialogue box, roguelike reward cards)
#[test]
fn dialogue_choices_and_one_region() {
    let b = blocks(fixture(&[
        ("Elder Mira", [311.0, 671.0, 190.0, 26.0]),
        ("The Moonstone Amulet was stolen from Stormhold last night.", [310.0, 730.0, 815.0, 30.0]),
        ("If the Shadow Guild sells it, the old seal will break.", [312.0, 772.0, 653.0, 27.0]),
        ("A. I will track down the thief tonight.", [360.0, 830.0, 446.0, 27.0]),
        ("B. That is not my problem.", [362.0, 870.0, 321.0, 27.0]),
        ("C. Pay me 500 gold first.", [361.0, 910.0, 297.0, 27.0]),
    ]));
    assert_eq!(b.len(), 5); // the two story lines merge, the marked options don't
    let (choice, strong) = find_choice(&b).unwrap();
    assert_eq!((choice.clone(), strong), (vec![vec![2], vec![3], vec![4]], true));
    assert_eq!(regions(&b, Some(&choice)), vec![vec![0, 1, 2, 3, 4]]);
}

#[test]
fn speaker_name_stays_separate() {
    // PaddleOCR boxes for the Japanese scene: name in a bigger font right above the first line
    let b = blocks(fixture(&[
        ("長老ミラ", [311.0, 668.0, 120.0, 34.0]),
        ("昨夜、ストームホールドから月光石の首飾りが盗まれた。", [310.0, 724.0, 760.0, 31.0]),
        ("影のギルドが売れば、古い封印が壊れてしまう。", [310.0, 766.0, 640.0, 31.0]),
    ]));
    assert_eq!(b.len(), 2);
    assert_eq!(b[1].text, "昨夜、ストームホールドから月光石の首飾りが盗まれた。影のギルドが売れば、古い封印が壊れてしまう。");
}

#[test]
fn row_wise_cards_merge_the_same() {
    // PaddleOCR order: all titles, then each description row across the three cards
    let b = blocks(fixture(&[
        ("Thunder Strike", [378.0, 496.0, 283.0, 27.0]),
        ("Iron Skin", [873.0, 496.0, 174.0, 27.0]),
        ("Your attacks chain lightning", [362.0, 572.0, 316.0, 25.0]),
        ("Take 30% less damage", [831.0, 572.0, 257.0, 25.0]),
        ("to 3 nearby enemies.", [400.0, 608.0, 239.0, 25.0]),
        ("for 10 seconds after a dash.", [803.0, 608.0, 313.0, 19.0]),
    ]));
    let texts: Vec<&str> = b.iter().map(|l| l.text.as_str()).collect();
    assert_eq!(texts, ["Thunder Strike", "Iron Skin", "Your attacks chain lightning to 3 nearby enemies.", "Take 30% less damage for 10 seconds after a dash."]);
}

#[test]
fn reward_cards_are_one_choice() {
    let b = blocks(fixture(&[
        ("Thunder Strike", [378.0, 496.0, 283.0, 27.0]),
        ("Your attacks chain lightning", [362.0, 572.0, 316.0, 25.0]),
        ("to 3 nearby enemies.", [400.0, 608.0, 239.0, 25.0]),
        ("Rare", [499.0, 733.0, 43.0, 16.0]),
        ("Choose a Blessing", [723.0, 169.0, 475.0, 50.0]),
        ("Iron Skin", [873.0, 496.0, 174.0, 27.0]),
        ("Take 30% less damage", [831.0, 572.0, 257.0, 25.0]),
        ("for 10 seconds after a dash.", [803.0, 608.0, 313.0, 19.0]),
        ("Rare", [939.0, 733.0, 43.0, 16.0]),
        ("Golden Touch", [1272.0, 496.0, 257.0, 27.0]),
        ("Enemies drop 50% more gold,", [1230.0, 572.0, 339.0, 25.0]),
        ("but you lose 10 max HP.", [1264.0, 608.0, 274.0, 25.0]),
        ("Rare", [1379.0, 733.0, 43.0, 16.0]),
    ]));
    let (choice, strong) = find_choice(&b).unwrap();
    assert!(strong);
    let cards: Vec<Vec<&str>> = choice.iter().map(|o| o.iter().map(|&i| b[i].text.as_str()).collect()).collect();
    assert_eq!(cards.len(), 3);
    assert_eq!(cards[1], ["Iron Skin", "Take 30% less damage for 10 seconds after a dash.", "Rare"]);
    let regs = regions(&b, Some(&choice));
    assert_eq!(regs.len(), 2); // the title on its own, all three cards together
    assert!(regs.iter().any(|r| r.len() == 1 && b[r[0]].text == "Choose a Blessing"));
}

/// Benchmark dump: `SCENES=<dir> NAMES=a,b cargo test --release ocr_json -- --ignored` writes <dir>/win_ocr.json
#[test]
#[ignore]
fn ocr_json() {
    let dir = std::path::PathBuf::from(std::env::var("SCENES").unwrap());
    let e = engine("en-US").unwrap();
    let mut out = serde_json::Map::new();
    for name in std::env::var("NAMES").unwrap().split(',') {
        let px = std::fs::read(dir.join(format!("{name}.bgra"))).unwrap();
        let mut best = f64::MAX;
        let mut lines = vec![];
        for _ in 0..5 {
            let t = std::time::Instant::now();
            lines = recognize(&e, &px, 1920, 1080).unwrap();
            best = best.min(t.elapsed().as_secs_f64() * 1000.0);
        }
        let texts: Vec<String> = lines.into_iter().map(|l| l.text).collect();
        out.insert(name.into(), serde_json::json!({ "ms": best, "lines": texts }));
    }
    std::fs::write(dir.join("win_ocr.json"), serde_json::to_string_pretty(&out).unwrap()).unwrap();
}
