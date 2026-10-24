#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
mod llm;
mod ocr;
mod paddle;
mod setup;

use eframe::egui::{
    self, Color32, Context, FontData, FontDefinitions, FontFamily, FontId, Frame, ProgressBar, Rect, RichText,
    Sense, Shadow, Stroke, StrokeKind, TextFormat, Ui, ViewportBuilder, ViewportId, pos2, text::LayoutJob, vec2,
};
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState, hotkey::HotKey};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Child;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tray_icon::menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem};
use tray_icon::{MouseButton, MouseButtonState, TrayIcon, TrayIconBuilder, TrayIconEvent};
use windows::Win32::Foundation::HWND;
use windows::Win32::UI::WindowsAndMessaging::*;

const BG: Color32 = Color32::from_rgb(0xF4, 0xEC, 0xDB);
const PAPER: Color32 = Color32::from_rgb(0xFB, 0xF6, 0xE9);
const EDGE: Color32 = Color32::from_rgb(0xE2, 0xD2, 0xB0);
const INK: Color32 = Color32::from_rgb(0x2E, 0x25, 0x1C);
const MUTED: Color32 = Color32::from_rgb(0x8A, 0x7A, 0x63);
const ORANGE: Color32 = Color32::from_rgb(0xE8, 0x6A, 0x10);
const ORANGE_SOFT: Color32 = Color32::from_rgb(0xFC, 0xE3, 0xC8);
const MARKER: Color32 = Color32::from_rgba_premultiplied(0x55, 0x37, 0x14, 0x55); // orange highlighter, ~33%
const OVERLAY_TITLE: &str = "PaperTranslateOverlay";
const SUGGEST_TITLE: &str = "PaperTranslateSuggest";

#[derive(Serialize, Deserialize, Clone, PartialEq)]
#[serde(default)]
struct Config {
    hotkey: String,
    /// "paddle" (PP-OCRv6 on the GPU, default) or "windows" (built-in OCR)
    ocr_engine: String,
    ocr_lang: String,
    target_lang: String,
    model_path: String,
    /// vision projector; empty when the model has none or it does not fit the VRAM budget
    mmproj_path: String,
    llama_server: String,
    highlight: bool,
    /// suggest on every screen without being asked (off: only from the ★ Suggest button, auto can misfire outside games)
    choice_auto: bool,
    interval_ms: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hotkey: "Ctrl+Shift+T".into(),
            ocr_engine: "paddle".into(),
            ocr_lang: "en-US".into(),
            target_lang: "Thai".into(),
            model_path: String::new(),
            mmproj_path: String::new(),
            llama_server: String::new(),
            highlight: true,
            choice_auto: false,
            interval_ms: 1200,
        }
    }
}

impl Config {
    fn path() -> PathBuf {
        setup::app_dir().join("config.json")
    }
    fn load() -> Self {
        std::fs::read_to_string(Self::path()).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
    }
    fn save(&self) {
        let _ = std::fs::write(Self::path(), serde_json::to_string_pretty(self).unwrap());
    }
    fn has_model(&self) -> bool {
        Path::new(&self.model_path).is_file()
    }
}

#[derive(Clone)]
struct Block {
    src: String,
    /// translation, important words wrapped in **
    text: String,
    rect: [f32; 4],
    line_h: f32,
    born: Instant,
}

/// The recommended pick inside a detected choice window (rects in physical pixels).
#[derive(Clone)]
struct ChoiceView {
    window: [f32; 4],
    best: [f32; 4],
    why: String,
    born: Instant,
}

#[derive(Default, Clone)]
struct Job {
    busy: bool,
    label: String,
    done: u64,
    total: u64,
    error: String,
}

#[derive(Default)]
struct Shared {
    cfg: Config,
    active: bool,
    show_config: bool,
    blocks: Vec<Block>,
    choice: Option<ChoiceView>,
    status: String,
    server: Option<Child>,
    ready: bool,
    /// the running server can see screenshots (mmproj loaded)
    vision: bool,
    /// which text reader the worker is using, for Settings
    reader: String,
    /// one suggestion was asked for with the ★ Suggest button or the tray
    ask_choice: bool,
    /// short message on the status pill, e.g. "No choices found"
    notice: Option<(String, Instant)>,
    /// the choice window (physical px) while options are on screen and nothing is suggested yet: shows ★ Suggest
    candidates: Option<[f32; 4]>,
    job: Job,
}

impl Shared {
    /// Start or stop translating; either way start from a clean overlay so old cards and outlines never stack.
    fn toggle(&mut self) {
        self.active = !self.active;
        self.blocks.clear();
        self.choice = None;
        self.candidates = None;
        self.status.clear();
    }

    /// Suggest a choice for the current screen; starts translating if it was off.
    fn ask(&mut self) {
        if !self.active {
            self.toggle();
        }
        self.choice = None;
        self.ask_choice = true;
    }
}

type S = Arc<Mutex<Shared>>;

fn boot_server(s: S, ctx: Context) {
    thread::spawn(move || {
        let cfg = {
            let mut g = s.lock().unwrap();
            if let Some(mut old) = g.server.take() {
                let _ = old.kill();
                let _ = old.wait(); // free the port before the new server binds it
            }
            g.ready = false;
            g.status = "Loading model…".into();
            g.cfg.clone()
        };
        ctx.request_repaint();
        let mmproj = Some(Path::new(&cfg.mmproj_path)).filter(|p| p.is_file());
        let r = setup::find_llama_server(&cfg.llama_server)
            .ok_or_else(|| "llama-server not found, open Settings → Change model".to_string())
            .and_then(|exe| setup::start_server(&exe, Path::new(&cfg.model_path), mmproj));
        let mut g = s.lock().unwrap();
        match r {
            Ok(child) => {
                g.server = Some(child);
                g.ready = true;
                g.vision = mmproj.is_some();
                g.status.clear();
            }
            Err(e) => g.status = e,
        }
        ctx.request_repaint();
    });
}

fn get_paddle(s: S, ctx: Context, dir: PathBuf) {
    s.lock().unwrap().job = Job { busy: true, label: "Downloading PaddleOCR…".into(), ..Default::default() };
    thread::spawn(move || {
        let r = setup::get_paddle(&dir, |done, total| {
            s.lock().unwrap().job = Job { busy: true, label: "Downloading PaddleOCR…".into(), done, total, error: String::new() };
            ctx.request_repaint();
        });
        s.lock().unwrap().job = Job { error: r.err().unwrap_or_default(), ..Default::default() };
        ctx.request_repaint();
    });
}

fn run_setup(s: S, ctx: Context, url: String, dest: PathBuf, mmproj: Option<(String, PathBuf)>, paddle: bool) {
    s.lock().unwrap().job = Job { busy: true, label: "Starting…".into(), ..Default::default() };
    thread::spawn(move || {
        let progress = |label: &str, done, total| {
            let mut g = s.lock().unwrap();
            g.job = Job { busy: true, label: label.into(), done, total, error: String::new() };
            ctx.request_repaint();
        };
        let result = (|| -> Result<(), String> {
            let configured = s.lock().unwrap().cfg.llama_server.clone();
            let exe = match setup::find_llama_server(&configured) {
                Some(p) => p,
                None => setup::install_llama(|d, t| progress("Installing llama.cpp (Vulkan)…", d, t))?,
            };
            if !dest.is_file() {
                std::fs::create_dir_all(dest.parent().unwrap()).map_err(|e| e.to_string())?;
                setup::download(&url, &dest, |d, t| progress("Downloading model…", d, t))?;
            }
            if let Some((url, path)) = mmproj.as_ref().filter(|m| !m.1.is_file()) {
                setup::download(url, path, |d, t| progress("Downloading vision module…", d, t))?;
            }
            if paddle {
                setup::get_paddle(dest.parent().unwrap(), |d, t| progress("Downloading PaddleOCR…", d, t))?;
            }
            let mut g = s.lock().unwrap();
            g.cfg.llama_server = exe.display().to_string();
            g.cfg.model_path = dest.display().to_string();
            g.cfg.mmproj_path = mmproj.map(|m| m.1.display().to_string()).unwrap_or_default();
            g.cfg.ocr_engine = if paddle { "paddle" } else { "windows" }.into();
            g.cfg.save();
            Ok(())
        })();
        let ok = result.is_ok();
        s.lock().unwrap().job = Job { error: result.err().unwrap_or_default(), ..Default::default() };
        if ok {
            boot_server(s.clone(), ctx.clone());
        }
        ctx.request_repaint();
    });
}

enum Reader {
    Paddle(paddle::Paddle),
    Windows(windows::Media::Ocr::OcrEngine),
}

/// Capture → OCR → translate only the lines not seen before → publish blocks for the overlay.
fn worker(s: S, ctx: Context) {
    let mut cache: HashMap<String, String> = HashMap::new();
    // option texts of the last choice window → the decision model's verdict (None = not a real choice)
    let mut picked: Option<(Vec<String>, Option<(usize, String)>)> = None;
    let mut reader: Option<(String, Reader)> = None;
    let mut cache_key = String::new();
    let mut last_frame: (Vec<u8>, Vec<ocr::Line>) = (vec![], vec![]);
    loop {
        let (active, ready, vision, ask, cfg) = {
            let g = s.lock().unwrap();
            (g.active, g.ready, g.vision, g.ask_choice, g.cfg.clone())
        };
        if !active {
            thread::sleep(Duration::from_millis(100));
            continue;
        }
        let tick = Instant::now();
        let mut scan = || -> Result<(), String> {
            let key = format!("{}|{}", cfg.target_lang, cfg.highlight);
            if key != cache_key || cache.len() > 2000 {
                (cache, picked, cache_key) = (HashMap::new(), None, key);
            }
            let dir = setup::paddle_dir(&cfg.model_path);
            let use_paddle = cfg.ocr_engine == "paddle" && setup::paddle_ready(&dir);
            let want = if use_paddle { format!("paddle|{}", dir.display()) } else { format!("windows|{}", cfg.ocr_lang) };
            if reader.as_ref().is_none_or(|r| r.0 != want) {
                let r = match use_paddle.then(|| paddle::Paddle::load(&dir)) {
                    Some(Ok(p)) => Reader::Paddle(p),
                    _ => Reader::Windows(ocr::engine(&cfg.ocr_lang)?),
                };
                s.lock().unwrap().reader = match &r {
                    Reader::Paddle(p) => format!("PaddleOCR v6 · {}", if p.gpu { "GPU" } else { "CPU" }),
                    Reader::Windows(_) => format!("Windows OCR · {}", cfg.ocr_lang),
                };
                (reader, last_frame) = (Some((want, r)), (vec![], vec![]));
            }
            // demo mode: hide the overlay from our own capture only, so screenshots and recordings still show it
            let demo: Vec<HWND> = std::env::var_os("PAPER_SHOW_IN_CAPTURE")
                .map(|_| [OVERLAY_TITLE, SUGGEST_TITLE].iter().filter_map(|t| unsafe { FindWindowW(None, &windows::core::HSTRING::from(*t)) }.ok()).collect())
                .unwrap_or_default();
            for &h in &demo {
                let _ = unsafe { SetWindowDisplayAffinity(h, WDA_EXCLUDEFROMCAPTURE) };
            }
            if !demo.is_empty() {
                thread::sleep(Duration::from_millis(40)); // let DWM apply it before the capture
            }
            let captured = ocr::capture_foreground();
            for &h in &demo {
                let _ = unsafe { SetWindowDisplayAffinity(h, WDA_NONE) };
            }
            let (px, [x, y, w, h]) = captured?;
            // a still screen is not read again (saves the GPU for the game)
            let print = ocr::fingerprint(&px, w, h);
            let lines = if ocr::same_frame(&print, &last_frame.0) {
                last_frame.1.clone()
            } else {
                let mut lines = match &mut reader.as_mut().unwrap().1 {
                    Reader::Paddle(p) => {
                        let lines = p.recognize(&px, w, h)?;
                        let auto = if p.last_scripts.is_empty() { String::new() } else { format!(" · auto: {}", p.last_scripts.join(", ")) };
                        s.lock().unwrap().reader = format!("PaddleOCR v6 · {}{auto}", if p.gpu { "GPU" } else { "CPU" });
                        lines
                    }
                    Reader::Windows(e) => ocr::recognize(e, &px, w, h)?,
                };
                for l in &mut lines {
                    l.rect[0] += x as f32;
                    l.rect[1] += y as f32;
                }
                last_frame = (print, lines.clone());
                lines
            };
            let blocks = ocr::blocks(lines);
            let (choice, strong) = match ocr::find_choice(&blocks) {
                Some((c, strong)) => (Some(c), strong),
                None => (None, false),
            };
            if ask && choice.is_none() {
                let mut g = s.lock().unwrap();
                g.ask_choice = false;
                g.notice = Some(("No choices found on this screen".into(), Instant::now()));
            }
            let window = choice.as_ref().map(|c| union(c.iter().flatten().map(|&i| blocks[i].rect)));
            let in_choice = |r: &[usize]| choice.iter().flatten().flatten().any(|i| r.contains(i));
            let max_h = |r: &[usize]| r.iter().map(|&i| blocks[i].line_h).fold(0.0, f32::max);
            let mut regions = ocr::regions(&blocks, choice.as_deref());
            // the choice window first, then big text (dialogue) before small UI chrome
            regions.sort_by(|a, b| in_choice(b).cmp(&in_choice(a)).then(max_h(b).total_cmp(&max_h(a))));
            for region in &regions {
                let mut new: Vec<String> = vec![];
                for &i in region {
                    if !cache.contains_key(&blocks[i].text) && !new.contains(&blocks[i].text) {
                        new.push(blocks[i].text.clone());
                    }
                }
                if !new.is_empty() && !ready {
                    return Err("Model is still loading…".into());
                }
                // one request per region: the model only sees text that belongs together
                for chunk in new.chunks(8) {
                    s.lock().unwrap().status = "Translating…".into();
                    ctx.request_repaint();
                    let tr = llm::translate(chunk, &cfg.target_lang, cfg.highlight)?;
                    for (src, t) in chunk.iter().zip(tr) {
                        cache.insert(src.clone(), t);
                    }
                    if !publish(&s, &blocks, &cache, choice_view(&blocks, choice.as_deref(), &picked), window) {
                        return Ok(()); // turned off mid-scan
                    }
                    ctx.request_repaint();
                }
                // decide right after the choice window is translated, once per new set of options
                // (sequential on purpose: two concurrent requests hang llama-server's Vulkan build with Gemma 4)
                if let Some(c) = choice.as_ref().filter(|_| in_choice(region) && (ask || cfg.choice_auto)) {
                    let options = option_texts(&blocks, c);
                    if ask || picked.as_ref().is_none_or(|p| p.0 != options) {
                        s.lock().unwrap().status = "Weighing the choices…".into();
                        ctx.request_repaint();
                        let window = union(c.iter().flatten().map(|&i| blocks[i].rect));
                        let mut near: Vec<usize> = (0..blocks.len()).filter(|i| !c.iter().flatten().any(|j| j == i)).collect();
                        near.sort_by(|&a, &b| dist(blocks[a].rect, window).total_cmp(&dist(blocks[b].rect, window)));
                        let scene: Vec<String> = near.iter().take(4).map(|&i| blocks[i].text.clone()).collect();
                        // the vision model decides from the screenshot; without one, the layout + text gate decides
                        // (a plain menu never gets that far: unmarked short labels with no question are filtered first)
                        let plausible = strong || ocr::worth_asking(&options, &scene);
                        // asked: the player says it is a choice, so just pick. Automatic: hybrid gate first — vision says
                        // whether it is a choice (14/15), text reasons about the best option (5/7)
                        let seen = (!ask && plausible && vision).then(|| llm::is_choice_vision(&ocr::bmp_thumbnail(&px, w, h, 896), &options));
                        let verdict = match seen {
                            _ if ask => llm::pick_choice(&options, &scene, &cfg.target_lang, false)?,
                            Some(Ok(false)) => None,
                            Some(Ok(true)) => llm::pick_choice(&options, &scene, &cfg.target_lang, false)?,
                            _ if plausible => llm::pick_choice(&options, &scene, &cfg.target_lang, !strong)?,
                            _ => None,
                        };
                        picked = Some((options, verdict));
                        s.lock().unwrap().ask_choice = false;
                    }
                }
            }
            publish(&s, &blocks, &cache, choice_view(&blocks, choice.as_deref(), &picked), window);
            Ok(())
        };
        if let Err(e) = scan() {
            s.lock().unwrap().status = e;
        }
        ctx.request_repaint();
        thread::sleep(Duration::from_millis(cfg.interval_ms).saturating_sub(tick.elapsed()));
    }
}

/// Shows every line that has a translation; returns false when translation was turned off.
fn publish(s: &S, lines: &[ocr::Line], cache: &HashMap<String, String>, choice: Option<ChoiceView>, window: Option<[f32; 4]>) -> bool {
    let mut g = s.lock().unwrap();
    if !g.active {
        return false;
    }
    let blocks: Vec<Block> = lines
        .iter()
        .filter_map(|l| {
            Some(Block {
                text: cache.get(&l.text)?.clone(),
                born: g.blocks.iter().find(|b| b.src == l.text).map_or_else(Instant::now, |b| b.born),
                rect: l.rect,
                line_h: l.line_h,
                src: l.text.clone(),
            })
        })
        .collect();
    g.status = if lines.is_empty() { "No text found on screen".into() } else { String::new() };
    g.blocks = blocks;
    let born = g.choice.as_ref().filter(|old| choice.as_ref().is_some_and(|c| c.why == old.why)).map(|old| old.born);
    g.choice = choice.map(|c| ChoiceView { born: born.unwrap_or(c.born), ..c });
    g.candidates = window.filter(|_| g.choice.is_none());
    true
}

fn union(rects: impl Iterator<Item = [f32; 4]>) -> [f32; 4] {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for r in rects {
        (x0, y0, x1, y1) = (x0.min(r[0]), y0.min(r[1]), x1.max(r[0] + r[2]), y1.max(r[1] + r[3]));
    }
    [x0, y0, x1 - x0, y1 - y0]
}

fn dist(a: [f32; 4], b: [f32; 4]) -> f32 {
    let dx = a[0].max(b[0]) - (a[0] + a[2]).min(b[0] + b[2]);
    let dy = a[1].max(b[1]) - (a[1] + a[3]).min(b[1] + b[3]);
    dx.max(0.0) + dy.max(0.0)
}

/// One string per option; a card's title, text and rarity are joined.
fn option_texts(b: &[ocr::Line], choice: &[Vec<usize>]) -> Vec<String> {
    choice.iter().map(|o| o.iter().map(|&i| b[i].text.as_str()).collect::<Vec<_>>().join(" — ")).collect()
}

/// The verdict applies only while the same options are still on screen.
fn choice_view(b: &[ocr::Line], choice: Option<&[Vec<usize>]>, picked: &Option<(Vec<String>, Option<(usize, String)>)>) -> Option<ChoiceView> {
    let c = choice?;
    let (options, verdict) = picked.as_ref()?;
    let (k, why) = verdict.as_ref()?;
    (*options == option_texts(b, c)).then(|| ChoiceView {
        window: union(c.iter().flatten().map(|&i| b[i].rect)),
        best: union(c[*k].iter().map(|&i| b[i].rect)),
        why: why.replace("**", ""),
        born: Instant::now(),
    })
}

struct App {
    s: S,
    _tray: TrayIcon,
    _menu: Vec<MenuItem>,
    hotkeys: GlobalHotKeyManager,
    hotkey: Option<HotKey>,
    overlay: HWND,
    /// the ★ Suggest window got its no-focus / no-capture styles
    suggest_styled: bool,
    shown: bool,
    opened_at: Option<f64>,
    draft: Config,
    recording: bool,
    msg: String,
    langs: Vec<String>,
    wizard: bool,
    specs: setup::Specs,
    pick: usize,
    custom_url: String,
    model_dir: String,
}

impl App {
    fn new(cc: &eframe::CreationContext) -> Self {
        let ctx = cc.egui_ctx.clone();
        style(&ctx);
        let overlay = unsafe {
            let h = FindWindowW(None, &windows::core::HSTRING::from(OVERLAY_TITLE)).unwrap_or_default();
            // keep our own boxes out of the OCR screenshots, and never steal focus from the game
            // (PAPER_SHOW_IN_CAPTURE=1 lets screenshots see the overlay, for debugging only)
            if std::env::var_os("PAPER_SHOW_IN_CAPTURE").is_none() {
                let _ = SetWindowDisplayAffinity(h, WDA_EXCLUDEFROMCAPTURE);
            }
            let ex = GetWindowLongPtrW(h, GWL_EXSTYLE);
            SetWindowLongPtrW(h, GWL_EXSTYLE, ex | (WS_EX_NOACTIVATE.0 | WS_EX_TOOLWINDOW.0) as isize);
            h
        };
        let cfg = Config::load();
        let s: S = Arc::new(Mutex::new(Shared { cfg: cfg.clone(), show_config: !cfg.has_model(), ..Default::default() }));

        let settings = MenuItem::new("Settings…", true, None);
        let toggle = MenuItem::new("Start / stop translating", true, None);
        let suggest = MenuItem::new("Suggest a choice", true, None);
        let quit = MenuItem::new("Quit", true, None);
        let menu = Menu::new();
        let _ = menu.append_items(&[&settings, &toggle, &suggest, &PredefinedMenuItem::separator(), &quit]);
        let tray = TrayIconBuilder::new()
            .with_menu(Box::new(menu))
            .with_tooltip("Paper Translate")
            .with_icon(tray_icon_image())
            .build()
            .expect("tray icon");
        let (sid, tid, gid, qid) = (settings.id().clone(), toggle.id().clone(), suggest.id().clone(), quit.id().clone());
        let (s2, c2) = (s.clone(), ctx.clone());
        MenuEvent::set_event_handler(Some(move |e: MenuEvent| {
            let mut g = s2.lock().unwrap();
            if e.id == sid {
                g.show_config = true;
            } else if e.id == tid {
                g.toggle();
            } else if e.id == gid {
                g.ask();
            } else if e.id == qid {
                if let Some(c) = g.server.as_mut() {
                    let _ = c.kill();
                }
                std::process::exit(0);
            }
            c2.request_repaint();
        }));
        let (s2, c2) = (s.clone(), ctx.clone());
        TrayIconEvent::set_event_handler(Some(move |e: TrayIconEvent| {
            if let TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, .. } = e {
                s2.lock().unwrap().show_config = true;
                c2.request_repaint();
            }
        }));
        let (s2, c2) = (s.clone(), ctx.clone());
        GlobalHotKeyEvent::set_event_handler(Some(move |e: GlobalHotKeyEvent| {
            if e.state == HotKeyState::Pressed {
                s2.lock().unwrap().toggle();
                c2.request_repaint();
            }
        }));

        let specs = setup::Specs::detect();
        let default_dir = std::env::var("LOCALAPPDATA").map(|d| format!("{d}\\PaperTranslate\\models")).unwrap_or_default();
        let mut app = Self {
            s: s.clone(),
            _tray: tray,
            _menu: vec![settings, toggle, suggest, quit],
            hotkeys: GlobalHotKeyManager::new().expect("hotkey manager"),
            hotkey: None,
            overlay,
            suggest_styled: false,
            shown: false,
            opened_at: None,
            draft: cfg.clone(),
            recording: false,
            msg: String::new(),
            langs: ocr::languages(),
            wizard: false,
            pick: specs.recommend().unwrap_or(0),
            specs,
            custom_url: String::new(),
            model_dir: default_dir,
        };
        if let Err(e) = rebind(&app.hotkeys, &mut app.hotkey, &cfg.hotkey) {
            app.msg = e;
        }
        if cfg.has_model() {
            boot_server(s.clone(), ctx.clone());
        }
        thread::spawn(move || worker(s, ctx));
        app
    }

    fn draw_overlay(&self, ui: &mut Ui, blocks: &[Block], choice: Option<&ChoiceView>, status: &str, highlight: bool) {
        let ppp = ui.ctx().pixels_per_point();
        let mut animating = false;
        let mut cards = vec![];
        for b in blocks {
            let t = ease(b.born.elapsed().as_secs_f32() / 0.3);
            animating |= t < 1.0;
            let src = Rect::from_min_size(pos2(b.rect[0], b.rect[1]) / ppp, vec2(b.rect[2], b.rect[3]) / ppp);
            let size = (b.line_h / ppp * 0.8).clamp(12.0, 30.0);
            let pad = vec2(8.0, 5.0);
            let galley = ui.painter().layout_job(rich(&b.text, size, src.width().max(160.0), highlight));
            // always cover the original text, grow if the translation is longer
            let rect = Rect::from_min_size(src.min - pad, galley.size() + pad * 2.0)
                .union(src.expand2(pad))
                .translate(vec2(0.0, (1.0 - t) * 8.0));
            let mut p = ui.painter().clone();
            p.set_opacity(t);
            p.add(Shadow { offset: [0, 3], blur: 12, spread: 0, color: Color32::from_black_alpha(70) }.as_shape(rect, 6));
            p.rect(rect, 6, PAPER, Stroke::new(1.0, EDGE), StrokeKind::Inside);
            p.galley(rect.min + pad, galley, INK);
            cards.push(rect);
        }
        if let Some(c) = choice {
            let t = ease(c.born.elapsed().as_secs_f32() / 0.35);
            animating |= t < 1.0;
            let mut p = ui.painter().clone();
            p.set_opacity(t);
            let pts = |r: [f32; 4]| Rect::from_min_size(pos2(r[0], r[1]) / ppp, vec2(r[2], r[3]) / ppp);
            // the whole choice window: soft halo + crisp outline that settles in
            let win = pts(c.window).expand(16.0 + (1.0 - t) * 10.0);
            p.rect_stroke(win, 16, Stroke::new(9.0, ORANGE.gamma_multiply(0.2)), StrokeKind::Outside);
            p.rect_stroke(win, 16, Stroke::new(2.0, ORANGE), StrokeKind::Outside);
            let best = pts(c.best).expand(9.0);
            p.rect_stroke(best, 10, Stroke::new(3.0, ORANGE), StrokeKind::Outside);
            let star = best.right_top();
            p.circle_filled(star, 12.0, ORANGE);
            p.text(star, egui::Align2::CENTER_CENTER, "★", FontId::proportional(14.0), Color32::WHITE);
            // short reason on a paper note above the window (below it when there is no room)
            let width = 340.0;
            let head = p.layout_no_wrap("Suggested choice".into(), FontId::proportional(12.0), ORANGE);
            let body = p.layout(c.why.clone(), FontId::proportional(15.0), INK, width - 30.0);
            let (hh, size) = (head.size().y, vec2(width, head.size().y + body.size().y + 20.0));
            // above the window, else beside or below it: the first spot that covers no translation
            let spots = [
                pos2(win.center().x - width / 2.0, win.top() - 12.0 - size.y),
                pos2(win.right() + 12.0, win.top()),
                pos2(win.left() - 12.0 - width, win.top()),
                pos2(win.center().x - width / 2.0, win.bottom() + 12.0),
            ];
            let note = (spots.iter().map(|&at| Rect::from_min_size(at, size)))
                .find(|r| ui.max_rect().contains_rect(*r) && !cards.iter().any(|c| c.intersects(*r)))
                .unwrap_or_else(|| Rect::from_min_size(spots[0], size));
            p.add(Shadow { offset: [0, 3], blur: 14, spread: 0, color: Color32::from_black_alpha(80) }.as_shape(note, 10));
            p.rect(note, 10, PAPER, Stroke::new(1.0, EDGE), StrokeKind::Inside);
            p.rect_filled(Rect::from_min_size(note.min, vec2(5.0, size.y)), egui::CornerRadius { nw: 10, sw: 10, ne: 0, se: 0 }, ORANGE);
            p.galley(note.min + vec2(16.0, 8.0), head, ORANGE);
            p.galley(note.min + vec2(16.0, 12.0 + hh), body, INK);
        }
        let hotkey = self.s.lock().unwrap().cfg.hotkey.clone();
        let text = if status.is_empty() { format!("● Translating · {hotkey} to stop") } else { format!("● {status}") };
        let p = ui.painter();
        let g = p.layout_no_wrap(text, FontId::proportional(13.0), INK);
        let pill = Rect::from_center_size(pos2(ui.max_rect().center().x, 26.0), g.size() + vec2(24.0, 12.0));
        p.add(Shadow { offset: [0, 2], blur: 10, spread: 0, color: Color32::from_black_alpha(50) }.as_shape(pill, 99));
        p.rect(pill, 99, PAPER, Stroke::new(1.0, EDGE), StrokeKind::Inside);
        p.galley(pill.min + vec2(12.0, 6.0), g, INK);
        if animating {
            ui.ctx().request_repaint();
        }
    }

    /// A small clickable window above the choice window (the overlay itself lets clicks through to the game).
    fn suggest_button(&mut self, ctx: &Context, window: [f32; 4]) {
        let ppp = ctx.pixels_per_point();
        let size = vec2(150.0, 40.0);
        // right-aligned just above where the choice outline will be drawn (16 pt outside the options)
        let at = pos2((window[0] + window[2]) / ppp + 16.0 - size.x, window[1] / ppp - 16.0 - size.y - 8.0).max(pos2(4.0, 4.0));
        let builder = ViewportBuilder::default()
            .with_title(SUGGEST_TITLE)
            .with_decorations(false)
            .with_transparent(true)
            .with_always_on_top()
            .with_taskbar(false)
            .with_resizable(false)
            .with_position(at)
            .with_inner_size(size);
        ctx.show_viewport_immediate(ViewportId::from_hash_of("suggest"), builder, |ui, _| {
            if !self.suggest_styled {
                // clicking must not take focus from the game, and OCR must not read the button
                unsafe {
                    if let Ok(h) = FindWindowW(None, &windows::core::HSTRING::from(SUGGEST_TITLE)) {
                        let ex = GetWindowLongPtrW(h, GWL_EXSTYLE);
                        SetWindowLongPtrW(h, GWL_EXSTYLE, ex | (WS_EX_NOACTIVATE.0 | WS_EX_TOOLWINDOW.0) as isize);
                        if std::env::var_os("PAPER_SHOW_IN_CAPTURE").is_none() {
                            let _ = SetWindowDisplayAffinity(h, WDA_EXCLUDEFROMCAPTURE);
                        }
                        self.suggest_styled = true;
                    }
                }
            }
            let rect = ui.max_rect().shrink(2.0);
            let r = ui.allocate_rect(rect, Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand);
            let (fill, ink) = if r.hovered() { (ORANGE, Color32::WHITE) } else { (PAPER, Color32::from_rgb(0xB3, 0x4A, 0x00)) };
            let p = ui.painter();
            p.rect(rect, 99, fill, Stroke::new(1.5, ORANGE), StrokeKind::Inside);
            p.text(rect.center(), egui::Align2::CENTER_CENTER, "★ Suggest", FontId::proportional(17.0), ink);
            if r.clicked() {
                self.s.lock().unwrap().ask();
            }
        });
    }

    fn settings_ui(&mut self, ui: &mut Ui, opened: f64) {
        let elapsed = (ui.input(|i| i.time) - opened) as f32;
        let a = |k: f32| ease((elapsed - k * 0.07) / 0.35); // staggered slide-in per card
        if a(5.0) < 1.0 {
            ui.ctx().request_repaint();
        }
        ui.painter().rect_filled(ui.max_rect(), 0, BG);
        let (job, ready, status) = {
            let g = self.s.lock().unwrap();
            (g.job.clone(), g.ready, g.status.clone())
        };
        egui::ScrollArea::vertical().show(ui, |ui| {
            Frame::new().inner_margin(20).show(ui, |ui| {
                ui.add_space((1.0 - a(0.0)) * 12.0);
                ui.horizontal(|ui| {
                    ui.label(RichText::new("●").color(ORANGE).size(22.0));
                    ui.label(RichText::new("Paper Translate").size(24.0).strong().color(INK));
                });
                ui.label(RichText::new("Live on-screen translation for games and apps").color(MUTED));
                ui.add_space(14.0);
                if self.wizard || !self.draft.has_model() {
                    self.wizard_ui(ui, &a, &job);
                } else {
                    self.prefs_ui(ui, &a, ready, &status);
                }
            });
        });
    }

    fn wizard_ui(&mut self, ui: &mut Ui, a: &dyn Fn(f32) -> f32, job: &Job) {
        let (vram_budget, ram_budget) = self.specs.budget();
        card(ui, a(1.0), |ui| {
            heading(ui, "Your PC");
            let sp = &self.specs;
            ui.label(RichText::new(format!("{} · {:.0} GB VRAM · {:.0} GB RAM", sp.gpu, sp.vram, sp.ram)).color(INK));
            ui.label(
                RichText::new(format!(
                    "While you play, Paper Translate uses about a quarter of your VRAM and RAM, so the game keeps the rest: \
                     {vram_budget:.1} GB VRAM and {ram_budget:.1} GB RAM for the model{}. Only quantizations with small quality loss \
                     (Q4_K_M and better) are offered.",
                    if sp.paddle_fits() { format!(", plus {:.2} GB for the PaddleOCR text reader", setup::PADDLE_VRAM) } else { ", text read by Windows OCR".into() }
                ))
                .size(12.0)
                .color(MUTED),
            );
        });
        card(ui, a(2.0), |ui| {
            heading(ui, "Choose a translation model");
            ui.add_space(6.0);
            let recommended = self.specs.recommend();
            for (i, m) in setup::MODELS.iter().enumerate() {
                let fit = self.specs.fit(m);
                let sel = fit.is_some() && self.pick == i && self.custom_url.trim().is_empty();
                let r = Frame::new()
                    .fill(if sel { ORANGE_SOFT } else { PAPER })
                    .stroke(Stroke::new(if sel { 1.5 } else { 1.0 }, if sel { ORANGE } else { EDGE }))
                    .corner_radius(8)
                    .inner_margin(10)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        if fit.is_none() {
                            ui.set_opacity(0.45);
                        }
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(m.name).strong().color(INK));
                            if Some(i) == recommended {
                                ui.label(RichText::new(" Best for your PC ").size(11.0).color(Color32::WHITE).background_color(ORANGE));
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                let size = match fit {
                                    Some((q, true)) => format!("{} + vision · {:.1} GB VRAM", q.label, q.vram + m.vision.as_ref().map_or(0.0, |v| v.vram)),
                                    Some((q, false)) => format!("{} · {:.1} GB VRAM", q.label, q.vram),
                                    None => format!("needs {:.1} GB VRAM", m.quants.last().map_or(0.0, |q| q.vram)),
                                };
                                ui.label(RichText::new(size).size(12.0).color(MUTED));
                            });
                        });
                        let note = if fit.is_some() { m.note } else { "Over your gaming budget: it would fight the game for VRAM or RAM." };
                        ui.label(RichText::new(note).size(12.0).color(MUTED));
                    })
                    .response;
                if fit.is_some() && r.interact(Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                    self.pick = i;
                    self.custom_url.clear();
                }
                ui.add_space(4.0);
            }
            ui.add_space(4.0);
            ui.label(RichText::new("Or paste any Hugging Face .gguf link").color(MUTED));
            ui.add(egui::TextEdit::singleline(&mut self.custom_url).hint_text("https://huggingface.co/…/model-Q4_K_M.gguf").desired_width(f32::INFINITY));
        });
        card(ui, a(3.0), |ui| {
            heading(ui, "Save models to");
            ui.horizontal(|ui| {
                if ui.button("Browse…").clicked()
                    && let Some(p) = rfd::FileDialog::new().set_directory(&self.model_dir).pick_folder()
                {
                    self.model_dir = p.display().to_string();
                }
                ui.add(egui::TextEdit::singleline(&mut self.model_dir).desired_width(f32::INFINITY));
            });
            ui.add_space(4.0);
            match setup::find_llama_server(&self.draft.llama_server) {
                Some(p) => ui.label(RichText::new(format!("✔ llama.cpp found: {}", p.display())).size(12.0).color(MUTED)),
                None => ui.label(RichText::new("llama.cpp not found, the Vulkan build will be installed automatically (works on NVIDIA, AMD and Intel).").size(12.0).color(MUTED)),
            };
        });
        card(ui, a(4.0), |ui| {
            if job.busy {
                ui.label(RichText::new(&job.label).color(INK));
                let frac = if job.total > 0 { job.done as f32 / job.total as f32 } else { 0.0 };
                ui.add(ProgressBar::new(frac).fill(ORANGE).text(format!("{:.0} / {:.0} MB", job.done as f64 / 1e6, job.total as f64 / 1e6)).animate(job.total == 0));
                return;
            }
            if !job.error.is_empty() {
                ui.label(RichText::new(&job.error).color(Color32::from_rgb(0xB0, 0x30, 0x20)));
            }
            let target = if self.custom_url.trim().is_empty() {
                let m = &setup::MODELS[self.pick];
                self.specs.fit(m).map(|(q, vision)| {
                    let dir = Path::new(self.model_dir.trim());
                    let mmproj = m.vision.as_ref().filter(|_| vision).map(|v| (setup::hf_url(m.repo, v.file), dir.join(v.local)));
                    (setup::hf_url(m.repo, q.file), q.file.to_string(), mmproj)
                })
            } else {
                setup::parse_custom_url(&self.custom_url).map(|(url, file)| (url, file, None))
            };
            let invalid = target.is_none();
            ui.horizontal(|ui| {
                let go = ui.add_enabled(!invalid && !self.model_dir.trim().is_empty(), primary("Download & start"));
                if go.clicked()
                    && let Some((url, file, mmproj)) = target
                {
                    let paddle = self.specs.paddle_fits();
                    run_setup(self.s.clone(), ui.ctx().clone(), url, Path::new(self.model_dir.trim()).join(file), mmproj, paddle);
                    self.wizard = false;
                }
                if self.draft.has_model() && ui.button("Cancel").clicked() {
                    self.wizard = false;
                }
            });
            if invalid && self.custom_url.trim().is_empty() {
                ui.label(RichText::new("No listed model fits this PC while gaming; paste a smaller .gguf link.").size(12.0).color(MUTED));
            } else if invalid {
                ui.label(RichText::new("That link must be a https://huggingface.co/… URL ending in .gguf").size(12.0).color(MUTED));
            }
        });
    }

    fn prefs_ui(&mut self, ui: &mut Ui, a: &dyn Fn(f32) -> f32, ready: bool, status: &str) {
        let job = self.s.lock().unwrap().job.clone();
        if job.busy || !job.error.is_empty() {
            return self.wizard_ui(ui, a, &job);
        }
        card(ui, a(1.0), |ui| {
            heading(ui, "Hotkey");
            hotkey_row(ui, &mut self.draft.hotkey, &mut self.recording, "start / stop translating anywhere");
        });
        card(ui, a(2.0), |ui| {
            heading(ui, "Languages");
            let paddle = self.draft.ocr_engine == "paddle";
            egui::Grid::new("langs").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                ui.label("Text reader");
                let name = if paddle { "PaddleOCR v6 (accurate)" } else { "Windows OCR (lightest)" };
                egui::ComboBox::from_id_salt("engine").selected_text(name).width(200.0).show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.draft.ocr_engine, "paddle".into(), "PaddleOCR v6 (accurate)");
                    ui.selectable_value(&mut self.draft.ocr_engine, "windows".into(), "Windows OCR (lightest)");
                });
                ui.end_row();
                if !paddle {
                    ui.label("Read text in");
                    egui::ComboBox::from_id_salt("ocr").selected_text(&self.draft.ocr_lang).width(200.0).show_ui(ui, |ui| {
                        for l in &self.langs {
                            ui.selectable_value(&mut self.draft.ocr_lang, l.clone(), l);
                        }
                    });
                    ui.end_row();
                }
                ui.label("Translate to");
                ui.add(egui::TextEdit::singleline(&mut self.draft.target_lang).desired_width(200.0));
                ui.end_row();
            });
            ui.label(RichText::new("Translate to accepts any language name, e.g. Thai, Japanese, English.").size(12.0).color(MUTED));
            let reader = self.s.lock().unwrap().reader.clone();
            if !reader.is_empty() {
                ui.label(RichText::new(format!("Reading with {reader}")).size(12.0).color(MUTED));
            }
            let dir = setup::paddle_dir(&self.draft.model_path);
            if paddle && !setup::paddle_ready(&dir) {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("PaddleOCR files are missing; Windows OCR is used until then.").size(12.0).color(MUTED));
                    if ui.button("Download (139 MB)").clicked() {
                        get_paddle(self.s.clone(), ui.ctx().clone(), dir);
                    }
                });
            }
            // Windows only ships OCR for languages that are installed; offer the common game languages
            let missing: Vec<(&str, &str)> = [("ja-JP", "Japanese"), ("zh-CN", "Chinese"), ("ko-KR", "Korean")]
                .into_iter()
                .filter(|(tag, _)| !self.langs.iter().any(|l| l == tag))
                .collect();
            if !missing.is_empty() && !paddle {
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new("Read games in").size(12.0).color(MUTED));
                    for (tag, name) in missing {
                        if ui.button(format!("+ {name} OCR")).on_hover_text("Installs Windows' OCR pack (asks for admin)").clicked() {
                            ocr::install_language(tag);
                        }
                    }
                    if ui.small_button("Refresh").clicked() {
                        self.langs = ocr::languages();
                    }
                });
            }
        });
        card(ui, a(3.0), |ui| {
            heading(ui, "Overlay");
            ui.checkbox(&mut self.draft.highlight, "Highlight names, items and key terms in orange");
            ui.label(RichText::new("When choices are on screen, a ★ Suggest button appears above them.").size(12.0).color(MUTED));
            ui.checkbox(&mut self.draft.choice_auto, "Suggest automatically (can also mark menus and lists outside games)");
            let mut secs = self.draft.interval_ms as f32 / 1000.0;
            ui.horizontal(|ui| {
                ui.label("Scan every");
                ui.add(egui::Slider::new(&mut secs, 0.5..=5.0).suffix(" s"));
            });
            self.draft.interval_ms = (secs * 1000.0) as u64;
        });
        card(ui, a(4.0), |ui| {
            heading(ui, "Model");
            let name = Path::new(&self.draft.model_path).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            ui.label(RichText::new(name).strong().color(INK));
            let (dot, text) = if ready { (Color32::from_rgb(0x3C, 0x9A, 0x4A), "Ready".to_string()) } else { (ORANGE, status.to_string()) };
            ui.horizontal(|ui| {
                ui.label(RichText::new("●").color(dot));
                ui.label(RichText::new(text).color(MUTED));
            });
            if ui.button("Change model…").clicked() {
                self.wizard = true;
            }
        });
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui.add(primary("Save")).clicked() {
                let hotkey = self.draft.hotkey.clone();
                self.msg = match rebind(&self.hotkeys, &mut self.hotkey, &hotkey) {
                    Ok(()) => {
                        let mut g = self.s.lock().unwrap();
                        // model fields belong to the setup flow, keep its latest values
                        self.draft.model_path = g.cfg.model_path.clone();
                        self.draft.llama_server = g.cfg.llama_server.clone();
                        g.cfg = self.draft.clone();
                        g.cfg.save();
                        "Saved ✓".into()
                    }
                    Err(e) => e,
                };
            }
            ui.label(RichText::new(&self.msg).color(MUTED));
        });
    }
}

impl eframe::App for App {
    fn logic(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        let (active, show_cfg) = {
            let g = self.s.lock().unwrap();
            (g.active, g.show_config)
        };
        if active != self.shown {
            self.shown = active;
            // Win32 directly: winit's show would activate the window and steal focus from the game
            unsafe {
                if active {
                    let (w, h) = (GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN));
                    let _ = SetWindowPos(self.overlay, Some(HWND_TOPMOST), 0, 0, w, h, SWP_NOACTIVATE | SWP_SHOWWINDOW);
                } else {
                    let _ = ShowWindow(self.overlay, SW_HIDE);
                }
            }
            ctx.request_repaint();
        }
        if show_cfg {
            // the settings window lives inside the root pass, which is skipped while the overlay is hidden
            ctx.send_viewport_cmd(egui::ViewportCommand::RequestPaintWhileHidden);
        }
    }

    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        let (active, show_cfg, blocks, choice, status, highlight, candidates) = {
            let g = self.s.lock().unwrap();
            let notice = g.notice.as_ref().filter(|n| n.1.elapsed() < Duration::from_secs(3)).map(|n| n.0.clone());
            if notice.is_some() {
                ui.ctx().request_repaint_after(Duration::from_millis(500)); // to hide it again
            }
            (g.active, g.show_config, g.blocks.clone(), g.choice.clone(), notice.unwrap_or_else(|| g.status.clone()), g.cfg.highlight, g.candidates)
        };
        if active {
            self.draw_overlay(ui, &blocks, choice.as_ref(), &status, highlight);
        }
        match candidates.filter(|_| active && choice.is_none()) {
            Some(window) => self.suggest_button(ui.ctx(), window),
            None => self.suggest_styled = false,
        }
        if !show_cfg {
            self.opened_at = None;
            return;
        }
        let ctx = ui.ctx().clone();
        let opened = match self.opened_at {
            Some(t) => t,
            None => {
                self.draft = self.s.lock().unwrap().cfg.clone();
                self.langs = ocr::languages();
                self.recording = false;
                self.msg.clear();
                *self.opened_at.insert(ctx.input(|i| i.time))
            }
        };
        let builder = ViewportBuilder::default()
            .with_title("Paper Translate")
            .with_inner_size([540.0, 720.0])
            .with_min_inner_size([460.0, 400.0])
            .with_icon(Arc::new(egui::IconData { rgba: icon_rgba(64), width: 64, height: 64 }));
        ctx.show_viewport_immediate(ViewportId::from_hash_of("settings"), builder, |ui, _| {
            if ui.input(|i| i.viewport().close_requested()) {
                self.s.lock().unwrap().show_config = false;
            }
            self.settings_ui(ui, opened);
        });
    }

    fn clear_color(&self, _: &egui::Visuals) -> [f32; 4] {
        [0.0; 4]
    }
}

/// Registers `text` in place of the hotkey in `slot`; the old one stays if the new one is invalid or taken.
fn rebind(m: &GlobalHotKeyManager, slot: &mut Option<HotKey>, text: &str) -> Result<(), String> {
    let hk: HotKey = text.parse().map_err(|e| format!("Invalid hotkey {text}: {e}"))?;
    if *slot == Some(hk) {
        return Ok(());
    }
    m.register(hk).map_err(|e| format!("Hotkey {text} is taken by another app ({e})"))?;
    if let Some(old) = slot.replace(hk) {
        let _ = m.unregister(old);
    }
    Ok(())
}

/// A hotkey button: click, then press the new combo (Esc cancels).
fn hotkey_row(ui: &mut Ui, text: &mut String, recording: &mut bool, hint: &str) {
    ui.horizontal(|ui| {
        let label = if *recording { "Press a key combo…".to_string() } else { text.clone() };
        if ui.add(egui::Button::new(RichText::new(label).monospace().color(INK)).min_size(vec2(180.0, 30.0))).clicked() {
            *recording = !*recording;
        }
        ui.label(RichText::new(hint).color(MUTED));
    });
    if !*recording {
        return;
    }
    let combo = ui.input(|i| {
        i.events.iter().find_map(|e| match e {
            egui::Event::Key { key, pressed: true, modifiers, .. } => Some((*key, *modifiers)),
            _ => None,
        })
    });
    if let Some((key, m)) = combo {
        if key != egui::Key::Escape {
            let mods = [(m.ctrl, "Ctrl"), (m.shift, "Shift"), (m.alt, "Alt")];
            let mut parts: Vec<&str> = mods.iter().filter(|x| x.0).map(|x| x.1).collect();
            parts.push(key.name());
            *text = parts.join("+");
        }
        *recording = false;
    }
}

fn card(ui: &mut Ui, t: f32, add: impl FnOnce(&mut Ui)) {
    ui.add_space((1.0 - t) * 18.0);
    ui.scope(|ui| {
        ui.set_opacity(t);
        Frame::new()
            .fill(PAPER)
            .stroke(Stroke::new(1.0, EDGE))
            .corner_radius(12)
            .inner_margin(14)
            .shadow(Shadow { offset: [0, 2], blur: 10, spread: 0, color: Color32::from_black_alpha(20) })
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                add(ui)
            });
    });
    ui.add_space(10.0);
}

fn heading(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text).size(16.0).strong().color(INK));
}

fn primary(text: &str) -> egui::Button<'_> {
    egui::Button::new(RichText::new(text).color(Color32::WHITE).strong()).fill(ORANGE).min_size(vec2(150.0, 32.0)).corner_radius(8)
}

fn ease(t: f32) -> f32 {
    1.0 - (1.0 - t.clamp(0.0, 1.0)).powi(3)
}

/// Text with **marked** words drawn as orange highlighter strokes.
fn rich(text: &str, size: f32, width: f32, highlight: bool) -> LayoutJob {
    let mut job = LayoutJob::default();
    job.wrap.max_width = width;
    for (i, part) in text.split("**").enumerate() {
        let hot = highlight && i % 2 == 1;
        let format = TextFormat {
            font_id: FontId::proportional(size),
            color: if hot { Color32::from_rgb(0xB3, 0x4A, 0x00) } else { INK },
            background: if hot { MARKER } else { Color32::TRANSPARENT },
            ..Default::default()
        };
        job.append(part, 0.0, format);
    }
    job
}

fn style(ctx: &Context) {
    let mut fonts = FontDefinitions::default();
    let dir = PathBuf::from(std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".into())).join("Fonts");
    // Segoe UI first for the look, then system fallbacks so Thai/CJK/Korean translations render
    for (i, name) in ["segoeui.ttf", "LeelawUI.ttf", "malgun.ttf", "YuGothM.ttc", "msyh.ttc"].into_iter().enumerate() {
        if let Ok(bytes) = std::fs::read(dir.join(name)) {
            fonts.font_data.insert(name.into(), Arc::new(FontData::from_owned(bytes)));
            let family = fonts.families.entry(FontFamily::Proportional).or_default();
            if i == 0 { family.insert(0, name.into()) } else { family.push(name.into()) }
        }
    }
    ctx.set_fonts(fonts);
    ctx.set_theme(egui::Theme::Light);
    let mut v = egui::Visuals::light();
    v.panel_fill = BG;
    v.window_fill = BG;
    v.extreme_bg_color = Color32::from_rgb(0xFF, 0xFC, 0xF5);
    v.selection.bg_fill = ORANGE_SOFT;
    v.selection.stroke = Stroke::new(1.0, ORANGE);
    v.hyperlink_color = ORANGE;
    for (w, fill) in [
        (&mut v.widgets.inactive, Color32::from_rgb(0xF0, 0xE4, 0xCB)),
        (&mut v.widgets.hovered, Color32::from_rgb(0xF6, 0xD9, 0xB5)),
        (&mut v.widgets.active, ORANGE_SOFT),
        (&mut v.widgets.open, ORANGE_SOFT),
    ] {
        w.weak_bg_fill = fill;
        w.bg_fill = fill;
        w.corner_radius = 8.into();
    }
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, ORANGE);
    ctx.set_visuals_of(egui::Theme::Light, v);
}

/// A paper sheet with an orange corner fold, drawn in code so there is no asset to ship.
fn icon_rgba(n: u32) -> Vec<u8> {
    let mut px = Vec::with_capacity((n * n * 4) as usize);
    let s = n as f32;
    for y in 0..n {
        for x in 0..n {
            let (fx, fy) = (x as f32 / s, y as f32 / s);
            let inside = (0.12..0.88).contains(&fx) && (0.06..0.94).contains(&fy);
            let fold = fx + (0.94 - fy) > 1.45; // bottom-right corner
            let line = inside && !fold && (0.22..0.70).contains(&fx) && [0.3, 0.45, 0.6].iter().any(|l| (fy - l).abs() < 0.035);
            let c: [u8; 4] = if !inside {
                [0, 0, 0, 0]
            } else if fold {
                [0xE8, 0x6A, 0x10, 255]
            } else if line {
                [0x8A, 0x7A, 0x63, 255]
            } else {
                [0xFB, 0xF6, 0xE9, 255]
            };
            px.extend_from_slice(&c);
        }
    }
    px
}

fn tray_icon_image() -> tray_icon::Icon {
    tray_icon::Icon::from_rgba(icon_rgba(32), 32, 32).expect("icon")
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: ViewportBuilder::default()
            .with_title(OVERLAY_TITLE)
            .with_decorations(false)
            .with_transparent(true)
            .with_always_on_top()
            .with_mouse_passthrough(true)
            .with_taskbar(false)
            .with_visible(false)
            .with_inner_size([1.0, 1.0]),
        ..Default::default()
    };
    eframe::run_native(OVERLAY_TITLE, options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}

