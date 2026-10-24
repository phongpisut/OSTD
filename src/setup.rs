use std::fs::{self, File};
use std::io::{Read, Write};
use std::os::windows::io::AsRawHandle;
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// llama-server port; PAPER_PORT overrides it when 8719 is taken (or to run a second instance).
pub fn port() -> u16 {
    std::env::var("PAPER_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8719)
}

pub struct Quant {
    pub file: &'static str,
    pub label: &'static str,
    /// GB of VRAM with -c 4096 (GPU weights from the GGUF header + measured KV/compute overhead)
    pub vram: f32,
    /// GB of system RAM for weights llama.cpp keeps on the CPU (input and per-layer embeddings)
    pub ram: f32,
}

/// A model's vision projector (mmproj): lets the same server look at screenshots to judge choices.
pub struct Vision {
    pub file: &'static str,
    /// saved under this name, since every repo calls its projector "mmproj-F16.gguf"
    pub local: &'static str,
    /// extra GB of VRAM, measured for Gemma 4 E2B
    pub vram: f32,
}

pub struct Model {
    pub name: &'static str,
    pub repo: &'static str,
    pub note: &'static str,
    pub vision: Option<Vision>,
    /// best quality first; nothing below Q4_K_M, where quality loss stops being small
    pub quants: &'static [Quant],
}

const fn q(file: &'static str, label: &'static str, vram: f32, ram: f32) -> Quant {
    Quant { file, label, vram, ram }
}

// Ordered by translation quality. VRAM = GPU weight bytes read from each file's GGUF header + overhead measured on an
// RTX 4060 Ti (Gemma 4 / Qwen 0.2 GB, Gemma 3 based Typhoon 0.7 GB). Q4_K_M of Gemma 4 E2B/E4B, Qwen3.5 2B and Typhoon measured directly.
pub const MODELS: &[Model] = &[
    Model {
        name: "Gemma 4 E4B",
        repo: "unsloth/gemma-4-E4B-it-GGUF",
        note: "Best wording here.",
        vision: Some(Vision { file: "mmproj-F16.gguf", local: "mmproj-gemma-4-E4B-F16.gguf", vram: 1.2 }),
        quants: &[
            q("gemma-4-E4B-it-Q8_0.gguf", "Q8_0 · near-lossless", 5.4, 3.7),
            q("gemma-4-E4B-it-Q6_K.gguf", "Q6_K · tiny loss", 4.3, 3.6),
            q("gemma-4-E4B-it-Q5_K_M.gguf", "Q5_K_M · small loss", 3.8, 2.4),
            q("gemma-4-E4B-it-Q4_K_M.gguf", "Q4_K_M · small loss", 3.3, 2.3),
        ],
    },
    Model {
        name: "Gemma 4 E2B",
        repo: "unsloth/gemma-4-E2B-it-GGUF",
        note: "Human-like wording, reliable key-term marking, fast. Its vision judged 5/5 test screens right.",
        vision: Some(Vision { file: "mmproj-F16.gguf", local: "mmproj-gemma-4-E2B-F16.gguf", vram: 1.2 }),
        quants: &[
            q("gemma-4-E2B-it-Q8_0.gguf", "Q8_0 · near-lossless", 2.8, 2.9),
            q("gemma-4-E2B-it-Q6_K.gguf", "Q6_K · tiny loss", 2.2, 2.8),
            q("gemma-4-E2B-it-Q5_K_M.gguf", "Q5_K_M · small loss", 1.95, 1.9),
            q("gemma-4-E2B-it-Q4_K_M.gguf", "Q4_K_M · small loss", 1.65, 1.8),
        ],
    },
    Model {
        name: "Typhoon Translate 1.5 4B",
        repo: "mradermacher/typhoon-translate1.5-4b-GGUF",
        note: "Thai specialist (EN↔TH).",
        vision: None,
        quants: &[
            q("typhoon-translate1.5-4b.Q8_0.gguf", "Q8_0 · near-lossless", 5.0, 0.4),
            q("typhoon-translate1.5-4b.Q6_K.gguf", "Q6_K · tiny loss", 4.0, 0.3),
            q("typhoon-translate1.5-4b.Q5_K_M.gguf", "Q5_K_M · small loss", 3.6, 0.3),
            q("typhoon-translate1.5-4b.Q4_K_M.gguf", "Q4_K_M · small loss", 3.2, 0.3),
        ],
    },
    Model {
        name: "Qwen3.5 2B",
        repo: "unsloth/Qwen3.5-2B-GGUF",
        note: "Light on RAM, 200+ languages. Plainer wording.",
        vision: None,
        quants: &[
            q("Qwen3.5-2B-Q8_0.gguf", "Q8_0 · near-lossless", 2.2, 0.5),
            q("Qwen3.5-2B-Q6_K.gguf", "Q6_K · tiny loss", 1.75, 0.4),
            q("Qwen3.5-2B-Q5_K_M.gguf", "Q5_K_M · small loss", 1.6, 0.4),
            q("Qwen3.5-2B-Q4_K_M.gguf", "Q4_K_M · small loss", 1.45, 0.4),
        ],
    },
    Model {
        name: "LFM2.5 1.2B",
        repo: "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
        note: "Very fast. Good for EN/JA/KO/ZH/EU, weak at Thai.",
        vision: None,
        // small models lose more per bit, so they stop at Q6_K
        quants: &[q("LFM2.5-1.2B-Instruct-Q8_0.gguf", "Q8_0 · near-lossless", 1.45, 0.15), q("LFM2.5-1.2B-Instruct-Q6_K.gguf", "Q6_K · tiny loss", 1.15, 0.1)],
    },
    Model {
        name: "Qwen3.5 0.8B",
        repo: "unsloth/Qwen3.5-0.8B-GGUF",
        note: "For integrated or very small GPUs. Rougher wording.",
        vision: None,
        quants: &[q("Qwen3.5-0.8B-Q8_0.gguf", "Q8_0 · near-lossless", 1.0, 0.3), q("Qwen3.5-0.8B-Q6_K.gguf", "Q6_K · tiny loss", 0.85, 0.2)],
    },
];

/// PaddleOCR v6 medium on DirectML at 960 px, measured peak.
pub const PADDLE_VRAM: f32 = 0.75;

pub struct Specs {
    pub gpu: String,
    pub vram: f32,
    pub ram: f32,
}

impl Specs {
    /// Biggest non-software GPU by dedicated memory (DXGI) and total RAM.
    pub fn detect() -> Self {
        use windows::Win32::Graphics::Dxgi::*;
        use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        let (mut gpu, mut vram) = (String::from("No GPU found"), 0.0);
        unsafe {
            if let Ok(f) = CreateDXGIFactory1::<IDXGIFactory1>() {
                let mut i = 0;
                while let Ok(a) = f.EnumAdapters1(i) {
                    i += 1;
                    let Ok(d) = a.GetDesc1() else { continue };
                    let gb = d.DedicatedVideoMemory as f32 / 1073741824.0; // GiB, as Windows shows it
                    if d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 && gb > vram {
                        vram = gb;
                        gpu = String::from_utf16_lossy(&d.Description).trim_end_matches('\0').to_string();
                    }
                }
            }
        }
        let mut mem = MEMORYSTATUSEX { dwLength: size_of::<MEMORYSTATUSEX>() as u32, ..Default::default() };
        let ram = unsafe { GlobalMemoryStatusEx(&mut mem) }.map_or(0.0, |_| mem.ullTotalPhys as f32 / 1073741824.0);
        Specs { gpu, vram, ram }
    }

    /// A quarter of VRAM and of RAM is what Paper Translate may use while a game runs; the rest stays with the game.
    fn share(&self) -> (f32, f32) {
        ((self.vram * 0.25).clamp(0.8, 8.0), (self.ram * 0.25).max(0.5))
    }

    /// PaddleOCR on the GPU needs ~0.75 GB; below a 2.5 GB share (GPUs under ~10 GB) the LLM needs it more, so those
    /// PCs read text with Windows OCR instead.
    pub fn paddle_fits(&self) -> bool {
        self.share().0 >= 2.5
    }

    /// What the translation model may use: the share minus PaddleOCR's VRAM when it runs.
    pub fn budget(&self) -> (f32, f32) {
        let (v, r) = self.share();
        (if self.paddle_fits() { v - PADDLE_VRAM } else { v }, r)
    }

    /// Highest-quality quant of `m` that fits the gaming budget, with its vision projector when that fits too.
    pub fn fit<'a>(&self, m: &'a Model) -> Option<(&'a Quant, bool)> {
        let (v, r) = self.budget();
        let extra = m.vision.as_ref().map(|x| x.vram);
        let with_vision = extra.and_then(|x| m.quants.iter().find(|q| q.vram + x <= v && q.ram <= r));
        with_vision.map(|q| (q, true)).or_else(|| m.quants.iter().find(|q| q.vram <= v && q.ram <= r).map(|q| (q, false)))
    }

    /// The best model that can judge choices from the screen; otherwise the best that fits at all.
    pub fn recommend(&self) -> Option<usize> {
        let fits = |vision: bool| MODELS.iter().position(|m| self.fit(m).is_some_and(|f| f.1 || !vision));
        fits(true).or_else(|| fits(false))
    }
}


pub fn hf_url(repo: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{file}?download=true")
}

/// Accepts a pasted Hugging Face page or file URL; returns (download url, file name).
pub fn parse_custom_url(url: &str) -> Option<(String, String)> {
    let url = url.trim().replace("/blob/", "/resolve/");
    let name = url.split('?').next()?.rsplit('/').next()?.to_string();
    (url.starts_with("https://huggingface.co/") && name.ends_with(".gguf")).then_some((url, name))
}

pub fn app_dir() -> PathBuf {
    let base = std::env::var("APPDATA").unwrap_or_else(|_| ".".into());
    let dir = Path::new(&base).join("PaperTranslate");
    let _ = fs::create_dir_all(&dir);
    dir
}

pub fn download(url: &str, dest: &Path, mut progress: impl FnMut(u64, u64)) -> Result<(), String> {
    let resp = ureq::get(url).header("User-Agent", "paper-translate").call().map_err(|e| format!("{url}: {e}"))?;
    let total = resp.headers().get("content-length").and_then(|v| v.to_str().ok()?.parse().ok()).unwrap_or(0);
    let part = dest.with_extension("part");
    let mut out = File::create(&part).map_err(|e| e.to_string())?;
    let mut body = resp.into_body().into_reader();
    let (mut buf, mut done) = (vec![0u8; 1 << 16], 0u64);
    loop {
        let n = body.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        done += n as u64;
        progress(done, total);
    }
    drop(out);
    if total > 0 && done != total {
        return Err(format!("download interrupted ({done}/{total} bytes)"));
    }
    fs::rename(&part, dest).map_err(|e| e.to_string())
}

/// PaddleOCR model files live next to the translation model.
pub fn paddle_dir(model_path: &str) -> PathBuf {
    Path::new(model_path).parent().map(Path::to_path_buf).unwrap_or_else(app_dir)
}

pub fn paddle_ready(dir: &Path) -> bool {
    crate::paddle::FILES[..3].iter().all(|f| dir.join(f.2).is_file())
}

pub fn get_paddle(dir: &Path, mut progress: impl FnMut(u64, u64)) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    for (repo, file, local) in crate::paddle::FILES {
        if !dir.join(local).is_file() {
            download(&hf_url(repo, file), &dir.join(local), &mut progress)?;
        }
    }
    Ok(())
}

pub fn find_llama_server(configured: &str) -> Option<PathBuf> {
    let p = PathBuf::from(configured);
    if !configured.is_empty() && p.is_file() {
        return Some(p);
    }
    let local = app_dir().join("llama").join("llama-server.exe");
    if local.is_file() {
        return Some(local);
    }
    std::env::var_os("PATH")
        .and_then(|paths| std::env::split_paths(&paths).map(|d| d.join("llama-server.exe")).find(|p| p.is_file()))
}

/// Downloads the newest llama.cpp Vulkan build (runs on NVIDIA, AMD and Intel GPUs).
pub fn install_llama(progress: impl FnMut(u64, u64)) -> Result<PathBuf, String> {
    let dir = app_dir().join("llama");
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    // "latest" can point at a non-binary tag, so scan recent releases for the asset.
    let releases: serde_json::Value = ureq::get("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10")
        .header("User-Agent", "paper-translate")
        .call()
        .and_then(|r| r.into_body().read_json())
        .map_err(|e| format!("GitHub: {e}"))?;
    let url = releases
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|r| r["assets"].as_array().into_iter().flatten())
        .find(|a| a["name"].as_str().is_some_and(|n| n.ends_with("-bin-win-vulkan-x64.zip")))
        .and_then(|a| a["browser_download_url"].as_str())
        .ok_or("no Windows Vulkan build found in recent llama.cpp releases")?
        .to_string();
    let zip_path = dir.join("llama.zip");
    download(&url, &zip_path, progress)?;
    let mut zip = zip::ZipArchive::new(File::open(&zip_path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    zip.extract(&dir).map_err(|e| e.to_string())?;
    let _ = fs::remove_file(&zip_path);
    find_llama_server("").ok_or_else(|| "llama-server.exe missing from the downloaded archive".into())
}

pub fn start_server(exe: &Path, model: &Path, mmproj: Option<&Path>) -> Result<Child, String> {
    let log = File::create(app_dir().join("llama-server.log")).map_err(|e| e.to_string())?;
    let mut cmd = Command::new(exe);
    if let Some(p) = mmproj {
        cmd.arg("--mmproj").arg(p);
    }
    let mut child = cmd
        .arg("-m")
        .arg(model)
        .args(["-ngl", "99", "-c", "4096", "--reasoning", "off", "--host", "127.0.0.1", "--port", &port().to_string()])
        .stdout(Stdio::null())
        .stderr(log)
        .creation_flags(0x0800_0000) // CREATE_NO_WINDOW
        .spawn()
        .map_err(|e| format!("cannot start llama-server: {e}"))?;
    kill_with_us(&child);
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(180) {
        if let Ok(Some(status)) = child.try_wait() {
            return Err(format!("llama-server exited ({status}), see {}", app_dir().join("llama-server.log").display()));
        }
        if ureq::get(&format!("http://127.0.0.1:{}/health", port())).call().is_ok() {
            return Ok(child);
        }
        std::thread::sleep(Duration::from_millis(300));
    }
    let _ = child.kill();
    Err("llama-server did not become ready in 3 minutes".into())
}

/// Puts the server in a kill-on-close job so it dies with this app, even on a crash.
fn kill_with_us(child: &Child) {
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::System::JobObjects::*;
    unsafe {
        let Ok(job) = CreateJobObjectW(None, None) else { return };
        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let _ = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            (&raw const info).cast(),
            size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        );
        let _ = AssignProcessToJobObject(job, HANDLE(child.as_raw_handle()));
        // job handle intentionally leaked: the OS closes it when we exit, which kills the server
    }
}

#[test]
fn custom_url_parsing() {
    let (u, n) = parse_custom_url("https://huggingface.co/a/b/blob/main/m-Q4_K_M.gguf").unwrap();
    assert_eq!((u.as_str(), n.as_str()), ("https://huggingface.co/a/b/resolve/main/m-Q4_K_M.gguf", "m-Q4_K_M.gguf"));
    assert!(parse_custom_url("https://evil.example/x.gguf").is_none());
}

#[test]
fn picks_best_quant_for_the_budget() {
    let pc = |vram, ram| Specs { gpu: String::new(), vram, ram };
    let file = |f: Option<(&Quant, bool)>| f.map(|(q, v)| (q.file, v));
    // 16 GB GPU / 32 GB RAM: 4 GB share, PaddleOCR takes 0.75 → E4B no longer fits, E2B Q5_K_M + vision does
    assert!(pc(16.0, 32.0).paddle_fits());
    assert!(pc(16.0, 32.0).fit(&MODELS[0]).is_none());
    assert_eq!(file(pc(16.0, 32.0).fit(&MODELS[1])), Some(("gemma-4-E2B-it-Q5_K_M.gguf", true)));
    assert_eq!(pc(16.0, 32.0).recommend(), Some(1));
    // 8 GB GPU / 16 GB RAM: 2 GB share → Windows OCR, no room for vision, E2B Q5_K_M text-only
    assert!(!pc(8.0, 16.0).paddle_fits());
    assert!(pc(8.0, 16.0).fit(&MODELS[0]).is_none());
    assert_eq!(file(pc(8.0, 16.0).fit(&MODELS[1])), Some(("gemma-4-E2B-it-Q5_K_M.gguf", false)));
    // 4 GB GPU: 1 GB → only the tiny model
    assert_eq!(MODELS.iter().filter(|m| pc(4.0, 16.0).fit(m).is_some()).count(), 1);
}
