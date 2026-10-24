# Paper Translate

Live on-screen translation overlay for Windows 11 games and apps. Press a hotkey and the text in the foreground
window is covered by paper-style cards in your language. Names, items and key terms are highlighted in orange.
When the game asks you to pick (A/B/C answers, ①②③, a roguelike reward screen), the whole choice window gets an
orange outline, the best option a ★, and a short reason appears next to it.

Reads English, Japanese and Chinese games, and translates into any language (Thai, Japanese, English…). Everything
runs locally: PaddleOCR (or Windows OCR) reads the screen, and a small quantized model in
[llama.cpp](https://github.com/ggml-org/llama.cpp) translates and judges choices.

## Install

Download `PaperTranslate-<version>-Setup.msi` from [Releases](https://github.com/phongpisut/OSTD/releases) and run it.
It installs for your user only (no admin) and adds a Start Menu shortcut.

On first launch, Paper Translate shows your GPU, VRAM and RAM, and recommends a model that fits beside a running game.
It downloads the model, the vision module and PaddleOCR from Hugging Face into a folder you choose. If `llama-server` is
not on `PATH`, it also installs the newest llama.cpp Vulkan build, which runs on NVIDIA, AMD and Intel.

After setup the app lives in the tray. Left-click the tray icon for Settings, and press **Ctrl+Shift+T**
(configurable) to start or stop translating.

## How it works

1. **Capture** the foreground window. A screen that hasn't changed is not read again.
2. **Read text** with PaddleOCR PP-OCRv6 medium on the GPU (ONNX Runtime + DirectML), or Windows OCR on PCs with
   less VRAM. Icon noise is dropped, and Japanese/Chinese words are joined without spaces.
3. **Context boundaries**: lines are merged into paragraphs, then into regions (a dialogue box, a panel, a choice
   window). Each region is translated in its own request, so the model only sees text that belongs together.
   Translations are cached.
4. **Choices**: the layout proposes candidates. Strong evidence is `A.` / `1)` / `①` / `▶` markers or 2–5 evenly
   spaced cards. A plain stacked list counts only when a question is nearby or the lines read like replies.
5. **Decision (hybrid)**: Gemma 4's vision looks at the screenshot plus the candidate options and says whether the
   game is really asking for a pick. If it is, the text prompt picks the best option and writes the reason.

## Benchmarks behind the choices

Measured on an RTX 4060 Ti with rendered game scenes: dialogue, reward cards, stylized fonts, small HUD text,
low-contrast glow text, and Japanese dialogue. The ground truth is known.

**Text reader** (character error rate over the five English scenes):

| Reader | CER | Junk lines | Time per frame |
|---|---|---|---|
| Windows OCR (CPU) | 8.4% (35% on stylized fonts) | 2 | 37 ms |
| PP-OCRv6 small (GPU) | 2.5% | 0 | 171 ms |
| **PP-OCRv6 medium (GPU, 960 px detection)** | **0.2–0.7%** | **0** | **~100–170 ms, 0.75 GB VRAM** |
| PP-OCRv6 medium (CPU) | 0.2% | 0 | 790 ms |

PP-OCRv6 also reads the Japanese scene (2–3% CER, only missing spaces) without any Windows language pack. Detecting on a
frame downscaled to 960 px was as accurate as full 1080p at less than half the VRAM; 800 px started losing small HUD text.

**Decision model** (15 screens: 7 real choices, 8 look-alikes such as quest logs, settings, tips, inventories):

| Model | "Is it a choice?" | Best pick | Time |
|---|---|---|---|
| [Laya](https://huggingface.co/convaiinnovations/laya) English, zero-shot (CPU) | 8/15 | always option 0 | 720 ms |
| Laya multilingual, zero-shot (CPU) | 10/15 | always option 0 | 249 ms |
| Gemma 4 E2B, text only | 10/15 | 5/7 | 0.6 s |
| Gemma 4 E2B, vision only | 14/15 | 2/7 | 0.7 s |
| **Gemma 4 E2B hybrid (vision gate + text pick)** | **14/15** | **5/7** | **~1.0 s** |

Laya is a fast encoder classifier, but zero-shot it calls most lists a choice and always picks the first option. Its
model card notes that the base checkpoints need fine-tuning on your own decisions, and it would also need a Python and
PyTorch runtime. So it is not used. Sub-1B vision models (LFM2.5-VL 450M, SmolVLM 500M, Qwen3.5 0.8B) answered
"choice" for every screen.

## Models

Paper Translate may use **a quarter of your VRAM and RAM** while you play. On GPUs with a share of at least 2.5 GB (about
10 GB VRAM and up), PaddleOCR takes 0.75 GB of it. Smaller GPUs use Windows OCR so the model keeps the room. For each
model the app picks the best quantization that fits (never below Q4_K_M), with the vision module when that fits too.

| Model | VRAM, Q4_K_M → Q8_0 | Vision (+1.2 GB) |
|---|---|---|
| Gemma 4 E4B | 3.3 (measured) → 5.4 GB | yes |
| Gemma 4 E2B | 1.65 (measured) → 2.8 GB | yes |
| Typhoon Translate 1.5 4B (Thai) | 3.2 (measured) → 5.0 GB | no |
| Qwen3.5 2B | 1.45 (measured) → 2.2 GB | no |
| LFM2.5 1.2B | 1.15 → 1.45 GB (Q6_K → Q8_0) | no |
| Qwen3.5 0.8B | 0.85 → 1.0 GB (Q6_K → Q8_0) | no |

On a 16 GB card the recommendation is Gemma 4 E2B Q5_K_M + vision + PaddleOCR (3.9 GB in total). Warm, a dialogue box
plus 3 options is translated in ~1.2 s, and the choice decision adds ~1 s once per new choice screen.

## Build from source

```bash
cargo build --release
```

To build the installer with WiX 5 (`dotnet tool install --global wix --version 5.0.2`):

```bash
wix build installer/PaperTranslate.wxs -d Version=0.1.0 -arch x64 -o dist/PaperTranslate-0.1.0-Setup.msi
```

## Limits

- Games must run in **borderless windowed** mode. Exclusive fullscreen cannot be overlaid or captured with GDI.
- Primary monitor only. Vertical (tategaki) Japanese and rotated text are not handled.
- Unmarked options closer together than about one line height merge into one paragraph and get no suggestion.
- llama-server handles one request at a time on purpose: two concurrent requests hung the Vulkan build with Gemma 4.
- The overlay is hidden from screen capture so OCR never reads its own cards. With `PAPER_SHOW_IN_CAPTURE=1` it is
  hidden only during the app's own capture, so screenshots and recordings show it. `PAPER_PORT` changes the
  llama-server port (default 8719).

## Tests

```bash
cargo test
```

The unit tests use real OCR output from the dialogue and reward-card scenes, plus the paragraph, Japanese and budget rules.

```bash
cargo test e2e -- --ignored --nocapture
```

```bash
cargo test vision_e2e -- --ignored --nocapture
```

```bash
cargo test --release paddle_bench -- --ignored --nocapture
```

These three run the real models. `vision_e2e` and `paddle_bench` need `SCENES=<folder with rendered .bgra scenes>`.
