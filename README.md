# Paper Translate

Live on-screen translation overlay for Windows 11 games and apps. Press a hotkey and the text in the foreground
window is covered by paper-style cards in your language. Names, items and key terms are highlighted in orange.
When the game asks you to pick (A/B/C answers, ①②③, a roguelike reward screen), the whole choice window gets an
orange outline, the best option a ★, and a short reason appears next to it.

Reads English, Japanese, Chinese and Korean games, and translates into any language (Thai, Japanese, English…).
Everything runs locally: Windows' built-in OCR reads the screen, and a small quantized model in
[llama.cpp](https://github.com/ggml-org/llama.cpp) translates.

## Install

Download `PaperTranslate-<version>-Setup.msi` from [Releases](https://github.com/phongpisut/OSTD/releases) and run it.
It installs for your user only (no admin) and adds a Start Menu shortcut.

On first launch, Paper Translate shows your GPU, VRAM and RAM, and recommends a model that fits beside a running game.
It downloads the model from Hugging Face into a folder you choose. If `llama-server` is not on `PATH`, it also installs
the newest llama.cpp Vulkan build, which runs on NVIDIA, AMD and Intel.

After setup the app lives in the tray. Left-click the tray icon for Settings, and press **Ctrl+Shift+T**
(configurable) to start or stop translating.

**Japanese, Chinese or Korean games:** Settings → Languages → **+ Japanese OCR** installs Windows' OCR pack (Windows asks
for admin). Then click Refresh and pick `ja-JP` under "Read text in".

## How it works

1. **OCR**: Windows.Media.Ocr reads the foreground window in ~30 ms per 1080p frame on the CPU. On the test scenes it
   read the same text as PaddleOCR PP-OCRv5 (ONNX, CPU), which took ~600 ms. Tokens that look like icon noise are
   dropped, and Japanese/Chinese words are joined without spaces.
2. **Context boundaries**: lines are merged into paragraphs, then into regions (a dialogue box, a panel, a choice
   window). Each region is translated in its own request, so the model only sees text that belongs together.
   Translations are cached.
3. **Choices**: the layout proposes candidates. Strong evidence is `A.` / `1)` / `①` / `▶` markers or 2–5 evenly
   spaced cards. A plain stacked list counts only when a question is nearby or the lines read like replies.
4. **Vision decision**: with Gemma 4 the same server loads its vision projector. The model looks at the screenshot
   plus the candidate options and decides whether the game is really asking for a pick, which option is best, and why.
   On five test screens (dialogue, reward cards, Japanese dialogue, narration, title menu) it was right 5/5 in ~0.8 s.
   Sub-1B vision models (LFM2.5-VL 450M, SmolVLM 500M, Qwen3.5 0.8B) said "choice" for everything. Models without
   vision fall back to a text-only decision.

## Models

The translator gets **a quarter of your VRAM and RAM** while you play. For each model the app picks the best
quantization that fits (never below Q4_K_M), with the vision projector when that fits too.

| Model | VRAM, Q4_K_M → Q8_0 | Vision (+1.2 GB) |
|---|---|---|
| Gemma 4 E4B | 3.3 (measured) → 5.4 GB | yes |
| Gemma 4 E2B | 1.65 (measured) → 2.8 GB | yes |
| Typhoon Translate 1.5 4B (Thai) | 3.2 (measured) → 5.0 GB | no |
| Qwen3.5 2B | 1.45 (measured) → 2.2 GB | no |
| LFM2.5 1.2B | 1.15 → 1.45 GB (Q6_K → Q8_0) | no |
| Qwen3.5 0.8B | 0.85 → 1.0 GB (Q6_K → Q8_0) | no |

VRAM = GPU weight size from each file's GGUF header plus KV/compute overhead measured on an RTX 4060 Ti. On a 16 GB
card that makes the recommendation Gemma 4 E2B Q8_0 + vision (4.0 GB). Warm, a dialogue box plus 3 options is
translated in ~1.2 s (E2B) or ~1.7 s (E4B), and the choice decision adds ~0.8 s.

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
- Primary monitor only. Vertical (tategaki) Japanese text is not handled.
- Unmarked options closer together than about one line height merge into one paragraph and get no suggestion.
- llama-server handles one request at a time on purpose: two concurrent requests hung the Vulkan build with Gemma 4.
- The overlay is hidden from screen capture so OCR never reads its own cards. With `PAPER_SHOW_IN_CAPTURE=1` it is
  hidden only during the app's own capture, so screenshots and recordings show it. `PAPER_PORT` changes the
  llama-server port (default 8719).

## Tests

```bash
cargo test
```

The unit tests use real OCR output from a dialogue scene and a reward-card scene, plus the Japanese joining and
marker rules.

```bash
cargo test e2e -- --ignored --nocapture
```

```bash
cargo test vision_e2e -- --ignored --nocapture
```

These two run the real models. `vision_e2e` needs `SCENES=<folder with rendered .bgra scenes>`.
