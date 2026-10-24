# Paper Translate

Live on-screen translation overlay for Windows 11 games and apps. Press a hotkey and the text in the foreground
window is covered by paper-style cards in your language. Names, items and key terms are highlighted in orange.
Press **Ctrl+Shift+T** and drag a box around the text you want translated, or just click to translate the whole game
window. When the game asks you to pick (A/B/C answers, ①②③, a roguelike reward screen), click **★ Suggest** under the
status pill. The choice window gets an orange outline, the best option a ★, and a short reason appears next to it. Quiz,
trivia and math choices get the correct answer, and a question with no options gets its answer in a note next to it.

Reads English, Japanese, Chinese, Korean, Russian (Cyrillic) and Thai games, switching readers automatically, and
translates into any language (Thai, Japanese, English…). Everything
runs locally: PaddleOCR (or Windows OCR) reads the screen, and a small quantized model in
[llama.cpp](https://github.com/ggml-org/llama.cpp) translates and judges choices.

## Install

Download `PaperTranslate-<version>-portable.zip` from [Releases](https://github.com/phongpisut/OSTD/releases), unzip it
anywhere you can write to, and run `paper-translate.exe`. There is no installer and no admin prompt.

Everything stays in that folder:
- `data\`: settings, llama.cpp, logs
- `models\`: downloaded models (another folder can be picked in the setup window)

To move or back it up, copy the folder. To remove it, quit from the tray and delete the folder. If the folder is
read-only (for example under Program Files), settings go to `%APPDATA%\PaperTranslate` and models to
`%LOCALAPPDATA%\PaperTranslate` instead.

On first launch, Paper Translate shows your GPU, VRAM and RAM, and recommends a model that fits beside a running game.
It downloads the model, the vision module and PaddleOCR from Hugging Face. If `llama-server` is not on `PATH`, it also
installs the newest llama.cpp Vulkan build into `data\llama`, which runs on NVIDIA, AMD and Intel.

After setup the app lives in the tray. Left-click the tray icon for Settings. The hotkey **Ctrl+Shift+T**
(configurable) works like this:
- Press it to start, then drag a box (a click means the whole window, right-click cancels).
- Press it again to stop.

Only the box is read and translated, and a dashed orange frame marks it. The box selection and the ★ Suggest button never
take focus from the game, and the tray menu has a "Suggest a choice" item too. Automatic suggestions can be turned on in Settings. They are off by default, because outside games
they can also mark menus and lists.

## How it works

1. **Capture** the dragged box, or the whole foreground window. A screen that hasn't changed is not read again.
2. **Read text** with PaddleOCR PP-OCRv6 medium on the GPU (ONNX Runtime + DirectML), or Windows OCR on PCs with
   less VRAM. PP-OCRv6 reads Chinese, Japanese and Latin scripts. Some lines it can't read: it returns few letters
   for its box, or a low confidence. Those lines are re-read on the CPU by small PP-OCRv5 Korean, Cyrillic and Thai
   readers (8–13 MB each, 2 threads), and a reader only takes over a line written in its own script. Icon noise and
   single glyphs are dropped, number-only lines (such as "1. 5" or "500") are kept for choices but not translated, and
   Japanese/Chinese words are joined without spaces.
3. **Context boundaries**: lines are merged into paragraphs, then into regions (a dialogue box, a panel, a choice
   window). Each region is translated in its own request, so the model only sees text that belongs together.
   Translations are cached.
4. **Choices**: the layout proposes candidates. Strong evidence is `A.` / `1)` / `①` / `▶` markers or 2–5 evenly
   spaced cards. A plain stacked list counts only when a question is nearby or the lines read like replies.
5. **Decision**: when you click ★ Suggest, the text prompt picks the best option and writes the reason. For quiz, trivia,
   math or riddle questions, the best option is the correct answer. With no options on screen, the question is answered
   directly: the line with a "?", or else the most prominent line. With automatic
   suggestions on, Gemma 4's vision first checks the screenshot plus the candidate options to confirm the game is
   really asking for a pick.

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

PP-OCRv6 also reads the Japanese scene (2–3% CER, only missing spaces) without any Windows language pack. With the
small per-script readers, Korean, Russian and Thai dialogue scenes read at 5.8%, 2.0% and 1.2% CER in ~0.4 s. Before,
they were unreadable (48–100%). English and Japanese scenes never wake the small readers. Detecting on a
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

**Key-term highlights** (10 game lines, 30 hand-labeled key words):

| Method | Precision | Recall | F1 | Cost |
|---|---|---|---|---|
| Laya English, zero-shot | 0.50 | 0.32 | 0.39 | 1.1 s/line CPU |
| Laya multilingual, zero-shot | 0.68 | 0.56 | 0.61 | 0.4 s/line CPU |
| Capitalized words + numbers rule | 1.00 | 0.68 | 0.81 | free, English only |
| **Gemma 4 E2B marking during translation** | **0.93** | **0.74** | **0.82** | +7% translation time |

Highlights stay with the translation model: marking costs it ~0.15 s per 10 lines and lands on the translated words
directly.

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

To build the portable zip (`dist\PaperTranslate-<version>-portable.zip`):

```bash
powershell -File portable/package.ps1
```

## Limits

- Games must run in **borderless windowed** mode. Exclusive fullscreen cannot be overlaid or captured with GDI.
- Primary monitor only. Vertical (tategaki) Japanese and rotated text are not handled.
- Unmarked options closer together than about one line height merge into one paragraph and get no suggestion.
- llama-server handles one request at a time on purpose: two concurrent requests hung the Vulkan build with Gemma 4.
- The overlay and the ★ Suggest button are hidden from screen capture, so OCR never reads them. With
  `PAPER_SHOW_IN_CAPTURE=1` they are hidden only during the app's own capture, so screenshots and recordings show them. `PAPER_PORT` changes the
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
