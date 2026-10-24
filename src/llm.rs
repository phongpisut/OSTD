use crate::setup::port;
use serde_json::{Value, json};
use std::time::Duration;

/// Runs a chat prompt through raw /completion with a JSON schema. Using /completion instead of the chat
/// endpoint makes the grammar bind from the first token, so models that open a thinking channel anyway
/// (Gemma 4) cannot leak it into the answer.
fn complete(system: &str, user: &str, schema: Value, n_predict: usize) -> Result<Value, String> {
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(90))).build().into();
    let post = |path: &str, body: Value| -> Result<Value, String> {
        (agent.post(&format!("http://127.0.0.1:{}{path}", port())).send_json(&body))
            .and_then(|r| r.into_body().read_json())
            .map_err(|e| format!("llama-server: {e}"))
    };
    let messages = json!([{ "role": "system", "content": system }, { "role": "user", "content": user }]);
    let prompt = post("/apply-template", json!({ "messages": messages }))?;
    let resp = post("/completion", json!({ "prompt": prompt["prompt"], "json_schema": schema, "temperature": 0.0, "n_predict": n_predict }))?;
    let content = resp["content"].as_str().ok_or("empty reply from model")?;
    serde_json::from_str(content).map_err(|e| format!("bad JSON from model: {e}"))
}

fn numbered(lines: &[String]) -> String {
    lines.iter().enumerate().map(|(i, l)| format!("{i}. {l}\n")).collect()
}

/// Translates all lines in one request; important words come back wrapped in ** when `highlight` is set.
pub fn translate(lines: &[String], target: &str, highlight: bool) -> Result<Vec<String>, String> {
    let n = lines.len();
    let mut sys = format!(
        "You are a professional video game localizer. Translate each numbered line into natural, fluent {target}, \
         the way a skilled human translator would: keep tone, slang and personality, and keep names consistent. \
         Return one translation per line, in order, without the numbers."
    );
    if highlight {
        sys += " In every translation, wrap the important words in **double asterisks**: character names, places, \
                items, skills, quest goals, key numbers and game terms. \
                Example of the marking: Bring the **Iron Key** to **Captain Mira** at **Stormhold**.";
    }
    let schema = json!({
        "type": "object",
        "properties": { "translations": { "type": "array", "items": { "type": "string", "minLength": 1 }, "minItems": n, "maxItems": n } },
        "required": ["translations"], "additionalProperties": false
    });
    let out = complete(&sys, &format!("Translate into {target}:\n{}", numbered(lines)), schema, 128 + 160 * n)?;
    let translated: Vec<String> = (out["translations"].as_array().into_iter().flatten())
        .enumerate()
        .map(|(i, v)| {
            let t = v.as_str().unwrap_or("");
            // small models sometimes echo the "3. " numbering back
            let num = format!("{i}. ");
            if t.starts_with(&num) && !lines[i].starts_with(&num) { t[num.len()..].to_string() } else { t.to_string() }
        })
        .collect();
    if translated.len() != n {
        return Err(format!("model returned {} lines for {n}", translated.len()));
    }
    Ok(translated)
}

/// Decision model. The layout already found candidate options; `confirm` is set when that evidence is weak (an
/// unmarked list), and then the model must also classify the text as a player choice. Returns (best, reason), or
/// None when it is not a choice. `scene` is the text nearest the options.
pub fn pick_choice(options: &[String], scene: &[String], target: &str, confirm: bool) -> Result<Option<(usize, String)>, String> {
    let sys = "You advise a player in a video game.";
    let user = format!(
        "Text near the options:\n{}\n\nLines:\n{}\n\
         Set kind to what these lines are: \"player_choice\" if the player must pick one of them now (dialogue answers, \
         rewards, upgrades, cards, a yes/no prompt), \"story\" for narration or spoken lines, \"menu\" for game menus or \
         settings, \"hud\" for status labels. Then set best to the number of the wisest or most beneficial line for the \
         player (rewards, safety, relationships, story progress) and give a one-sentence reason in {target} in why.",
        scene.join("\n"),
        numbered(options)
    );
    let schema = json!({
        "type": "object",
        "properties": {
            "kind": { "enum": ["player_choice", "story", "menu", "hud"] },
            "best": { "enum": (0..options.len()).collect::<Vec<_>>() },
            "why": { "type": "string", "minLength": 8, "maxLength": 200 }
        },
        "required": ["kind", "best", "why"], "additionalProperties": false
    });
    let out = complete(sys, &user, schema, 220)?;
    if confirm && out["kind"] != "player_choice" {
        return Ok(None);
    }
    let best = out["best"].as_u64().map(|b| b as usize).filter(|&b| b < options.len()).ok_or("model gave no choice")?;
    Ok(Some((best, out["why"].as_str().unwrap_or("").to_string())))
}

fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            out.push(if i <= c.len() { T[(n >> (18 - 6 * i) & 63) as usize] as char } else { '=' });
        }
    }
    out
}

/// Vision decision: the model looks at the screenshot (BMP) and the candidate options from OCR, decides whether
/// the game is asking for a pick right now and which option is best. Gemma 4 E2B got 5/5 test screens right;
/// sub-1B vision models said "choice" for everything.
pub fn pick_choice_vision(screen_bmp: &[u8], options: &[String], target: &str) -> Result<Option<(usize, String)>, String> {
    // a label pick ("player_choice" / "story" / ...) scored 5/5 at every image size; a yes/no flipped on menus
    let text = format!(
        "The text detector found these lines on screen:\n{}\nLook at the screenshot and set kind to what these lines are: \
         \"player_choice\" if the game is waiting for the player to pick one of them now (dialogue answers, rewards, upgrades, \
         cards), \"story\" for narration or spoken lines, \"menu\" for title or settings menus, \"hud\" for status labels. Then set \
         best to the number of the wisest line and give a one-sentence reason in {target} in why.",
        numbered(options)
    );
    let schema = json!({
        "type": "object",
        "properties": {
            "kind": { "enum": ["player_choice", "story", "menu", "hud"] },
            "best": { "enum": (0..options.len()).collect::<Vec<_>>() },
            "why": { "type": "string", "maxLength": 200 }
        },
        "required": ["kind", "best", "why"], "additionalProperties": false
    });
    let body = json!({
        "messages": [{ "role": "user", "content": [
            { "type": "image_url", "image_url": { "url": format!("data:image/bmp;base64,{}", base64(screen_bmp)) } },
            { "type": "text", "text": text },
        ] }],
        "temperature": 0.0,
        "max_tokens": 250,
        "response_format": { "type": "json_schema", "json_schema": { "name": "decision", "schema": schema } },
    });
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(60))).build().into();
    let resp: Value = (agent.post(&format!("http://127.0.0.1:{}/v1/chat/completions", port())).send_json(&body))
        .and_then(|r| r.into_body().read_json())
        .map_err(|e| format!("llama-server: {e}"))?;
    let content = resp["choices"][0]["message"]["content"].as_str().unwrap_or("");
    let out: Value = serde_json::from_str(content).map_err(|e| format!("bad JSON from vision model: {e}"))?;
    if out["kind"] != "player_choice" {
        return Ok(None);
    }
    let best = out["best"].as_u64().map(|b| b as usize).filter(|&b| b < options.len()).ok_or("model gave no choice")?;
    Ok(Some((best, out["why"].as_str().unwrap_or("").to_string())))
}

#[test]
fn base64_matches_rfc4648() {
    assert_eq!((base64(b"Man"), base64(b"Ma"), base64(b"M")), ("TWFu".into(), "TWE=".into(), "TQ==".into()));
}

/// Vision e2e on rendered scenes: `SCENES=<dir with dialogue/cards/narration/menu/japanese .bgra> cargo test vision_e2e -- --ignored --nocapture`
#[test]
#[ignore]
fn vision_e2e() {
    use crate::setup::*;
    let dir = std::path::PathBuf::from(std::env::var("LOCALAPPDATA").unwrap()).join("PaperTranslate").join("models");
    let exe = find_llama_server("").unwrap();
    let mut server = start_server(&exe, &dir.join("gemma-4-E2B-it-Q4_K_M.gguf"), Some(&dir.join("mmproj-gemma-4-E2B-F16.gguf"))).unwrap();
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let scenes = [
        ("dialogue", s(&["A. I will track down the thief tonight.", "B. That is not my problem.", "C. Pay me 500 gold first."]), true),
        ("cards", s(&["Thunder Strike — Your attacks chain lightning to 3 nearby enemies. — Rare", "Iron Skin — Take 30% less damage for 10 seconds after a dash. — Rare", "Golden Touch — Enemies drop 50% more gold, but you lose 10 max HP. — Rare"]), true),
        ("narration", s(&["We sailed for three days before the storm hit.", "Half the crew was lost at sea.", "Only the old map kept us alive."]), false),
        ("menu", s(&["Continue", "New Game", "Settings", "Quit"]), false),
        ("japanese", s(&["① 今夜、泥棒を追いかける。", "② 私には関係ない。", "③ 先に金貨500枚を払え。"]), true),
    ];
    let mut right = 0;
    for (name, options, truth) in &scenes {
        if !crate::ocr::marker(&options[0]) && !crate::ocr::worth_asking(options, &[]) {
            println!("{name}: filtered by layout before the model");
            right += !*truth as usize;
            continue;
        }
        let px = std::fs::read(std::path::Path::new(&std::env::var("SCENES").unwrap()).join(format!("{name}.bgra"))).unwrap();
        let t = std::time::Instant::now();
        let bmp = crate::ocr::bmp_thumbnail(&px, 1920, 1080, 896);
        if let Ok(out) = std::env::var("THUMB_OUT") {
            std::fs::write(std::path::Path::new(&out).join(format!("{name}_thumb.bmp")), &bmp).unwrap();
        }
        let r = pick_choice_vision(&bmp, options, "Thai");
        println!("{name}: {r:?} in {:?}", t.elapsed());
        right += (r.as_ref().is_ok_and(|v| v.is_some() == *truth)) as usize;
    }
    let _ = server.kill();
    assert_eq!(right, scenes.len());
}

/// Real run: installs llama.cpp, downloads the default model, translates and checks the choice gate.
/// `cargo test e2e -- --ignored --nocapture` (E2E_MODEL=<file in models dir> to try another, PAPER_PORT to avoid a running app)
#[test]
#[ignore]
fn e2e() {
    use crate::setup::*;
    let exe = find_llama_server("").unwrap_or_else(|| install_llama(|_, _| {}).unwrap());
    let (m, q) = (&MODELS[1], MODELS[1].quants.last().unwrap());
    let dir = std::path::PathBuf::from(std::env::var("LOCALAPPDATA").unwrap()).join("PaperTranslate").join("models");
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join(std::env::var("E2E_MODEL").unwrap_or(q.file.into()));
    if !model.is_file() {
        download(&hf_url(m.repo, q.file), &model, |_, _| {}).unwrap();
    }
    let mut server = start_server(&exe, &model, None).unwrap();
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let scene = s(&["Elder Mira", "The Moonstone Amulet was stolen from Stormhold last night. If the Shadow Guild sells it, the old seal will break."]);
    let options = s(&["A. I will track down the thief tonight.", "B. That is not my problem.", "C. Pay me 500 gold first."]);
    let t0 = std::time::Instant::now();
    let tr = translate(&[scene.clone(), options.clone()].concat(), "Thai", true);
    println!("{tr:#?} in {:?}", t0.elapsed());
    let jp = s(&["長老ミラ", "昨夜、ストームホールドから月光石の首飾りが盗まれた。影のギルドが売れば、古い封印が壊れてしまう。", "① 今夜、泥棒を追いかける。", "② 私には関係ない。", "③ 先に金貨500枚を払え。"]);
    println!("japanese → Thai: {:#?}", translate(&jp, "Thai", true));
    let t0 = std::time::Instant::now();
    let dialogue = pick_choice(&options, &scene, "Thai", false);
    println!("dialogue: {dialogue:?} in {:?}", t0.elapsed());
    let cards = pick_choice(
        &s(&[
            "Thunder Strike — Your attacks chain lightning to 3 nearby enemies. — Rare",
            "Iron Skin — Take 30% less damage for 10 seconds after a dash. — Rare",
            "Golden Touch — Enemies drop 50% more gold, but you lose 10 max HP. — Rare",
        ]),
        &s(&["Choose a Blessing"]),
        "Thai",
        false,
    );
    println!("cards: {cards:?}");
    let unmarked = pick_choice(&s(&["Sure, I'll help you.", "Not today."]), &s(&["Guard", "Will you help me find my sword?"]), "Thai", true);
    println!("unmarked answers: {unmarked:?}");
    let narration = pick_choice(&s(&["Rain kept falling on the old harbor.", "Nobody came to the docks that night."]), &s(&["Chapter 3"]), "Thai", true);
    println!("narration: {narration:?}");
    let menu = pick_choice(&s(&["Continue", "New Game", "Settings", "Quit"]), &s(&["Version 1.0.4"]), "Thai", true);
    println!("menu: {menu:?}");
    let _ = server.kill();
    assert_eq!(tr.unwrap().len(), 5);
    assert!(dialogue.unwrap().is_some() && cards.unwrap().is_some());
    assert!(unmarked.unwrap().is_some() && narration.unwrap().is_none());
    let _ = menu; // menus are filtered before the model by ocr::worth_asking
}
