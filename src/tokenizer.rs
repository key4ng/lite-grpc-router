use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;
use tonic::Streaming;

use crate::proto::common::GetTokenizerChunk;

const MAX_BUNDLE_SIZE: usize = 200 * 1024 * 1024; // 200 MB

pub async fn load_tokenizer_from_rpc(
    mut stream: Streaming<GetTokenizerChunk>,
) -> Result<LoadedTokenizer> {
    let mut data = Vec::new();
    let mut sha256 = String::new();

    while let Some(chunk) = stream.message().await.context("stream error")? {
        if data.len() + chunk.data.len() > MAX_BUNDLE_SIZE {
            bail!("tokenizer bundle exceeds {MAX_BUNDLE_SIZE} bytes");
        }
        data.extend_from_slice(&chunk.data);
        if !chunk.sha256.is_empty() {
            sha256 = chunk.sha256;
        }
    }

    if data.is_empty() {
        bail!("empty tokenizer stream");
    }

    if !sha256.is_empty() {
        let computed = format!("{:x}", Sha256::digest(&data));
        if !computed.eq_ignore_ascii_case(&sha256) {
            bail!("tokenizer SHA-256 mismatch: expected {sha256}, got {computed}");
        }
    }

    let cursor = Cursor::new(&data);
    let mut archive = zip::ZipArchive::new(cursor).context("failed to open tokenizer zip")?;
    let temp_dir = tempfile::tempdir().context("failed to create temp dir")?;
    archive
        .extract(temp_dir.path())
        .context("failed to extract tokenizer zip")?;

    load_from_dir(temp_dir.path())
}

pub async fn load_tokenizer_from_path_or_hub(source: &str) -> Result<LoadedTokenizer> {
    let path = Path::new(source);
    if path.exists() {
        let dir = if path.is_file() {
            path.parent().context("invalid tokenizer path")?
        } else {
            path
        };
        return load_from_dir(dir);
    }

    // Treat as HuggingFace model ID
    let api = hf_hub::api::tokio::ApiBuilder::from_env()
        .with_progress(true)
        .build()
        .context("failed to build HuggingFace Hub client")?;
    let repo = api.model(source.to_string());

    let tokenizer_path = repo
        .get("tokenizer.json")
        .await
        .context("failed to download tokenizer.json from HuggingFace")?;

    // Best-effort: download tokenizer_config.json for chat template
    let _ = repo.get("tokenizer_config.json").await;

    let dir = tokenizer_path.parent().context("invalid cache path")?;
    load_from_dir(dir)
}

fn load_from_dir(dir: &Path) -> Result<LoadedTokenizer> {
    let tokenizer_json = dir.join("tokenizer.json");
    if !tokenizer_json.exists() {
        bail!(
            "tokenizer.json not found in extracted archive (contents: {:?})",
            std::fs::read_dir(dir)?
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect::<Vec<_>>()
        );
    }

    let tokenizer = Tokenizer::from_file(&tokenizer_json)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    let chat_template = load_chat_template(dir);

    Ok(LoadedTokenizer {
        tokenizer: Arc::new(tokenizer),
        chat_template,
    })
}

pub struct LoadedTokenizer {
    pub tokenizer: Arc<Tokenizer>,
    pub chat_template: Option<String>,
}

fn load_chat_template(dir: &Path) -> Option<String> {
    let config_path = dir.join("tokenizer_config.json");
    let data = std::fs::read_to_string(&config_path).ok()?;
    let config: serde_json::Value = serde_json::from_str(&data).ok()?;
    config.get("chat_template")?.as_str().map(|s| s.to_string())
}

pub fn apply_chat_template(
    template: &str,
    messages: &[serde_json::Value],
    add_generation_prompt: bool,
) -> Result<String> {
    let mut env = minijinja::Environment::new();
    env.add_template("chat", template)
        .context("invalid chat template")?;
    let tmpl = env.get_template("chat").context("template not found")?;
    let result = tmpl
        .render(minijinja::context! {
            messages => messages,
            add_generation_prompt => add_generation_prompt,
        })
        .context("failed to render chat template")?;
    Ok(result)
}
