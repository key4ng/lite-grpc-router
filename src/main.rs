#![allow(dead_code)]

mod client;
mod parsers;
mod router;
mod streaming;
mod tokenizer;
mod types;
mod worker;

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use axum::routing::{get, post};
use axum::Router;
use clap::Parser;

use crate::client::SglangClient;
use crate::router::AppState;
use crate::worker::{Worker, WorkerPool};

// Proto modules: hierarchy must match proto package names so tonic's
// generated cross-package super:: references resolve correctly.
pub mod smg {
    pub mod grpc {
        pub mod common {
            tonic::include_proto!("smg.grpc.common");
        }
    }
}
pub mod sglang {
    pub mod grpc {
        pub mod scheduler {
            tonic::include_proto!("sglang.grpc.scheduler");
        }
    }
}

// Convenience re-exports
pub mod proto {
    pub use crate::sglang::grpc::scheduler as sglang;
    pub use crate::smg::grpc::common;
}

#[derive(Parser)]
#[command(name = "lite-grpc", about = "Minimal gRPC router for SGLang")]
struct Cli {
    /// SGLang backend gRPC URLs (can specify multiple)
    #[arg(long = "backend", required = true)]
    backends: Vec<String>,

    /// Port to listen on
    #[arg(long, default_value = "8080")]
    port: u16,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.backends.is_empty() {
        bail!("at least one --backend URL is required");
    }

    eprintln!("lite-grpc starting...");
    eprintln!("Backends: {:?}", cli.backends);

    let mut workers = Vec::new();
    for url in &cli.backends {
        eprint!("Connecting to {url}... ");
        let client = SglangClient::connect(url)
            .await
            .with_context(|| format!("failed to connect to {url}"))?;

        let model_info = client.get_model_info().await
            .with_context(|| format!("failed to get model info from {url}"))?;
        eprintln!("OK (model: {})", model_info.served_model_name);

        workers.push(Arc::new(Worker::new(client, model_info.served_model_name)));
    }

    let model_name = workers[0].model_name.clone();

    eprint!("Loading tokenizer from {}... ", cli.backends[0]);
    let tokenizer_stream = workers[0].client.get_tokenizer().await
        .context("failed to start tokenizer download")?;
    let loaded = tokenizer::load_tokenizer_from_rpc(tokenizer_stream).await
        .context("failed to load tokenizer")?;
    eprintln!("OK (chat template: {})", if loaded.chat_template.is_some() { "found" } else { "not found" });

    let pool = WorkerPool::new(workers);
    let state = Arc::new(AppState {
        pool,
        tokenizer: loaded.tokenizer,
        chat_template: loaded.chat_template,
        model_name,
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(router::chat_completions))
        .route("/v1/models", get(router::list_models))
        .route("/health", get(router::health))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", cli.port);
    eprintln!("Listening on {addr}");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
