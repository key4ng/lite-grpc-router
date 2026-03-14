use anyhow::{Context, Result};
use tonic::transport::Channel;

use crate::proto::sglang::sglang_scheduler_client::SglangSchedulerClient;

pub struct SglangClient {
    client: SglangSchedulerClient<Channel>,
    url: String,
}

impl SglangClient {
    pub async fn connect(url: &str) -> Result<Self> {
        let endpoint = if url.starts_with("http") {
            url.to_string()
        } else {
            format!("http://{url}")
        };

        let channel = Channel::from_shared(endpoint.clone())
            .context("invalid endpoint URL")?
            .connect()
            .await
            .with_context(|| format!("failed to connect to {endpoint}"))?;

        Ok(Self {
            client: SglangSchedulerClient::new(channel),
            url: url.to_string(),
        })
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub async fn generate(
        &self,
        req: crate::proto::sglang::GenerateRequest,
    ) -> Result<tonic::Streaming<crate::proto::sglang::GenerateResponse>> {
        let mut client = self.client.clone();
        let response = client
            .generate(tonic::Request::new(req))
            .await
            .context("generate RPC failed")?;
        Ok(response.into_inner())
    }

    pub async fn get_tokenizer(
        &self,
    ) -> Result<tonic::Streaming<crate::proto::common::GetTokenizerChunk>> {
        let mut client = self.client.clone();
        let response = client
            .get_tokenizer(tonic::Request::new(
                crate::proto::common::GetTokenizerRequest {},
            ))
            .await
            .context("get_tokenizer RPC failed")?;
        Ok(response.into_inner())
    }

    pub async fn get_model_info(&self) -> Result<crate::proto::sglang::GetModelInfoResponse> {
        let mut client = self.client.clone();
        let response = client
            .get_model_info(tonic::Request::new(
                crate::proto::sglang::GetModelInfoRequest {},
            ))
            .await
            .context("get_model_info RPC failed")?;
        Ok(response.into_inner())
    }

    pub async fn health_check(&self) -> Result<bool> {
        let mut client = self.client.clone();
        let response = client
            .health_check(tonic::Request::new(
                crate::proto::sglang::HealthCheckRequest {},
            ))
            .await
            .context("health_check RPC failed")?;
        Ok(response.into_inner().healthy)
    }
}
