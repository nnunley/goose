use anyhow::Result;
use async_stream::try_stream;
use async_trait::async_trait;
use futures::TryStreamExt;
use reqwest::{Client, Response};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io;
use std::time::Duration;
use tokio::pin;
use tokio_stream::StreamExt;
use tokio_util::codec::{FramedRead, LinesCodec};
use tokio_util::io::StreamReader;

use super::api_client::{ApiClient, AuthMethod};
use super::base::{ConfigKey, ModelInfo, Provider, ProviderMetadata, ProviderUsage, Usage};
use super::embedding::{EmbeddingCapable, EmbeddingRequest, EmbeddingResponse, EmbeddingService, EmbeddingCapabilities, EmbeddingModel, EmbeddingResult, EmbeddingUsage};
use super::errors::ProviderError;
use super::formats::openai::{create_request, get_usage, response_to_message};
use super::utils::{
    emit_debug_trace, get_model, handle_response_openai_compat, handle_status_openai_compat,
    ImageFormat,
};
use crate::config::custom_providers::CustomProviderConfig;
use crate::impl_provider_default;
use crate::conversation::message::Message;
use crate::model::ModelConfig;
use crate::providers::base::MessageStream;
use crate::providers::formats::openai::response_to_streaming_message;
use rmcp::model::Tool;

pub const OPEN_AI_DEFAULT_MODEL: &str = "gpt-4o";
pub const OPEN_AI_KNOWN_MODELS: &[&str] = &[
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "o1",
    "o3",
    "o4-mini",
];

pub const OPEN_AI_DOC_URL: &str = "https://platform.openai.com/docs/models";

#[derive(Debug, serde::Serialize)]
pub struct OpenAiProvider {
    #[serde(skip)]
    client: Client,
    #[serde(skip)]
    api_client: Option<ApiClient>,
    host: String,
    base_path: String,
    api_key: String,
    organization: Option<String>,
    project: Option<String>,
    model: ModelConfig,
    custom_headers: Option<HashMap<String, String>>,
    supports_streaming: bool,
}

impl_provider_default!(OpenAiProvider);

impl OpenAiProvider {
    pub fn from_env(model: ModelConfig) -> Result<Self> {
        let config = crate::config::Config::global();
        let api_key: String = config.get_secret("OPENAI_API_KEY")?;
        let host: String = config
            .get_param("OPENAI_HOST")
            .unwrap_or_else(|_| "https://api.openai.com".to_string());
        let base_path: String = config
            .get_param("OPENAI_BASE_PATH")
            .unwrap_or_else(|_| "v1/chat/completions".to_string());
        let organization: Option<String> = config.get_param("OPENAI_ORGANIZATION").ok();
        let project: Option<String> = config.get_param("OPENAI_PROJECT").ok();
        let custom_headers: Option<HashMap<String, String>> = config
            .get_secret("OPENAI_CUSTOM_HEADERS")
            .or_else(|_| config.get_param("OPENAI_CUSTOM_HEADERS"))
            .ok()
            .map(parse_custom_headers);
        let timeout_secs: u64 = config.get_param("OPENAI_TIMEOUT").unwrap_or(600);
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()?;

        Ok(Self {
            client,
            api_client: None,
            host,
            base_path,
            api_key,
            organization,
            project,
            model,
            custom_headers,
            supports_streaming: true,
        })
    }

    pub fn from_custom_config(model: ModelConfig, config: CustomProviderConfig) -> Result<Self> {
        let global_config = crate::config::Config::global();
        let api_key: String = global_config
            .get_secret(&config.api_key_env)
            .map_err(|_e| anyhow::anyhow!("Missing API key: {}", config.api_key_env))?;

        let url = url::Url::parse(&config.base_url)
            .map_err(|e| anyhow::anyhow!("Invalid base URL '{}': {}", config.base_url, e))?;

        let host = format!("{}://{}", url.scheme(), url.host_str().unwrap_or(""));
        let base_path = url.path().trim_start_matches('/').to_string();
        let base_path = if base_path.is_empty() {
            "v1/chat/completions".to_string()
        } else {
            base_path
        };

        let timeout_secs = config.timeout_seconds.unwrap_or(600);
        
        // Create a regular client for compatibility with existing methods
        let mut client_builder = Client::builder().timeout(std::time::Duration::from_secs(timeout_secs));
        
        // Add custom headers to client if present
        if let Some(headers) = &config.headers {
            let mut header_map = reqwest::header::HeaderMap::new();
            for (key, value) in headers {
                let header_name = reqwest::header::HeaderName::from_bytes(key.as_bytes())?;
                let header_value = reqwest::header::HeaderValue::from_str(value)?;
                header_map.insert(header_name, header_value);
            }
            client_builder = client_builder.default_headers(header_map);
        }
        
        let client = client_builder.build()?;

        // Create ApiClient for advanced features if needed
        let auth = AuthMethod::BearerToken(api_key.clone());
        let mut api_client =
            ApiClient::with_timeout(host.clone(), auth, std::time::Duration::from_secs(timeout_secs))?;

        // Add custom headers to ApiClient if present
        if let Some(headers) = &config.headers {
            let mut header_map = reqwest::header::HeaderMap::new();
            for (key, value) in headers {
                let header_name = reqwest::header::HeaderName::from_bytes(key.as_bytes())?;
                let header_value = reqwest::header::HeaderValue::from_str(value)?;
                header_map.insert(header_name, header_value);
            }
            api_client = api_client.with_headers(header_map)?;
        }

        Ok(Self {
            client,
            api_client: Some(api_client),
            host,
            base_path,
            api_key,
            organization: None,
            project: None,
            model,
            custom_headers: config.headers,
            supports_streaming: config.supports_streaming.unwrap_or(true),
        })
    }

    /// Helper function to add OpenAI-specific headers to a request
    fn add_headers(&self, mut request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        // Add organization header if present
        if let Some(org) = &self.organization {
            request = request.header("OpenAI-Organization", org);
        }

        // Add project header if present
        if let Some(project) = &self.project {
            request = request.header("OpenAI-Project", project);
        }

        // Add custom headers if present
        if let Some(custom_headers) = &self.custom_headers {
            for (key, value) in custom_headers {
                request = request.header(key, value);
            }
        }

        request
    }

    async fn post(&self, payload: &Value) -> Result<Response, ProviderError> {
        let base_url = url::Url::parse(&self.host)
            .map_err(|e| ProviderError::RequestFailed(format!("Invalid base URL: {e}")))?;
        let url = base_url.join(&self.base_path).map_err(|e| {
            ProviderError::RequestFailed(format!("Failed to construct endpoint URL: {e}"))
        })?;

        let request = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key));

        let request = self.add_headers(request);

        Ok(request.json(&payload).send().await?)
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::with_models(
            "openai",
            "OpenAI",
            "GPT-4 and other OpenAI models, including OpenAI compatible ones",
            OPEN_AI_DEFAULT_MODEL,
            vec![
                ModelInfo::new("gpt-4o", 128000),
                ModelInfo::new("gpt-4o-mini", 128000),
                ModelInfo::new("gpt-4-turbo", 128000),
                ModelInfo::new("gpt-3.5-turbo", 16385),
                ModelInfo::new("o1", 200000),
                ModelInfo::new("o3", 200000),
                ModelInfo::new("o4-mini", 128000),
            ],
            OPEN_AI_DOC_URL,
            vec![
                ConfigKey::new("OPENAI_API_KEY", true, true, None),
                ConfigKey::new("OPENAI_HOST", true, false, Some("https://api.openai.com")),
                ConfigKey::new("OPENAI_BASE_PATH", true, false, Some("v1/chat/completions")),
                ConfigKey::new("OPENAI_ORGANIZATION", false, false, None),
                ConfigKey::new("OPENAI_PROJECT", false, false, None),
                ConfigKey::new("OPENAI_CUSTOM_HEADERS", false, true, None),
                ConfigKey::new("OPENAI_TIMEOUT", false, false, Some("600")),
            ],
        )
    }

    fn get_model_config(&self) -> ModelConfig {
        self.model.clone()
    }

    #[tracing::instrument(
        skip(self, system, messages, tools),
        fields(model_config, input, output, input_tokens, output_tokens, total_tokens)
    )]
    async fn complete(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        let payload = create_request(&self.model, system, messages, tools, &ImageFormat::OpenAi)?;

        // Make request
        let response = handle_response_openai_compat(self.post(&payload).await?).await?;

        // Parse response
        let message = response_to_message(&response)?;
        let usage = response.get("usage").map(get_usage).unwrap_or_else(|| {
            tracing::debug!("Failed to get usage data");
            Usage::default()
        });
        let model = get_model(&response);
        emit_debug_trace(&self.model, &payload, &response, &usage);
        Ok((message, ProviderUsage::new(model, usage)))
    }

    /// Fetch supported models from OpenAI; returns Err on any failure, Ok(None) if no data
    async fn fetch_supported_models_async(&self) -> Result<Option<Vec<String>>, ProviderError> {
        // List available models via OpenAI API
        let base_url =
            url::Url::parse(&self.host).map_err(|e| ProviderError::RequestFailed(e.to_string()))?;
        let url = base_url
            .join(&self.base_path.replace("v1/chat/completions", "v1/models"))
            .map_err(|e| ProviderError::RequestFailed(e.to_string()))?;
        let mut request = self.client.get(url).bearer_auth(&self.api_key);
        if let Some(org) = &self.organization {
            request = request.header("OpenAI-Organization", org);
        }
        if let Some(project) = &self.project {
            request = request.header("OpenAI-Project", project);
        }
        if let Some(headers) = &self.custom_headers {
            for (key, value) in headers {
                request = request.header(key, value);
            }
        }
        let response = request.send().await?;
        let json: serde_json::Value = response.json().await?;
        if let Some(err_obj) = json.get("error") {
            let msg = err_obj
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            return Err(ProviderError::Authentication(msg.to_string()));
        }
        let data = json.get("data").and_then(|v| v.as_array()).ok_or_else(|| {
            ProviderError::UsageError("Missing data field in JSON response".into())
        })?;
        let mut models: Vec<String> = data
            .iter()
            .filter_map(|m| m.get("id").and_then(|v| v.as_str()).map(str::to_string))
            .collect();
        models.sort();
        Ok(Some(models))
    }



    fn supports_streaming(&self) -> bool {
        self.supports_streaming
    }

    async fn stream(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        let mut payload =
            create_request(&self.model, system, messages, tools, &ImageFormat::OpenAi)?;
        payload["stream"] = serde_json::Value::Bool(true);
        payload["stream_options"] = json!({
            "include_usage": true,
        });

        let response = handle_status_openai_compat(self.post(&payload).await?).await?;

        let stream = response.bytes_stream().map_err(io::Error::other);

        let model_config = self.model.clone();
        // Wrap in a line decoder and yield lines inside the stream
        Ok(Box::pin(try_stream! {
            let stream_reader = StreamReader::new(stream);
            let framed = FramedRead::new(stream_reader, LinesCodec::new()).map_err(anyhow::Error::from);

            let message_stream = response_to_streaming_message(framed);
            pin!(message_stream);
            while let Some(message) = message_stream.next().await {
                let (message, usage) = message.map_err(|e| ProviderError::RequestFailed(format!("Stream decode error: {}", e)))?;
                super::utils::emit_debug_trace(&model_config, &payload, &message, &usage.as_ref().map(|f| f.usage).unwrap_or_default());
                yield (message, usage);
            }
        }))
    }
}

fn parse_custom_headers(s: String) -> HashMap<String, String> {
    s.split(',')
        .filter_map(|header| {
            let mut parts = header.splitn(2, '=');
            let key = parts.next().map(|s| s.trim().to_string())?;
            let value = parts.next().map(|s| s.trim().to_string())?;
            Some((key, value))
        })
        .collect()
}

#[async_trait]
impl EmbeddingService for OpenAiProvider {
    fn embedding_capabilities(&self) -> Option<EmbeddingCapabilities> {
        Some(EmbeddingCapabilities {
            models: vec![
                EmbeddingModel {
                    name: "text-embedding-3-small".to_string(),
                    dimensions: 1536,
                    max_input_tokens: 8192,
                    cost_per_token: Some(0.00002),
                },
                EmbeddingModel {
                    name: "text-embedding-3-large".to_string(),
                    dimensions: 3072,
                    max_input_tokens: 8192,
                    cost_per_token: Some(0.00013),
                },
                EmbeddingModel {
                    name: "text-embedding-ada-002".to_string(),
                    dimensions: 1536,
                    max_input_tokens: 8192,
                    cost_per_token: Some(0.0001),
                },
            ],
            default_model: "text-embedding-3-small".to_string(),
            max_batch_size: 2048,
            supports_custom_dimensions: true,
        })
    }

    async fn create_embeddings_with_model(
        &self,
        texts: Vec<String>,
        model: &str,
    ) -> Result<EmbeddingResult, ProviderError> {
        if texts.is_empty() {
            let capabilities = EmbeddingService::embedding_capabilities(self).unwrap();
            let model_info = capabilities.models.into_iter()
                .find(|m| m.name == model)
                .unwrap_or_else(|| EmbeddingModel {
                    name: model.to_string(),
                    dimensions: 1536,
                    max_input_tokens: 8192,
                    cost_per_token: None,
                });
            
            return Ok(EmbeddingResult {
                embeddings: vec![],
                model: model_info,
                usage: EmbeddingUsage {
                    tokens: Some(0),
                    embeddings_count: 0,
                },
            });
        }

        let request = EmbeddingRequest {
            input: texts.clone(),
            model: model.to_string(),
        };

        // Construct embeddings endpoint URL
        let base_url = url::Url::parse(&self.host)
            .map_err(|e| ProviderError::RequestFailed(format!("Invalid base URL: {e}")))?;
        let url = base_url
            .join("v1/embeddings")
            .map_err(|e| ProviderError::RequestFailed(format!("Failed to construct embeddings URL: {e}")))?;

        let req = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request);

        let req = self.add_headers(req);

        let response = req
            .send()
            .await
            .map_err(|e| ProviderError::RequestFailed(format!("Failed to send embedding request: {e}")))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(ProviderError::RequestFailed(format!("Embedding API error: {}", error_text)));
        }

        let embedding_response: EmbeddingResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::RequestFailed(format!("Failed to parse embedding response: {e}")))?;

        let embeddings: Vec<Vec<f32>> = embedding_response
            .data
            .into_iter()
            .map(|d| d.embedding)
            .collect();

        // Get model info
        let model_info = self.get_embedding_model_info(model)
            .ok_or_else(|| ProviderError::ExecutionError(format!("Unknown embedding model: {}", model)))?;

        Ok(EmbeddingResult {
            embeddings,
            model: model_info,
            usage: EmbeddingUsage {
                tokens: None, // OpenAI doesn't return token usage for embeddings
                embeddings_count: texts.len(),
            },
        })
    }
}

#[async_trait]
impl EmbeddingCapable for OpenAiProvider {
    async fn create_embeddings(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Get embedding model from env var or use default
        let embedding_model = std::env::var("GOOSE_EMBEDDING_MODEL")
            .unwrap_or_else(|_| "text-embedding-3-small".to_string());

        let request = EmbeddingRequest {
            input: texts,
            model: embedding_model,
        };

        // Construct embeddings endpoint URL
        let base_url =
            url::Url::parse(&self.host).map_err(|e| anyhow::anyhow!("Invalid base URL: {e}"))?;
        let url = base_url
            .join("v1/embeddings")
            .map_err(|e| anyhow::anyhow!("Failed to construct embeddings URL: {e}"))?;

        let req = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request);

        let req = self.add_headers(req);

        let response = req
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send embedding request: {e}"))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Embedding API error: {}", error_text));
        }

        let embedding_response: EmbeddingResponse = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse embedding response: {e}"))?;

        Ok(embedding_response
            .data
            .into_iter()
            .map(|d| d.embedding)
            .collect())
    }
}
