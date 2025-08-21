use super::base::{ConfigKey, Provider, ProviderMetadata, ProviderUsage, Usage};
use super::embedding::{EmbeddingCapable, EmbeddingService, EmbeddingCapabilities, EmbeddingModel, EmbeddingResult, EmbeddingUsage};
use super::errors::ProviderError;
use super::utils::{get_model, handle_response_openai_compat};
use crate::config::custom_providers::CustomProviderConfig;
use crate::impl_provider_default;
use crate::conversation::message::Message;
use crate::model::ModelConfig;
use crate::providers::formats::openai::{create_request, get_usage, response_to_message};
use crate::utils::safe_truncate;
use anyhow::Result;
use async_trait::async_trait;
use futures::future::join_all;
use regex::Regex;
use reqwest::Client;
use rmcp::model::Tool;
use serde_json::Value;
use std::time::Duration;
use url::Url;

pub const OLLAMA_HOST: &str = "localhost";
pub const OLLAMA_TIMEOUT: u64 = 600; // seconds
pub const OLLAMA_DEFAULT_PORT: u16 = 11434;
pub const OLLAMA_DEFAULT_MODEL: &str = "qwen3";
// Ollama can run many models, we only provide the default
pub const OLLAMA_KNOWN_MODELS: &[&str] = &[OLLAMA_DEFAULT_MODEL];
pub const OLLAMA_DOC_URL: &str = "https://ollama.com/library";

#[derive(serde::Serialize)]
pub struct OllamaProvider {
    #[serde(skip)]
    client: Client,
    host: String,
    model: ModelConfig,
    supports_streaming: bool,
    embedding_model: Option<String>,
}

impl_provider_default!(OllamaProvider);

impl OllamaProvider {
    pub fn from_env(model: ModelConfig) -> Result<Self> {
        let config = crate::config::Config::global();
        let host: String = config
            .get_param("OLLAMA_HOST")
            .unwrap_or_else(|_| OLLAMA_HOST.to_string());

        // Check for separate embedding model
        let embedding_model = config
            .get_param("GOOSE_EMBEDDINGS_MODEL")
            .ok();

        let timeout: Duration =
            Duration::from_secs(config.get_param("OLLAMA_TIMEOUT").unwrap_or(OLLAMA_TIMEOUT));

        let client = Client::builder().timeout(timeout).build()?;

        Ok(Self {
            client,
            host,
            model,
            supports_streaming: true,
            embedding_model,
        })
    }

    /// Get the base URL for Ollama API calls
    fn get_base_url(&self) -> Result<Url, ProviderError> {
        // OLLAMA_HOST is sometimes just the 'host' or 'host:port' without a scheme
        let base = if self.host.starts_with("http://") || self.host.starts_with("https://") {
            &self.host
        } else {
            &format!("http://{}", self.host)
        };

        let mut base_url = Url::parse(base)
            .map_err(|e| ProviderError::RequestFailed(format!("Invalid base URL: {e}")))?;

        // Set the default port if missing
        // Don't add default port if:
        // 1. URL explicitly ends with standard ports (:80 or :443)
        // 2. URL uses HTTPS (which implicitly uses port 443)
        let explicit_default_port = self.host.ends_with(":80") || self.host.ends_with(":443");
        let is_https = base_url.scheme() == "https";

        if base_url.port().is_none() && !explicit_default_port && !is_https {
            base_url.set_port(Some(OLLAMA_DEFAULT_PORT)).map_err(|_| {
                ProviderError::RequestFailed("Failed to set default port".to_string())
            })?;
        }

        Ok(base_url)
    }

    pub fn from_custom_config(model: ModelConfig, config: CustomProviderConfig) -> Result<Self> {
        let timeout = Duration::from_secs(config.timeout_seconds.unwrap_or(OLLAMA_TIMEOUT));

        // Parse and normalize the custom URL
        let base =
            if config.base_url.starts_with("http://") || config.base_url.starts_with("https://") {
                config.base_url.clone()
            } else {
                format!("http://{}", config.base_url)
            };

        let mut base_url = Url::parse(&base)
            .map_err(|e| anyhow::anyhow!("Invalid base URL '{}': {}", config.base_url, e))?;

        // Set default port if missing and not using standard ports
        let explicit_default_port =
            config.base_url.ends_with(":80") || config.base_url.ends_with(":443");
        let is_https = base_url.scheme() == "https";

        if base_url.port().is_none() && !explicit_default_port && !is_https {
            base_url
                .set_port(Some(OLLAMA_DEFAULT_PORT))
                .map_err(|_| anyhow::anyhow!("Failed to set default port"))?;
        }

        let client = Client::builder().timeout(timeout).build()?;

        Ok(Self {
            client,
            host: base_url.to_string(),
            model,
            supports_streaming: config.supports_streaming.unwrap_or(true),
            embedding_model: None, // Custom config doesn't support separate embedding models yet
        })
    }

    /// Get the embedding model name - either from GOOSE_EMBEDDINGS_MODEL or fallback to chat model
    fn get_embedding_model(&self) -> &str {
        self.embedding_model.as_deref().unwrap_or(&self.model.model_name)
    }

    async fn post(&self, payload: &Value) -> Result<Value, ProviderError> {
        // TODO: remove this later when the UI handles provider config refresh
        let base_url = self.get_base_url()?;

        let url = base_url.join("v1/chat/completions").map_err(|e| {
            ProviderError::RequestFailed(format!("Failed to construct endpoint URL: {e}"))
        })?;

        let response = self.client.post(url).json(payload).send().await?;

        handle_response_openai_compat(response).await
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            "ollama",
            "Ollama",
            "Local open source models",
            OLLAMA_DEFAULT_MODEL,
            OLLAMA_KNOWN_MODELS.to_vec(),
            OLLAMA_DOC_URL,
            vec![
                ConfigKey::new("OLLAMA_HOST", true, false, Some(OLLAMA_HOST)),
                ConfigKey::new(
                    "OLLAMA_TIMEOUT",
                    false,
                    false,
                    Some(&(OLLAMA_TIMEOUT.to_string())),
                ),
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
        let config = crate::config::Config::global();
        let goose_mode = config.get_param("GOOSE_MODE").unwrap_or("auto".to_string());
        let filtered_tools = if goose_mode == "chat" { &[] } else { tools };

        let payload = create_request(
            &self.model,
            system,
            messages,
            filtered_tools,
            &super::utils::ImageFormat::OpenAi,
        )?;
        let response = self.post(&payload).await?;
        let message = response_to_message(&response)?;

        let usage = response.get("usage").map(get_usage).unwrap_or_else(|| {
            tracing::debug!("Failed to get usage data");
            Usage::default()
        });
        let model = get_model(&response);
        super::utils::emit_debug_trace(&self.model, &payload, &response, &usage);
        Ok((message, ProviderUsage::new(model, usage)))
    }

    /// Generate a session name based on the conversation history
    /// This override filters out reasoning tokens that some Ollama models produce
    async fn generate_session_name(&self, messages: &[Message]) -> Result<String, ProviderError> {
        let context = self.get_initial_user_messages(messages);
        let message = Message::user().with_text(self.create_session_name_prompt(&context));
        let result = self
            .complete(
                "You are a title generator. Output only the requested title of 4 words or less, with no additional text, reasoning, or explanations.",
                &[message],
                &[],
            )
            .await?;

        let mut description = result.0.as_concat_text();
        description = Self::filter_reasoning_tokens(&description);

        Ok(safe_truncate(&description, 100))
    }

    fn supports_streaming(&self) -> bool {
        self.supports_streaming
    }
}

impl OllamaProvider {
    /// Filter out reasoning tokens and thinking patterns from model responses
    fn filter_reasoning_tokens(text: &str) -> String {
        let mut filtered = text.to_string();

        // Remove common reasoning patterns
        let reasoning_patterns = [
            r"<think>.*?</think>",
            r"<thinking>.*?</thinking>",
            r"Let me think.*?\n",
            r"I need to.*?\n",
            r"First, I.*?\n",
            r"Okay, .*?\n",
            r"So, .*?\n",
            r"Well, .*?\n",
            r"Hmm, .*?\n",
            r"Actually, .*?\n",
            r"Based on.*?I think",
            r"Looking at.*?I would say",
        ];

        for pattern in reasoning_patterns {
            if let Ok(re) = Regex::new(pattern) {
                filtered = re.replace_all(&filtered, "").to_string();
            }
        }
        // Remove any remaining thinking markers
        filtered = filtered
            .replace("<think>", "")
            .replace("</think>", "")
            .replace("<thinking>", "")
            .replace("</thinking>", "");
        // Clean up extra whitespace
        filtered = filtered
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

        filtered
    }
}

#[async_trait]
impl EmbeddingService for OllamaProvider {
    fn embedding_capabilities(&self) -> Option<EmbeddingCapabilities> {
        // For Ollama, we'll create a basic capability based on the embedding model
        // In the future, this could query the Ollama API for available embedding models
        let embedding_model = self.get_embedding_model();
        
        // We could detect dimensions dynamically, but for now use known values
        // to avoid blocking calls in this sync method
        let dimensions = match embedding_model.as_ref() {
            "nomic-embed-text" => 768,
            "mxbai-embed-large" => 1024,
            _ => 1024, // Default assumption - will be detected dynamically in create_embeddings
        };
        
        Some(EmbeddingCapabilities {
            models: vec![
                EmbeddingModel {
                    name: embedding_model.to_string(),
                    dimensions,
                    max_input_tokens: 2048,
                    cost_per_token: None, // Ollama is typically free
                },
            ],
            default_model: embedding_model.to_string(),
            max_batch_size: 32,
            supports_custom_dimensions: false,
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
                    dimensions: 1024,
                    max_input_tokens: 2048,
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

        let embed_url = self.get_base_url()?.join("api/embed").map_err(|_| {
            ProviderError::RequestFailed("Failed to construct endpoint URL".to_string())
        })?;

        let futures = texts
            .iter()
            .map(|text| {
                let payload = serde_json::json!({
                    "model": model,
                    "input": text,
                });

                let client = &self.client;
                let url = embed_url.clone();

                async move {
                    let response = client.post(url).json(&payload).send().await?;

                    let response_json: Value = handle_response_openai_compat(response).await?;
                    if let Some(embedding) = response_json.get("embeddings").and_then(|v| v.as_array()).and_then(|v| v.get(0))
                    {
                        if let Some(embedding_vec) = embedding.as_array() {
                            return Ok(embedding_vec
                                .iter()
                                .filter_map(|v| v.as_f64())
                                .map(|v| v as f32)
                                .collect::<Vec<f32>>());
                        }
                    }
                    Err(ProviderError::RequestFailed(
                        "Invalid embedding response format".to_string(),
                    ))
                }
            })
            .collect::<Vec<_>>();

        let results = join_all(futures).await;
        let mut embeddings = Vec::with_capacity(results.len());

        for result in results {
            embeddings.push(result?);
        }

        // Detect dimensions from first embedding
        let detected_dimensions = embeddings.first()
            .map(|e| e.len())
            .unwrap_or(1024);

        // Create model info with detected dimensions
        let model_info = EmbeddingModel {
            name: model.to_string(),
            dimensions: detected_dimensions,
            max_input_tokens: 2048,
            cost_per_token: None,
        };

        Ok(EmbeddingResult {
            embeddings,
            model: model_info,
            usage: EmbeddingUsage {
                tokens: None, // Ollama doesn't return token usage for embeddings
                embeddings_count: texts.len(),
            },
        })
    }
}

#[async_trait]
impl EmbeddingCapable for OllamaProvider {
    async fn create_embeddings(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let embed_url = self.get_base_url()?.join("api/embed").map_err(|_| {
            anyhow::anyhow!("Failed to construct endpoint URL")
        })?;

        let futures = texts
            .into_iter()
            .map(|text| {
                let payload = serde_json::json!({
                    "model": self.get_embedding_model(),
                    "input": text,
                });

                let client = &self.client;
                let url = embed_url.clone();

                async move {
                    let response = client.post(url).json(&payload).send().await
                        .map_err(|e| anyhow::anyhow!("Request failed: {}", e))?;

                    let response_json: Value = handle_response_openai_compat(response).await
                        .map_err(|e| anyhow::anyhow!("Response handling failed: {}", e))?;
                    
                    if let Some(embedding) = response_json.get("embeddings").and_then(|v| v.as_array()).and_then(|v| v.get(0))
                    {
                        if let Some(embedding_vec) = embedding.as_array() {
                            return Ok(embedding_vec
                                .iter()
                                .filter_map(|v| v.as_f64())
                                .map(|v| v as f32)
                                .collect::<Vec<f32>>());
                        }
                    }
                    Err(anyhow::anyhow!("Invalid embedding response format"))
                }
            })
            .collect::<Vec<_>>();

        let results = join_all(futures).await;
        let mut embeddings = Vec::with_capacity(results.len());

        for result in results {
            embeddings.push(result?);
        }

        Ok(embeddings)
    }
}
