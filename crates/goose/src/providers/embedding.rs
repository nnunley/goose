use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use super::errors::ProviderError;
use std::sync::Arc;
use crate::model::ModelConfig;

// Legacy types for backwards compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub input: Vec<String>,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub data: Vec<EmbeddingData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub embedding: Vec<f32>,
}

// New enhanced embedding types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingModel {
    /// Model identifier (e.g., "text-embedding-3-small")
    pub name: String,
    /// Embedding vector dimensions
    pub dimensions: usize,
    /// Maximum input tokens per request
    pub max_input_tokens: usize,
    /// Cost per token (optional, for usage tracking)
    pub cost_per_token: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingCapabilities {
    /// Available embedding models
    pub models: Vec<EmbeddingModel>,
    /// Default model to use
    pub default_model: String,
    /// Maximum batch size for embedding requests
    pub max_batch_size: usize,
    /// Whether this provider supports custom dimensions
    pub supports_custom_dimensions: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResult {
    /// The embedding vectors
    pub embeddings: Vec<Vec<f32>>,
    /// Model used for embeddings
    pub model: EmbeddingModel,
    /// Usage information
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmbeddingUsage {
    /// Number of tokens processed
    pub tokens: Option<i32>,
    /// Number of embedding vectors created
    pub embeddings_count: usize,
}

/// Enhanced embedding service trait with metadata and model selection
#[async_trait]
pub trait EmbeddingService: Send + Sync {
    /// Get embedding capabilities and models
    fn embedding_capabilities(&self) -> Option<EmbeddingCapabilities>;
    
    /// Check if embeddings are supported
    fn supports_embeddings(&self) -> bool {
        self.embedding_capabilities().is_some()
    }
    
    /// Create embeddings with explicit model selection
    async fn create_embeddings_with_model(
        &self,
        texts: Vec<String>,
        model: &str,
    ) -> Result<EmbeddingResult, ProviderError>;
    
    /// Create embeddings with default model
    async fn create_embeddings(&self, texts: Vec<String>) -> Result<EmbeddingResult, ProviderError> {
        let capabilities = self.embedding_capabilities()
            .ok_or_else(|| ProviderError::ExecutionError("Embeddings not supported".to_string()))?;
        
        self.create_embeddings_with_model(texts, &capabilities.default_model).await
    }
    
    /// Get model info for a specific model name
    fn get_embedding_model_info(&self, model: &str) -> Option<EmbeddingModel> {
        self.embedding_capabilities()?
            .models
            .into_iter()
            .find(|m| m.name == model)
    }
}

/// Legacy trait for backwards compatibility
#[async_trait]
pub trait EmbeddingCapable {
    async fn create_embeddings(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>>;
}


/// Create an embedding service directly from provider name and model config
pub fn create_embedding_service(name: &str, model: ModelConfig) -> Result<Arc<dyn EmbeddingService>> {
    use super::{
        ollama::OllamaProvider,
        openai::OpenAiProvider,
        litellm::LiteLLMProvider,
        databricks::DatabricksProvider,
    };
    
    match name {
        "openai" => Ok(Arc::new(OpenAiProvider::from_env(model)?)),
        "ollama" => Ok(Arc::new(OllamaProvider::from_env(model)?)),
        "litellm" => Ok(Arc::new(LiteLLMProvider::from_env(model)?)),
        "databricks" => Ok(Arc::new(DatabricksProvider::from_env(model)?)),
        _ => Err(anyhow::anyhow!("Provider '{}' does not support embeddings", name)),
    }
}
