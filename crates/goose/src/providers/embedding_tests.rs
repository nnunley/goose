#[cfg(test)]
mod tests {
    use super::super::embedding::*;
    use super::super::errors::ProviderError;
    use async_trait::async_trait;

    // Mock provider for testing
    struct MockEmbeddingService {
        dimensions: usize,
        supports_embeddings: bool,
    }

    #[async_trait]
    impl EmbeddingService for MockEmbeddingService {
        fn embedding_capabilities(&self) -> Option<EmbeddingCapabilities> {
            if !self.supports_embeddings {
                return None;
            }
            
            Some(EmbeddingCapabilities {
                models: vec![
                    EmbeddingModel {
                        name: "mock-small".to_string(),
                        dimensions: self.dimensions,
                        max_input_tokens: 1000,
                        cost_per_token: Some(0.0001),
                    },
                    EmbeddingModel {
                        name: "mock-large".to_string(),
                        dimensions: self.dimensions * 2,
                        max_input_tokens: 2000,
                        cost_per_token: Some(0.0002),
                    },
                ],
                default_model: "mock-small".to_string(),
                max_batch_size: 100,
                supports_custom_dimensions: false,
            })
        }

        async fn create_embeddings_with_model(
            &self,
            texts: Vec<String>,
            model: &str,
        ) -> Result<EmbeddingResult, ProviderError> {
            if !self.supports_embeddings {
                return Err(ProviderError::ExecutionError(
                    "This provider does not support embeddings".to_string(),
                ));
            }

            let model_info = self.get_embedding_model_info(model)
                .ok_or_else(|| ProviderError::ExecutionError(format!("Unknown model: {}", model)))?;

            // Create mock embeddings with the correct dimensions
            let embeddings = texts
                .iter()
                .map(|_| vec![0.1_f32; model_info.dimensions])
                .collect();

            Ok(EmbeddingResult {
                embeddings,
                model: model_info,
                usage: EmbeddingUsage {
                    tokens: Some(texts.len() as i32 * 10),
                    embeddings_count: texts.len(),
                },
            })
        }
    }

    #[test]
    fn test_embedding_model_creation() {
        let model = EmbeddingModel {
            name: "test-model".to_string(),
            dimensions: 1536,
            max_input_tokens: 8192,
            cost_per_token: Some(0.0001),
        };

        assert_eq!(model.name, "test-model");
        assert_eq!(model.dimensions, 1536);
        assert_eq!(model.max_input_tokens, 8192);
        assert_eq!(model.cost_per_token, Some(0.0001));
    }

    #[test]
    fn test_embedding_capabilities() {
        let capabilities = EmbeddingCapabilities {
            models: vec![
                EmbeddingModel {
                    name: "small".to_string(),
                    dimensions: 512,
                    max_input_tokens: 1000,
                    cost_per_token: None,
                },
            ],
            default_model: "small".to_string(),
            max_batch_size: 50,
            supports_custom_dimensions: true,
        };

        assert_eq!(capabilities.models.len(), 1);
        assert_eq!(capabilities.default_model, "small");
        assert_eq!(capabilities.max_batch_size, 50);
        assert!(capabilities.supports_custom_dimensions);
    }

    #[tokio::test]
    async fn test_mock_provider_with_embeddings() {
        let provider = MockEmbeddingService {
            dimensions: 768,
            supports_embeddings: true,
        };

        // Test capabilities
        let caps = provider.embedding_capabilities().unwrap();
        assert_eq!(caps.models.len(), 2);
        assert_eq!(caps.models[0].dimensions, 768);
        assert_eq!(caps.models[1].dimensions, 1536);

        // Test embedding creation with default model
        let texts = vec!["Hello".to_string(), "World".to_string()];
        let result = provider.create_embeddings(texts.clone()).await.unwrap();
        
        assert_eq!(result.embeddings.len(), 2);
        assert_eq!(result.embeddings[0].len(), 768);
        assert_eq!(result.model.name, "mock-small");
        assert_eq!(result.usage.embeddings_count, 2);

        // Test with specific model
        let result = provider
            .create_embeddings_with_model(texts, "mock-large")
            .await
            .unwrap();
        
        assert_eq!(result.embeddings[0].len(), 1536);
        assert_eq!(result.model.name, "mock-large");
    }

    #[tokio::test]
    async fn test_provider_without_embeddings() {
        let provider = MockEmbeddingService {
            dimensions: 768,
            supports_embeddings: false,
        };

        // Test no capabilities
        assert!(provider.embedding_capabilities().is_none());
        assert!(!provider.supports_embeddings());

        // Test embedding creation fails
        let texts = vec!["Hello".to_string()];
        let result = provider.create_embeddings(texts).await;
        
        assert!(result.is_err());
        match result {
            Err(ProviderError::ExecutionError(msg)) => {
                assert!(msg.contains("Embeddings not supported") || msg.contains("does not support embeddings"), "Unexpected error message: {}", msg);
            }
            _ => panic!("Expected ExecutionError, got: {:?}", result),
        }
    }

    #[tokio::test]
    async fn test_unknown_model_error() {
        let provider = MockEmbeddingService {
            dimensions: 768,
            supports_embeddings: true,
        };

        let texts = vec!["Hello".to_string()];
        let result = provider
            .create_embeddings_with_model(texts, "unknown-model")
            .await;

        assert!(result.is_err());
        match result {
            Err(ProviderError::ExecutionError(msg)) => {
                assert!(msg.contains("Unknown model"));
            }
            _ => panic!("Expected ExecutionError for unknown model"),
        }
    }

    #[test]
    fn test_embedding_usage_default() {
        let usage = EmbeddingUsage::default();
        assert_eq!(usage.tokens, None);
        assert_eq!(usage.embeddings_count, 0);
    }
}