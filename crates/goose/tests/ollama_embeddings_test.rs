#[cfg(test)]
mod ollama_embedding_tests {
    use anyhow::Result;
    use goose::config::Config;
    use goose::providers::base::Provider;
    use goose::providers::embedding::{EmbeddingService, EmbeddingCapabilities};
    use goose::providers::ollama::OllamaProvider;
    use goose::model::ModelConfig;
    use temp_env;

    fn is_ollama_available() -> bool {
        // Check if Ollama is running by trying to connect to it
        let config = Config::global();
        if let Ok(host) = config.get_param::<String>("OLLAMA_HOST") {
            // Try to connect to Ollama's API endpoint
            let client = reqwest::blocking::Client::new();
            if let Ok(response) = client.get(format!("{}/api/version", host)).send() {
                return response.status().is_success();
            }
        }
        false
    }

    #[tokio::test]
    async fn test_ollama_embedding_capabilities() -> Result<()> {
        if !is_ollama_available() {
            eprintln!("Skipping Ollama embedding test - Ollama server not available");
            return Ok(());
        }

        temp_env::with_vars(
            vec![
                ("OLLAMA_HOST", Some("http://localhost:11434")),
                ("GOOSE_EMBEDDING_MODEL", Some("nomic-embed-text:latest")),
            ],
            || async {
                let model_config = ModelConfig::default();
                let provider = OllamaProvider::from_env(model_config)?;
                
                // Test that embeddings are supported
                assert!(provider.supports_embeddings(), "Ollama should support embeddings");
                
                // Get embedding capabilities
                let capabilities = provider.embedding_capabilities()
                    .expect("Should have embedding capabilities");
                
                // Verify capabilities
                assert!(!capabilities.models.is_empty(), "Should have at least one embedding model");
                assert_eq!(capabilities.default_model, "nomic-embed-text:latest");
                assert!(capabilities.max_batch_size > 0);
                
                Ok::<(), anyhow::Error>(())
            },
        ).await?;
        
        Ok(())
    }

    #[tokio::test]
    async fn test_ollama_create_embeddings() -> Result<()> {
        if !is_ollama_available() {
            eprintln!("Skipping Ollama embedding creation test - Ollama server not available");
            return Ok(());
        }

        temp_env::with_vars(
            vec![
                ("OLLAMA_HOST", Some("http://localhost:11434")),
                ("GOOSE_EMBEDDING_MODEL", Some("nomic-embed-text:latest")),
            ],
            || async {
                let model_config = ModelConfig::default();
                let provider = OllamaProvider::from_env(model_config)?;
                
                // Test embedding creation
                let texts = vec![
                    "Hello, world!".to_string(),
                    "This is a test of embeddings".to_string(),
                ];
                
                let result = provider.create_embeddings(texts.clone()).await?;
                
                // Verify results
                assert_eq!(result.embeddings.len(), texts.len(), "Should have embeddings for all texts");
                assert!(!result.embeddings[0].is_empty(), "Embedding vectors should not be empty");
                assert_eq!(result.model.name, "nomic-embed-text:latest");
                assert_eq!(result.usage.embeddings_count, texts.len());
                
                // Verify embedding dimensions match expected
                let expected_dim = 768; // nomic-embed-text dimension
                assert_eq!(
                    result.embeddings[0].len(), 
                    expected_dim, 
                    "Embedding dimension should be {}", 
                    expected_dim
                );
                
                Ok::<(), anyhow::Error>(())
            },
        ).await?;
        
        Ok(())
    }

    #[tokio::test]
    async fn test_ollama_embedding_dimension_detection() -> Result<()> {
        if !is_ollama_available() {
            eprintln!("Skipping Ollama dimension detection test - Ollama server not available");
            return Ok(());
        }

        // Test different models with different dimensions
        let test_cases = vec![
            ("nomic-embed-text:latest", 768),
            ("mxbai-embed-large:latest", 1024),
        ];

        for (model, expected_dim) in test_cases {
            temp_env::with_vars(
                vec![
                    ("OLLAMA_HOST", Some("http://localhost:11434")),
                    ("GOOSE_EMBEDDING_MODEL", Some(model)),
                ],
                || async {
                    let model_config = ModelConfig::default();
                    match OllamaProvider::from_env(model_config) {
                        Ok(provider) => {
                            // Try to create embeddings
                            match provider.create_embeddings(vec!["test".to_string()]).await {
                                Ok(result) => {
                                    assert_eq!(
                                        result.embeddings[0].len(),
                                        expected_dim,
                                        "Model {} should have dimension {}",
                                        model,
                                        expected_dim
                                    );
                                },
                                Err(e) => {
                                    eprintln!("Model {} not available: {}", model, e);
                                }
                            }
                        },
                        Err(e) => {
                            eprintln!("Failed to create provider for model {}: {}", model, e);
                        }
                    }
                    Ok::<(), anyhow::Error>(())
                },
            ).await?;
        }
        
        Ok(())
    }

    #[tokio::test]
    async fn test_ollama_embedding_error_handling() -> Result<()> {
        // Test with invalid host
        temp_env::with_vars(
            vec![
                ("OLLAMA_HOST", Some("http://invalid-host:11434")),
                ("GOOSE_EMBEDDING_MODEL", Some("nomic-embed-text:latest")),
            ],
            || async {
                let model_config = ModelConfig::default();
                let provider = OllamaProvider::from_env(model_config)?;
                
                // This should fail gracefully
                let result = provider.create_embeddings(vec!["test".to_string()]).await;
                assert!(result.is_err(), "Should fail with invalid host");
                
                Ok::<(), anyhow::Error>(())
            },
        ).await?;
        
        Ok(())
    }
}