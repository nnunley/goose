#[cfg(test)]
mod ollama_vectordb_integration_tests {
    use anyhow::Result;
    use goose::agents::tool_vectordb::{ToolRecord, ToolVectorDB};
    use goose::config::Config;
    use goose::providers::base::Provider;
    use goose::providers::embedding::EmbeddingService;
    use goose::providers::ollama::OllamaProvider;
    use goose::model::ModelConfig;
    use tempfile::tempdir;
    use temp_env;

    fn is_ollama_available() -> bool {
        let config = Config::global();
        if let Ok(host) = config.get_param::<String>("OLLAMA_HOST") {
            let client = reqwest::blocking::Client::new();
            if let Ok(response) = client.get(format!("{}/api/version", host)).send() {
                return response.status().is_success();
            }
        }
        false
    }

    #[tokio::test]
    async fn test_ollama_tool_vectordb_integration() -> Result<()> {
        if !is_ollama_available() {
            eprintln!("Skipping Ollama VectorDB integration test - Ollama server not available");
            return Ok(());
        }

        let temp_dir = tempdir()?;
        let db_path = temp_dir.path().join("test_ollama_tools.db");

        temp_env::with_vars(
            vec![
                ("OLLAMA_HOST", Some("http://localhost:11434")),
                ("GOOSE_EMBEDDING_MODEL", Some("nomic-embed-text:latest")),
                ("GOOSE_EMBEDDING_MODEL_PROVIDER", Some("ollama")),
                ("GOOSE_VECTOR_DB_PATH", Some(db_path.to_str().unwrap())),
            ],
            || async {
                // Create Ollama provider
                let model_config = ModelConfig::default();
                let provider = OllamaProvider::from_env(model_config)?;
                
                // Create vector database
                let vectordb = ToolVectorDB::new(Some("ollama_test_tools".to_string())).await?;
                
                // Create some test tools
                let tools = vec![
                    ToolRecord {
                        tool_name: "file_reader".to_string(),
                        description: "Reads content from files on the filesystem".to_string(),
                        schema: r#"{"type": "object", "properties": {"path": {"type": "string"}}}"#.to_string(),
                        vector: vec![], // Will be filled by indexing
                        extension_name: "filesystem".to_string(),
                    },
                    ToolRecord {
                        tool_name: "web_search".to_string(),
                        description: "Searches the web for information using a search engine".to_string(),
                        schema: r#"{"type": "object", "properties": {"query": {"type": "string"}}}"#.to_string(),
                        vector: vec![], // Will be filled by indexing
                        extension_name: "web".to_string(),
                    },
                    ToolRecord {
                        tool_name: "calculator".to_string(),
                        description: "Performs mathematical calculations and arithmetic operations".to_string(),
                        schema: r#"{"type": "object", "properties": {"expression": {"type": "string"}}}"#.to_string(),
                        vector: vec![], // Will be filled by indexing
                        extension_name: "math".to_string(),
                    },
                ];
                
                // Index tools with Ollama embeddings
                vectordb.index_tools(tools).await?;
                
                // Test searching for tools
                let query = "I need to read a file from disk";
                let query_embedding = provider.create_embeddings(vec![query.to_string()]).await?;
                let results = vectordb.search_tools(
                    query_embedding.embeddings[0].clone(),
                    2,
                    None,
                ).await?;
                
                // Verify results
                assert!(!results.is_empty(), "Should find some tools");
                assert_eq!(results[0].tool_name, "file_reader", "File reader should be the top result");
                
                // Test with extension filter
                let web_results = vectordb.search_tools(
                    query_embedding.embeddings[0].clone(),
                    10,
                    Some("web"),
                ).await?;
                
                assert_eq!(web_results.len(), 1, "Should only find web extension tools");
                assert_eq!(web_results[0].tool_name, "web_search");
                
                // Test another query
                let math_query = "calculate the sum of numbers";
                let math_embedding = provider.create_embeddings(vec![math_query.to_string()]).await?;
                let math_results = vectordb.search_tools(
                    math_embedding.embeddings[0].clone(),
                    1,
                    None,
                ).await?;
                
                assert_eq!(math_results[0].tool_name, "calculator", "Calculator should be the top result for math query");
                
                Ok::<(), anyhow::Error>(())
            },
        ).await?;
        
        Ok(())
    }

    #[tokio::test]
    async fn test_ollama_vectordb_migration() -> Result<()> {
        if !is_ollama_available() {
            eprintln!("Skipping Ollama VectorDB migration test - Ollama server not available");
            return Ok(());
        }

        let temp_dir = tempdir()?;
        let db_path = temp_dir.path().join("test_migration.db");

        temp_env::with_vars(
            vec![
                ("OLLAMA_HOST", Some("http://localhost:11434")),
                ("GOOSE_EMBEDDING_MODEL", Some("nomic-embed-text:latest")),
                ("GOOSE_EMBEDDING_MODEL_PROVIDER", Some("ollama")),
                ("GOOSE_VECTOR_DB_PATH", Some(db_path.to_str().unwrap())),
            ],
            || async {
                // Create vector database and add some tools
                let vectordb = ToolVectorDB::new(Some("migration_test".to_string())).await?;
                
                let tool = ToolRecord {
                    tool_name: "test_tool".to_string(),
                    description: "A test tool for migration".to_string(),
                    schema: "{}".to_string(),
                    vector: vec![],
                    extension_name: "test".to_string(),
                };
                
                vectordb.index_tools(vec![tool]).await?;
                
                // Drop and recreate to test migration/persistence
                drop(vectordb);
                
                let vectordb2 = ToolVectorDB::new(Some("migration_test".to_string())).await?;
                
                // Create a query embedding
                let provider = OllamaProvider::from_env(ModelConfig::default())?;
                let query_embedding = provider.create_embeddings(vec!["test".to_string()]).await?;
                
                // Should still find the tool
                let results = vectordb2.search_tools(
                    query_embedding.embeddings[0].clone(),
                    10,
                    None,
                ).await?;
                
                assert_eq!(results.len(), 1, "Should find the persisted tool");
                assert_eq!(results[0].tool_name, "test_tool");
                
                Ok::<(), anyhow::Error>(())
            },
        ).await?;
        
        Ok(())
    }

    #[cfg(feature = "vectordb-sqlite")]
    #[tokio::test]
    async fn test_ollama_sqlite_vectordb() -> Result<()> {
        if !is_ollama_available() {
            eprintln!("Skipping Ollama SQLite VectorDB test - Ollama server not available");
            return Ok(());
        }

        use goose::agents::sqlite_vectordb::{ToolVectorDB, sqlite_impl::ToolRecord as LWToolRecord};

        let temp_dir = tempdir()?;
        let db_path = temp_dir.path().join("test_sqlite.db");

        temp_env::with_vars(
            vec![
                ("OLLAMA_HOST", Some("http://localhost:11434")),
                ("GOOSE_EMBEDDING_MODEL", Some("nomic-embed-text:latest")),
            ],
            || async {
                // Create Ollama provider
                let provider = OllamaProvider::from_env(ModelConfig::default())?;
                
                // Create SQLite vector database
                let dimension = 768; // nomic-embed-text dimension
                let vectordb = ToolVectorDB::new(&db_path, dimension, 1000).await?;
                
                // Create test data
                let tools = vec![
                    ("code_search", "Search for code patterns in the repository"),
                    ("file_write", "Write content to a file"),
                    ("terminal_execute", "Execute commands in the terminal"),
                ];
                
                // Generate embeddings and add to DB
                for (name, desc) in &tools {
                    let embedding_result = provider.create_embeddings(vec![desc.to_string()]).await?;
                    let tool = LWToolRecord {
                        id: format!("tool_{}", name),
                        name: name.to_string(),
                        description: desc.to_string(),
                        embedding: embedding_result.embeddings[0].clone(),
                        metadata: std::collections::HashMap::new(),
                    };
                    vectordb.add_tool(tool).await?;
                }
                
                // Test search
                let query = "I need to find specific code patterns";
                let query_embedding = provider.create_embeddings(vec![query.to_string()]).await?;
                let results = vectordb.search_similar(
                    &query_embedding.embeddings[0],
                    2,
                    None,
                ).await?;
                
                assert!(!results.is_empty(), "Should find tools");
                assert_eq!(results[0].name, "code_search", "Code search should be top result");
                
                // Test stats
                let stats = vectordb.stats().await?;
                assert_eq!(stats["total_tools"], 3);
                assert_eq!(stats["dimension"], dimension);
                
                Ok::<(), anyhow::Error>(())
            },
        ).await?;
        
        Ok(())
    }
}