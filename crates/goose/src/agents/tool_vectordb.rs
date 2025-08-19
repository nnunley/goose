use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use chrono::Local;

// Common ToolRecord structure used by all implementations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolRecord {
    pub tool_name: String,
    pub description: String,
    pub schema: String,
    pub vector: Vec<f32>,
    pub extension_name: String,
}

// Common interface for vector database operations
#[async_trait]
pub trait VectorDBAdapter: Send + Sync {
    async fn index_tools(&self, tools: Vec<ToolRecord>) -> Result<()>;
    async fn search_tools(
        &self,
        query_vector: Vec<f32>,
        k: usize,
        extension_name: Option<&str>,
    ) -> Result<Vec<ToolRecord>>;
    async fn remove_tool(&self, tool_name: &str) -> Result<()>;
    #[cfg(test)]
    async fn clear_tools(&self) -> Result<()>;
}

// ==============================================================================
// SQLite Implementation - HNSW + SQLite
// ==============================================================================
#[cfg(feature = "vectordb-sqlite")]
pub use sqlite_impl::*;

#[cfg(feature = "vectordb-sqlite")]
mod sqlite_impl {
    use super::*;
    use crate::agents::sqlite_vectordb::sqlite_impl::{
        SqliteVectorDB as SqliteDB, 
        ToolRecord as SqliteToolRecord
    };
    use std::collections::HashMap;
    use std::path::Path;

    pub struct SqliteAdapter {
        db: SqliteDB,
    }

    impl SqliteAdapter {
        pub async fn new(table_name: Option<String>) -> Result<Self> {
            let db_path = Self::get_db_path(table_name)?;
            let dimension = 1536; // Default OpenAI embedding dimension
            let max_elements = 10000; // Reasonable default
            
            let db = SqliteDB::new(db_path, dimension, max_elements).await?;
            Ok(Self { db })
        }

        fn get_db_path(table_name: Option<String>) -> Result<std::path::PathBuf> {
            use etcetera::base_strategy::{BaseStrategy, Xdg};
            use anyhow::Context;

            let config = crate::config::Config::global();

            // Check for custom database path override
            if let Ok(custom_path) = config.get_param::<String>("GOOSE_VECTOR_DB_PATH") {
                let mut path = std::path::PathBuf::from(custom_path);
                if let Some(name) = table_name {
                    path = path.join(format!("{}.db", name));
                } else {
                    path = path.join("tools.db");
                }
                return Ok(path);
            }

            // Fall back to default XDG-based path
            let data_dir = Xdg::new()
                .context("Failed to determine base strategy")?
                .data_dir();

            let mut path = data_dir.join("goose").join("tool_db");
            if let Some(name) = table_name {
                path = path.join(format!("{}.db", name));
            } else {
                path = path.join("tools.db");
            }

            Ok(path)
        }
    }

    #[async_trait]
    impl VectorDBAdapter for SqliteAdapter {
        async fn index_tools(&self, tools: Vec<ToolRecord>) -> Result<()> {
            for tool in tools {
                let sqlite_tool = SqliteToolRecord {
                    id: tool.tool_name.clone(),
                    name: tool.tool_name,
                    description: tool.description,
                    embedding: tool.vector,
                    metadata: {
                        let mut metadata = HashMap::new();
                        metadata.insert("schema".to_string(), tool.schema);
                        metadata.insert("extension_name".to_string(), tool.extension_name);
                        metadata
                    },
                };
                self.db.add_tool(sqlite_tool).await?;
            }
            Ok(())
        }

        async fn search_tools(
            &self,
            query_vector: Vec<f32>,
            k: usize,
            extension_name: Option<&str>,
        ) -> Result<Vec<ToolRecord>> {
            let filter = extension_name.map(|ext| {
                let mut filter = HashMap::new();
                filter.insert("extension_name".to_string(), ext.to_string());
                filter
            });

            let results = self.db.search_similar(&query_vector, k, filter).await?;
            
            let mut tools = Vec::new();
            for result in results {
                let schema = result.metadata.get("schema").cloned().unwrap_or_default();
                let extension_name = result.metadata.get("extension_name").cloned().unwrap_or_default();
                
                tools.push(ToolRecord {
                    tool_name: result.name,
                    description: result.description,
                    schema,
                    vector: result.embedding,
                    extension_name,
                });
            }
            
            Ok(tools)
        }

        async fn remove_tool(&self, tool_name: &str) -> Result<()> {
            self.db.remove_tool(tool_name).await?;
            Ok(())
        }

        #[cfg(test)]
        async fn clear_tools(&self) -> Result<()> {
            self.db.clear().await
        }
    }
}

// ==============================================================================
// Factory Functions with Feature Precedence
// ==============================================================================

// Public factory function that creates the appropriate adapter based on feature precedence
pub async fn create_vector_db_adapter(table_name: Option<String>) -> Result<Box<dyn VectorDBAdapter>> {
    
    #[cfg(feature = "vectordb-sqlite")]
    {
        let adapter = sqlite_impl::SqliteAdapter::new(table_name).await?;
        Ok(Box::new(adapter))
    }
    #[cfg(not(feature = "vectordb-sqlite"))]
    {
        Err(anyhow::anyhow!(
            "No vector database feature enabled. This should not happen with default features."
        ))
    }
}

// Convenience function for the default adapter
pub async fn create_default_vector_db() -> Result<Box<dyn VectorDBAdapter>> {
    create_vector_db_adapter(None).await
}

// Generate a unique table ID based on timestamp
pub fn generate_table_id() -> String {
    Local::now().format("%Y%m%d_%H%M%S").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_adapter_creation() -> Result<()> {
        let adapter = create_vector_db_adapter(Some("test_adapter".to_string())).await?;
        
        // Clear any existing data
        adapter.clear_tools().await?;
        
        // Test basic operations
        let test_tools = vec![
            ToolRecord {
                tool_name: "test_tool".to_string(),
                description: "A test tool".to_string(),
                schema: r#"{"type": "object"}"#.to_string(),
                vector: vec![0.1; 1536],
                extension_name: "test".to_string(),
            },
        ];

        adapter.index_tools(test_tools).await?;

        let query_vector = vec![0.1; 1536];
        let results = adapter.search_tools(query_vector, 1, None).await?;
        
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tool_name, "test_tool");

        // Clean up
        adapter.remove_tool("test_tool").await?;

        Ok(())
    }

    // Tests from test_lightweight_vectordb.rs
    #[cfg(feature = "vectordb-sqlite")]
    mod lightweight_vectordb_tests {
        use crate::agents::sqlite_vectordb::sqlite_impl::{SqliteVectorDB, ToolRecord};
        use anyhow::Result;
        use std::collections::HashMap;
        use tempfile::TempDir;

        #[tokio::test]
        async fn test_lightweight_vectordb_basic_operations() -> Result<()> {
            // Create a temporary directory for the test database
            let temp_dir = TempDir::new()?;
            let db_path = temp_dir.path().join("test.db");
            
            // Create a new lightweight vector database
            let db = SqliteVectorDB::new(&db_path, 128, 1000).await?;
            
            // Create a test tool
            let mut metadata = HashMap::new();
            metadata.insert("schema".to_string(), r#"{"type": "object"}"#.to_string());
            metadata.insert("extension_name".to_string(), "test_extension".to_string());
            
            let tool = ToolRecord {
                id: "test_tool_1".to_string(),
                name: "test_tool".to_string(),
                description: "A test tool for vector database".to_string(),
                embedding: vec![0.1; 128], // 128-dimensional test embedding
                metadata,
            };
            
            // Add the tool
            db.add_tool(tool.clone()).await?;
            
            // Search for similar tools
            let query_embedding = vec![0.1; 128]; // Same as our tool for exact match
            let results = db.search_similar(&query_embedding, 5, None).await?;
            
            // Verify we found our tool
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].name, "test_tool");
            assert_eq!(results[0].description, "A test tool for vector database");
            
            // Test filtering by extension
            let mut filter = HashMap::new();
            filter.insert("extension_name".to_string(), "test_extension".to_string());
            let filtered_results = db.search_similar(&query_embedding, 5, Some(filter)).await?;
            assert_eq!(filtered_results.len(), 1);
            
            // Test filtering with non-matching extension
            let mut wrong_filter = HashMap::new();
            wrong_filter.insert("extension_name".to_string(), "wrong_extension".to_string());
            let no_results = db.search_similar(&query_embedding, 5, Some(wrong_filter)).await?;
            assert_eq!(no_results.len(), 0);
            
            // Remove the tool
            db.remove_tool("test_tool_1").await?;
            
            // Verify it's gone
            let empty_results = db.search_similar(&query_embedding, 5, None).await?;
            assert_eq!(empty_results.len(), 0);
            
            Ok(())
        }
        
        #[tokio::test]
        async fn test_lightweight_vectordb_multiple_tools() -> Result<()> {
            let temp_dir = TempDir::new()?;
            let db_path = temp_dir.path().join("test_multi.db");
            
            let db = SqliteVectorDB::new(&db_path, 128, 1000).await?;
            
            // Add multiple tools with different embeddings
            for i in 0..10 {
                let mut metadata = HashMap::new();
                metadata.insert("schema".to_string(), format!(r#"{{"id": {}}}"#, i));
                metadata.insert("extension_name".to_string(), format!("extension_{}", i % 3));
                
                let mut embedding = vec![0.0; 128];
                embedding[i % 128] = 1.0; // Different embeddings for each tool
                
                let tool = ToolRecord {
                    id: format!("tool_{}", i),
                    name: format!("Tool {}", i),
                    description: format!("Description for tool {}", i),
                    embedding,
                    metadata,
                };
                
                db.add_tool(tool).await?;
            }
            
            // Search for all tools
            let query = vec![0.5; 128];
            let results = db.search_similar(&query, 20, None).await?;
            assert_eq!(results.len(), 10);
            
            // Test filtering by extension
            let mut filter = HashMap::new();
            filter.insert("extension_name".to_string(), "extension_0".to_string());
            let filtered = db.search_similar(&query, 20, Some(filter)).await?;
            assert_eq!(filtered.len(), 4); // tools 0, 3, 6, 9
            
            // Clear all tools
            db.clear().await?;
            
            // Verify empty
            let empty = db.search_similar(&query, 20, None).await?;
            assert_eq!(empty.len(), 0);
            
            Ok(())
        }
    }

    // Tests from test_heterogeneous_vectors.rs
    mod heterogeneous_vectors_tests {
        use super::*;
        use crate::providers::embedding::{EmbeddingService, EmbeddingCapabilities, EmbeddingModel, EmbeddingResult, EmbeddingUsage};
        use crate::providers::errors::ProviderError;
        use async_trait::async_trait;
        use anyhow::Result;

        // Input structure for heterogeneous vector tests
        #[derive(Debug, Clone)]
        #[allow(dead_code)]
        pub struct ToolInput {
            pub tool_name: String,
            pub description: String,
            pub schema: String,
            pub extension_name: String,
        }

        // Mock provider that can change dimensions
        struct MockVariableProvider {
            current_dimension: usize,
        }

        #[async_trait]
        impl EmbeddingService for MockVariableProvider {
            fn embedding_capabilities(&self) -> Option<EmbeddingCapabilities> {
                Some(EmbeddingCapabilities {
                    models: vec![
                        EmbeddingModel {
                            name: format!("mock-{}", self.current_dimension),
                            dimensions: self.current_dimension,
                            max_input_tokens: 1000,
                            cost_per_token: None,
                        },
                    ],
                    default_model: format!("mock-{}", self.current_dimension),
                    max_batch_size: 10,
                    supports_custom_dimensions: false,
                })
            }

            async fn create_embeddings_with_model(
                &self,
                texts: Vec<String>,
                model: &str,
            ) -> Result<EmbeddingResult, ProviderError> {
                let embeddings = texts.iter()
                    .map(|_| vec![0.5_f32; self.current_dimension])
                    .collect();

                Ok(EmbeddingResult {
                    embeddings,
                    model: EmbeddingModel {
                        name: model.to_string(),
                        dimensions: self.current_dimension,
                        max_input_tokens: 1000,
                        cost_per_token: None,
                    },
                    usage: EmbeddingUsage {
                        tokens: None,
                        embeddings_count: texts.len(),
                    },
                })
            }
        }


        #[tokio::test]
        async fn test_heterogeneous_vectors_same_session() -> Result<()> {
            println!("\n=== Test 1: Attempting to mix vector sizes in same table ===");
            
            let adapter = create_vector_db_adapter(Some("test_heterogeneous_same".to_string())).await?;
            adapter.clear_tools().await?;

            // First, index tools with 512 dimensions
            let provider_512 = MockVariableProvider { current_dimension: 512 };
            let embedding_result_512 = provider_512.create_embeddings_with_model(
                vec![
                    "tool_512_1 First tool with 512 dimensions {}".to_string(),
                    "tool_512_2 Second tool with 512 dimensions {}".to_string(),
                ],
                "mock-512"
            ).await.map_err(|e| anyhow::anyhow!("Embedding error: {}", e))?;

            let tools_512 = vec![
                ToolRecord {
                    tool_name: "tool_512_1".to_string(),
                    description: "First tool with 512 dimensions".to_string(),
                    schema: r#"{"type": "object"}"#.to_string(),
                    vector: embedding_result_512.embeddings[0].clone(),
                    extension_name: "test".to_string(),
                },
                ToolRecord {
                    tool_name: "tool_512_2".to_string(),
                    description: "Second tool with 512 dimensions".to_string(),
                    schema: r#"{"type": "object"}"#.to_string(),
                    vector: embedding_result_512.embeddings[1].clone(),
                    extension_name: "test".to_string(),
                },
            ];

            println!("Indexing tools with 512 dimensions...");
            adapter.index_tools(tools_512).await?;
            println!("✓ Successfully indexed 512-dimensional tools");

            // Now try to add tools with different dimensions (768)
            let provider_768 = MockVariableProvider { current_dimension: 768 };
            let embedding_result_768 = provider_768.create_embeddings_with_model(
                vec!["tool_768_1 Tool with 768 dimensions {}".to_string()],
                "mock-768"
            ).await.map_err(|e| anyhow::anyhow!("Embedding error: {}", e))?;

            let tools_768 = vec![
                ToolRecord {
                    tool_name: "tool_768_1".to_string(),
                    description: "Tool with 768 dimensions".to_string(),
                    schema: r#"{"type": "object"}"#.to_string(),
                    vector: embedding_result_768.embeddings[0].clone(),
                    extension_name: "test".to_string(),
                },
            ];

            println!("\nAttempting to index tools with 768 dimensions...");
            let result = adapter.index_tools(tools_768).await;
            
            match result {
                Ok(_) => println!("✗ Unexpectedly succeeded - table should reject different dimensions"),
                Err(e) => println!("✓ Expected error: {}", e),
            }

            // Verify we can still search with 512 dimensions
            let query_vec = vec![0.5; 512];
            let results = adapter.search_tools(query_vec, 5, None).await?;
            println!("\nSearch results with 512-d query: {} tools found", results.len());
            for tool in &results {
                println!("  - {}", tool.tool_name);
            }

            Ok(())
        }

        #[tokio::test]
        async fn test_schema_migration_new_table() -> Result<()> {
            println!("\n=== Test 2: Schema changes with new table ===");
            
            // Start with 512 dimensions
            let adapter1 = create_vector_db_adapter(Some("test_migration_v1".to_string())).await?;
            adapter1.clear_tools().await?;

            let provider_512 = MockVariableProvider { current_dimension: 512 };
            let embedding_result_512 = provider_512.create_embeddings_with_model(
                vec!["legacy_tool_1 Tool from v1 with 512 dimensions {}".to_string()],
                "mock-512"
            ).await.map_err(|e| anyhow::anyhow!("Embedding error: {}", e))?;

            let tools_v1 = vec![
                ToolRecord {
                    tool_name: "legacy_tool_1".to_string(),
                    description: "Tool from v1 with 512 dimensions".to_string(),
                    schema: r#"{"type": "object"}"#.to_string(),
                    vector: embedding_result_512.embeddings[0].clone(),
                    extension_name: "v1".to_string(),
                },
            ];

            println!("Creating v1 table with 512 dimensions...");
            adapter1.index_tools(tools_v1).await?;
            println!("✓ v1 table created");

            // Simulate a "migration" by using a different table
            let adapter2 = create_vector_db_adapter(Some("test_migration_v2".to_string())).await?;
            adapter2.clear_tools().await?;

            let provider_768 = MockVariableProvider { current_dimension: 768 };
            let embedding_result_768 = provider_768.create_embeddings_with_model(
                vec!["new_tool_1 Tool from v2 with 768 dimensions {}".to_string()],
                "mock-768"
            ).await.map_err(|e| anyhow::anyhow!("Embedding error: {}", e))?;

            let tools_v2 = vec![
                ToolRecord {
                    tool_name: "new_tool_1".to_string(),
                    description: "Tool from v2 with 768 dimensions".to_string(),
                    schema: r#"{"type": "object"}"#.to_string(),
                    vector: embedding_result_768.embeddings[0].clone(),
                    extension_name: "v2".to_string(),
                },
            ];

            println!("\nCreating v2 table with 768 dimensions...");
            adapter2.index_tools(tools_v2).await?;
            println!("✓ v2 table created");

            // Verify both tables work independently
            let results_v1 = adapter1.search_tools(vec![0.5; 512], 5, None).await?;
            let results_v2 = adapter2.search_tools(vec![0.5; 768], 5, None).await?;

            println!("\nResults:");
            println!("  v1 table (512-d): {} tools", results_v1.len());
            println!("  v2 table (768-d): {} tools", results_v2.len());

            Ok(())
        }

        #[tokio::test]
        async fn test_dimension_detection_and_validation() -> Result<()> {
            println!("\n=== Test 3: Dimension detection and validation ===");
            
            let adapter = create_vector_db_adapter(Some("test_dimension_detection".to_string())).await?;
            adapter.clear_tools().await?;

            // Test 1: Empty table adapts to first dimension
            let provider_1024 = MockVariableProvider { current_dimension: 1024 };
            let embedding_result_1024 = provider_1024.create_embeddings_with_model(
                vec!["tool_1024 Tool with 1024 dimensions {}".to_string()],
                "mock-1024"
            ).await.map_err(|e| anyhow::anyhow!("Embedding error: {}", e))?;

            let tools_1024 = vec![
                ToolRecord {
                    tool_name: "tool_1024".to_string(),
                    description: "Tool with 1024 dimensions".to_string(),
                    schema: r#"{"type": "object"}"#.to_string(),
                    vector: embedding_result_1024.embeddings[0].clone(),
                    extension_name: "test".to_string(),
                },
            ];

            println!("Indexing first tool with 1024 dimensions...");
            adapter.index_tools(tools_1024).await?;
            println!("✓ Table adapted to 1024 dimensions");

            // Test 2: Try different dimension - should fail
            let provider_2048 = MockVariableProvider { current_dimension: 2048 };
            let embedding_result_2048 = provider_2048.create_embeddings_with_model(
                vec!["tool_2048 Tool with 2048 dimensions {}".to_string()],
                "mock-2048"
            ).await.map_err(|e| anyhow::anyhow!("Embedding error: {}", e))?;

            let tools_2048 = vec![
                ToolRecord {
                    tool_name: "tool_2048".to_string(),
                    description: "Tool with 2048 dimensions".to_string(),
                    schema: r#"{"type": "object"}"#.to_string(),
                    vector: embedding_result_2048.embeddings[0].clone(),
                    extension_name: "test".to_string(),
                },
            ];

            println!("\nAttempting to add tool with 2048 dimensions...");
            match adapter.index_tools(tools_2048).await {
                Ok(_) => println!("✗ Should have failed!"),
                Err(e) => println!("✓ Expected error: {}", e),
            }

            // Test 3: Same dimension should work
            let more_embedding_1024 = provider_1024.create_embeddings_with_model(
                vec!["tool_1024_extra Another tool with 1024 dimensions {}".to_string()],
                "mock-1024"
            ).await.map_err(|e| anyhow::anyhow!("Embedding error: {}", e))?;

            let more_tools_1024 = vec![
                ToolRecord {
                    tool_name: "tool_1024_extra".to_string(),
                    description: "Another tool with 1024 dimensions".to_string(),
                    schema: r#"{"type": "object"}"#.to_string(),
                    vector: more_embedding_1024.embeddings[0].clone(),
                    extension_name: "test".to_string(),
                },
            ];

            println!("\nAdding more tools with matching 1024 dimensions...");
            adapter.index_tools(more_tools_1024).await?;
            println!("✓ Successfully added tools with matching dimensions");

            Ok(())
        }

        #[tokio::test]
        async fn test_migration_strategy() -> Result<()> {
            println!("\n=== Test 4: Migration strategy demonstration ===");
            
            // Simulate a real migration scenario
            println!("Scenario: Upgrading from 1536-d OpenAI embeddings to 3072-d embeddings");
            
            // Step 1: Original table with 1536 dimensions
            let adapter_old = create_vector_db_adapter(Some("production_tools".to_string())).await?;
            adapter_old.clear_tools().await?;
            
            let provider_1536 = MockVariableProvider { current_dimension: 1536 };
            let embedding_result_1536 = provider_1536.create_embeddings_with_model(
                vec![
                    "file_reader Reads files from the filesystem {}".to_string(),
                    "http_client Makes HTTP requests {}".to_string(),
                ],
                "mock-1536"
            ).await.map_err(|e| anyhow::anyhow!("Embedding error: {}", e))?;

            let production_tools = vec![
                ToolRecord {
                    tool_name: "file_reader".to_string(),
                    description: "Reads files from the filesystem".to_string(),
                    schema: r#"{"type": "object", "properties": {"path": {"type": "string"}}}"#.to_string(),
                    vector: embedding_result_1536.embeddings[0].clone(),
                    extension_name: "core".to_string(),
                },
                ToolRecord {
                    tool_name: "http_client".to_string(),
                    description: "Makes HTTP requests".to_string(),
                    schema: r#"{"type": "object", "properties": {"url": {"type": "string"}}}"#.to_string(),
                    vector: embedding_result_1536.embeddings[1].clone(),
                    extension_name: "core".to_string(),
                },
            ];

            println!("\n1. Current production table (1536-d):");
            adapter_old.index_tools(production_tools.clone()).await?;
            let results = adapter_old.search_tools(vec![0.5; 1536], 10, None).await?;
            println!("   - {} tools indexed", results.len());

            // Step 2: Create new table with new dimensions
            let adapter_new = create_vector_db_adapter(Some("production_tools_v2".to_string())).await?;
            adapter_new.clear_tools().await?;
            
            let provider_3072 = MockVariableProvider { current_dimension: 3072 };
            let embedding_result_3072 = provider_3072.create_embeddings_with_model(
                vec![
                    "file_reader Reads files from the filesystem {}".to_string(),
                    "http_client Makes HTTP requests {}".to_string(),
                ],
                "mock-3072"
            ).await.map_err(|e| anyhow::anyhow!("Embedding error: {}", e))?;

            let production_tools_new = vec![
                ToolRecord {
                    tool_name: "file_reader".to_string(),
                    description: "Reads files from the filesystem".to_string(),
                    schema: r#"{"type": "object", "properties": {"path": {"type": "string"}}}"#.to_string(),
                    vector: embedding_result_3072.embeddings[0].clone(),
                    extension_name: "core".to_string(),
                },
                ToolRecord {
                    tool_name: "http_client".to_string(),
                    description: "Makes HTTP requests".to_string(),
                    schema: r#"{"type": "object", "properties": {"url": {"type": "string"}}}"#.to_string(),
                    vector: embedding_result_3072.embeddings[1].clone(),
                    extension_name: "core".to_string(),
                },
            ];
            
            println!("\n2. Creating new table with 3072-d embeddings:");
            adapter_new.index_tools(production_tools_new).await?;
            let results_new = adapter_new.search_tools(vec![0.5; 3072], 10, None).await?;
            println!("   - {} tools re-indexed with new dimensions", results_new.len());

            println!("\n3. Migration complete!");
            println!("   - Old table remains functional for rollback");
            println!("   - New table ready with updated embeddings");
            println!("   - Can run both in parallel during transition");

            Ok(())
        }

        #[tokio::test] 
        async fn test_provider_switching_implications() -> Result<()> {
            println!("\n=== Test 5: Provider switching implications ===");
            
            // Test switching between providers with different dimensions
            let adapter = create_vector_db_adapter(Some("test_provider_switch".to_string())).await?;
            adapter.clear_tools().await?;

            // Start with OpenAI-like provider (1536 dimensions)
            let openai_like = MockVariableProvider { current_dimension: 1536 };
            let embedding_result_openai = openai_like.create_embeddings_with_model(
                vec!["calculator Performs mathematical calculations {}".to_string()],
                "mock-1536"
            ).await.map_err(|e| anyhow::anyhow!("Embedding error: {}", e))?;

            let tools = vec![
                ToolRecord {
                    tool_name: "calculator".to_string(),
                    description: "Performs mathematical calculations".to_string(),
                    schema: r#"{"type": "object"}"#.to_string(),
                    vector: embedding_result_openai.embeddings[0].clone(),
                    extension_name: "math".to_string(),
                },
            ];

            println!("1. Indexing with OpenAI-like provider (1536-d)...");
            adapter.index_tools(tools.clone()).await?;
            println!("   ✓ Success");

            // Try to switch to Ollama-like provider (768 dimensions)
            let ollama_like = MockVariableProvider { current_dimension: 768 };
            let embedding_result_ollama = ollama_like.create_embeddings_with_model(
                vec!["calculator Performs mathematical calculations {}".to_string()],
                "mock-768"
            ).await.map_err(|e| anyhow::anyhow!("Embedding error: {}", e))?;

            let tools_ollama = vec![
                ToolRecord {
                    tool_name: "calculator".to_string(),
                    description: "Performs mathematical calculations".to_string(),
                    schema: r#"{"type": "object"}"#.to_string(),
                    vector: embedding_result_ollama.embeddings[0].clone(),
                    extension_name: "math".to_string(),
                },
            ];
            
            println!("\n2. Attempting to index with Ollama-like provider (768-d)...");
            match adapter.index_tools(tools_ollama).await {
                Ok(_) => println!("   ✗ Should have failed!"),
                Err(e) => {
                    println!("   ✓ Cannot switch providers: {}", e);
                    println!("   → Need to create new table or re-index all data");
                }
            }

            // Demonstrate successful search with matching dimensions
            println!("\n3. Search still works with original dimensions:");
            let results = adapter.search_tools(vec![0.5; 1536], 5, None).await?;
            println!("   - Found {} tools", results.len());

            Ok(())
        }
    }
}