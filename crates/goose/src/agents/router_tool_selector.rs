use rmcp::model::Tool;
use rmcp::model::{Content, ErrorCode, ErrorData};

use anyhow::{Result, Context};
use async_trait::async_trait;
use serde::Serialize;
use serde_json::Value;
use std::borrow::Cow;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::env;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::agents::tool_vectordb::{create_vector_db_adapter, VectorDBAdapter};
use crate::conversation::message::Message;
use crate::model::ModelConfig;
use crate::prompt_template::render_global_file;
use crate::providers::{base::Provider, embedding::{EmbeddingService, create_embedding_service}};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterToolSelectionStrategy {
    Llm,
    Vector,
}

#[derive(Serialize)]
struct ToolSelectorContext {
    tools: String,
    query: String,
}

#[async_trait]
pub trait RouterToolSelector: Send + Sync {
    async fn select_tools(&self, params: Value) -> Result<Vec<Content>, ErrorData>;
    async fn index_tools(&self, tools: &[Tool], extension_name: &str) -> Result<(), ErrorData>;
    async fn remove_tool(&self, tool_name: &str) -> Result<(), ErrorData>;
    async fn record_tool_call(&self, tool_name: &str) -> Result<(), ErrorData>;
    async fn get_recent_tool_calls(&self, limit: usize) -> Result<Vec<String>, ErrorData>;
    fn selector_type(&self) -> RouterToolSelectionStrategy;
}

pub struct VectorToolSelector {
    vector_db: Arc<RwLock<Box<dyn VectorDBAdapter>>>,
    embedding_service: Arc<dyn EmbeddingService>,
    recent_tool_calls: Arc<RwLock<VecDeque<String>>>,
}

impl VectorToolSelector {
    pub async fn new(embedding_service: Arc<dyn EmbeddingService>, table_name: String) -> Result<Self> {
        let vector_db = create_vector_db_adapter(Some(table_name)).await?;

        Ok(Self {
            vector_db: Arc::new(RwLock::new(vector_db)),
            embedding_service,
            recent_tool_calls: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
        })
    }
}

#[async_trait]
impl RouterToolSelector for VectorToolSelector {
    async fn select_tools(&self, params: Value) -> Result<Vec<Content>, ErrorData> {
        let query = params
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ErrorData::new(ErrorCode::INVALID_PARAMS, "Missing 'query' parameter".to_string(), None))?;

        let k = params.get("k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

        // Extract extension_name from params if present
        let extension_name = params.get("extension_name").and_then(|v| v.as_str());

        // Check if provider supports embeddings
        if !self.embedding_service.supports_embeddings() {
            return Err(ErrorData::new(ErrorCode::INTERNAL_ERROR, 
                "Embedding provider does not support embeddings".to_string(),
                None
            ));
        }

        let embedding_result = self.embedding_service.create_embeddings(
            vec![query.to_string()]
        )
            .await
            .map_err(|e| {
                ErrorData::new(ErrorCode::INTERNAL_ERROR, format!("Failed to generate query embedding: {}", e), None)
            })?;

        let query_embedding = embedding_result.embeddings
            .into_iter()
            .next()
            .ok_or_else(|| ErrorData::new(ErrorCode::INTERNAL_ERROR, "No embedding returned".to_string(), None))?;

        let vector_db = self.vector_db.read().await;
        let tools = vector_db
            .search_tools(query_embedding, k, extension_name)
            .await
            .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, format!("Failed to search tools: {}", e), None))?;

        let selected_tools: Vec<Content> = tools
            .into_iter()
            .map(|tool| {
                let text = format!(
                    "Tool: {}\nDescription: {}\nSchema: {}",
                    tool.tool_name, tool.description, tool.schema
                );
                Content::text(text)
            })
            .collect();

        Ok(selected_tools)
    }

    async fn index_tools(&self, tools: &[Tool], extension_name: &str) -> Result<(), ErrorData> {
        let texts_to_embed: Vec<String> = tools
            .iter()
            .map(|tool| {
                let schema_str = serde_json::to_string_pretty(&tool.input_schema)
                    .unwrap_or_else(|_| "{}".to_string());
                format!(
                    "{} {} {}",
                    tool.name,
                    tool.description
                        .as_ref()
                        .map(|d| d.as_ref())
                        .unwrap_or_default(),
                    schema_str
                )
            })
            .collect();

        if !self.embedding_service.supports_embeddings() {
            return Err(ErrorData::new(ErrorCode::INTERNAL_ERROR, 
                "Embedding provider does not support embeddings".to_string(),
                None
            ));
        }

        let embedding_result = self.embedding_service.create_embeddings(
            texts_to_embed
        )
            .await
            .map_err(|e| {
                ErrorData::new(ErrorCode::INTERNAL_ERROR, format!("Failed to generate tool embeddings: {}", e), None)
            })?;

        // Create tool records
        let tool_records: Vec<crate::agents::tool_vectordb::ToolRecord> = tools
            .iter()
            .zip(embedding_result.embeddings.into_iter())
            .map(|(tool, vector)| {
                let schema_str = serde_json::to_string_pretty(&tool.input_schema)
                    .unwrap_or_else(|_| "{}".to_string());
                crate::agents::tool_vectordb::ToolRecord {
                    tool_name: tool.name.to_string(),
                    description: tool
                        .description
                        .as_ref()
                        .map(|d| d.to_string())
                        .unwrap_or_default(),
                    schema: schema_str,
                    vector,
                    extension_name: extension_name.to_string(),
                }
            })
            .collect();

        // Get vector_db lock
        let vector_db = self.vector_db.read().await;

        // Filter out tools that already exist in the database
        let mut new_tool_records = Vec::new();
        for record in tool_records {
            // Check if tool exists by searching for it
            let existing_tools = vector_db
                .search_tools(record.vector.clone(), 1, Some(&record.extension_name))
                .await
                .map_err(|e| {
                    ErrorData::new(ErrorCode::INTERNAL_ERROR, format!("Failed to search for existing tools: {}", e), None)
                })?;

            // Only add if no exact match found
            if !existing_tools
                .iter()
                .any(|t| t.tool_name == record.tool_name)
            {
                new_tool_records.push(record);
            }
        }

        // Only index if there are new tools to add
        if !new_tool_records.is_empty() {
            vector_db
                .index_tools(new_tool_records)
                .await
                .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, format!("Failed to index tools: {}", e), None))?;
        }

        Ok(())
    }

    async fn remove_tool(&self, tool_name: &str) -> Result<(), ErrorData> {
        let vector_db = self.vector_db.read().await;
        vector_db.remove_tool(tool_name).await.map_err(|e| {
            ErrorData::new(ErrorCode::INTERNAL_ERROR, format!("Failed to remove tool {}: {}", tool_name, e), None)
        })?;
        Ok(())
    }

    async fn record_tool_call(&self, tool_name: &str) -> Result<(), ErrorData> {
        let mut recent_calls = self.recent_tool_calls.write().await;
        if recent_calls.len() >= 100 {
            recent_calls.pop_front();
        }
        recent_calls.push_back(tool_name.to_string());
        Ok(())
    }

    async fn get_recent_tool_calls(&self, limit: usize) -> Result<Vec<String>, ErrorData> {
        let recent_calls = self.recent_tool_calls.read().await;
        Ok(recent_calls.iter().rev().take(limit).cloned().collect())
    }

    fn selector_type(&self) -> RouterToolSelectionStrategy {
        RouterToolSelectionStrategy::Vector
    }
}

pub struct LLMToolSelector {
    llm_provider: Arc<dyn Provider>,
    tool_strings: Arc<RwLock<HashMap<String, String>>>, // extension_name -> tool_string
    recent_tool_calls: Arc<RwLock<VecDeque<String>>>,
}

impl LLMToolSelector {
    pub async fn new(provider: Arc<dyn Provider>) -> Result<Self> {
        Ok(Self {
            llm_provider: provider.clone(),
            tool_strings: Arc::new(RwLock::new(HashMap::new())),
            recent_tool_calls: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
        })
    }
}

#[async_trait]
impl RouterToolSelector for LLMToolSelector {
    async fn select_tools(&self, params: Value) -> Result<Vec<Content>, ErrorData> {
        let query = params
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ErrorData::new(ErrorCode::INVALID_PARAMS, "Missing 'query' parameter".to_string(), None))?;

        let extension_name = params
            .get("extension_name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Get relevant tool strings based on extension_name
        let tool_strings = self.tool_strings.read().await;
        let relevant_tools = if let Some(ext) = &extension_name {
            tool_strings.get(ext).cloned()
        } else {
            // If no extension specified, use all tools
            Some(
                tool_strings
                    .values()
                    .cloned()
                    .collect::<Vec<String>>()
                    .join("\n"),
            )
        };

        if let Some(tools) = relevant_tools {
            // Use template to generate the prompt
            let context = ToolSelectorContext {
                tools: tools.clone(),
                query: query.to_string(),
            };

            let user_prompt =
                render_global_file("router_tool_selector.md", &context).map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, format!("Failed to render prompt template: {}", e), None))?;

            let user_message = Message::user().with_text(&user_prompt);
            let response = self
                .llm_provider
                .complete("system", &[user_message], &[])
                .await
                .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, format!("Failed to search tools: {}", e), None))?;

            // Extract just the message content from the response
            let (message, _usage) = response;
            let text = message.content[0].as_text().unwrap_or_default();

            // Split the response into individual tool entries
            let tool_entries: Vec<Content> = text
                .split("\n\n")
                .filter(|entry| entry.trim().starts_with("Tool:"))
                .map(|entry| Content::text(entry.trim().to_string()))
                .collect();

            Ok(tool_entries)
        } else {
            Ok(vec![])
        }
    }

    async fn index_tools(&self, tools: &[Tool], extension_name: &str) -> Result<(), ErrorData> {
        let mut tool_strings = self.tool_strings.write().await;

        for tool in tools {
            let tool_string = format!(
                "Tool: {}\nDescription: {}\nSchema: {}",
                tool.name,
                tool.description
                    .as_ref()
                    .map(|d| d.as_ref())
                    .unwrap_or_default(),
                serde_json::to_string_pretty(&tool.input_schema)
                    .unwrap_or_else(|_| "{}".to_string())
            );

            // Use the provided extension_name instead of parsing from tool name
            let entry = tool_strings.entry(extension_name.to_string()).or_default();

            // Check if this tool already exists in the entry
            if !entry.contains(&format!("Tool: {}", tool.name)) {
                if !entry.is_empty() {
                    entry.push_str("\n\n");
                }
                entry.push_str(&tool_string);
            }
        }

        Ok(())
    }
    
    async fn remove_tool(&self, tool_name: &str) -> Result<(), ErrorData> {
        let mut tool_strings = self.tool_strings.write().await;
        if let Some(extension_name) = tool_name.split("__").next() {
            tool_strings.remove(extension_name);
        }
        Ok(())
    }

    async fn record_tool_call(&self, tool_name: &str) -> Result<(), ErrorData> {
        let mut recent_calls = self.recent_tool_calls.write().await;
        if recent_calls.len() >= 100 {
            recent_calls.pop_front();
        }
        recent_calls.push_back(tool_name.to_string());
        Ok(())
    }

    async fn get_recent_tool_calls(&self, limit: usize) -> Result<Vec<String>, ErrorData> {
        let recent_calls = self.recent_tool_calls.read().await;
        Ok(recent_calls.iter().rev().take(limit).cloned().collect())
    }

    fn selector_type(&self) -> RouterToolSelectionStrategy {
        RouterToolSelectionStrategy::Llm
    }
}

// Helper function to create a boxed tool selector
pub async fn create_tool_selector(
    provider: Arc<dyn Provider>,
    strategy: Option<RouterToolSelectionStrategy>,
    table_name: Option<String>,
) -> Result<Box<dyn RouterToolSelector>> {
    match strategy {
        Some(RouterToolSelectionStrategy::Vector) => {
            #[cfg(feature = "vectordb-sqlite")]
            {
            // Create embedding provider for vector search
            let embedding_model = env::var("GOOSE_EMBEDDING_MODEL")
                .unwrap_or_else(|_| "text-embedding-3-small".to_string());
            let embedding_service_name =
                env::var("GOOSE_EMBEDDING_MODEL_PROVIDER").unwrap_or_else(|_| "openai".to_string());

            let model_config = ModelConfig::new(embedding_model.as_str())
                .context("Failed to create model config for embedding service")?;
            let embedding_service = create_embedding_service(&embedding_service_name, model_config)
                .context(format!(
                    "Failed to create {} service for embeddings. If using OpenAI, make sure OPENAI_API_KEY env var is set.",
                    embedding_service_name
                ))?;
            
            let selector = VectorToolSelector::new(embedding_service, table_name.unwrap_or_else(|| "tool_selector".to_string())).await?;
            Ok(Box::new(selector))
            }
            #[cfg(not(feature = "vectordb-sqlite"))]
            {
                anyhow::bail!("Vector tool selection requires the 'vectordb-sqlite' feature to be enabled");
            }
        }
        Some(RouterToolSelectionStrategy::Llm) => {
            let selector = LLMToolSelector::new(provider).await?;
            Ok(Box::new(selector))
        }
        None => {
            let selector = LLMToolSelector::new(provider).await?;
            Ok(Box::new(selector))
        }
    }
}
