use rmcp::model::Tool;
use rmcp::model::{Content, ErrorCode, ErrorData};

use anyhow::{Result, Context};
use async_trait::async_trait;
use serde::Serialize;
use serde_json::Value;
use std::collections::VecDeque;
use std::env;
use std::sync::Arc;
use dashmap::DashMap;
use tokio::sync::{RwLock, mpsc, oneshot};

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
}

// Actor messages for VectorToolSelector
#[derive(Debug)]
pub enum VectorToolMessage {
    SelectTools {
        params: Value,
        response: oneshot::Sender<Result<Vec<Content>, ErrorData>>,
    },
    IndexTools {
        tools: Vec<Tool>,
        extension_name: String,
        response: oneshot::Sender<Result<(), ErrorData>>,
    },
    RemoveTool {
        tool_name: String,
        response: oneshot::Sender<Result<(), ErrorData>>,
    },
    RecordToolCall {
        tool_name: String,
        response: oneshot::Sender<Result<(), ErrorData>>,
    },
    GetRecentToolCalls {
        limit: usize,
        response: oneshot::Sender<Result<Vec<String>, ErrorData>>,
    },
}

pub struct VectorToolActor {
    vector_db: Box<dyn VectorDBAdapter>,
    embedding_service: Arc<dyn EmbeddingService>,
    recent_tool_calls: VecDeque<String>,
    receiver: mpsc::Receiver<VectorToolMessage>,
}

impl VectorToolActor {
    pub fn new(
        vector_db: Box<dyn VectorDBAdapter>,
        embedding_service: Arc<dyn EmbeddingService>,
        receiver: mpsc::Receiver<VectorToolMessage>,
    ) -> Self {
        Self {
            vector_db,
            embedding_service,
            recent_tool_calls: VecDeque::with_capacity(100),
            receiver,
        }
    }

    pub async fn run(mut self) {
        while let Some(msg) = self.receiver.recv().await {
            match msg {
                VectorToolMessage::SelectTools { params, response } => {
                    let result = self.handle_select_tools(params).await;
                    let _ = response.send(result);
                }
                VectorToolMessage::IndexTools {
                    tools,
                    extension_name,
                    response,
                } => {
                    let result = self.handle_index_tools(&tools, &extension_name).await;
                    let _ = response.send(result);
                }
                VectorToolMessage::RemoveTool { tool_name, response } => {
                    let result = self.handle_remove_tool(&tool_name).await;
                    let _ = response.send(result);
                }
                VectorToolMessage::RecordToolCall { tool_name, response } => {
                    let result = self.handle_record_tool_call(&tool_name).await;
                    let _ = response.send(result);
                }
                VectorToolMessage::GetRecentToolCalls { limit, response } => {
                    let result = self.handle_get_recent_tool_calls(limit).await;
                    let _ = response.send(result);
                }
            }
        }
    }

    async fn handle_select_tools(&self, params: Value) -> Result<Vec<Content>, ErrorData> {
        let query = params
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ErrorData::new(ErrorCode::INVALID_PARAMS, "Missing 'query' parameter".to_string(), None))?;

        let k = params.get("k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let extension_name = params.get("extension_name").and_then(|v| v.as_str());

        if !self.embedding_service.supports_embeddings() {
            return Err(ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                "Embedding provider does not support embeddings".to_string(),
                None,
            ));
        }

        let embedding_result = self
            .embedding_service
            .create_embeddings(vec![query.to_string()])
            .await
            .map_err(|e| {
                ErrorData::new(
                    ErrorCode::INTERNAL_ERROR,
                    format!("Failed to generate query embedding: {}", e),
                    None,
                )
            })?;

        let query_embedding = embedding_result
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| ErrorData::new(ErrorCode::INTERNAL_ERROR, "No embedding returned".to_string(), None))?;

        let tools = self
            .vector_db
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

    async fn handle_index_tools(&self, tools: &[Tool], extension_name: &str) -> Result<(), ErrorData> {
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
            return Err(ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                "Embedding provider does not support embeddings".to_string(),
                None,
            ));
        }

        let embedding_result = self
            .embedding_service
            .create_embeddings(texts_to_embed)
            .await
            .map_err(|e| {
                ErrorData::new(
                    ErrorCode::INTERNAL_ERROR,
                    format!("Failed to generate tool embeddings: {}", e),
                    None,
                )
            })?;

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

        // Filter out tools that already exist in the database
        let mut new_tool_records = Vec::new();
        for record in tool_records {
            let existing_tools = self
                .vector_db
                .search_tools(record.vector.clone(), 1, Some(&record.extension_name))
                .await
                .map_err(|e| {
                    ErrorData::new(
                        ErrorCode::INTERNAL_ERROR,
                        format!("Failed to search for existing tools: {}", e),
                        None,
                    )
                })?;

            if !existing_tools.iter().any(|t| t.tool_name == record.tool_name) {
                new_tool_records.push(record);
            }
        }

        if !new_tool_records.is_empty() {
            self.vector_db.index_tools(new_tool_records).await.map_err(|e| {
                ErrorData::new(
                    ErrorCode::INTERNAL_ERROR,
                    format!("Failed to index tools: {}", e),
                    None,
                )
            })?
        }

        Ok(())
    }

    async fn handle_remove_tool(&self, tool_name: &str) -> Result<(), ErrorData> {
        self.vector_db.remove_tool(tool_name).await.map_err(|e| {
            ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                format!("Failed to remove tool {}: {}", tool_name, e),
                None,
            )
        })?;
        Ok(())
    }

    async fn handle_record_tool_call(&mut self, tool_name: &str) -> Result<(), ErrorData> {
        if self.recent_tool_calls.len() >= 100 {
            self.recent_tool_calls.pop_front();
        }
        self.recent_tool_calls.push_back(tool_name.to_string());
        Ok(())
    }

    async fn handle_get_recent_tool_calls(&self, limit: usize) -> Result<Vec<String>, ErrorData> {
        Ok(self
            .recent_tool_calls
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect())
    }
}

pub struct VectorToolSelector {
    sender: mpsc::Sender<VectorToolMessage>,
}

impl VectorToolSelector {
    pub async fn new(embedding_service: Arc<dyn EmbeddingService>, table_name: String) -> Result<Self> {
        let vector_db = create_vector_db_adapter(Some(table_name)).await?;
        let (sender, receiver) = mpsc::channel(100);
        
        // Spawn the actor task
        let actor = VectorToolActor::new(vector_db, embedding_service, receiver);
        tokio::spawn(actor.run());

        Ok(Self { sender })
    }
}

#[async_trait]
impl RouterToolSelector for VectorToolSelector {
    async fn select_tools(&self, params: Value) -> Result<Vec<Content>, ErrorData> {
        let (response_sender, response_receiver) = oneshot::channel();
        
        if let Err(_) = self
            .sender
            .send(VectorToolMessage::SelectTools {
                params,
                response: response_sender,
            })
            .await
        {
            return Err(ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                "Vector tool selector actor is not available".to_string(),
                None,
            ));
        }

        response_receiver
            .await
            .map_err(|_| {
                ErrorData::new(
                    ErrorCode::INTERNAL_ERROR,
                    "Failed to receive response from vector tool selector".to_string(),
                    None,
                )
            })?
    }

    async fn index_tools(&self, tools: &[Tool], extension_name: &str) -> Result<(), ErrorData> {
        let (response_sender, response_receiver) = oneshot::channel();
        
        if let Err(_) = self
            .sender
            .send(VectorToolMessage::IndexTools {
                tools: tools.to_vec(),
                extension_name: extension_name.to_string(),
                response: response_sender,
            })
            .await
        {
            return Err(ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                "Vector tool selector actor is not available".to_string(),
                None,
            ));
        }

        response_receiver
            .await
            .map_err(|_| {
                ErrorData::new(
                    ErrorCode::INTERNAL_ERROR,
                    "Failed to receive response from vector tool selector".to_string(),
                    None,
                )
            })?
    }

    async fn remove_tool(&self, tool_name: &str) -> Result<(), ErrorData> {
        let (response_sender, response_receiver) = oneshot::channel();
        
        if let Err(_) = self
            .sender
            .send(VectorToolMessage::RemoveTool {
                tool_name: tool_name.to_string(),
                response: response_sender,
            })
            .await
        {
            return Err(ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                "Vector tool selector actor is not available".to_string(),
                None,
            ));
        }

        response_receiver
            .await
            .map_err(|_| {
                ErrorData::new(
                    ErrorCode::INTERNAL_ERROR,
                    "Failed to receive response from vector tool selector".to_string(),
                    None,
                )
            })?
    }

    async fn record_tool_call(&self, tool_name: &str) -> Result<(), ErrorData> {
        let (response_sender, response_receiver) = oneshot::channel();
        
        if let Err(_) = self
            .sender
            .send(VectorToolMessage::RecordToolCall {
                tool_name: tool_name.to_string(),
                response: response_sender,
            })
            .await
        {
            return Err(ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                "Vector tool selector actor is not available".to_string(),
                None,
            ));
        }

        response_receiver
            .await
            .map_err(|_| {
                ErrorData::new(
                    ErrorCode::INTERNAL_ERROR,
                    "Failed to receive response from vector tool selector".to_string(),
                    None,
                )
            })?
    }

    async fn get_recent_tool_calls(&self, limit: usize) -> Result<Vec<String>, ErrorData> {
        let (response_sender, response_receiver) = oneshot::channel();
        
        if let Err(_) = self
            .sender
            .send(VectorToolMessage::GetRecentToolCalls {
                limit,
                response: response_sender,
            })
            .await
        {
            return Err(ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                "Vector tool selector actor is not available".to_string(),
                None,
            ));
        }

        response_receiver
            .await
            .map_err(|_| {
                ErrorData::new(
                    ErrorCode::INTERNAL_ERROR,
                    "Failed to receive response from vector tool selector".to_string(),
                    None,
                )
            })?
    }
}

pub struct LLMToolSelector {
    llm_provider: Arc<dyn Provider>,
    tool_strings: Arc<DashMap<String, String>>, // extension_name -> tool_string
    recent_tool_calls: Arc<RwLock<VecDeque<String>>>,
}

impl LLMToolSelector {
    pub async fn new(provider: Arc<dyn Provider>) -> Result<Self> {
        Ok(Self {
            llm_provider: provider.clone(),
            tool_strings: Arc::new(DashMap::new()),
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
        let relevant_tools = if let Some(ext) = &extension_name {
            self.tool_strings.get(ext).map(|entry| entry.value().clone())
        } else {
            // If no extension specified, use all tools
            let all_tools: Vec<String> = self.tool_strings
                .iter()
                .map(|entry| entry.value().clone())
                .collect();
            if all_tools.is_empty() {
                None
            } else {
                Some(all_tools.join("\n"))
            }
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

            // Use DashMap's entry API to update or insert
            self.tool_strings
                .entry(extension_name.to_string())
                .and_modify(|existing_tools| {
                    // Check if this tool already exists in the entry
                    if !existing_tools.contains(&format!("Tool: {}", tool.name)) {
                        if !existing_tools.is_empty() {
                            existing_tools.push_str("\n\n");
                        }
                        existing_tools.push_str(&tool_string);
                    }
                })
                .or_insert(tool_string);
        }

        Ok(())
    }
    
    async fn remove_tool(&self, tool_name: &str) -> Result<(), ErrorData> {
        if let Some(extension_name) = tool_name.split("__").next() {
            self.tool_strings.remove(extension_name);
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
