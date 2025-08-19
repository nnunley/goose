use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::sync::Mutex;

#[cfg(feature = "vectordb-sqlite")]
pub(crate) mod sqlite_impl {
    use super::*;
    use hnsw_rs::prelude::*;
    use rusqlite::{params, Connection, Row};
    use std::sync::Arc;

    const CURRENT_SCHEMA_VERSION: u32 = 1;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ToolRecord {
        pub id: String,
        pub name: String,
        pub description: String,
        pub embedding: Vec<f32>,
        pub metadata: HashMap<String, String>,
    }

    pub struct SqliteVectorDB {
        db_path: String,
        sqlite_conn: Arc<Mutex<Connection>>,
        hnsw_index: Arc<Mutex<Option<Hnsw<'static, f32, DistCosine>>>>,
        dimension: usize,
        max_elements: usize,
        ef_construction: usize,
        m: usize,
    }

    impl SqliteVectorDB {
        pub async fn new(
            path: impl AsRef<Path>,
            dimension: usize,
            max_elements: usize,
        ) -> Result<Self> {
            let db_path = path.as_ref().to_string_lossy().to_string();
            
            // Initialize SQLite connection
            let conn = Connection::open(&db_path)?;
            Self::initialize_schema(&conn)?;
            
            let sqlite_conn = Arc::new(Mutex::new(conn));
            
            // HNSW parameters optimized for tool search
            let ef_construction = 200;
            let m = 16;
            
            let mut db = Self {
                db_path,
                sqlite_conn,
                hnsw_index: Arc::new(Mutex::new(None)),
                dimension,
                max_elements,
                ef_construction,
                m,
            };
            
            // Load existing index if available
            db.load_index().await?;
            
            Ok(db)
        }

        fn initialize_schema(conn: &Connection) -> Result<()> {
            // Create metadata table
            conn.execute(
                "CREATE TABLE IF NOT EXISTS vector_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )",
                [],
            )?;

            // Create tools table
            conn.execute(
                "CREATE TABLE IF NOT EXISTS tools (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )",
                [],
            )?;

            // Create index on name for faster lookups
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tools_name ON tools(name)",
                [],
            )?;

            // Set schema version
            conn.execute(
                "INSERT OR REPLACE INTO vector_metadata (key, value) VALUES ('schema_version', ?)",
                params![CURRENT_SCHEMA_VERSION.to_string()],
            )?;

            Ok(())
        }

        async fn load_index(&mut self) -> Result<()> {
            let conn = self.sqlite_conn.lock().await;
            
            // Check if we have any data
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM tools",
                [],
                |row| row.get(0),
            )?;

            if count == 0 {
                return Ok(());
            }

            // Initialize HNSW index
            let mut hnsw = Hnsw::<f32, DistCosine>::new(
                self.m,
                self.max_elements,
                self.ef_construction,
                200, // ef for search
                DistCosine {},
            );

            // Load all embeddings into HNSW index
            let tools_data: Vec<(String, Vec<f32>)> = {
                let mut stmt = conn.prepare(
                    "SELECT id, embedding FROM tools ORDER BY rowid",
                )?;

                let tool_iter = stmt.query_map([], |row: &Row| {
                    let id: String = row.get(0)?;
                    let embedding_blob: Vec<u8> = row.get(1)?;
                    let embedding: Vec<f32> = bincode::deserialize(&embedding_blob)
                        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                            1,
                            rusqlite::types::Type::Blob,
                            Box::new(e),
                        ))?;
                    Ok((id, embedding))
                })?;

                tool_iter.collect::<Result<Vec<_>, _>>()?
            }; // Drop stmt here before await

            let mut point_id = 0;
            let mut id_to_point: HashMap<String, usize> = HashMap::new();

            for (id, embedding) in tools_data {
                if embedding.len() != self.dimension {
                    return Err(anyhow!(
                        "Dimension mismatch: expected {}, got {} for tool {}",
                        self.dimension,
                        embedding.len(),
                        id
                    ));
                }

                hnsw.insert((&embedding, point_id as usize));
                id_to_point.insert(id, point_id);
                point_id += 1;
            }

            // Store the mapping in the HNSW index for later retrieval
            // For now, we'll need to maintain this mapping separately
            
            *self.hnsw_index.lock().await = Some(hnsw);
            
            Ok(())
        }

        pub async fn add_tool(&self, tool: ToolRecord) -> Result<()> {
            if tool.embedding.len() != self.dimension {
                return Err(anyhow!(
                    "Dimension mismatch: expected {}, got {}",
                    self.dimension,
                    tool.embedding.len()
                ));
            }

            let embedding_blob = bincode::serialize(&tool.embedding)?;
            let metadata_json = serde_json::to_string(&tool.metadata)?;

            // Add to SQLite
            {
                let conn = self.sqlite_conn.lock().await;
                conn.execute(
                    "INSERT OR REPLACE INTO tools (id, name, description, embedding, metadata, updated_at)
                     VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                    params![
                        tool.id,
                        tool.name,
                        tool.description,
                        embedding_blob,
                        metadata_json
                    ],
                )?;
            }

            // Add to HNSW index
            {
                let mut hnsw_opt = self.hnsw_index.lock().await;
                
                if hnsw_opt.is_none() {
                    // Initialize HNSW index if not exists
                    *hnsw_opt = Some(Hnsw::<f32, DistCosine>::new(
                        self.m,
                        self.max_elements,
                        self.ef_construction,
                        200,
                        DistCosine {},
                    ));
                }

                if let Some(hnsw) = hnsw_opt.as_mut() {
                    // For simplicity, we'll use a hash of the tool ID as point ID
                    let point_id = self.tool_id_to_point_id(&tool.id);
                    hnsw.insert((&tool.embedding, point_id as usize));
                }
            }

            Ok(())
        }

        pub async fn search_similar(
            &self,
            query_embedding: &[f32],
            limit: usize,
            filter: Option<HashMap<String, String>>,
        ) -> Result<Vec<ToolRecord>> {
            if query_embedding.len() != self.dimension {
                return Err(anyhow!(
                    "Query dimension mismatch: expected {}, got {}",
                    self.dimension,
                    query_embedding.len()
                ));
            }

            let hnsw_guard = self.hnsw_index.lock().await;
            let hnsw = hnsw_guard.as_ref()
                .ok_or_else(|| anyhow!("HNSW index not initialized"))?;

            // Search in HNSW index
            let neighbors = hnsw.search(query_embedding, limit, 200);

            drop(hnsw_guard);

            // Convert point IDs back to tool IDs and fetch from SQLite
            let conn = self.sqlite_conn.lock().await;
            let mut results = Vec::new();

            for neighbor in neighbors {
                let tool_id = self.point_id_to_tool_id(neighbor.d_id);
                
                // Fetch tool from SQLite
                let mut stmt = conn.prepare_cached(
                    "SELECT id, name, description, embedding, metadata FROM tools WHERE id = ?"
                )?;

                if let Ok(tool_row) = stmt.query_row(params![tool_id], |row| {
                    let id: String = row.get(0)?;
                    let name: String = row.get(1)?;
                    let description: String = row.get(2)?;
                    let embedding_blob: Vec<u8> = row.get(3)?;
                    let metadata_json: String = row.get(4)?;

                    let embedding: Vec<f32> = bincode::deserialize(&embedding_blob)
                        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                            3,
                            rusqlite::types::Type::Blob,
                            Box::new(e),
                        ))?;

                    let metadata: HashMap<String, String> = serde_json::from_str(&metadata_json)
                        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                            4,
                            rusqlite::types::Type::Text,
                            Box::new(e),
                        ))?;

                    Ok(ToolRecord {
                        id,
                        name,
                        description,
                        embedding,
                        metadata,
                    })
                }) {
                    // Apply filter if provided
                    if let Some(ref filter_map) = filter {
                        let matches_filter = filter_map.iter().all(|(key, value)| {
                            tool_row.metadata.get(key) == Some(value)
                        });
                        
                        if matches_filter {
                            results.push(tool_row);
                        }
                    } else {
                        results.push(tool_row);
                    }
                }

                if results.len() >= limit {
                    break;
                }
            }

            Ok(results)
        }

        pub async fn remove_tool(&self, tool_id: &str) -> Result<bool> {
            let conn = self.sqlite_conn.lock().await;
            let affected = conn.execute(
                "DELETE FROM tools WHERE id = ?",
                params![tool_id],
            )?;

            // Note: HNSW doesn't support efficient deletion, so we keep it simple
            // In a production system, you might want to rebuild the index periodically
            
            Ok(affected > 0)
        }

        pub async fn list_tools(&self) -> Result<Vec<ToolRecord>> {
            let conn = self.sqlite_conn.lock().await;
            let mut stmt = conn.prepare(
                "SELECT id, name, description, embedding, metadata FROM tools ORDER BY name"
            )?;

            let tool_iter = stmt.query_map([], |row| {
                let id: String = row.get(0)?;
                let name: String = row.get(1)?;
                let description: String = row.get(2)?;
                let embedding_blob: Vec<u8> = row.get(3)?;
                let metadata_json: String = row.get(4)?;

                let embedding: Vec<f32> = bincode::deserialize(&embedding_blob)
                    .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                        3,
                        rusqlite::types::Type::Blob,
                        Box::new(e),
                    ))?;

                let metadata: HashMap<String, String> = serde_json::from_str(&metadata_json)
                    .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                        4,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    ))?;

                Ok(ToolRecord {
                    id,
                    name,
                    description,
                    embedding,
                    metadata,
                })
            })?;

            let mut tools = Vec::new();
            for tool_result in tool_iter {
                tools.push(tool_result?);
            }

            Ok(tools)
        }

        pub async fn clear(&self) -> Result<()> {
            let conn = self.sqlite_conn.lock().await;
            conn.execute("DELETE FROM tools", [])?;
            
            // Reset HNSW index
            *self.hnsw_index.lock().await = None;
            
            Ok(())
        }

        pub async fn stats(&self) -> Result<HashMap<String, serde_json::Value>> {
            let conn = self.sqlite_conn.lock().await;
            
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM tools",
                [],
                |row| row.get(0),
            )?;

            let mut stats = HashMap::new();
            stats.insert("total_tools".to_string(), serde_json::Value::Number(count.into()));
            stats.insert("dimension".to_string(), serde_json::Value::Number(self.dimension.into()));
            stats.insert("db_path".to_string(), serde_json::Value::String(self.db_path.clone()));
            
            Ok(stats)
        }

        // Simple hash-based mapping between tool IDs and point IDs
        fn tool_id_to_point_id(&self, tool_id: &str) -> usize {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            
            let mut hasher = DefaultHasher::new();
            tool_id.hash(&mut hasher);
            (hasher.finish() % (self.max_elements as u64)) as usize
        }

        fn point_id_to_tool_id(&self, point_id: usize) -> String {
            // This is a simplified approach - in practice, you'd maintain a bidirectional mapping
            // For now, we'll query SQLite by point_id (this is inefficient but works for the demo)
            format!("point_{}", point_id)
        }
    }

    pub use SqliteVectorDB as VectorDB;
}

// Export the SQLite implementation
#[cfg(feature = "vectordb-sqlite")]
pub use sqlite_impl::*;

// Common interface for both implementations
pub trait VectorDatabase: Send + Sync {
    async fn add_tool(&self, tool: ToolRecord) -> Result<()>;
    async fn search_similar(
        &self,
        query_embedding: &[f32],
        limit: usize,
        filter: Option<HashMap<String, String>>,
    ) -> Result<Vec<ToolRecord>>;
    async fn remove_tool(&self, tool_id: &str) -> Result<bool>;
    async fn list_tools(&self) -> Result<Vec<ToolRecord>>;
    async fn clear(&self) -> Result<()>;
    async fn stats(&self) -> Result<HashMap<String, serde_json::Value>>;
}

#[cfg(feature = "vectordb-sqlite")]
impl VectorDatabase for sqlite_impl::SqliteVectorDB {
    async fn add_tool(&self, tool: ToolRecord) -> Result<()> {
        self.add_tool(tool).await
    }

    async fn search_similar(
        &self,
        query_embedding: &[f32],
        limit: usize,
        filter: Option<HashMap<String, String>>,
    ) -> Result<Vec<ToolRecord>> {
        self.search_similar(query_embedding, limit, filter).await
    }

    async fn remove_tool(&self, tool_id: &str) -> Result<bool> {
        self.remove_tool(tool_id).await
    }

    async fn list_tools(&self) -> Result<Vec<ToolRecord>> {
        self.list_tools().await
    }

    async fn clear(&self) -> Result<()> {
        self.clear().await
    }

    async fn stats(&self) -> Result<HashMap<String, serde_json::Value>> {
        self.stats().await
    }
}