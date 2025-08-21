use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::sync::Mutex;

#[cfg(feature = "vectordb-sqlite")]
pub(crate) mod sqlite_impl {
    use super::*;
    use hnsw_rs::prelude::*;
    use rusqlite::{params, Connection, Row};
    use r2d2::{Pool, PooledConnection};
    use r2d2_sqlite::SqliteConnectionManager;
    use std::sync::Arc;
    use tokio::sync::{mpsc, oneshot};
    use farmhash;

    const CURRENT_SCHEMA_VERSION: u32 = 1;

    /// Generic storage record used internally by SqliteVectorDB
    /// All specific record types convert to this for storage
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct VectorStorageRecord {
        pub id: String,
        pub content: String,
        pub embedding: Vec<f32>,
        pub metadata: HashMap<String, String>,
    }

    impl super::VectorRecord for VectorStorageRecord {
        fn id(&self) -> &str {
            &self.id
        }

        fn embedding(&self) -> &[f32] {
            &self.embedding
        }

        fn metadata(&self) -> HashMap<String, String> {
            self.metadata.clone()
        }
    }

    /// Legacy ToolRecord for backward compatibility
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ToolRecord {
        pub id: String,
        pub name: String,
        pub description: String,
        pub embedding: Vec<f32>,
        pub metadata: HashMap<String, String>,
    }

    impl super::VectorRecord for ToolRecord {
        fn id(&self) -> &str {
            &self.id
        }

        fn embedding(&self) -> &[f32] {
            &self.embedding
        }

        fn metadata(&self) -> HashMap<String, String> {
            self.metadata.clone()
        }
    }

    /// Convert ToolRecord to VectorStorageRecord
    impl From<ToolRecord> for VectorStorageRecord {
        fn from(tool: ToolRecord) -> Self {
            let mut metadata = tool.metadata;
            metadata.insert("name".to_string(), tool.name);
            metadata.insert("record_type".to_string(), "tool".to_string());
            
            Self {
                id: tool.id,
                content: tool.description,
                embedding: tool.embedding,
                metadata,
            }
        }
    }

    /// Convert VectorStorageRecord back to ToolRecord
    impl From<VectorStorageRecord> for ToolRecord {
        fn from(storage: VectorStorageRecord) -> Self {
            let mut metadata = storage.metadata;
            let name = metadata.remove("name").unwrap_or_else(|| "unknown".to_string());
            metadata.remove("record_type"); // Remove internal metadata
            
            Self {
                id: storage.id,
                name,
                description: storage.content,
                embedding: storage.embedding,
                metadata,
            }
        }
    }

    /// Messages for the HNSW actor
    #[derive(Debug)]
    pub enum HnswMessage {
        Insert {
            embedding: Vec<f32>,
            point_id: usize,
            response: oneshot::Sender<Result<(), anyhow::Error>>,
        },
        Search {
            query: Vec<f32>,
            k: usize,
            response: oneshot::Sender<Result<Vec<(usize, f32)>, anyhow::Error>>,
        },
        Initialize {
            dimension: usize,
            max_elements: usize,
            response: oneshot::Sender<Result<(), anyhow::Error>>,
        },
        LoadFromDb {
            embeddings: Vec<(String, Vec<f32>)>,
            content_to_point_mapping: HashMap<String, usize>,
            response: oneshot::Sender<Result<(), anyhow::Error>>,
        },
    }

    /// Actor managing the HNSW index
    pub struct HnswActor {
        index: Option<Hnsw<'static, f32, DistCosine>>,
        receiver: mpsc::Receiver<HnswMessage>,
        dimension: usize,
        max_elements: usize,
        ef_construction: usize,
        m: usize,
    }

    impl HnswActor {
        pub fn new(
            receiver: mpsc::Receiver<HnswMessage>,
            dimension: usize,
            max_elements: usize,
        ) -> Self {
            Self {
                index: None,
                receiver,
                dimension,
                max_elements,
                ef_construction: 200,
                m: 16,
            }
        }

        pub async fn run(mut self) {
            while let Some(msg) = self.receiver.recv().await {
                match msg {
                    HnswMessage::Initialize { dimension, max_elements, response } => {
                        let result = self.handle_initialize(dimension, max_elements);
                        let _ = response.send(result);
                    }
                    HnswMessage::Insert { embedding, point_id, response } => {
                        let result = self.handle_insert(embedding, point_id);
                        let _ = response.send(result);
                    }
                    HnswMessage::Search { query, k, response } => {
                        let result = self.handle_search(query, k);
                        let _ = response.send(result);
                    }
                    HnswMessage::LoadFromDb { embeddings, content_to_point_mapping, response } => {
                        let result = self.handle_load_from_db(embeddings, content_to_point_mapping);
                        let _ = response.send(result);
                    }
                }
            }
        }

        fn handle_initialize(&mut self, dimension: usize, max_elements: usize) -> Result<(), anyhow::Error> {
            self.dimension = dimension;
            self.max_elements = max_elements;
            
            let index = Hnsw::<f32, DistCosine>::new(
                self.m,
                max_elements,
                self.ef_construction,
                200, // ef for search
                DistCosine {},
            );
            
            self.index = Some(index);
            Ok(())
        }

        fn handle_insert(&mut self, embedding: Vec<f32>, point_id: usize) -> Result<(), anyhow::Error> {
            if let Some(ref mut index) = self.index {
                if embedding.len() != self.dimension {
                    return Err(anyhow!(
                        "Dimension mismatch: expected {}, got {}",
                        self.dimension,
                        embedding.len()
                    ));
                }
                index.insert((&embedding, point_id));
                Ok(())
            } else {
                Err(anyhow!("HNSW index not initialized"))
            }
        }

        fn handle_search(&self, query: Vec<f32>, k: usize) -> Result<Vec<(usize, f32)>, anyhow::Error> {
            if let Some(ref index) = self.index {
                if query.len() != self.dimension {
                    return Err(anyhow!(
                        "Query dimension mismatch: expected {}, got {}",
                        self.dimension,
                        query.len()
                    ));
                }
                
                let neighbors = index.search(&query, k, 200);
                let results = neighbors
                    .into_iter()
                    .map(|neighbor| (neighbor.d_id, neighbor.distance))
                    .collect();
                Ok(results)
            } else {
                Err(anyhow!("HNSW index not initialized"))
            }
        }

        fn handle_load_from_db(
            &mut self,
            embeddings: Vec<(String, Vec<f32>)>,
            _content_to_point_mapping: HashMap<String, usize>,
        ) -> Result<(), anyhow::Error> {
            if self.index.is_none() {
                return Err(anyhow!("HNSW index not initialized"));
            }

            if let Some(ref mut index) = self.index {
                for (content, embedding) in embeddings {
                    if embedding.len() != self.dimension {
                        continue; // Skip mismatched dimensions
                    }
                    
                    // Use content hash as point ID for consistent mapping
                    let point_id = farmhash::hash64(content.as_bytes()) as usize % self.max_elements;
                    index.insert((&embedding, point_id));
                }
            }
            
            Ok(())
        }
    }

    /// Optimized SqliteVectorDB with connection pooling and HNSW actor
    pub struct SqliteVectorDB {
        db_path: String,
        pool: Pool<SqliteConnectionManager>,
        hnsw_sender: mpsc::Sender<HnswMessage>,
        dimension: usize,
        max_elements: usize,
    }

    impl SqliteVectorDB {
        pub async fn new(
            path: impl AsRef<Path>,
            dimension: usize,
            max_elements: usize,
        ) -> Result<Self> {
            let db_path = path.as_ref().to_string_lossy().to_string();
            
            // Initialize connection pool
            let manager = SqliteConnectionManager::file(&db_path);
            let pool = Pool::new(manager)?;
            
            // Initialize schema using a connection from the pool
            {
                let conn = pool.get()?;
                Self::initialize_schema(&conn)?;
            }
            
            // Create HNSW actor
            let (hnsw_sender, hnsw_receiver) = mpsc::channel(100);
            let hnsw_actor = HnswActor::new(hnsw_receiver, dimension, max_elements);
            
            // Initialize the HNSW index
            let (init_sender, init_receiver) = oneshot::channel();
            if let Err(_) = hnsw_sender.send(HnswMessage::Initialize {
                dimension,
                max_elements,
                response: init_sender,
            }).await {
                return Err(anyhow!("Failed to send initialization message to HNSW actor"));
            }
            
            // Spawn the HNSW actor
            tokio::spawn(hnsw_actor.run());
            
            // Wait for initialization to complete
            init_receiver.await??;
            
            let db = Self {
                db_path,
                pool,
                hnsw_sender,
                dimension,
                max_elements,
            };
            
            // Load existing index if available
            db.load_index().await?;
            
            Ok(db)
        }

        fn initialize_schema(conn: &PooledConnection<SqliteConnectionManager>) -> Result<()> {
            // Create metadata table
            conn.execute(
                "CREATE TABLE IF NOT EXISTS vector_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )",
                [],
            )?;

            // Create vector_records table (generic storage)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS vector_records (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )",
                [],
            )?;

            // Create index on content for faster lookups
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vector_records_content ON vector_records(content)",
                [],
            )?;

            // Set schema version
            conn.execute(
                "INSERT OR REPLACE INTO vector_metadata (key, value) VALUES ('schema_version', ?)",
                params![CURRENT_SCHEMA_VERSION.to_string()],
            )?;

            Ok(())
        }

        async fn load_index(&self) -> Result<()> {
            let conn = self.pool.get()?;
            
            // Check if we have any data
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM vector_records",
                [],
                |row| row.get(0),
            )?;

            if count == 0 {
                return Ok(());
            }

            // Load all embeddings data
            let records_data: Vec<(String, Vec<f32>)> = {
                let mut stmt = conn.prepare(
                    "SELECT content, embedding FROM vector_records ORDER BY rowid",
                )?;

                let record_iter = stmt.query_map([], |row: &Row| {
                    let content: String = row.get(0)?;
                    let embedding_blob: Vec<u8> = row.get(1)?;
                    let embedding: Vec<f32> = bincode::deserialize(&embedding_blob)
                        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                            1,
                            rusqlite::types::Type::Blob,
                            Box::new(e),
                        ))?;
                    Ok((content, embedding))
                })?;

                record_iter.collect::<Result<Vec<_>, _>>()?
            };

            // Send data to HNSW actor for loading
            let (load_sender, load_receiver) = oneshot::channel();
            let content_to_point_mapping = HashMap::new(); // We'll use content hash for point IDs

            if let Err(_) = self.hnsw_sender.send(HnswMessage::LoadFromDb {
                embeddings: records_data,
                content_to_point_mapping,
                response: load_sender,
            }).await {
                return Err(anyhow!("Failed to send load message to HNSW actor"));
            }

            load_receiver.await??;
            Ok(())
        }

        /// Generic method to add any record that can be converted to VectorStorageRecord
        pub async fn add_storage_record(&self, record: VectorStorageRecord) -> Result<()> {
            if record.embedding.len() != self.dimension {
                return Err(anyhow!(
                    "Dimension mismatch: expected {}, got {}",
                    self.dimension,
                    record.embedding.len()
                ));
            }

            let embedding_blob = bincode::serialize(&record.embedding)?;
            let metadata_json = serde_json::to_string(&record.metadata)?;

            // Add to SQLite
            {
                let conn = self.sqlite_conn.lock().await;
                conn.execute(
                    "INSERT OR REPLACE INTO vector_records (id, content, embedding, metadata, updated_at)
                     VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                    params![
                        record.id,
                        record.content,
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
                    // Use content-based hash as point ID for consistent mapping
                    let record_type = record.metadata.get("record_type")
                        .cloned()
                        .unwrap_or_else(|| "unknown".to_string());
                    let point_id = self.content_to_point_id(&record.content, &record_type);
                    hnsw.insert((&record.embedding, point_id));
                }
            }

            Ok(())
        }

        /// Legacy method for backward compatibility
        pub async fn add_tool(&self, tool: ToolRecord) -> Result<()> {
            let storage_record: VectorStorageRecord = tool.into();
            self.add_storage_record(storage_record).await
        }

        /// Generic search method returning VectorStorageRecord
        pub async fn search_storage_records(
            &self,
            query_embedding: &[f32],
            limit: usize,
            filter: Option<HashMap<String, String>>,
        ) -> Result<Vec<VectorStorageRecord>> {
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

            // Convert point IDs back to record IDs and fetch from SQLite
            let mut results = Vec::new();

            for neighbor in neighbors {
                let record_id = match self.point_id_to_record_id(neighbor.d_id).await {
                    Ok(id) => id,
                    Err(_) => continue, // Skip if we can't find the record
                };
                
                // Fetch record from SQLite
                let storage_record = {
                    let conn = self.sqlite_conn.lock().await;
                    let mut stmt = conn.prepare_cached(
                        "SELECT id, content, embedding, metadata FROM vector_records WHERE id = ?"
                    )?;

                    stmt.query_row(params![record_id], |row| {
                        let id: String = row.get(0)?;
                        let content: String = row.get(1)?;
                        let embedding_blob: Vec<u8> = row.get(2)?;
                        let metadata_json: String = row.get(3)?;

                        let embedding: Vec<f32> = bincode::deserialize(&embedding_blob)
                            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                                2,
                                rusqlite::types::Type::Blob,
                                Box::new(e),
                            ))?;

                        let metadata: HashMap<String, String> = serde_json::from_str(&metadata_json)
                            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                                3,
                                rusqlite::types::Type::Text,
                                Box::new(e),
                            ))?;

                        Ok(VectorStorageRecord {
                            id,
                            content,
                            embedding,
                            metadata,
                        })
                    })
                };

                if let Ok(storage_record) = storage_record {
                    // Apply filter if provided
                    if let Some(ref filter_map) = filter {
                        let matches_filter = filter_map.iter().all(|(key, value)| {
                            storage_record.metadata.get(key) == Some(value)
                        });
                        
                        if matches_filter {
                            results.push(storage_record);
                        }
                    } else {
                        results.push(storage_record);
                    }
                }

                if results.len() >= limit {
                    break;
                }
            }

            Ok(results)
        }

        /// Legacy method for backward compatibility
        pub async fn search_similar(
            &self,
            query_embedding: &[f32],
            limit: usize,
            filter: Option<HashMap<String, String>>,
        ) -> Result<Vec<ToolRecord>> {
            // Use the new generic search method and convert results
            let storage_records = self.search_storage_records(query_embedding, limit, filter).await?;
            let tool_records: Vec<ToolRecord> = storage_records
                .into_iter()
                .filter_map(|record| {
                    // Only convert records that are actually tools
                    if record.metadata.get("record_type") == Some(&"tool".to_string()) {
                        Some(record.into())
                    } else {
                        None
                    }
                })
                .collect();
            Ok(tool_records)
        }

        /// Generic method to remove any record
        pub async fn remove_record(&self, record_id: &str) -> Result<bool> {
            let conn = self.sqlite_conn.lock().await;
            let affected = conn.execute(
                "DELETE FROM vector_records WHERE id = ?",
                params![record_id],
            )?;

            // Note: HNSW doesn't support efficient deletion, so we keep it simple
            // In a production system, you might want to rebuild the index periodically
            
            Ok(affected > 0)
        }

        /// Legacy method for backward compatibility
        pub async fn remove_tool(&self, tool_id: &str) -> Result<bool> {
            self.remove_record(tool_id).await
        }

        /// Generic method to list all storage records
        pub async fn list_storage_records(&self) -> Result<Vec<VectorStorageRecord>> {
            let conn = self.sqlite_conn.lock().await;
            let mut stmt = conn.prepare(
                "SELECT id, content, embedding, metadata FROM vector_records ORDER BY id"
            )?;

            let record_iter = stmt.query_map([], |row| {
                let id: String = row.get(0)?;
                let content: String = row.get(1)?;
                let embedding_blob: Vec<u8> = row.get(2)?;
                let metadata_json: String = row.get(3)?;

                let embedding: Vec<f32> = bincode::deserialize(&embedding_blob)
                    .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                        2,
                        rusqlite::types::Type::Blob,
                        Box::new(e),
                    ))?;

                let metadata: HashMap<String, String> = serde_json::from_str(&metadata_json)
                    .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                        3,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    ))?;

                Ok(VectorStorageRecord {
                    id,
                    content,
                    embedding,
                    metadata,
                })
            })?;

            let mut records = Vec::new();
            for record_result in record_iter {
                records.push(record_result?);
            }

            Ok(records)
        }

        /// Legacy method for backward compatibility
        pub async fn list_tools(&self) -> Result<Vec<ToolRecord>> {
            let storage_records = self.list_storage_records().await?;
            let tool_records: Vec<ToolRecord> = storage_records
                .into_iter()
                .filter_map(|record| {
                    // Only convert records that are actually tools
                    if record.metadata.get("record_type") == Some(&"tool".to_string()) {
                        Some(record.into())
                    } else {
                        None
                    }
                })
                .collect();
            Ok(tool_records)
        }

        pub async fn clear(&self) -> Result<()> {
            let conn = self.sqlite_conn.lock().await;
            conn.execute("DELETE FROM vector_records", [])?;
            
            // Reset HNSW index
            *self.hnsw_index.lock().await = None;
            
            Ok(())
        }

        pub async fn stats(&self) -> Result<HashMap<String, serde_json::Value>> {
            let conn = self.sqlite_conn.lock().await;
            
            let total_count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM vector_records",
                [],
                |row| row.get(0),
            )?;

            // Count by record type
            let mut stmt = conn.prepare("SELECT metadata FROM vector_records")?;
            let metadata_iter = stmt.query_map([], |row| {
                let metadata_json: String = row.get(0)?;
                let metadata: HashMap<String, String> = serde_json::from_str(&metadata_json)
                    .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    ))?;
                Ok(metadata.get("record_type").cloned().unwrap_or_else(|| "unknown".to_string()))
            })?;

            let mut type_counts = HashMap::new();
            for record_type in metadata_iter {
                let record_type = record_type?;
                *type_counts.entry(record_type).or_insert(0) += 1;
            }

            let mut stats = HashMap::new();
            stats.insert("total_records".to_string(), serde_json::Value::Number(total_count.into()));
            stats.insert("dimension".to_string(), serde_json::Value::Number(self.dimension.into()));
            stats.insert("db_path".to_string(), serde_json::Value::String(self.db_path.clone()));
            
            // Add type breakdown
            for (record_type, count) in type_counts {
                stats.insert(format!("{}_count", record_type), serde_json::Value::Number(count.into()));
            }
            
            Ok(stats)
        }

        // Simple hash-based mapping between record IDs and point IDs
        fn record_id_to_point_id(&self, record_id: &str) -> usize {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            
            let mut hasher = DefaultHasher::new();
            record_id.hash(&mut hasher);
            (hasher.finish() % (self.max_elements as u64)) as usize
        }

        async fn point_id_to_record_id(&self, point_id: usize) -> Result<String> {
            // Since we're using content-based hashing, we need to find the record
            // whose content hashes to this point_id
            let conn = self.sqlite_conn.lock().await;
            let mut stmt = conn.prepare(
                "SELECT id, content FROM vector_records"
            )?;

            let record_iter = stmt.query_map([], |row| {
                let id: String = row.get(0)?;
                let content: String = row.get(1)?;
                Ok((id, content))
            })?;

            for record_result in record_iter {
                let (id, content) = record_result?;
                // Get record type from database to calculate point ID
                let record_type = {
                    let mut stmt = conn.prepare("SELECT metadata FROM vector_records WHERE id = ?")?;
                    let metadata_json: String = stmt.query_row([&id], |row| row.get(0))?;
                    let metadata: HashMap<String, String> = serde_json::from_str(&metadata_json)
                        .map_err(|e| anyhow!("Failed to parse metadata: {}", e))?;
                    metadata.get("record_type").cloned().unwrap_or_else(|| "unknown".to_string())
                };
                
                if self.content_to_point_id(&content, &record_type) == point_id {
                    return Ok(id);
                }
            }

            Err(anyhow!("No record found for point_id {}", point_id))
        }

        fn content_to_point_id(&self, content: &str, record_type: &str) -> usize {
            let combined = format!("{}:{}", record_type, content);
            let hash = farmhash::hash64(combined.as_bytes());
            (hash % (self.max_elements as u64)) as usize
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::collections::HashMap;

    #[cfg(feature = "vectordb-sqlite")]
    mod generic_vectordb_tests {
        use super::*;
        use crate::agents::message_vectordb::MessageRecord;
        use crate::agents::document_vectordb::DocumentRecord;

        #[tokio::test]
        async fn test_vector_storage_record_conversions() {
            // Test ToolRecord conversion
            let mut metadata = HashMap::new();
            metadata.insert("schema".to_string(), "test_schema".to_string());
            
            let tool_record = sqlite_impl::ToolRecord {
                id: "tool_1".to_string(),
                name: "test_tool".to_string(),
                description: "A test tool".to_string(),
                embedding: vec![0.1, 0.2, 0.3],
                metadata: metadata.clone(),
            };

            let storage_record: sqlite_impl::VectorStorageRecord = tool_record.clone().into();
            assert_eq!(storage_record.id, "tool_1");
            assert_eq!(storage_record.content, "A test tool");
            assert_eq!(storage_record.embedding, vec![0.1, 0.2, 0.3]);
            assert_eq!(storage_record.metadata.get("name"), Some(&"test_tool".to_string()));
            assert_eq!(storage_record.metadata.get("record_type"), Some(&"tool".to_string()));

            // Convert back
            let converted_tool: sqlite_impl::ToolRecord = storage_record.into();
            assert_eq!(converted_tool.id, tool_record.id);
            assert_eq!(converted_tool.name, tool_record.name);
            assert_eq!(converted_tool.description, tool_record.description);
            assert_eq!(converted_tool.embedding, tool_record.embedding);
        }

        #[tokio::test]
        async fn test_message_record_conversions() {
            let message_record = MessageRecord {
                message_id: "msg_1".to_string(),
                session_id: "session_1".to_string(),
                content: "Hello world".to_string(),
                role: "user".to_string(),
                timestamp: 1234567890,
                embedding: vec![0.1, 0.2, 0.3],
                metadata: HashMap::new(),
            };

            let storage_record: sqlite_impl::VectorStorageRecord = message_record.clone().into();
            assert_eq!(storage_record.id, "msg_1");
            assert_eq!(storage_record.content, "Hello world");
            assert_eq!(storage_record.metadata.get("record_type"), Some(&"message".to_string()));
            assert_eq!(storage_record.metadata.get("session_id"), Some(&"session_1".to_string()));
            assert_eq!(storage_record.metadata.get("role"), Some(&"user".to_string()));

            // Convert back
            let converted_message: MessageRecord = storage_record.into();
            assert_eq!(converted_message.message_id, message_record.message_id);
            assert_eq!(converted_message.session_id, message_record.session_id);
            assert_eq!(converted_message.content, message_record.content);
            assert_eq!(converted_message.role, message_record.role);
            assert_eq!(converted_message.timestamp, message_record.timestamp);
        }

        #[tokio::test]
        async fn test_document_record_conversions() {
            let doc_record = DocumentRecord {
                document_id: "doc_1".to_string(),
                file_path: "/path/to/file.txt".to_string(),
                title: "Test Document".to_string(),
                content: "Document content".to_string(),
                content_type: "text".to_string(),
                chunk_index: 0,
                total_chunks: 1,
                file_size: 1024,
                last_modified: 1234567890,
                embedding: vec![0.1, 0.2, 0.3],
                metadata: HashMap::new(),
            };

            let storage_record: sqlite_impl::VectorStorageRecord = doc_record.clone().into();
            assert_eq!(storage_record.id, "doc_1");
            assert_eq!(storage_record.content, "Document content");
            assert_eq!(storage_record.metadata.get("record_type"), Some(&"document".to_string()));
            assert_eq!(storage_record.metadata.get("file_path"), Some(&"/path/to/file.txt".to_string()));
            assert_eq!(storage_record.metadata.get("content_type"), Some(&"text".to_string()));

            // Convert back
            let converted_doc: DocumentRecord = storage_record.into();
            assert_eq!(converted_doc.document_id, doc_record.document_id);
            assert_eq!(converted_doc.file_path, doc_record.file_path);
            assert_eq!(converted_doc.content, doc_record.content);
            assert_eq!(converted_doc.content_type, doc_record.content_type);
            assert_eq!(converted_doc.chunk_index, doc_record.chunk_index);
        }

        #[tokio::test]
        async fn test_generic_vectordb_operations() -> Result<()> {
            let temp_dir = TempDir::new()?;
            let db_path = temp_dir.path().join("generic_test.db");
            
            let db = sqlite_impl::SqliteVectorDB::new(&db_path, 3, 1000).await?;

            // Test adding different record types to the same database
            let tool_record = sqlite_impl::ToolRecord {
                id: "tool_1".to_string(),
                name: "test_tool".to_string(),
                description: "A test tool".to_string(),
                embedding: vec![0.1, 0.2, 0.3],
                metadata: HashMap::new(),
            };

            let message_record = MessageRecord {
                message_id: "msg_1".to_string(),
                session_id: "session_1".to_string(),
                content: "Hello world".to_string(),
                role: "user".to_string(),
                timestamp: 1234567890,
                embedding: vec![0.4, 0.5, 0.6],
                metadata: HashMap::new(),
            };

            let doc_record = DocumentRecord {
                document_id: "doc_1".to_string(),
                file_path: "/test.txt".to_string(),
                title: "Test".to_string(),
                content: "Document content".to_string(),
                content_type: "text".to_string(),
                chunk_index: 0,
                total_chunks: 1,
                file_size: 100,
                last_modified: 1234567890,
                embedding: vec![0.7, 0.8, 0.9],
                metadata: HashMap::new(),
            };

            // Add records using generic interface
            let tool_db: &dyn VectorDatabase<sqlite_impl::ToolRecord> = &db;
            tool_db.add_record(tool_record.clone()).await?;

            let msg_db: &dyn VectorDatabase<MessageRecord> = &db;
            msg_db.add_record(message_record.clone()).await?;

            let doc_db: &dyn VectorDatabase<DocumentRecord> = &db;
            doc_db.add_record(doc_record.clone()).await?;

            // Test searching for each type
            let tool_results = tool_db.search_similar(&vec![0.1, 0.2, 0.3], 5, None).await?;
            assert_eq!(tool_results.len(), 1);
            assert_eq!(tool_results[0].name, "test_tool");

            let msg_results = msg_db.search_similar(&vec![0.4, 0.5, 0.6], 5, None).await?;
            assert_eq!(msg_results.len(), 1);
            assert_eq!(msg_results[0].content, "Hello world");

            let doc_results = doc_db.search_similar(&vec![0.7, 0.8, 0.9], 5, None).await?;
            assert_eq!(doc_results.len(), 1);
            assert_eq!(doc_results[0].content, "Document content");

            // Test stats show all record types
            let stats = db.stats().await?;
            assert_eq!(stats["total_records"], 3);
            assert_eq!(stats["tool_count"], 1);
            assert_eq!(stats["message_count"], 1);
            assert_eq!(stats["document_count"], 1);

            Ok(())
        }

        #[tokio::test]
        async fn test_type_filtering_in_search() -> Result<()> {
            let temp_dir = TempDir::new()?;
            let db_path = temp_dir.path().join("filter_test.db");
            
            let db = sqlite_impl::SqliteVectorDB::new(&db_path, 2, 1000).await?;

            // Add multiple records of different types with similar embeddings
            let similar_embedding = vec![0.5, 0.5];

            let tool_record = sqlite_impl::ToolRecord {
                id: "tool_1".to_string(),
                name: "similar_tool".to_string(),
                description: "Similar content".to_string(),
                embedding: similar_embedding.clone(),
                metadata: HashMap::new(),
            };

            let message_record = MessageRecord {
                message_id: "msg_1".to_string(),
                session_id: "session_1".to_string(),
                content: "Similar content".to_string(),
                role: "user".to_string(),
                timestamp: 1234567890,
                embedding: similar_embedding.clone(),
                metadata: HashMap::new(),
            };

            // Add using generic interface
            let tool_db: &dyn VectorDatabase<sqlite_impl::ToolRecord> = &db;
            tool_db.add_record(tool_record).await?;

            let msg_db: &dyn VectorDatabase<MessageRecord> = &db;
            msg_db.add_record(message_record).await?;

            // Search for tools should only return tools
            let tool_results = tool_db.search_similar(&similar_embedding, 10, None).await?;
            assert_eq!(tool_results.len(), 1);
            assert_eq!(tool_results[0].name, "similar_tool");

            // Search for messages should only return messages
            let msg_results = msg_db.search_similar(&similar_embedding, 10, None).await?;
            assert_eq!(msg_results.len(), 1);
            assert_eq!(msg_results[0].content, "Similar content");

            Ok(())
        }

        #[tokio::test]
        async fn test_session_filtering_for_messages() -> Result<()> {
            let temp_dir = TempDir::new()?;
            let db_path = temp_dir.path().join("session_filter_test.db");
            
            let db = sqlite_impl::SqliteVectorDB::new(&db_path, 2, 1000).await?;

            let msg_db: &dyn VectorDatabase<MessageRecord> = &db;

            // Add messages from different sessions
            let msg1 = MessageRecord {
                message_id: "msg_1".to_string(),
                session_id: "session_a".to_string(),
                content: "Message from session A".to_string(),
                role: "user".to_string(),
                timestamp: 1234567890,
                embedding: vec![0.1, 0.2],
                metadata: HashMap::new(),
            };

            let msg2 = MessageRecord {
                message_id: "msg_2".to_string(),
                session_id: "session_b".to_string(),
                content: "Message from session B".to_string(),
                role: "user".to_string(),
                timestamp: 1234567891,
                embedding: vec![0.2, 0.3],
                metadata: HashMap::new(),
            };

            msg_db.add_record(msg1).await?;
            msg_db.add_record(msg2).await?;

            // Filter by session
            let mut filter = HashMap::new();
            filter.insert("session_id".to_string(), "session_a".to_string());
            
            let filtered_results = msg_db.search_similar(&vec![0.1, 0.2], 10, Some(filter)).await?;
            assert_eq!(filtered_results.len(), 1);
            assert_eq!(filtered_results[0].session_id, "session_a");

            Ok(())
        }
    }
}

// Export the SQLite implementation
#[cfg(feature = "vectordb-sqlite")]
pub use sqlite_impl::SqliteVectorDB;

// Type aliases for specific use cases - now properly specialized
#[cfg(feature = "vectordb-sqlite")]
pub type ToolVectorDB = sqlite_impl::SqliteVectorDB;

#[cfg(feature = "vectordb-sqlite")]
pub type MessageVectorDB = sqlite_impl::SqliteVectorDB;

#[cfg(feature = "vectordb-sqlite")]
pub type DocumentVectorDB = sqlite_impl::SqliteVectorDB;

// Generic trait for serializable data with embeddings
pub trait VectorRecord: Send + Sync + Clone + Serialize + for<'de> Deserialize<'de> {
    fn id(&self) -> &str;
    fn embedding(&self) -> &[f32];
    fn metadata(&self) -> HashMap<String, String>;
}

// Common interface for vector databases - generic over record type
#[async_trait]
pub trait VectorDatabase<T: VectorRecord>: Send + Sync {
    async fn add_record(&self, record: T) -> Result<()>;
    async fn search_similar(
        &self,
        query_embedding: &[f32],
        limit: usize,
        filter: Option<HashMap<String, String>>,
    ) -> Result<Vec<T>>;
    async fn remove_record(&self, record_id: &str) -> Result<bool>;
    async fn list_records(&self) -> Result<Vec<T>>;
    async fn clear(&self) -> Result<()>;
    async fn stats(&self) -> Result<HashMap<String, serde_json::Value>>;
}

#[cfg(feature = "vectordb-sqlite")]
#[async_trait]
impl VectorDatabase<sqlite_impl::ToolRecord> for sqlite_impl::SqliteVectorDB {
    async fn add_record(&self, record: sqlite_impl::ToolRecord) -> Result<()> {
        self.add_tool(record).await
    }

    async fn search_similar(
        &self,
        query_embedding: &[f32],
        limit: usize,
        filter: Option<HashMap<String, String>>,
    ) -> Result<Vec<sqlite_impl::ToolRecord>> {
        self.search_similar(query_embedding, limit, filter).await
    }

    async fn remove_record(&self, record_id: &str) -> Result<bool> {
        self.remove_tool(record_id).await
    }

    async fn list_records(&self) -> Result<Vec<sqlite_impl::ToolRecord>> {
        self.list_tools().await
    }

    async fn clear(&self) -> Result<()> {
        self.clear().await
    }

    async fn stats(&self) -> Result<HashMap<String, serde_json::Value>> {
        self.stats().await
    }
}

#[cfg(feature = "vectordb-sqlite")]
#[async_trait]
impl VectorDatabase<crate::agents::message_vectordb::MessageRecord> for sqlite_impl::SqliteVectorDB {
    async fn add_record(&self, record: crate::agents::message_vectordb::MessageRecord) -> Result<()> {
        let storage_record: sqlite_impl::VectorStorageRecord = record.into();
        self.add_storage_record(storage_record).await
    }

    async fn search_similar(
        &self,
        query_embedding: &[f32],
        limit: usize,
        filter: Option<HashMap<String, String>>,
    ) -> Result<Vec<crate::agents::message_vectordb::MessageRecord>> {
        let storage_records = self.search_storage_records(query_embedding, limit, filter).await?;
        let message_records: Vec<crate::agents::message_vectordb::MessageRecord> = storage_records
            .into_iter()
            .filter_map(|record| {
                // Only convert records that are actually messages
                if record.metadata.get("record_type") == Some(&"message".to_string()) {
                    Some(record.into())
                } else {
                    None
                }
            })
            .collect();
        Ok(message_records)
    }

    async fn remove_record(&self, record_id: &str) -> Result<bool> {
        self.remove_record(record_id).await
    }

    async fn list_records(&self) -> Result<Vec<crate::agents::message_vectordb::MessageRecord>> {
        let storage_records = self.list_storage_records().await?;
        let message_records: Vec<crate::agents::message_vectordb::MessageRecord> = storage_records
            .into_iter()
            .filter_map(|record| {
                // Only convert records that are actually messages
                if record.metadata.get("record_type") == Some(&"message".to_string()) {
                    Some(record.into())
                } else {
                    None
                }
            })
            .collect();
        Ok(message_records)
    }

    async fn clear(&self) -> Result<()> {
        self.clear().await
    }

    async fn stats(&self) -> Result<HashMap<String, serde_json::Value>> {
        self.stats().await
    }
}

#[cfg(feature = "vectordb-sqlite")]
#[async_trait]
impl VectorDatabase<crate::agents::document_vectordb::DocumentRecord> for sqlite_impl::SqliteVectorDB {
    async fn add_record(&self, record: crate::agents::document_vectordb::DocumentRecord) -> Result<()> {
        let storage_record: sqlite_impl::VectorStorageRecord = record.into();
        self.add_storage_record(storage_record).await
    }

    async fn search_similar(
        &self,
        query_embedding: &[f32],
        limit: usize,
        filter: Option<HashMap<String, String>>,
    ) -> Result<Vec<crate::agents::document_vectordb::DocumentRecord>> {
        let storage_records = self.search_storage_records(query_embedding, limit, filter).await?;
        let document_records: Vec<crate::agents::document_vectordb::DocumentRecord> = storage_records
            .into_iter()
            .filter_map(|record| {
                // Only convert records that are actually documents
                if record.metadata.get("record_type") == Some(&"document".to_string()) {
                    Some(record.into())
                } else {
                    None
                }
            })
            .collect();
        Ok(document_records)
    }

    async fn remove_record(&self, record_id: &str) -> Result<bool> {
        self.remove_record(record_id).await
    }

    async fn list_records(&self) -> Result<Vec<crate::agents::document_vectordb::DocumentRecord>> {
        let storage_records = self.list_storage_records().await?;
        let document_records: Vec<crate::agents::document_vectordb::DocumentRecord> = storage_records
            .into_iter()
            .filter_map(|record| {
                // Only convert records that are actually documents
                if record.metadata.get("record_type") == Some(&"document".to_string()) {
                    Some(record.into())
                } else {
                    None
                }
            })
            .collect();
        Ok(document_records)
    }

    async fn clear(&self) -> Result<()> {
        self.clear().await
    }

    async fn stats(&self) -> Result<HashMap<String, serde_json::Value>> {
        self.stats().await
    }
}