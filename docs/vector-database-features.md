# Vector Database Feature Flags

Goose supports two different vector database implementations through feature flags, allowing you to choose between performance and resource usage based on your needs.

## Available Features

### `vectordb-sqlite` (Default)
- **Dependencies**: HNSW + SQLite (~5MB total)
- **Benefits**: 
  - 98% smaller dependency footprint
  - Much faster build times
  - Lower memory usage
  - Simpler deployment
- **Use Case**: Most users, development, resource-constrained environments

### `vectordb-lancedb` (Removed)
- **Dependencies**: LanceDB + Arrow + DataFusion (~300MB total)
- **Benefits**:
  - More mature vector database with advanced features
  - Better performance for very large datasets (>100K tools)
- **Use Case**: Enterprise deployments with massive tool catalogs

## Usage

### Using SQLite Vector DB (Default)
```bash
# Default build uses SQLite implementation
cargo build

# Explicitly specify SQLite
cargo build --features vectordb-sqlite --no-default-features
```

### LanceDB Support Removed
LanceDB support has been removed to reduce dependency footprint. The SQLite implementation provides equivalent functionality with much smaller dependencies.

### Runtime Configuration
The SQLite implementation supports the following environment variables:

```bash
# Custom database path (optional)
export GOOSE_VECTOR_DB_PATH="/custom/path/to/vectordb"
```

## Migration

For existing installations with LanceDB data:

1. **Fresh Install**: Uses SQLite implementation by default
2. **Existing Data**: Any existing vector databases will be recreated with SQLite backend
3. **No Automatic Migration**: LanceDB data is not migrated - tools will be re-indexed on first use

## Performance

| Metric | SQLite Implementation |
|--------|----------------------|
| Dependencies | ~5MB |
| Build Time | ~30s |
| Memory Usage | ~50MB |
| Search Speed | Good (up to 100K tools) |
| Startup Time | Fast |

## API

The SQLite implementation provides a unified API through the `VectorDBAdapter` trait:

```rust
use goose::agents::tool_vectordb::create_default_vector_db;

let db = create_default_vector_db().await?;
db.index_tools(tools).await?;
let results = db.search_tools(query, limit, filter).await?;
```

## Troubleshooting

### Build Issues
If you encounter compilation issues:
```bash
# Clean rebuild
cargo clean
cargo build
```

### Database Issues
If you have database problems, you can force a clean start:
```bash
# Remove existing database
rm -rf ~/.local/share/goose/tool_db

# Rebuild
cargo build --features vectordb-sqlite
```

## Development

When developing and testing:

```bash
# Test SQLite implementation
cargo test --features vectordb-sqlite

# Test vector database functionality
cargo test vectordb

# Test Ollama integration (requires Ollama running)
cargo test ollama_vectordb_integration_test
```