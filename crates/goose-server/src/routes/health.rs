use axum::{routing::get, Json, Router};
use serde::Serialize;
use std::collections::HashMap;

#[derive(Serialize)]
struct StatusResponse {
    status: &'static str,
}

#[derive(Serialize)]
struct FeaturesResponse {
    features: HashMap<String, bool>,
}

/// Simple status endpoint that returns 200 OK when the server is running
async fn status() -> Json<StatusResponse> {
    Json(StatusResponse { status: "ok" })
}

/// Features endpoint that returns which optional features are enabled
async fn features() -> Json<FeaturesResponse> {
    let mut features_map = HashMap::new();
    
    // Check for vectordb-sqlite feature
    #[cfg(feature = "vectordb-sqlite")]
    features_map.insert("vectordb-sqlite".to_string(), true);
    #[cfg(not(feature = "vectordb-sqlite"))]
    features_map.insert("vectordb-sqlite".to_string(), false);
    
    Json(FeaturesResponse {
        features: features_map,
    })
}

/// Configure health check routes
pub fn routes() -> Router {
    Router::new()
        .route("/status", get(status))
        .route("/features", get(features))
}
