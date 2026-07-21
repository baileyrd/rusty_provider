pub mod errors;
pub mod routes;
pub mod state;

use axum::routing::{get, patch, post};
use axum::Router as AxumRouter;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::state::AppState;

/// Builds the full axum app (routes + middleware) over the given state.
/// Shared by `main` (serving on a real listener) and integration tests
/// (serving on an ephemeral port via the same `axum::serve` path).
pub fn build_app(state: AppState) -> AxumRouter {
    AxumRouter::new()
        .route("/health", get(routes::health))
        .route("/v1/models", get(routes::list_models))
        .route("/v1/usage", get(routes::usage_stats))
        .route("/metrics", get(routes::metrics))
        .route("/v1/chat/completions", post(routes::chat_completions))
        .route(
            "/v1/admin/clients",
            get(routes::admin_list_clients).post(routes::admin_create_client),
        )
        .route(
            "/v1/admin/clients/:name",
            patch(routes::admin_update_client).delete(routes::admin_delete_client),
        )
        .route(
            "/v1/admin/clients/:name/reset-spend",
            post(routes::admin_reset_client_spend),
        )
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state)
}
