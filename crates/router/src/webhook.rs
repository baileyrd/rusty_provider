use std::time::Duration;

use serde::Serialize;

use crate::config::BudgetPeriod;

/// A budget-related event this router can push to an operator's own
/// endpoint, so crossing a budget surfaces as more than a `402` on the
/// client's next request and a Prometheus counter.
#[derive(Debug, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum WebhookEvent {
    BudgetExceeded {
        client: String,
        spent_usd: f64,
        budget_usd: f64,
        period: BudgetPeriod,
    },
    BudgetReset {
        client: String,
        budget_usd: f64,
        period: BudgetPeriod,
    },
}

/// Fires `[webhook]`-configured POSTs on budget events. Delivery is
/// fire-and-forget (spawned, not awaited) so a slow or unreachable
/// receiver never adds latency to the request that triggered the event --
/// same non-blocking contract as `record_usage`'s persistence writes. A
/// delivery failure is only logged, never surfaced to the client.
pub(crate) struct WebhookNotifier {
    client: reqwest::Client,
    url: String,
    /// The exact value to send as this POST's `Authorization` header
    /// (e.g. `"Bearer <token>"`), so the receiver can verify the request
    /// came from this router. `None` sends no `Authorization` header.
    auth_header: Option<String>,
}

impl WebhookNotifier {
    pub(crate) fn new(url: String, auth_header: Option<String>, timeout_secs: u64) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(timeout_secs))
                .build()
                .expect("reqwest client should build with a timeout configured"),
            url,
            auth_header,
        }
    }

    fn send(&self, event: WebhookEvent) {
        let client = self.client.clone();
        let url = self.url.clone();
        let auth_header = self.auth_header.clone();
        tokio::spawn(async move {
            let mut req = client.post(&url).json(&event);
            if let Some(auth) = auth_header {
                req = req.header(reqwest::header::AUTHORIZATION, auth);
            }
            if let Err(e) = req.send().await {
                tracing::warn!(%url, error = %e, "budget webhook delivery failed");
            }
        });
    }

    /// A client's tracked spend just reached or passed its configured
    /// `budget_usd` as a result of a request that was already let through
    /// (the request that pushed it over is charged before this fires, not
    /// blocked by it -- the `402` starts on the *next* request).
    pub(crate) fn notify_budget_exceeded(
        &self,
        client_name: &str,
        spent_usd: f64,
        budget_usd: f64,
        period: BudgetPeriod,
    ) {
        self.send(WebhookEvent::BudgetExceeded {
            client: client_name.to_string(),
            spent_usd,
            budget_usd,
            period,
        });
    }

    /// An operator manually reset a client's spend via the admin API
    /// (`POST /v1/admin/clients/{name}/reset-spend`).
    pub(crate) fn notify_budget_reset(
        &self,
        client_name: &str,
        budget_usd: f64,
        period: BudgetPeriod,
    ) {
        self.send(WebhookEvent::BudgetReset {
            client: client_name.to_string(),
            budget_usd,
            period,
        });
    }
}
