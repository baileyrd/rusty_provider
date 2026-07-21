//! Pure logic for per-client spend budgets (`[[clients]].budget_usd`):
//! parsing budget settings out of config, and the calendar math behind
//! `budget_period = "monthly"`. `Router` (in `lib.rs`) owns the actual
//! state -- in-memory by default, or backed by `Persistence` when
//! `[persistence]` is configured -- and calls into this module for the
//! parts that don't depend on which storage is active.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config::{BudgetPeriod, ClientConfig};

#[derive(Debug, Clone, Copy)]
pub struct ClientBudgetSetting {
    pub budget_usd: f64,
    pub period: BudgetPeriod,
}

/// One client's in-memory spend, scoped to whichever period key was
/// current the last time it was touched. A key of `0` (what
/// `period_key_at` always returns for `BudgetPeriod::Total`) never
/// changes, so total-period spend simply accumulates forever; a
/// `BudgetPeriod::Monthly` client's key changes across a calendar-month
/// boundary, at which point `spent_usd` resets to zero.
#[derive(Debug, Default)]
pub struct SpendState {
    pub period_key: i64,
    pub spent_usd: f64,
}

/// `client_name -> budget_usd`/`budget_period`, for every `[[clients]]`
/// entry that has `budget_usd` set. Clients without a configured budget
/// are absent here, and treated as unrestricted by every caller.
pub fn settings_from_clients(clients: &[ClientConfig]) -> HashMap<String, ClientBudgetSetting> {
    clients
        .iter()
        .filter_map(|c| {
            c.budget_usd.map(|budget_usd| {
                (
                    c.name.clone(),
                    ClientBudgetSetting {
                        budget_usd,
                        period: c.budget_period,
                    },
                )
            })
        })
        .collect()
}

pub fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// A value that changes exactly when `period` should reset: always `0`
/// for `Total` (so it never resets), or `year * 12 + month` for `Monthly`
/// (so it changes precisely at each calendar-month boundary).
pub fn period_key_at(period: BudgetPeriod, now_unix: i64) -> i64 {
    match period {
        BudgetPeriod::Total => 0,
        BudgetPeriod::Monthly => {
            let (year, month) = year_month_from_unix(now_unix);
            year as i64 * 12 + month as i64
        }
    }
}

/// Converts Unix seconds (UTC) to a `(year, month)` pair via Howard
/// Hinnant's `civil_from_days` algorithm
/// (<https://howardhinnant.github.io/date_algorithms.html>), since this
/// otherwise has no date/time dependency for a single calendar
/// computation.
fn year_month_from_unix(unix_secs: i64) -> (i32, u32) {
    let days = unix_secs.div_euclid(86_400);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m as u32)
}

/// Resets `state` to zero spend if `current_key` doesn't match its
/// stored period key (i.e. a rollover happened since it was last
/// touched).
pub fn roll_period_if_needed(state: &mut SpendState, current_key: i64) {
    if state.period_key != current_key {
        state.period_key = current_key;
        state.spent_usd = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn client(name: &str, budget_usd: Option<f64>, period: BudgetPeriod) -> ClientConfig {
        ClientConfig {
            name: name.to_string(),
            api_key_env: format!("{}_KEY", name.to_uppercase()),
            requests_per_minute: 60,
            budget_usd,
            budget_period: period,
        }
    }

    // --- settings_from_clients ---------------------------------------------------

    #[test]
    fn settings_from_clients_skips_clients_with_no_budget() {
        let settings = settings_from_clients(&[client("acme", None, BudgetPeriod::Total)]);
        assert!(settings.is_empty());
    }

    #[test]
    fn settings_from_clients_includes_clients_with_a_budget() {
        let settings = settings_from_clients(&[client("acme", Some(10.0), BudgetPeriod::Monthly)]);
        let setting = &settings["acme"];
        assert_eq!(setting.budget_usd, 10.0);
        assert_eq!(setting.period, BudgetPeriod::Monthly);
    }

    // --- year_month_from_unix ----------------------------------------------------
    // Reference epoch values are well-known: 2000-01-01T00:00:00Z ==
    // 946684800, 2024-01-01T00:00:00Z == 1704067200.

    #[test]
    fn year_month_from_unix_matches_known_reference_epochs() {
        assert_eq!(year_month_from_unix(0), (1970, 1)); // 1970-01-01
        assert_eq!(year_month_from_unix(946_684_800), (2000, 1));
        assert_eq!(year_month_from_unix(1_704_067_200), (2024, 1));
    }

    #[test]
    fn year_month_from_unix_crosses_a_month_boundary() {
        assert_eq!(year_month_from_unix(1_706_745_599), (2024, 1)); // 2024-01-31T23:59:59Z
        assert_eq!(year_month_from_unix(1_706_745_600), (2024, 2)); // 2024-02-01T00:00:00Z
    }

    #[test]
    fn year_month_from_unix_handles_a_leap_year_february() {
        assert_eq!(year_month_from_unix(1_709_164_800), (2024, 2)); // 2024-02-29T00:00:00Z
        assert_eq!(year_month_from_unix(1_709_251_200), (2024, 3)); // 2024-03-01T00:00:00Z
    }

    #[test]
    fn year_month_from_unix_crosses_a_year_boundary() {
        assert_eq!(year_month_from_unix(1_703_980_799), (2023, 12)); // 2023-12-31T23:59:59Z
        assert_eq!(year_month_from_unix(1_704_067_200), (2024, 1)); // 2024-01-01T00:00:00Z
    }

    // --- period_key_at -------------------------------------------------------------

    #[test]
    fn period_key_at_total_is_always_zero() {
        assert_eq!(period_key_at(BudgetPeriod::Total, 0), 0);
        assert_eq!(period_key_at(BudgetPeriod::Total, 1_800_000_000), 0);
    }

    #[test]
    fn period_key_at_monthly_changes_across_a_month_boundary() {
        let jan = period_key_at(BudgetPeriod::Monthly, 1_704_067_200); // 2024-01-01
        let feb = period_key_at(BudgetPeriod::Monthly, 1_706_745_600); // 2024-02-01
        assert_ne!(jan, feb);
    }

    #[test]
    fn period_key_at_monthly_is_stable_within_a_month() {
        let start = period_key_at(BudgetPeriod::Monthly, 1_704_067_200); // 2024-01-01
        let mid = period_key_at(BudgetPeriod::Monthly, 1_704_067_200 + 15 * 86_400);
        assert_eq!(start, mid);
    }

    // --- roll_period_if_needed ----------------------------------------------------

    #[test]
    fn roll_period_if_needed_resets_spend_on_a_new_key() {
        let mut state = SpendState {
            period_key: 100,
            spent_usd: 42.0,
        };
        roll_period_if_needed(&mut state, 101);
        assert_eq!(state.period_key, 101);
        assert_eq!(state.spent_usd, 0.0);
    }

    #[test]
    fn roll_period_if_needed_leaves_spend_alone_on_the_same_key() {
        let mut state = SpendState {
            period_key: 100,
            spent_usd: 42.0,
        };
        roll_period_if_needed(&mut state, 100);
        assert_eq!(state.spent_usd, 42.0);
    }
}
