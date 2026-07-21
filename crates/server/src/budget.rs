use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use rp_router::{BudgetPeriod, ClientConfig};

#[derive(Debug, Clone, Copy)]
struct BudgetSetting {
    budget_usd: f64,
    period: BudgetPeriod,
}

/// Tracked spend for one client, scoped to whichever period key was
/// current the last time it was touched. A key of `0` (the value
/// `period_key_at` always returns for `BudgetPeriod::Total`) never
/// changes, so total-period spend simply accumulates forever; a
/// `BudgetPeriod::Monthly` client's key changes across a calendar-month
/// boundary, at which point `spent_usd` resets to zero.
#[derive(Debug, Default)]
struct SpendState {
    period_key: i64,
    spent_usd: f64,
}

/// Tracks each configured client's spend (in the same USD terms as this
/// router's own `cost_usd` computation) against its optional `budget_usd`
/// cap. Clients with no configured budget are always unrestricted.
pub struct ClientBudgets {
    settings: HashMap<String, BudgetSetting>,
    spend: Mutex<HashMap<String, SpendState>>,
}

/// The client's current-period spend and configured cap, returned when a
/// budget check fails.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetExceeded {
    pub spent_usd: f64,
    pub budget_usd: f64,
}

impl ClientBudgets {
    pub fn from_clients(clients: &[ClientConfig]) -> Self {
        let settings = clients
            .iter()
            .filter_map(|c| {
                c.budget_usd.map(|budget_usd| {
                    (
                        c.name.clone(),
                        BudgetSetting {
                            budget_usd,
                            period: c.budget_period,
                        },
                    )
                })
            })
            .collect();
        Self {
            settings,
            spend: Mutex::new(HashMap::new()),
        }
    }

    /// `Ok(())` if `client_name` has no configured budget, or hasn't yet
    /// reached it for the current period. `Err(BudgetExceeded)` if it has.
    pub fn check(&self, client_name: &str) -> Result<(), BudgetExceeded> {
        self.check_at(client_name, now_unix())
    }

    /// Adds `cost_usd` to `client_name`'s tracked spend for the current
    /// period. A no-op for clients with no configured budget — there's
    /// nothing to track against.
    pub fn record(&self, client_name: &str, cost_usd: f64) {
        self.record_at(client_name, cost_usd, now_unix())
    }

    fn check_at(&self, client_name: &str, now_unix: i64) -> Result<(), BudgetExceeded> {
        let Some(setting) = self.settings.get(client_name) else {
            return Ok(());
        };
        let mut spend = self.spend.lock().unwrap();
        let state = spend.entry(client_name.to_string()).or_default();
        roll_period_if_needed(state, period_key_at(setting.period, now_unix));
        if state.spent_usd >= setting.budget_usd {
            Err(BudgetExceeded {
                spent_usd: state.spent_usd,
                budget_usd: setting.budget_usd,
            })
        } else {
            Ok(())
        }
    }

    fn record_at(&self, client_name: &str, cost_usd: f64, now_unix: i64) {
        let Some(setting) = self.settings.get(client_name) else {
            return;
        };
        let mut spend = self.spend.lock().unwrap();
        let state = spend.entry(client_name.to_string()).or_default();
        roll_period_if_needed(state, period_key_at(setting.period, now_unix));
        state.spent_usd += cost_usd;
    }
}

fn roll_period_if_needed(state: &mut SpendState, current_key: i64) {
    if state.period_key != current_key {
        state.period_key = current_key;
        state.spent_usd = 0.0;
    }
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// A value that changes exactly when `period` should reset: always `0`
/// for `Total` (so it never resets), or `year * 12 + month` for `Monthly`
/// (so it changes precisely at each calendar-month boundary).
fn period_key_at(period: BudgetPeriod, now_unix: i64) -> i64 {
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
        // 2024-01-31T23:59:59Z, one second before February.
        assert_eq!(year_month_from_unix(1_706_745_599), (2024, 1));
        // 2024-02-01T00:00:00Z.
        assert_eq!(year_month_from_unix(1_706_745_600), (2024, 2));
    }

    #[test]
    fn year_month_from_unix_handles_a_leap_year_february() {
        // 2024-02-29T00:00:00Z -- 2024 is a leap year, so this date exists.
        assert_eq!(year_month_from_unix(1_709_164_800), (2024, 2));
        // 2024-03-01T00:00:00Z, the day after.
        assert_eq!(year_month_from_unix(1_709_251_200), (2024, 3));
    }

    #[test]
    fn year_month_from_unix_crosses_a_year_boundary() {
        // 2023-12-31T23:59:59Z.
        assert_eq!(year_month_from_unix(1_703_980_799), (2023, 12));
        // 2024-01-01T00:00:00Z.
        assert_eq!(year_month_from_unix(1_704_067_200), (2024, 1));
    }

    // --- ClientBudgets: unrestricted clients --------------------------------------

    #[test]
    fn check_is_ok_for_a_client_with_no_configured_budget() {
        let budgets = ClientBudgets::from_clients(&[client("acme", None, BudgetPeriod::Total)]);
        assert!(budgets.check("acme").is_ok());
    }

    #[test]
    fn check_is_ok_for_an_entirely_unknown_client_name() {
        let budgets = ClientBudgets::from_clients(&[]);
        assert!(budgets.check("nobody").is_ok());
    }

    #[test]
    fn record_is_a_no_op_for_a_client_with_no_configured_budget() {
        let budgets = ClientBudgets::from_clients(&[client("acme", None, BudgetPeriod::Total)]);
        budgets.record_at("acme", 1_000_000.0, 0);
        assert!(budgets.check_at("acme", 0).is_ok());
    }

    // --- ClientBudgets: total period -----------------------------------------------

    #[test]
    fn total_period_accumulates_spend_across_multiple_records() {
        let budgets =
            ClientBudgets::from_clients(&[client("acme", Some(10.0), BudgetPeriod::Total)]);
        budgets.record_at("acme", 4.0, 0);
        budgets.record_at("acme", 4.0, 0);
        assert!(budgets.check_at("acme", 0).is_ok());
        budgets.record_at("acme", 4.0, 0);
        assert_eq!(
            budgets.check_at("acme", 0).unwrap_err(),
            BudgetExceeded {
                spent_usd: 12.0,
                budget_usd: 10.0
            }
        );
    }

    #[test]
    fn total_period_check_fails_exactly_at_the_budget_not_only_over_it() {
        let budgets =
            ClientBudgets::from_clients(&[client("acme", Some(10.0), BudgetPeriod::Total)]);
        budgets.record_at("acme", 10.0, 0);
        assert!(budgets.check_at("acme", 0).is_err());
    }

    #[test]
    fn total_period_never_resets_across_different_now_unix_values() {
        let budgets =
            ClientBudgets::from_clients(&[client("acme", Some(5.0), BudgetPeriod::Total)]);
        budgets.record_at("acme", 5.0, 0);
        // A "now" far in the future (a different month/year) must not
        // reset a Total-period client's accumulated spend.
        assert!(budgets.check_at("acme", 1_800_000_000).is_err());
    }

    #[test]
    fn total_period_independent_clients_do_not_share_spend() {
        let budgets = ClientBudgets::from_clients(&[
            client("acme", Some(10.0), BudgetPeriod::Total),
            client("globex", Some(10.0), BudgetPeriod::Total),
        ]);
        budgets.record_at("acme", 10.0, 0);
        assert!(budgets.check_at("acme", 0).is_err());
        assert!(budgets.check_at("globex", 0).is_ok());
    }

    // --- ClientBudgets: monthly period -----------------------------------------------

    #[test]
    fn monthly_period_accumulates_within_the_same_month() {
        let budgets =
            ClientBudgets::from_clients(&[client("acme", Some(10.0), BudgetPeriod::Monthly)]);
        let jan_1 = 1_704_067_200; // 2024-01-01T00:00:00Z
        let jan_15 = jan_1 + 14 * 86_400;
        budgets.record_at("acme", 6.0, jan_1);
        budgets.record_at("acme", 6.0, jan_15);
        assert!(budgets.check_at("acme", jan_15).is_err());
    }

    #[test]
    fn monthly_period_resets_when_the_month_rolls_over() {
        let budgets =
            ClientBudgets::from_clients(&[client("acme", Some(10.0), BudgetPeriod::Monthly)]);
        let jan_31 = 1_706_745_599; // 2024-01-31T23:59:59Z
        let feb_1 = 1_706_745_600; // 2024-02-01T00:00:00Z
        budgets.record_at("acme", 10.0, jan_31);
        assert!(budgets.check_at("acme", jan_31).is_err());
        // A new month: the cap must no longer be exceeded.
        assert!(budgets.check_at("acme", feb_1).is_ok());
    }

    #[test]
    fn monthly_period_record_after_rollover_starts_a_fresh_period() {
        let budgets =
            ClientBudgets::from_clients(&[client("acme", Some(10.0), BudgetPeriod::Monthly)]);
        let jan_31 = 1_706_745_599;
        let feb_1 = 1_706_745_600;
        budgets.record_at("acme", 10.0, jan_31);
        budgets.record_at("acme", 3.0, feb_1);
        assert!(budgets.check_at("acme", feb_1).is_ok());
        budgets.record_at("acme", 7.0, feb_1);
        assert!(budgets.check_at("acme", feb_1).is_err());
    }
}
