use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Instant;

/// A token bucket with a given capacity, refilling continuously at
/// `capacity` tokens per 60 seconds (i.e. `capacity` is a requests-per-
/// minute rate). Refill happens lazily, computed from elapsed wall-clock
/// time on each access rather than on a timer.
struct TokenBucket {
    capacity: f64,
    tokens: f64,
    last_refill: Instant,
}

impl TokenBucket {
    fn new(capacity: f64) -> Self {
        Self {
            capacity,
            tokens: capacity,
            last_refill: Instant::now(),
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed_secs = now.duration_since(self.last_refill).as_secs_f64();
        self.last_refill = now;
        self.tokens = (self.tokens + elapsed_secs * (self.capacity / 60.0)).min(self.capacity);
    }

    /// Seconds from now until this bucket is back to full capacity --
    /// always computable regardless of whether the most recent
    /// `try_acquire` succeeded or failed, for `X-RateLimit-Reset`.
    fn reset_secs(&self) -> f64 {
        if self.capacity <= 0.0 {
            0.0
        } else {
            (self.capacity - self.tokens) / (self.capacity / 60.0)
        }
    }

    /// Consume one token if available. `Ok`/`Err` both carry a
    /// [`RateLimitStatus`] snapshot -- the only difference is `remaining`
    /// (`0` on failure) and `retry_after_secs` (`0.0` on success).
    fn try_acquire(&mut self) -> Result<RateLimitStatus, RateLimitStatus> {
        self.refill();
        let limit = self.capacity.max(0.0) as u32;
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            Ok(RateLimitStatus {
                limit,
                remaining: self.tokens as u32,
                reset_secs: self.reset_secs(),
                retry_after_secs: 0.0,
            })
        } else if self.capacity <= 0.0 {
            // A zero/negative capacity bucket never refills usefully;
            // report a fixed cooldown rather than an infinite/undefined wait.
            Err(RateLimitStatus {
                limit,
                remaining: 0,
                reset_secs: 60.0,
                retry_after_secs: 60.0,
            })
        } else {
            Err(RateLimitStatus {
                limit,
                remaining: 0,
                reset_secs: self.reset_secs(),
                retry_after_secs: (1.0 - self.tokens) / (self.capacity / 60.0),
            })
        }
    }
}

/// A snapshot of one rate-limit bucket's state as of a `RateLimiter::check`
/// call, carrying everything needed for `X-RateLimit-*` response headers
/// (and, on failure, `Retry-After`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateLimitStatus {
    /// The bucket's configured capacity, in requests per minute.
    pub limit: u32,
    /// Tokens left in the bucket right after this check -- always `0` on
    /// a failed check.
    pub remaining: u32,
    /// Seconds from now until the bucket is back to full capacity.
    pub reset_secs: f64,
    /// Seconds from now until at least one token will be available again
    /// -- `0.0` on a successful check, since one already was.
    pub retry_after_secs: f64,
}

/// A named collection of independent token-bucket rate limiters, keyed by
/// an arbitrary identity string (a client name, provider name, or IP
/// address). Each key's capacity (requests per minute) is supplied by the
/// caller at check time rather than fixed on construction, so one
/// `RateLimiter` can back many identities with different limits — e.g.
/// every configured client, or every provider, each with its own rate.
#[derive(Default)]
pub struct RateLimiter {
    buckets: RwLock<HashMap<String, TokenBucket>>,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Attempt to consume one token from `key`'s bucket, creating it with
    /// `requests_per_minute` capacity if this is the first time `key` has
    /// been seen. `Err` if the bucket is currently empty; either way, the
    /// returned [`RateLimitStatus`] carries what a caller needs for
    /// `X-RateLimit-*` (and, on `Err`, `Retry-After`) response headers.
    pub fn check(
        &self,
        key: &str,
        requests_per_minute: u32,
    ) -> Result<RateLimitStatus, RateLimitStatus> {
        let mut buckets = self.buckets.write().unwrap();
        let bucket = buckets
            .entry(key.to_string())
            .or_insert_with(|| TokenBucket::new(requests_per_minute as f64));
        bucket.try_acquire()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn allows_up_to_capacity_then_rejects() {
        let limiter = RateLimiter::new();
        assert!(limiter.check("a", 3).is_ok());
        assert!(limiter.check("a", 3).is_ok());
        assert!(limiter.check("a", 3).is_ok());

        let status = limiter.check("a", 3).unwrap_err();
        assert_eq!(status.limit, 3);
        assert_eq!(status.remaining, 0);
        assert!(
            status.retry_after_secs > 0.0 && status.retry_after_secs <= 60.0,
            "unexpected retry_after_secs: {}",
            status.retry_after_secs
        );
    }

    #[test]
    fn independent_keys_have_independent_buckets() {
        let limiter = RateLimiter::new();
        assert!(limiter.check("a", 1).is_ok());
        assert!(limiter.check("a", 1).is_err());
        // "b" has never been touched, so it starts with a full bucket
        // regardless of "a" being exhausted.
        assert!(limiter.check("b", 1).is_ok());
    }

    #[test]
    fn refills_over_time() {
        let limiter = RateLimiter::new();
        // 6000 requests/minute = 100 tokens/sec, so a short sleep refills
        // meaningfully without slowing the test suite down.
        for _ in 0..6000 {
            limiter.check("fast", 6000).unwrap();
        }
        assert!(limiter.check("fast", 6000).is_err());

        sleep(Duration::from_millis(50));
        assert!(limiter.check("fast", 6000).is_ok());
    }

    #[test]
    fn zero_capacity_always_rejects_with_fixed_cooldown() {
        let limiter = RateLimiter::new();
        let status = limiter.check("none", 0).unwrap_err();
        assert_eq!(status.limit, 0);
        assert_eq!(status.remaining, 0);
        assert_eq!(status.retry_after_secs, 60.0);
        assert_eq!(status.reset_secs, 60.0);
        assert!(limiter.check("none", 0).is_err());
    }

    #[test]
    fn first_use_of_a_key_seeds_a_full_bucket() {
        let limiter = RateLimiter::new();
        // The very first check for a key should succeed even though no
        // time has passed to "earn" a token -- new buckets start full.
        assert!(limiter.check("fresh", 5).is_ok());
    }

    #[test]
    fn successful_check_reports_limit_and_decremented_remaining() {
        let limiter = RateLimiter::new();
        let first = limiter.check("a", 5).unwrap();
        assert_eq!(first.limit, 5);
        assert_eq!(first.remaining, 4);
        assert_eq!(first.retry_after_secs, 0.0);

        let second = limiter.check("a", 5).unwrap();
        assert_eq!(second.remaining, 3);
    }

    #[test]
    fn reset_secs_reflects_the_deficit_left_by_this_check() {
        let limiter = RateLimiter::new();
        // Capacity 10 => refills at 10/60 tokens/sec. Consuming the first
        // token from a freshly-seeded full bucket leaves a 1-token
        // deficit, which takes 1 / (10/60) = 6 seconds to refill.
        let status = limiter.check("fresh", 10).unwrap();
        assert!(
            (status.reset_secs - 6.0).abs() < 0.01,
            "expected ~6.0, got {}",
            status.reset_secs
        );
    }
}
