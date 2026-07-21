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

    /// Consume one token if available. On failure, returns the number of
    /// seconds until a token will next be available.
    fn try_acquire(&mut self) -> Result<(), f64> {
        self.refill();
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            Ok(())
        } else if self.capacity <= 0.0 {
            // A zero/negative capacity bucket never refills usefully;
            // report a fixed cooldown rather than an infinite/undefined wait.
            Err(60.0)
        } else {
            Err((1.0 - self.tokens) / (self.capacity / 60.0))
        }
    }
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
    /// been seen. Returns `Err(seconds_until_retry)` if the bucket is
    /// currently empty.
    pub fn check(&self, key: &str, requests_per_minute: u32) -> Result<(), f64> {
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

        let retry_after = limiter.check("a", 3).unwrap_err();
        assert!(
            retry_after > 0.0 && retry_after <= 60.0,
            "unexpected retry_after: {retry_after}"
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
        assert_eq!(limiter.check("none", 0), Err(60.0));
        assert!(limiter.check("none", 0).is_err());
    }

    #[test]
    fn first_use_of_a_key_seeds_a_full_bucket() {
        let limiter = RateLimiter::new();
        // The very first check for a key should succeed even though no
        // time has passed to "earn" a token -- new buckets start full.
        assert!(limiter.check("fresh", 5).is_ok());
    }
}
