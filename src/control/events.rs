use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crossbeam_channel::Sender;
use serde::Serialize;
use serde_json::{Value, json};

#[derive(Debug, Clone, Serialize)]
pub struct EventEnvelope {
    pub event: String,
    pub sequence: u64,
    pub revision: u64,
    pub source: String,
    pub data: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initiating_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initiating_request_id: Option<Value>,
}

#[derive(Debug)]
struct Subscriber {
    patterns: Vec<String>,
    outbound: Sender<Value>,
    dropped: u64,
}

#[derive(Debug, Default)]
pub struct EventHub {
    revision: AtomicU64,
    sequence: AtomicU64,
    subscribers: Mutex<HashMap<String, Subscriber>>,
}

impl EventHub {
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn revision(&self) -> u64 {
        self.revision.load(Ordering::Acquire)
    }

    pub fn next_revision(&self) -> u64 {
        self.revision.fetch_add(1, Ordering::AcqRel) + 1
    }

    pub fn register(&self, session_id: String, outbound: Sender<Value>) {
        self.subscribers
            .lock()
            .expect("event subscribers poisoned")
            .insert(
                session_id,
                Subscriber {
                    patterns: Vec::new(),
                    outbound,
                    dropped: 0,
                },
            );
    }

    pub fn remove(&self, session_id: &str) {
        self.subscribers
            .lock()
            .expect("event subscribers poisoned")
            .remove(session_id);
    }

    pub fn subscribe(
        &self,
        session_id: &str,
        patterns: Vec<String>,
    ) -> Result<Vec<String>, &'static str> {
        if patterns.is_empty() || patterns.iter().any(|pattern| !valid_pattern(pattern)) {
            return Err("events must contain one or more non-empty event patterns");
        }
        let mut subscribers = self.subscribers.lock().expect("event subscribers poisoned");
        let subscriber = subscribers
            .get_mut(session_id)
            .ok_or("control session is not registered")?;
        for pattern in patterns {
            if !subscriber.patterns.contains(&pattern) {
                subscriber.patterns.push(pattern);
            }
        }
        subscriber.patterns.sort();
        Ok(subscriber.patterns.clone())
    }

    pub fn unsubscribe(&self, session_id: &str, patterns: Option<&[String]>) -> Vec<String> {
        let mut subscribers = self.subscribers.lock().expect("event subscribers poisoned");
        let Some(subscriber) = subscribers.get_mut(session_id) else {
            return Vec::new();
        };
        match patterns {
            Some(patterns) => subscriber
                .patterns
                .retain(|registered| !patterns.contains(registered)),
            None => subscriber.patterns.clear(),
        }
        subscriber.patterns.clone()
    }

    pub fn status(&self, session_id: &str) -> Value {
        let subscribers = self.subscribers.lock().expect("event subscribers poisoned");
        let subscriber = subscribers.get(session_id);
        json!({
            "revision": self.revision(),
            "next_sequence": self.sequence.load(Ordering::Acquire) + 1,
            "subscriptions": subscriber.map(|item| item.patterns.clone()).unwrap_or_default(),
            "dropped_events": subscriber.map(|item| item.dropped).unwrap_or_default(),
        })
    }

    pub fn diagnostics(&self) -> Value {
        let subscribers = self.subscribers.lock().expect("event subscribers poisoned");
        json!({
            "revision": self.revision(),
            "event_sequence": self.sequence.load(Ordering::Acquire),
            "connected_sessions": subscribers.len(),
            "subscribed_sessions": subscribers.values().filter(|item| !item.patterns.is_empty()).count(),
            "dropped_events": subscribers.values().map(|item| item.dropped).sum::<u64>(),
        })
    }

    pub fn publish(
        &self,
        event: impl Into<String>,
        source: impl Into<String>,
        revision: u64,
        data: Value,
        initiating_session_id: Option<String>,
        initiating_request_id: Option<Value>,
    ) {
        let event = event.into();
        let sequence = self.sequence.fetch_add(1, Ordering::AcqRel) + 1;
        let notification = json!({
            "jsonrpc": "2.0",
            "method": "events.event",
            "params": EventEnvelope {
                event: event.clone(),
                sequence,
                revision,
                source: source.into(),
                data,
                initiating_session_id,
                initiating_request_id,
            }
        });
        let mut subscribers = self.subscribers.lock().expect("event subscribers poisoned");
        for subscriber in subscribers.values_mut() {
            if subscriber
                .patterns
                .iter()
                .any(|pattern| pattern_matches(pattern, &event))
                && subscriber.outbound.try_send(notification.clone()).is_err()
            {
                subscriber.dropped += 1;
            }
        }
    }
}

fn valid_pattern(pattern: &str) -> bool {
    !pattern.trim().is_empty()
        && !pattern.chars().any(char::is_whitespace)
        && pattern.matches('*').count() <= 1
        && (!pattern.contains('*') || pattern.ends_with('*'))
}

fn pattern_matches(pattern: &str, event: &str) -> bool {
    pattern == "*"
        || pattern == event
        || pattern
            .strip_suffix('*')
            .is_some_and(|prefix| event.starts_with(prefix))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subscriptions_match_prefixes_and_track_revisions() {
        let hub = EventHub::shared();
        let (tx, rx) = crossbeam_channel::bounded(2);
        hub.register("session".into(), tx);
        hub.subscribe("session", vec!["viewer.camera.*".into()])
            .expect("subscribe");
        let revision = hub.next_revision();
        hub.publish(
            "viewer.camera.changed",
            "viewer:active",
            revision,
            json!({"zoom": 2}),
            None,
            None,
        );
        let notification = rx.recv().expect("event notification");
        assert_eq!(notification["params"]["sequence"], 1);
        assert_eq!(notification["params"]["revision"], 1);
        assert_eq!(notification["params"]["event"], "viewer.camera.changed");
    }

    #[test]
    fn slow_subscribers_drop_events_without_blocking_publishers() {
        let hub = EventHub::shared();
        let (tx, _rx) = crossbeam_channel::bounded(1);
        hub.register("session".into(), tx);
        hub.subscribe("session", vec!["*".into()])
            .expect("subscribe");
        const EVENT_COUNT: u64 = 10_000;
        let started = std::time::Instant::now();
        for index in 0..EVENT_COUNT {
            hub.publish("pressure", "app", 0, json!({"index":index}), None, None);
        }
        let elapsed = started.elapsed();
        println!(
            "slow-subscriber pressure: events={EVENT_COUNT} total_us={} average_ns={}",
            elapsed.as_micros(),
            elapsed.as_nanos() / u128::from(EVENT_COUNT),
        );
        assert_eq!(hub.status("session")["dropped_events"], EVENT_COUNT - 1);
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "publishing into a saturated subscriber took {elapsed:?}; publishers must not block"
        );
    }
}
