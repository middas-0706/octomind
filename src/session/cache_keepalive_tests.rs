// Copyright 2026 Muvon Un Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;
use crate::session::chat::test_support;

/// Anthropic's ping interval for the 5m conversation cache (0.9 × TTL).
const FIVE_MINUTE_INTERVAL: Duration = Duration::from_secs(270);

fn message(role: &str, content: &str, cached: bool, cache_ttl: Option<&str>) -> Message {
	Message {
		role: role.to_string(),
		content: content.to_string(),
		cached,
		cache_ttl: cache_ttl.map(str::to_string),
		..Message::default()
	}
}

fn user_message(cached: bool) -> Message {
	message("user", "hello", cached, None)
}

/// Rewrite penalty of a prefix that is all tail, as on Opus/Sonnet 5.5.
const FULL_TAIL_PENALTY: f64 = 24.0;

fn keepalive_config() -> Config {
	let mut config = test_support::fake_provider_config();
	config.cache_keepalive_enabled = true;
	config
}

fn session(model: &str, messages: Vec<Message>) -> ChatSession {
	let mut session = ChatSession::for_tests(messages);
	session.model = model.to_string();
	session
}

fn minutes(n: u64) -> Duration {
	Duration::from_secs(n * 60)
}

// --- pricing bound ---------------------------------------------------------

#[test]
fn penalty_follows_the_price_ratio() {
	// Opus 5.5: write $5, read $0.20 → (5 − 0.2) / 0.2 = 24 — a float that
	// lands just below 24, so the bound must still floor to 24.
	assert_eq!(max_pings(rewrite_penalty_in_pings(5.00, 0.20, 1.0)), 24);
	// Fable 5.1: write $12.50, read $0.25 → 49.
	assert_eq!(max_pings(rewrite_penalty_in_pings(12.50, 0.25, 1.0)), 49);
	// A 0.1× read model (Opus 5): write $6.25, read $0.50 → 11.5 → 11.
	assert_eq!(max_pings(rewrite_penalty_in_pings(6.25, 0.50, 1.0)), 11);
}

#[test]
fn penalty_scales_with_the_expiring_share() {
	// Only a quarter of the prefix would be re-written: 24 × 0.25 = 6.
	assert_eq!(max_pings(rewrite_penalty_in_pings(5.00, 0.20, 0.25)), 6);
}

#[test]
fn penalty_is_zero_without_a_cache_discount() {
	assert_eq!(
		rewrite_penalty_in_pings(5.00, 0.0, 1.0),
		0.0,
		"free reads have no meaningful bound"
	);
	assert_eq!(
		rewrite_penalty_in_pings(2.00, 2.00, 1.0),
		0.0,
		"no write premium: never ping"
	);
}

#[test]
fn tail_share_excludes_the_long_cached_system_prompt() {
	let system = message("system", &"rules ".repeat(2_000), true, None);
	let snapshot = [system, message("user", "hello", true, None)];
	let share = tail_share(&snapshot, None, "anthropic:claude-sonnet-5-5");
	assert!(
		share > 0.0 && share < 0.1,
		"the short tail is a small share, got {}",
		share
	);
}

#[test]
fn tail_share_starts_after_the_last_long_marker() {
	let system = message("system", "rules", true, None);
	let summary = message("user", &"summary ".repeat(2_000), true, Some("1h"));
	let snapshot = [system, summary, message("user", "hello", true, None)];
	let share = tail_share(&snapshot, None, "anthropic:claude-sonnet-5-5");
	assert!(share < 0.1, "the 1h summary stays warm on its own, got {}", share);
}

#[test]
fn tail_share_is_zero_when_the_newest_marker_is_the_system_prompt() {
	let snapshot = [message("system", "rules", true, None)];
	assert_eq!(tail_share(&snapshot, None, "anthropic:claude-sonnet-5-5"), 0.0);
}

#[test]
fn pings_to_bridge_counts_whole_intervals() {
	assert_eq!(pings_to_bridge(minutes(20), FIVE_MINUTE_INTERVAL), 4);
	assert_eq!(
		pings_to_bridge(Duration::from_secs(200), FIVE_MINUTE_INTERVAL),
		0,
		"a wait shorter than one interval needs no ping"
	);
}

// --- budget ----------------------------------------------------------------

#[test]
fn budget_without_history_is_the_ski_rental_bound() {
	assert_eq!(ping_budget(&[], None, FIVE_MINUTE_INTERVAL, FULL_TAIL_PENALTY), 24);
}

#[test]
fn budget_bridges_an_affordable_known_wake_exactly() {
	let history = [minutes(200)];
	assert_eq!(
		ping_budget(&history, Some(minutes(20)), FIVE_MINUTE_INTERVAL, FULL_TAIL_PENALTY),
		4
	);
}

#[test]
fn budget_ignores_a_known_wake_beyond_the_bound() {
	// 3h needs 40 pings > 24: the wake is not worth bridging, so the budget
	// falls back to the bound (no history yet).
	assert_eq!(
		ping_budget(&[], Some(minutes(180)), FIVE_MINUTE_INTERVAL, FULL_TAIL_PENALTY),
		24
	);
}

#[test]
fn budget_covers_waits_the_session_keeps_having() {
	let history = [minutes(20), minutes(20), minutes(20)];
	assert_eq!(ping_budget(&history, None, FIVE_MINUTE_INTERVAL, FULL_TAIL_PENALTY), 4);
}

#[test]
fn budget_is_zero_when_waits_always_outlast_the_bound() {
	let history = [minutes(180), minutes(240)];
	assert_eq!(
		ping_budget(&history, None, FIVE_MINUTE_INTERVAL, FULL_TAIL_PENALTY),
		0,
		"pinging only adds cost when the cache expires anyway"
	);
}

#[test]
fn budget_is_zero_for_waits_shorter_than_one_interval() {
	let history = [Duration::from_secs(30), Duration::from_secs(90)];
	assert_eq!(ping_budget(&history, None, FIVE_MINUTE_INTERVAL, FULL_TAIL_PENALTY), 0);
}

#[test]
fn wait_history_keeps_kinds_apart() {
	let mut history = WaitHistory::default();
	history.record(WaitKind::Reply, minutes(10));
	history.record(WaitKind::Tools, minutes(3));
	assert_eq!(history.samples(WaitKind::Reply), &[minutes(10)]);
	assert_eq!(history.samples(WaitKind::Tools), &[minutes(3)]);
}

// --- spawn -----------------------------------------------------------------

#[tokio::test]
async fn spawn_returns_none_when_disabled() {
	let mut config = keepalive_config();
	config.cache_keepalive_enabled = false;
	let chat = session("anthropic:claude-sonnet-5-5", vec![user_message(true)]);
	assert!(KeepaliveHandle::spawn(&chat, &config, WaitKind::Reply).is_none());
}

#[tokio::test]
async fn spawn_returns_none_when_no_message_is_cached() {
	let chat = session("anthropic:claude-sonnet-5-5", vec![user_message(false)]);
	assert!(KeepaliveHandle::spawn(&chat, &keepalive_config(), WaitKind::Reply).is_none());
}

#[tokio::test]
async fn spawn_returns_none_for_a_structured_output_session() {
	// The API rejects a pre-warm that carries an output schema.
	let mut chat = session("anthropic:claude-sonnet-5-5", vec![user_message(true)]);
	chat.schema = Some(serde_json::json!({"type": "object"}));
	assert!(KeepaliveHandle::spawn(&chat, &keepalive_config(), WaitKind::Reply).is_none());
}

#[tokio::test]
async fn spawn_returns_none_for_unparseable_model() {
	let chat = session("no-provider-prefix", vec![user_message(true)]);
	assert!(KeepaliveHandle::spawn(&chat, &keepalive_config(), WaitKind::Reply).is_none());
}

#[tokio::test]
async fn spawn_returns_none_for_provider_without_keepalive_policy() {
	// OpenAI manages its prompt cache server-side; pinging it would burn
	// money for nothing.
	let chat = session("openai:gpt-5", vec![user_message(true)]);
	assert!(KeepaliveHandle::spawn(&chat, &keepalive_config(), WaitKind::Reply).is_none());
}

#[tokio::test]
async fn finish_records_a_completed_wait_and_skips_an_abandoned_one() {
	let config = keepalive_config();
	let mut chat = session("anthropic:claude-sonnet-5-5", vec![user_message(true)]);

	let handle = KeepaliveHandle::spawn(&chat, &config, WaitKind::Tools).expect("cached Anthropic session spawns");
	// The first ping is one interval away, so finishing now sends nothing.
	assert_eq!(handle.finish(&mut chat, &config, true).await, 0);
	assert_eq!(chat.wait_history.samples(WaitKind::Tools).len(), 1);

	let handle = KeepaliveHandle::spawn(&chat, &config, WaitKind::Tools).expect("spawns again");
	handle.finish(&mut chat, &config, false).await;
	assert_eq!(
		chat.wait_history.samples(WaitKind::Tools).len(),
		1,
		"an abandoned wait must not feed the budget"
	);
}

// --- run -------------------------------------------------------------------

fn profile(model: &str) -> ModelProfile {
	session(model, Vec::new()).model_profile(&keepalive_config())
}

#[tokio::test]
async fn run_sends_nothing_with_a_zero_budget() {
	let (_tx, rx) = watch::channel(false);
	let exchanges = run(
		vec![user_message(true)],
		profile("anthropic:claude-sonnet-5-5"),
		keepalive_config(),
		Duration::from_millis(1),
		0,
		rx,
	)
	.await;
	assert!(exchanges.is_empty());
}

#[tokio::test]
async fn run_stops_on_cancel_during_interval() {
	let (tx, rx) = watch::channel(false);
	let task = tokio::spawn(run(
		vec![user_message(true)],
		profile("anthropic:claude-sonnet-5-5"),
		keepalive_config(),
		FIVE_MINUTE_INTERVAL,
		24,
		rx,
	));
	tokio::time::sleep(Duration::from_millis(25)).await;
	tx.send(true).expect("cancel signal sent");
	let exchanges = task.await.expect("keepalive task finished cleanly");
	assert!(exchanges.is_empty());
}

#[tokio::test]
async fn run_stops_when_cancel_sender_drops() {
	let (tx, rx) = watch::channel(false);
	drop(tx);
	let exchanges = run(
		vec![user_message(true)],
		profile("anthropic:claude-sonnet-5-5"),
		keepalive_config(),
		FIVE_MINUTE_INTERVAL,
		24,
		rx,
	)
	.await;
	assert!(exchanges.is_empty());
}

#[tokio::test]
async fn run_spends_at_most_its_budget_on_failing_pings() {
	// Unparseable model: every ping fails before any network call, so only
	// the budget can end the loop.
	let (_tx, rx) = watch::channel(false);
	let exchanges = run(
		vec![user_message(true)],
		profile("no-provider-prefix"),
		keepalive_config(),
		Duration::from_millis(5),
		3,
		rx,
	)
	.await;
	assert!(exchanges.is_empty());
}

#[tokio::test]
#[serial_test::serial]
async fn run_stops_after_a_ping_that_misses_the_cache() {
	let _env = test_support::ENV_LOCK.lock().await;
	let url = test_support::spawn_stub(vec![
		test_support::final_response("ok"),
		test_support::final_response("ok"),
	])
	.await;
	std::env::set_var("OLLAMA_API_URL", &url);

	// The stub reports no cache read: the first ping re-wrote the prefix, so
	// the keepalive must stop instead of paying that every interval.
	let (_tx, rx) = watch::channel(false);
	let exchanges = run(
		vec![user_message(true)],
		profile("ollama:fake-model"),
		keepalive_config(),
		Duration::from_millis(10),
		5,
		rx,
	)
	.await;

	std::env::remove_var("OLLAMA_API_URL");

	assert_eq!(
		exchanges.len(),
		1,
		"the missed ping is billed once, then the loop stops"
	);
}
