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

//! Prompt cache keepalive while the session waits.
//!
//! While the session waits — for the user's reply or for a long tool run —
//! a cache pre-warm request (no output generated, only a cache read billed)
//! replays the cached prefix so the provider's TTL keeps resetting and the
//! next real request still hits cache.
//!
//! How many pings to spend is an economic decision, never a fixed cap. Each
//! ping costs one cache read of the whole prefix; letting the entry expire
//! costs a write instead of a read of the tail that expires with it (tools,
//! system prompt and 1h-marked messages stay warm on their own). That makes
//! this a ski rental problem: `max_pings = (write − read) / read × tail share`
//! is the most that can ever pay off. Within that bound the budget minimizes
//! the expected cost over the waits this session has already seen; a known
//! wake (a scheduled turn) is bridged exactly when it is affordable.
//!
//! Read-only on session state: the snapshot is owned by the keepalive task and
//! `session.messages` is never mutated by a ping. [`KeepaliveHandle::finish`]
//! folds ping costs and the measured wait back into the session.
//!
//! Provider-aware: a ping only fires when the resolved provider returns
//! `Some(KeepalivePolicy)` and has pricing. Today that is Anthropic and
//! OpenRouter Claude routes.

use crate::config::{Config, ModelProfile};
use crate::providers::{ChatCompletionParams, ProviderExchange, ProviderFactory};
use crate::session::chat::{ChatSession, CostTracker};
use crate::session::Message;
use anyhow::Result;
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tokio::task::JoinHandle;

/// TTL marker value for the long (1h) Anthropic cache entry.
const LONG_CACHE_TTL: &str = "1h";

/// Prices are decimal dollars stored as binary floats: (5.00 − 0.20) / 0.20
/// lands at 23.999…, so the floor needs this much slack to give 24.
const PRICE_RATIO_TOLERANCE: f64 = 1e-9;

/// What the session waits on while a keepalive runs. Each kind keeps its own
/// history: how fast a user replies says nothing about how long builds take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitKind {
	/// Waiting for the next user / client / scheduled message.
	Reply,
	/// Waiting for the current turn's tool calls to finish.
	Tools,
}

/// Observed waits per kind (runtime-only, never persisted). Feeds the
/// expected-cost budget; empty history falls back to the ski-rental bound.
#[derive(Debug, Clone, Default)]
pub struct WaitHistory {
	reply: Vec<Duration>,
	tools: Vec<Duration>,
}

impl WaitHistory {
	pub fn record(&mut self, kind: WaitKind, wait: Duration) {
		self.samples_mut(kind).push(wait);
	}

	pub fn samples(&self, kind: WaitKind) -> &[Duration] {
		match kind {
			WaitKind::Reply => &self.reply,
			WaitKind::Tools => &self.tools,
		}
	}

	fn samples_mut(&mut self, kind: WaitKind) -> &mut Vec<Duration> {
		match kind {
			WaitKind::Reply => &mut self.reply,
			WaitKind::Tools => &mut self.tools,
		}
	}
}

/// A running keepalive. Finish it with [`KeepaliveHandle::finish`]: dropping
/// it stops the pings (the cancel channel closes) but loses their cost.
pub struct KeepaliveHandle {
	cancel_tx: watch::Sender<bool>,
	task: JoinHandle<Vec<ProviderExchange>>,
	kind: WaitKind,
	started: Instant,
}

impl KeepaliveHandle {
	/// Start a keepalive for a wait of `kind`. Returns `None` when keepalive
	/// is disabled, nothing is cached, the session requests structured output
	/// (the API rejects a pre-warm with an output schema), or the provider has
	/// no keepalive policy or pricing. A zero ping budget still returns a
	/// handle: the wait is measured either way, so the next budget learns.
	pub fn spawn(chat_session: &ChatSession, config: &Config, kind: WaitKind) -> Option<Self> {
		if !config.cache_keepalive_enabled || chat_session.schema.is_some() {
			return None;
		}
		let messages = &chat_session.session.messages;
		// The ping replays exactly the prefix the last request cached: up to
		// its newest marker. Anything after it (the assistant reply, the
		// runtime memory pack) was never cached and would only add input.
		let last_cached = messages.iter().rposition(|m| m.cached)?;
		let snapshot = messages[..=last_cached].to_vec();
		let profile = chat_session.model_profile(config);
		let (provider, actual_model) = ProviderFactory::get_provider_for_model(&profile.model).ok()?;
		let long_ttl = snapshot[last_cached].cache_ttl.as_deref() == Some(LONG_CACHE_TTL);
		let policy = provider.keepalive_policy(&actual_model, long_ttl)?;
		let pricing = provider.get_model_pricing(&actual_model)?;

		let penalty = rewrite_penalty_in_pings(
			pricing.cache_write_price_per_1m,
			pricing.cache_read_price_per_1m,
			tail_share(&snapshot, chat_session.cached_tools.as_deref(), &profile.model),
		);
		let known_wake = match kind {
			WaitKind::Reply => crate::mcp::orchestration::next_due_in(),
			WaitKind::Tools => None,
		};
		let budget = ping_budget(
			chat_session.wait_history.samples(kind),
			known_wake,
			policy.interval,
			penalty,
		);

		let (cancel_tx, cancel_rx) = watch::channel(false);
		let ping_loop = run(snapshot, profile, config.clone(), policy.interval, budget, cancel_rx);
		// A spawned task drops the task-local session: without it the ping
		// would assemble the global tool list instead of this session's, and
		// a different tools block misses the cache.
		let session_id = crate::session::context::current_session_id();
		let task = tokio::spawn(async move {
			match session_id {
				Some(id) => crate::session::context::with_session_id(id, ping_loop).await,
				None => ping_loop.await,
			}
		});

		crate::log_debug!(
			"Cache keepalive started: kind={:?}, interval={}s, budget={} ping(s), rewrite penalty={:.1} ping(s)",
			kind,
			policy.interval.as_secs(),
			budget,
			penalty
		);

		Some(Self {
			cancel_tx,
			task,
			kind,
			started: Instant::now(),
		})
	}

	/// Stop the task and fold its ping costs into the session. `waited`
	/// records the elapsed time as a completed wait — pass `false` when the
	/// wait ended without its answer (disconnect, unload, session exit), so
	/// an abandoned wait never skews the next budget. Returns the ping count.
	pub async fn finish(self, chat_session: &mut ChatSession, config: &Config, waited: bool) -> usize {
		let elapsed = self.started.elapsed();
		let _ = self.cancel_tx.send(true);
		let exchanges = match self.task.await {
			Ok(exchanges) => exchanges,
			Err(e) => {
				crate::log_debug!("Cache keepalive task join error: {}", e);
				Vec::new()
			}
		};
		if waited {
			chat_session.wait_history.record(self.kind, elapsed);
		}
		for exchange in &exchanges {
			if let Err(e) = CostTracker::track_exchange_cost(chat_session, exchange, config) {
				crate::log_debug!("Failed to track keepalive cost: {}", e);
			}
		}
		crate::log_debug!(
			"Cache keepalive finished: kind={:?}, waited={}s, {} ping(s)",
			self.kind,
			elapsed.as_secs(),
			exchanges.len()
		);
		exchanges.len()
	}
}

/// Share of the pinged prefix that an expiry would actually re-write. A ping
/// reads the whole prefix, but the tools, the system prompt and any 1h-marked
/// message stay warm in their own long entry; only the tail after the last
/// long boundary expires with the newest marker. Without cached tool
/// definitions the tools are left out of both sides, which overstates the
/// share slightly (more pings), never understates it.
fn tail_share(snapshot: &[Message], tools: Option<&[crate::mcp::McpFunction]>, model: &str) -> f64 {
	let newest = snapshot.len() - 1;
	// The provider gives the cached system prompt the long TTL on every request.
	let long_end = snapshot[..newest]
		.iter()
		.rposition(|m| m.cache_ttl.as_deref() == Some(LONG_CACHE_TTL))
		.or_else(|| (snapshot[0].role == "system" && snapshot[0].cached).then_some(0))
		.map_or(0, |i| i + 1);
	let total = crate::session::estimate_sent_full_context_tokens(snapshot, tools, model) as f64;
	let long = crate::session::estimate_sent_full_context_tokens(&snapshot[..long_end], tools, model) as f64;
	if total <= 0.0 {
		return 0.0;
	}
	((total - long) / total).clamp(0.0, 1.0)
}

/// What letting the entry expire costs, in pings. A ping is one cache read of
/// the whole prefix; an expiry makes the next request write the tail instead
/// of reading it.
fn rewrite_penalty_in_pings(write_price: f64, read_price: f64, tail_share: f64) -> f64 {
	if read_price <= 0.0 {
		return 0.0;
	}
	((write_price - read_price) / read_price * tail_share).max(0.0)
}

/// Ski-rental bound: the most pings that can beat the rewrite.
fn max_pings(penalty: f64) -> u32 {
	(penalty + PRICE_RATIO_TOLERANCE).floor() as u32
}

/// Pings needed to keep the entry warm until `wait` ends. The interval is
/// below the TTL, so the entry outlives the last ping by at least one gap.
fn pings_to_bridge(wait: Duration, interval: Duration) -> u32 {
	(wait.as_millis() / interval.as_millis().max(1)) as u32
}

/// Pings to spend on this wait, in `0..=max_pings(penalty)`. A known wake
/// that is affordable is bridged exactly — a turn is certain then. Otherwise
/// pick the budget with the lowest total cost over past waits of this kind,
/// in pings: a wait bridged within the budget costs its pings, a longer one
/// costs the whole budget plus the rewrite penalty. No history yet means the
/// ski-rental bound, which is never worse than twice the optimum.
fn ping_budget(history: &[Duration], known_wake: Option<Duration>, interval: Duration, penalty: f64) -> u32 {
	let max = max_pings(penalty);
	if let Some(wake) = known_wake {
		let needed = pings_to_bridge(wake, interval);
		if needed <= max {
			return needed;
		}
	}
	if history.is_empty() {
		return max;
	}
	let expected_cost = |budget: u32| -> f64 {
		history
			.iter()
			.map(|wait| {
				let needed = pings_to_bridge(*wait, interval);
				if needed <= budget {
					f64::from(needed)
				} else {
					f64::from(budget) + penalty
				}
			})
			.sum()
	};
	// `min_by` keeps the first minimum, so ties choose the smaller budget.
	(0..=max)
		.min_by(|a, b| expected_cost(*a).total_cmp(&expected_cost(*b)))
		.unwrap_or(0)
}

async fn run(
	snapshot: Vec<Message>,
	profile: ModelProfile,
	config: Config,
	interval: Duration,
	budget: u32,
	mut cancel_rx: watch::Receiver<bool>,
) -> Vec<ProviderExchange> {
	let mut exchanges = Vec::new();

	for _ in 0..budget {
		tokio::select! {
			_ = tokio::time::sleep(interval) => {}
			res = cancel_rx.changed() => {
				// channel closed or cancel signalled
				if res.is_err() || *cancel_rx.borrow() {
					break;
				}
			}
		}

		// Re-check right before firing: a ping must never race the real
		// request that ended the wait.
		if *cancel_rx.borrow() {
			break;
		}

		// Best-effort: failures are logged and the next interval retries.
		// A keepalive must never abort the session.
		match send_ping(&snapshot, &profile, &config, cancel_rx.clone()).await {
			Ok(exchange) => {
				// A ping that read nothing re-wrote the prefix at the write
				// price: its request differs from the real one. Stop instead of
				// paying that again every interval.
				let missed = exchange.usage.as_ref().is_none_or(|u| u.cache_read_tokens == 0);
				exchanges.push(exchange);
				if missed {
					crate::log_info!("Cache keepalive: ping missed the cache, stopping");
					break;
				}
			}
			Err(e) => {
				crate::log_debug!("Cache keepalive ping failed: {}", e);
			}
		}
	}

	exchanges
}

async fn send_ping(
	messages: &[Message],
	profile: &ModelProfile,
	config: &Config,
	cancel_rx: watch::Receiver<bool>,
) -> Result<ProviderExchange> {
	let (provider, actual_model) = ProviderFactory::get_provider_for_model(&profile.model)?;

	// Same model, effort, tools and vision as the real request: any of them
	// differing changes the cached prefix and the ping would miss.
	let chat_params = ChatCompletionParams::new(
		messages,
		&actual_model,
		profile.temperature,
		profile.top_p,
		profile.top_k,
		profile.max_tokens,
		config,
	)
	.with_max_retries(0)
	.with_cancellation_token(cancel_rx)
	.with_reasoning_effort(profile.reasoning_effort)
	.with_vision(crate::session::model_utils::provider_supports_vision(
		provider.as_ref(),
		&actual_model,
	))
	.with_cache_prewarm();

	let octolib_params = chat_params
		.to_octolib_params(provider.name())
		.await
		.map_err(|e| anyhow::anyhow!("Failed to convert keepalive ping parameters: {}", e))?;

	let response = provider.chat_completion(octolib_params).await?;
	Ok(response.exchange)
}

#[cfg(test)]
#[path = "cache_keepalive_tests.rs"]
mod tests;
