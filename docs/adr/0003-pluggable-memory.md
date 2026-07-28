# ADR-0003: Pluggable memory

Status: Proposed
Date: 2026-07-28

## Context

Every harness makes you configure your knowledge separately. Claude Code
has its memory, Cursor has its own, an editor over ACP has none. The same
notes, decisions and preferences get set up repeatedly, in different
formats, and drift apart. The intent behind this router is that the
environment around a model — tools, knowledge, preferences — is configured
once here and inherited by whatever is driving.

Model access is already solved that way. `Provider` is a narrow trait in
`rp-core` with three adapters behind it, and the router never branches on
provider identity. Memory should arrive the same way, and specifically it
should *not* arrive as an integration with one product.

There is a real backend to design against — `remind_me`, a hybrid
FTS5-plus-vector store with a knowledge graph, vitality scoring and a
synthesis layer, reachable over MCP. It is a good worked example and a bad
thing to hardwire. Most memory backends in this ecosystem are MCP servers,
which is a fact the design should exploit rather than a reason to bind to
any one of them.

The seam already exists. `Router::apply_web_search` runs before dispatch,
mutates the request to add retrieved context, and is a silent no-op when
unconfigured. Memory is that shape, plus a write path.

## Decision

Add a `Memory` port to `rp-core`, adapters in a new `crates/memory`
(`rp-memory`), and a `[[memory]]` config array. Recall runs as a
pre-dispatch pipeline stage next to web search. `remind_me` is a config
block, not a code path.

### The port

```rust
#[async_trait]
pub trait Memory: Send + Sync {
    async fn recall(&self, query: &RecallQuery) -> Result<Vec<Recollection>, MemoryError>;
    async fn remember(&self, item: &Remembrance) -> Result<(), MemoryError>;
    fn capabilities(&self) -> MemoryCapabilities;
}

pub struct RecallQuery {
    pub text: String,
    pub token_budget: u32,
    /// Opaque namespace, typically the matched `[[clients]]` name.
    pub scope: Option<String>,
    pub limit: u32,
}

pub struct Recollection {
    pub text: String,
    pub id: Option<String>,
    pub score: Option<f32>,
    pub source: Option<String>,
}

pub struct Remembrance {
    pub text: String,
    pub scope: Option<String>,
    pub tags: Vec<String>,
}

pub struct MemoryCapabilities {
    pub writes: bool,
    pub scoping: bool,
}
```

The trait is the narrow waist. Backend richness — entity graphs, triples,
vitality decay, wiki synthesis, retrieval strategy profiles — stays in
adapter config and never reaches the port. This is the same rule the
provider adapters already follow for parameters like `min_p` and
`cache_control`: an unsupported capability degrades quietly instead of
erroring, because recall is an enrichment, not a correctness requirement.

`capabilities()` exists so the router can report honestly at startup
rather than guess. A backend with `scoping: false` configured alongside
more than one `[[clients]]` entry logs a warning once, because silently
sharing one person's memories with another caller is a surprise worth
making loud.

### Adapters

`mcp` is the adapter that matters. Most memory backends already speak MCP,
so one generic adapter with configurable tool names and argument mapping
covers the category:

```toml
[[memory]]
name = "personal"
adapter = "mcp"
url = "http://127.0.0.1:5199/mcp"
auth_env = "REMIND_ME_TOKEN"
search_tool = "remind_me_search"
store_tool  = "remind_me_add"

[memory.search_args]        # maps port fields onto that server's schema
query = "{{text}}"
token_budget = "{{budget}}"
limit = "{{limit}}"
```

`http` follows: a plain REST endpoint with mapped request and response,
the same shape as the existing moderation and web-search clients. A
built-in SQLite adapter is worth adding eventually so the feature works
with nothing else installed, but it is not needed to prove the port.

Adapters live in their own crate rather than in `rp-router`. Moderation
and web search are router modules because each has exactly one
implementation; memory is multi-implementation by definition. It also
stops the router — already 6,468 lines — from absorbing another
subsystem.

### Config

```toml
[[memory]]
name = "personal"           # required; identifies the entry in logs and metrics
adapter = "mcp"
recall = "auto"             # auto | tool | off   (default: auto)
write_back = "off"          # off  | tool | exchange (default: off)
token_budget = 800
limit = 10
scope_by_client = true
timeout_ms = 2000
```

An array from the first commit, not a single table. Personal notes and
project knowledge are different stores and will both be wanted;
singular-to-plural is a breaking config change later. Absent means memory
is off, matching every other optional subsystem.

`recall = "auto"` injects on every request — the mode that makes knowledge
follow you into a client that knows nothing about this router.
`recall = "tool"` exposes search as a tool and lets the model decide:
cheaper and more precise, but only works where the model has tools.

`write_back` defaults to `off` deliberately. Writing every exchange into a
memory store poisons it with transient chatter; `tool` (expose a
`remember` tool, let the model choose) is the mode worth using, and
`exchange` exists for people who genuinely want everything.

### Pipeline placement

Recall runs immediately after `apply_web_search` and therefore *before*
guardrails and moderation. That ordering is deliberate: recalled text is
content the model will see, so it should be subject to the same redaction
and classification as anything the user typed. Placing memory later would
create a hole in content policy.

The query is the latest user message's plain text, via
`MessageContent::as_plain_text()`, truncated — the same input web search
already uses.

Recollections are injected as a single `Role::System` message inserted
immediately before the last user message, not merged into the user's own
words. Two reasons: the model can tell recalled context from what the
person actually said, and a late insertion leaves the static prefix
(system prompt, earlier turns) byte-identical, so Anthropic cache
breakpoints and OpenAI's automatic prefix caching still hit. Prepending to
the system prompt would invalidate the cacheable prefix on every request.
A `placement = "user_message"` option covers models that handle system
messages poorly.

### Failure semantics

Fail open, always, with a timeout. An unreachable, slow or erroring memory
backend means the request proceeds without recall — never a failed
request, never added latency beyond `timeout_ms`. This matches the
existing rule for the three auxiliary HTTP backends, which ARCHITECTURE.md
already states: "their own unavailability never blocks or fails the
request that triggered them."

`remember` failures are logged and dropped on the same basis.

Metrics: `rp_memory_recall_total{name,outcome}` and
`rp_memory_recall_duration_seconds{name}`, so a slow backend on the hot
path is visible rather than felt.

## Alternatives considered

**Integrate `remind_me` directly.** Fastest path, and it would work well.
Rejected because the stated goal is one place to configure your
environment, and binding to one store makes that place opinionated in a
way it does not need to be. The port costs little more than the direct
integration; the generic MCP adapter *is* the `remind_me` integration.

**Treat memory as nothing but MCP tools.** Tempting, since these backends
are MCP servers and a general MCP port is wanted anyway. Rejected because
it only delivers tool-driven recall. Automatic injection is what makes
memory work in a client that has no idea this router exists, and that is
the case the whole vision rests on. Memory also has a distinct failure
posture — an unavailable memory store should silently degrade, while an
unavailable MCP tool the model explicitly called should be reported back
to it.

**Put adapters in `rp-router` beside moderation and web search.** Fewer
moving parts, and consistent with how the existing auxiliary clients are
organised. Rejected because those are single-implementation clients and
this is not, and because the router is already the crate most in need of
not growing.

**A vector store as the port instead.** Modelling embeddings, collections
and similarity directly would be a lower-level and more honest interface
for some backends. Rejected because it forecloses backends that are not
vector stores — a wiki, a graph, a notes folder — and because the router
has no business owning retrieval strategy.

## Consequences

- Injected recall becomes part of the request before `Router::dispatch`
  computes its cache key, so `[cache]` hit rates will fall wherever recall
  is on. This is already true of web search; memory makes it common rather
  than occasional.
- Recall on every request costs tokens on every request. An 800-token
  budget is not free at volume, which is the main argument for
  `recall = "tool"` on high-traffic clients and `auto` on interactive
  ones.
- Some backends learn from being queried. `remind_me`, for example,
  reinforces co-retrieval associations on every search regardless of flags.
  Automatic recall would train that graph on machine-generated queries
  rather than human intent. `RecallQuery` should carry a `reinforce: bool`
  hint that adapters map where the backend supports it; where it does not,
  this is a genuine reason to prefer `recall = "tool"`.
- Scoping is only as good as the backend. Where `capabilities().scoping`
  is false, multi-client deployments share one memory. The warning covers
  the surprise; it does not fix it.
- `rp-acp` gets this for free — the ACP turn loop already calls the same
  pre-dispatch stages, so an agent session inherits recall with no changes
  in that crate.

## Phasing

1. Port in `rp-core`, `mcp` adapter in `rp-memory`, `[[memory]]` config,
   recall stage in the pipeline, `recall = "auto"` only. Tests: a wiremock
   MCP server proving injection placement and budget trimming, and a dead
   backend proving the request still succeeds.
2. `write_back` and the `remember` tool.
3. `recall = "tool"` mode, which lands naturally with the MCP tool work.
4. `http` and built-in SQLite adapters.

Phase 1 is the whole thesis: configure knowledge once, and any client
pointed at this router inherits it.
