# Odon Control Protocol v1

Status: experimental implementation; no stable compatibility promise yet

## Transport and framing

Version 1 uses UTF-8 JSON-RPC 2.0 messages over a persistent loopback TCP
connection. Each message is one JSON object followed by `\n`. Authenticated
connections use a bounded worker pool, requests may execute concurrently, and
responses are correlated by JSON-RPC request ID rather than response order.

Each GUI binds an ephemeral loopback endpoint and atomically publishes a
user-private runtime manifest. The manifest contains an instance ID, endpoint,
protocol versions, PID, and a random 256-bit bearer token. Clients remove stale
manifests only after checking that their endpoint is unreachable.

## Handshake

The first JSON-RPC request on every connection must be:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "system.hello",
  "params": {
    "token": "token-from-private-runtime-manifest",
    "client": {"name": "odon-client", "version": "0.1.0"},
    "protocol_versions": [1]
  }
}
```

The response selects a protocol version and returns instance/session IDs,
application and control API versions, capabilities, permission policy, and the
maximum inline payload. Requests made before a successful handshake fail with
`HANDSHAKE_REQUIRED`; invalid authentication closes the connection.

## Events, revisions, and tasks

`events.subscribe` accepts exact names and suffix-wildcard patterns. Odon emits
bounded JSON-RPC notifications using method `events.event`; each carries a
global sequence, state revision, source ID, payload, and optional initiating
session/request IDs. Slow subscribers drop events rather than block egui and can
inspect drop counts with `events.get_status`.

Successful mutations increment the shared revision. Mutating application
commands accept `if_revision` and return `CONFLICT` if it is stale.

`tasks.start` submits an existing application command and immediately returns a
retained task snapshot. `tasks.get`, `tasks.list`, `tasks.cancel`, and
`tasks.forget` manage it. Open operations settle when the viewer is ready;
screenshot operations settle when output is written. Progress and terminal
states are emitted as `tasks.*` events. Cancellation is cooperative: queued work
can be cancelled, but an indivisible operation already running on the UI thread
cannot be forcibly interrupted.

## Referenced data, layers, and UI

Large data is registered by URI through `data.resources.*`, with explicit
format, axes, units, scale, translation, ownership, metadata, and provenance.
`viewer.layers.*` manages stable external layer IDs, style, order, visibility,
and lifecycle without putting arrays in JSON.

`ui.extensions.*` and `ui.contributions.*` retain bounded, versioned component
trees. Rust renders the supported controls through egui and emits extension
events. Native command actions and supported layer bindings execute locally.
Supported hosts are right tabs, left sections, top-bar actions, a status bar,
canvas controls, and a dedicated project-card extension host.

Session-owned descriptors are cleaned when their authenticated connection
closes. Project-owned data-resource and layer descriptors persist in project
JSON and are restored when that project opens. A project-owned layer cannot
reference a session-temporary resource.

## Introspection

- `system.get_capabilities` returns the negotiated protocol version and
  capability names.
- `system.list_methods` and `system.describe_methods` return method metadata and
  request schemas.
- `system.describe_events` describes the event envelope and event families.
- `ui.describe_schema` returns the native declarative UI vocabulary and limits.

Application methods are admitted and validated through the central Rust control
registry. The Python SDK and MCP adapter both use this boundary; MCP retains
only its client-specific tool descriptions and result formatting. Methods are
currently provisional.

## Errors

Application failures use JSON-RPC error responses:

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "error": {
    "code": -32602,
    "message": "zoom must be finite and greater than zero",
    "data": {"kind": "INVALID_PARAMS", "method": "set_camera"}
  }
}
```

Defined initial kinds include `PARSE_ERROR`, `INVALID_REQUEST`,
`METHOD_NOT_FOUND`, `INVALID_PARAMS`, `APPLICATION`, `AUTHENTICATION_FAILED`,
`HANDSHAKE_REQUIRED`, `INCOMPATIBLE_PROTOCOL`, `NOT_READY`, `UNSUPPORTED`,
`CONFLICT`, `CANCELLED`, `RESOURCE_NOT_FOUND`, `PERMISSION_DENIED`, `WRONG_MODE`,
`RESOURCE_LIMIT`, `TIMEOUT`, and `INTERNAL`.

## Compatibility bridge

Explicit development/test listeners may accept the previous envelope:

```json
{"method": "get_camera", "params": {}}
```

and return `{"ok": true, "result": ...}`. Published authenticated instances
reject legacy envelopes. The compatibility path will not gain v1 features.
