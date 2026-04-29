# ModelKit Privacy Statement

ModelKit collects limited, anonymous telemetry to help improve the
product. This page describes exactly what is collected, what is not,
and how to control it.

## Data category

All ModelKit telemetry is classified as **Optional** under Microsoft's
data categorization model. None of it is required to run any feature;
it exists solely to support product improvement.

A first-run interactive prompt asks for consent before any event is
sent. The prompt defaults to accept — pressing Enter enables telemetry.
You can decline explicitly at the prompt, or change your answer later
by editing `%USERPROFILE%\.winml\config.json`. Telemetry is
automatically disabled in non-interactive contexts (non-TTY stdin,
CI/CD pipelines) regardless of stored consent; those contexts do not
see the prompt and default to off.

## Events collected

When telemetry is enabled, ModelKit emits three event types:

### ModelKitHeartbeat

Sent once per CLI invocation, just before the requested command runs.
Carries only context attributes (OS, architecture, app version, device
ID) — no per-event payload.

### ModelKitAction

Sent once per command completion.

| Attribute | Description |
|---|---|
| `invoked_from` | `Script` or `Interactive`, based on whether stdin is a TTY. |
| `action_name` | Click subcommand name (e.g., `build`, `analyze`). |
| `device` | Target device type, if the subcommand accepts `--device` (e.g., `NPU`, `GPU`). |
| `ep` | Execution provider, if the subcommand accepts `--ep` (e.g., `QNNExecutionProvider`). |
| `duration_ms` | Wall-clock execution time in milliseconds. |
| `success` | Whether the command completed without raising. |

### ModelKitError

Sent only when a command raises an unhandled exception.

| Attribute | Description |
|---|---|
| `exception_type` | Exception class name (e.g., `ValueError`). |
| `exception_message` | The exception message, with absolute paths trimmed to package-relative, truncated to 200 characters, and with emails, GUIDs, IPv4/IPv6 addresses, and long opaque tokens replaced by `<scrubbed>`. |
| `exception_stack` | A list of frames, each `{file, line, function}`. File paths are package-relative. No source line text, no local variable values. |

## Common context attributes

Every event carries these attributes (populated by the telemetry module,
not by the command code):

| Attribute | Description |
|---|---|
| `device_id` | SHA256 hash of a randomly generated UUID, persisted per machine. Enables counting distinct users without identifying them. |
| `id_status` | `EXISTING`, `NEW`, or `FAILED` — how the device ID was obtained on this run. |
| `os.name`, `os.version`, `os.release`, `os.arch` | Operating system and architecture (e.g., `Windows`, `10.0.26200`, `11`, `AMD64`). |
| `app_version` | ModelKit package version. |
| `app_instance_id` | A random UUID generated for this process only; not persisted. |
| `initTs` | Epoch timestamp when telemetry was initialized. |

## Data **never** collected

- Model file paths or model contents
- User names, machine names, or IP addresses (IPs appearing in exception messages are scrubbed)
- Command-line arguments or option values (e.g., `--model path/to/file.onnx`)
- Source code text in tracebacks
- Local variable values in tracebacks
- HuggingFace tokens, API keys, or session tokens (long opaque tokens in exception messages are scrubbed)
- File system contents
- Email addresses (scrubbed from exception messages if present)

## Controlling telemetry

### Consent

On the first run of any command, ModelKit prompts:

```
Enable telemetry? [Y/n]
```

The default is `Y` (telemetry enabled) — pressing Enter accepts. Your
answer is persisted to `%USERPROFILE%\.winml\config.json` under
`telemetry.consent` and the prompt is not shown again.

### Changing your decision

Edit `%USERPROFILE%\.winml\config.json` directly:

```json
{
  "telemetry": {
    "consent": "disabled"
  }
}
```

| Goal | Edit |
|---|---|
| Opt out | Set `telemetry.consent` to `"disabled"` (or delete the file). |
| Opt in | Set `telemetry.consent` to `"enabled"`. |
| Re-show the prompt on next run | Delete the file, or remove the `telemetry.consent` field. |

There are no CLI subcommands, per-invocation flags, or environment
variables for consent — the config file is the single source of truth.

### CI / CD

Telemetry is automatically disabled when any of these environment
variables are set, and no prompt is shown:

`CI`, `TF_BUILD`, `GITHUB_ACTIONS`, `JENKINS_URL`,
`CODEBUILD_BUILD_ID`, `BUILDKITE`,
`SYSTEM_TEAMFOUNDATIONCOLLECTIONURI`.

### Cache directory

Events that fail to send (e.g., transient network errors) are cached
locally and retried on the next run. The cache file lives at:

`%USERPROFILE%\.winml\telemetry\modelkit.cache`

The cache is append-only on failure and drain-then-resend on recovery.
When telemetry is disabled, the cache is cleared so a disabled session
never resends events the user has since opted out of.

## Dev installs

ModelKit installed from source (`pip install -e .`) or run directly
from a checkout never sends telemetry. The InstrumentationKey is blank
in source and is only populated by the official build pipeline. Only
official binary releases are capable of sending telemetry, and only
after the user has seen the first-run prompt.
