# T5 Translation Export: KV Cache Study

## Problem Summary

`wmk build` for `google-t5/t5-small` with `translation` task fails with:

```
Error: Build failed: Only tuples, lists and Variables are supported as JIT inputs/outputs.
Dictionaries and strings are also accepted, but their usage is not recommended.
Here, received an input of unsupported type: EncoderDecoderCache
```

## Root Cause Confirmed: KV Cache

T5 is an encoder-decoder model. When `use_cache=True` (the default), the decoder stack creates an `EncoderDecoderCache` object containing two `DynamicCache` instances (self-attention + cross-attention). This object is returned in `past_key_values` output.

`torch.onnx.export` (TorchScript-based legacy exporter) cannot trace this non-tensor type.

### Why the Optimum patcher doesn't help

ModelKit attempts to use Optimum's `ModelPatcher` for tracing compatibility, but it fails because Optimum maps T5 to `text2text-generation` task, not `translation`:

```
ValueError: t5 doesn't support task translation for the onnx backend.
Supported tasks are: feature-extraction, feature-extraction-with-past,
text2text-generation, text2text-generation-with-past.
```

Even if the task mapping were fixed, Optimum's patcher only handles cache for the split encoder/decoder export approach (not the monolithic export that ModelKit uses).

## Cache Architecture

T5-small produces per-layer KV cache with shape `[batch, heads, seq_len, d_kv]`:
- **6 decoder layers**, each with:
  - Self-attention: key `[1, 8, dec_seq, 64]` + value `[1, 8, dec_seq, 64]`
  - Cross-attention: key `[1, 8, enc_seq, 64]` + value `[1, 8, enc_seq, 64]`
- Total: **24 KV tensors** per inference step

Cross-attention KV is computed once from encoder output and reused for all decoder steps. Self-attention KV grows by 1 each step (appending the new token's KV).

## Approaches Evaluated

### Approach 1: Disable cache (`use_cache=False`)

**Verified working.** Exports successfully to a 293.7 MB ONNX model.

```python
config.use_cache = False
# Export produces: input_ids, attention_mask, decoder_input_ids -> logits
```

- **Pro**: Simplest fix, works today with minimal code changes
- **Con**: No KV cache means the decoder recomputes all attention from scratch every step during autoregressive generation. For a 128-token translation with 6 layers, that's O(n^2) attention vs O(n) with cache. Roughly 64x more attention compute at the last step.
- **Recommendation**: Suitable as an interim fix to unblock export. Not suitable for production inference.

### Approach 2: Encoder/decoder split with flattened KV cache

**Verified working.** All three components export and run correctly in ONNX Runtime.

Split the model into 3 ONNX files:

| Component | Inputs | Outputs | Verified Size |
|---|---|---|---|
| **Encoder** | `input_ids [1,16]`, `attention_mask [1,16]` | `encoder_hidden_states [1,16,512]` | 134.8 MB |
| **Decoder (no past)** | `decoder_input_ids [1,1]`, `encoder_hidden_states [1,16,512]`, `attention_mask [1,16]` | `logits [1,1,32128]` + 24 present KV tensors | 221.6 MB |
| **Decoder (with past)** | `decoder_input_ids [1,1]`, `encoder_hidden_states [1,16,512]`, `attention_mask [1,16]` + 24 past KV tensors | `logits [1,1,32128]` + 24 present KV tensors | 209.6 MB |

The wrapper approach flattens `EncoderDecoderCache` into individual tensors:

```python
class T5DecoderWithPast(nn.Module):
    def forward(self, decoder_input_ids, encoder_hidden_states, attention_mask, *past_kv_flat):
        # Reconstruct EncoderDecoderCache from flat tensors
        self_attn_cache = DynamicCache()
        cross_attn_cache = DynamicCache()
        for i in range(self.num_layers):
            idx = i * 4
            self_attn_cache.update(past_kv_flat[idx], past_kv_flat[idx+1], i)
            cross_attn_cache.update(past_kv_flat[idx+2], past_kv_flat[idx+3], i)
        past_kv = EncoderDecoderCache(self_attn_cache, cross_attn_cache)

        out = self.model(decoder_input_ids=decoder_input_ids,
                         encoder_outputs=(encoder_hidden_states,),
                         attention_mask=attention_mask,
                         past_key_values=past_kv, use_cache=True)

        # Flatten output cache back to tensors
        result = [out.logits]
        cache = out.past_key_values
        for i in range(self.num_layers):
            result.extend([
                cache.self_attention_cache.layers[i].keys,
                cache.self_attention_cache.layers[i].values,
                cache.cross_attention_cache.layers[i].keys,
                cache.cross_attention_cache.layers[i].values,
            ])
        return tuple(result)
```

KV tensor naming convention (matching Optimum):
```
Input:  past.{layer}.decoder.key, past.{layer}.decoder.value
        past.{layer}.encoder.key, past.{layer}.encoder.value
Output: present.{layer}.decoder.key, present.{layer}.decoder.value
        present.{layer}.encoder.key, present.{layer}.encoder.value
```

- **Pro**: Efficient autoregressive generation. Well-established pattern (Optimum uses this). All components verified to export and run in ORT with correct numerics (max encoder diff: 0.000002).
- **Con**: Requires multi-model pipeline support in ModelKit. Current pipeline is monolithic (one model in, one ONNX out). Needs changes to export, optimize, and inference stages.

### Approach 3: Fixed context_length static KV (SELECTED)

**Target design:** A ring-buffer static cache where every ONNX input/output has a fixed shape.

#### Requirements

1. **Zero-initialized on start**: Cache tensors are `[1, heads, context_length, d_kv]`, filled with zeros.
2. **Fixed I/O shape**: Every generation step sends and receives exactly `context_length` KV cache per layer. No dynamic sequence dimensions.
3. **Ring-buffer eviction**: When a new token's KV is computed, it replaces the **oldest** entry in the cache. The cache never grows — it shifts.
4. **Fully static ONNX graph**: All inputs and outputs have compile-time-known shapes. This is critical for NPU/QNN targets.

#### Decoder ONNX I/O (per generation step)

```
Inputs:
  decoder_input_ids        [1, 1]                        # single new token
  encoder_hidden_states    [1, enc_seq, 512]              # from encoder (constant)
  attention_mask           [1, enc_seq]                   # encoder mask (constant)
  past.{i}.decoder.key     [1, 8, context_length, 64]    # static self-attn KV (i=0..5)
  past.{i}.decoder.value   [1, 8, context_length, 64]
  past.{i}.encoder.key     [1, 8, enc_seq, 64]           # static cross-attn KV (constant after step 1)
  past.{i}.encoder.value   [1, 8, enc_seq, 64]

Outputs:
  logits                   [1, 1, vocab_size]
  present.{i}.decoder.key  [1, 8, context_length, 64]    # updated self-attn KV (oldest evicted)
  present.{i}.decoder.value [1, 8, context_length, 64]
  present.{i}.encoder.key  [1, 8, enc_seq, 64]           # pass-through
  present.{i}.encoder.value [1, 8, enc_seq, 64]
```

#### Why transformers `StaticCache` was not used

The initial study tested `StaticCache` directly via `EncoderDecoderCache(StaticCache(...), StaticCache(...))` and got:

```
The size of tensor a (32) must match the size of tensor b (16)
```

This failed because **`StaticCache` is designed for `torch.compile`, not ONNX export**, and **is not a ring buffer**:

1. **Not a ring buffer.** `StaticCache` writes to sequential positions tracked by `cache_position`. It never evicts oldest entries. There is no shift/rotation semantic — it's a write-once pre-allocated buffer, not a circular buffer.

2. **In-place mutation via `index_copy_`.** `StaticCache.update()` does `self.keys.index_copy_(2, cache_position, key_states)`. In-place mutation is not traceable by `torch.jit.trace` — the tracer records the mutation target as a constant, so the write is baked in rather than flowing from input to output.

3. **Attention mask mismatch with T5.** T5's mask preparation computes mask size from `StaticCache.max_cache_len` (e.g., 32) but the encoder attention mask is sized to `enc_seq` (e.g., 16). These are added in the attention computation, causing the dimension error. T5 was never validated with `StaticCache` for encoder-decoder architectures.

4. **`torch.compile`-specific APIs.** Uses `torch._dynamo.mark_static_address` and lazy initialization designed for dynamo, not ONNX tracing.

**A custom `RingBufferCache` is needed** that:
- Uses `torch.cat([buffer[:, :, 1:, :], new_kv], dim=2)` (slice + cat) instead of `index_copy_` — traceable, produces new tensor, no in-place mutation
- Is initialized from input tensors (ONNX inputs), not module state — avoids constant-folding
- Returns the same fixed shape on every `update()` call
- Integrates with `EncoderDecoderCache` interface so T5's attention layers work unchanged

#### Why DynamicCache also doesn't work

Tested: a wrapper that reconstructs `DynamicCache` from flat input tensors, calls the model, and extracts flat tensors from the output. Self-attention KV inputs survive as real ONNX I/O (the `torch.cat` in `update()` creates a data-dependency the tracer can't fold). However:

- **`encoder_hidden_states` is constant-folded.** When cross-attention cache is pre-populated, `EncoderDecoderCache.is_updated[i] = True` (Python bool, constant during tracing). The model skips recomputing cross-attention from encoder output, so `encoder_hidden_states` has no consumers and the tracer drops it from the ONNX graph. The model is baked to one specific encoder output.
- **Not a ring buffer.** `DynamicCache.update()` concatenates: `cat([past, new], dim=2)`. Output shape is `past_seq + 1`, not `context_length`. Each step the sequence dimension grows.
- **TracerWarnings.** `DynamicCache.__init__` creates empty tensors registered as constants; boolean checks become constant branches.

#### Why `RingBufferCache` must be a custom cache (not DynamicCache, not StaticCache)

| Requirement | DynamicCache | StaticCache | RingBufferCache (custom) |
|---|---|---|---|
| Fixed output shape | No (concat grows) | Yes (pre-allocated) | Yes (slice + cat) |
| Ring-buffer eviction | No | No | Yes |
| Traceable by `torch.jit.trace` | Partially (TracerWarnings) | No (`index_copy_`) | Yes (`cat` on sliced input tensors) |
| No in-place mutation | Yes | No | Yes |
| Input tensors → ONNX inputs | Yes | No (module state) | Yes |
| Works with `EncoderDecoderCache` | Yes | Broken (mask mismatch) | Yes (implements Cache interface) |

#### `RingBufferCache` design

The cache has two distinct roles, cleanly separated:

**Role 1 — Inside export wrapper (traced into ONNX):**
Constructed from flat input tensors. `update()` does slice+cat to produce updated buffer. Both input and output KV have shape `[1, 8, context_length, 64]`. The shift logic lives in the ONNX graph.

**Role 2 — At inference time (Python, between ORT calls):**
Initialized with zeros. Passed to `transformers.pipeline`'s generate loop as `past_key_values`. Same `update()` logic (in-place mutation is fine here — not traced). Also needs to satisfy the `EncoderDecoderCache` interface for `prepare_inputs_for_generation` and `_update_model_kwargs_for_generation` to work.

```python
class RingBufferCache(Cache):
    """Fixed-size ring-buffer KV cache for self-attention.

    update() evicts oldest position, appends new → constant shape.
    Traceable (slice + cat, no index_copy_).
    """
    def __init__(self, kv_pairs: list[tuple[Tensor, Tensor]]):
        # kv_pairs[i] = (key_i, value_i), each [batch, heads, context_length, d_kv]
        self._kv = list(kv_pairs)

    @classmethod
    def from_zeros(cls, num_layers, batch, heads, context_length, d_kv, dtype=torch.float32):
        """Create zero-initialized cache for inference start."""
        kv = [(torch.zeros(batch, heads, context_length, d_kv, dtype=dtype),
               torch.zeros(batch, heads, context_length, d_kv, dtype=dtype))
              for _ in range(num_layers)]
        return cls(kv)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        old_k, old_v = self._kv[layer_idx]
        # Shift left (evict oldest), append new at end → same shape
        new_k = torch.cat([old_k[:, :, 1:, :], key_states], dim=2)
        new_v = torch.cat([old_v[:, :, 1:, :], value_states], dim=2)
        self._kv[layer_idx] = (new_k, new_v)
        return new_k, new_v  # [batch, heads, context_length, d_kv]

    def get_seq_length(self, layer_idx=0):
        return self._kv[layer_idx][0].shape[2]  # always context_length

    def __len__(self):
        return len(self._kv)

    def __getitem__(self, layer_idx):
        return self._kv[layer_idx]


class RingBufferEncoderDecoderCache(EncoderDecoderCache):
    """Wraps a RingBufferCache (self-attn) + DynamicCache (cross-attn).

    Cross-attention: is_updated forced to False so T5 always recomputes
    from encoder_hidden_states (prevents constant-folding).
    """
    def __init__(self, self_attention_cache, cross_attention_cache):
        super().__init__(self_attention_cache, cross_attention_cache)
        # Force cross-attention recomputation every step
        for layer_idx in self.is_updated:
            self.is_updated[layer_idx] = False

    @classmethod
    def from_zeros(cls, config, context_length, batch=1, dtype=torch.float32):
        """Create zero-initialized cache for pipeline inference start."""
        self_attn = RingBufferCache.from_zeros(
            config.num_layers, batch, config.num_heads,
            context_length, config.d_kv, dtype)
        cross_attn = DynamicCache()  # empty, populated on first step
        return cls(self_attn, cross_attn)
```

#### ONNX I/O (updated)

Both input and output self-attention KV have identical shape `[1, 8, context_length, 64]`:

```
Inputs:
  decoder_input_ids        [1, 1]
  encoder_hidden_states    [1, enc_seq, 512]
  attention_mask           [1, enc_seq]
  past.{i}.decoder.key     [1, 8, context_length, 64]   # i=0..5
  past.{i}.decoder.value   [1, 8, context_length, 64]

Outputs:
  logits                   [1, 1, vocab_size]
  present.{i}.decoder.key  [1, 8, context_length, 64]   # shifted + new token inserted
  present.{i}.decoder.value [1, 8, context_length, 64]
```

Cross-attention KV is **not** in the ONNX I/O. It is always recomputed from `encoder_hidden_states` (because `is_updated[i] = False`), which keeps `encoder_hidden_states` as a real ONNX input.

#### Export wrapper

```python
class T5DecoderWithRingBuffer(nn.Module):
    def __init__(self, model, num_layers):
        super().__init__()
        self.model = model
        self.num_layers = num_layers

    def forward(self, decoder_input_ids, encoder_hidden_states, attention_mask,
                *past_self_attn_flat):
        # past_self_attn_flat: key_0, value_0, key_1, value_1, ...
        # each [1, 8, context_length, 64]
        kv_pairs = [(past_self_attn_flat[i*2], past_self_attn_flat[i*2+1])
                    for i in range(self.num_layers)]
        self_attn_cache = RingBufferCache(kv_pairs)
        cross_attn_cache = DynamicCache()  # empty → forces recomputation
        cache = RingBufferEncoderDecoderCache(self_attn_cache, cross_attn_cache)

        out = self.model(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=(encoder_hidden_states,),
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )

        # Extract updated self-attention KV (each [1, 8, context_length, 64])
        result = [out.logits]
        updated = out.past_key_values.self_attention_cache
        for i in range(self.num_layers):
            k, v = updated[i]
            result.extend([k, v])
        return tuple(result)
```

#### Inference-time pipeline integration

```python
# Initialize zero cache for generate() loop
cache = RingBufferEncoderDecoderCache.from_zeros(config, context_length=128)

# Pipeline calls model.generate() which passes cache as past_key_values.
# generate loop: prepare_inputs_for_generation receives cache,
# forward() receives cache, returns Seq2SeqLMOutput with updated cache,
# _update_model_kwargs carries it to next step.
#
# The ONNX-backed forward():
#   1. Flatten cache.self_attention_cache → flat tensors
#   2. Run ORT session
#   3. Reconstruct RingBufferEncoderDecoderCache from outputs
#   4. Return Seq2SeqLMOutput(logits=..., past_key_values=cache)
```

#### Initial test results

| Approach | Status | Notes |
|---|---|---|
| `StaticCache` based `EncoderDecoderCache` | **Failed** | Attention mask mismatch, not a ring buffer, not traceable |
| `DynamicCache` in wrapper | **Broken** | `encoder_hidden_states` constant-folded, output shape grows |
| `use_cache=False` monolithic | **Works** | No cache I/O, O(n²) decode |
| `RingBufferCache` (ring buffer) | **ONNX export works, inference broken** | Ring buffer shift breaks T5's `memory_position = arange(key_length)` invariant (see below). |
| `StaticWriteCache` (append-only, scatter) | **Verified end-to-end** | `wmk config` + `wmk build` + `transformers.pipeline` exact match vs PyTorch reference. |

#### Why RingBufferCache breaks T5 inference (position bias problem)

T5 assumes **KV index = sequence position**. This is hardcoded into three coupled mechanisms:

1. **`get_seq_length()`** → `past_key_values_length` → `cache_position = [past_length, ...]`
2. **Causal mask**: built from `cache_position`, masks KV indices > current position
3. **Relative position bias**: `bias[query_pos - key_pos]` where `query_pos = cache_position` and `key_pos = KV_index`

With RingBufferCache (`context_length=128`):
- Step 0: `get_seq_length()=128` → `cache_position=[128]`
- The current token's KV is appended at KV index 128
- Position bias: `128 - [0..128]` = distances `[128, 127, ..., 0]`
- But the reference model at step 0 has: `0 - [0]` = distance `[0]`

The distances are completely wrong. Position bias for distance 128 ≠ distance 0.

If we try `cache_position=[0]` (override), then the causal mask says "attend to KV index ≤ 0". But the current token is at KV index 128 (past 128 + 1 current). **The token can't even attend to itself** — everything is masked.

**The ring buffer shift fundamentally breaks the KV_index = sequence_position invariant** that T5 relies on. There is no combination of `cache_position`, `decoder_attention_mask`, or `get_seq_length()` that fixes this without modifying T5's attention internals.

**Verified**: RingBufferCache logits diff vs reference = 3.27 at step 0 (same token picked by luck, different distribution). Without any cache (`use_cache=False`), diff = 0.000000.

## Multi-Model Build: Applicability of the SD Subfolder Pattern

### How SD multi-component build works (reny/sd branch)

The Stable Diffusion commits (`de34c59`, `c155ecb`, `e2f062e`) solved multi-component export via:

1. **`subfolder` parameter** threaded through the full pipeline: `WinMLAutoModel.from_pretrained` → `generate_hf_build_config` → `resolve_loader_config` → `load_hf_model`. Each component (text_encoder, vae, unet) is built independently.

2. **Wrapper `nn.Module` classes** (`StableDiffusionVaeDecoder`, `StableDiffusionUNet`) that expose the correct `forward()` for ONNX export. Registered via `MODEL_CLASS_MAPPING` keyed by `(model_type, task)`.

3. **Per-component `WinMLBuildConfig`** with explicit `InputTensorSpec`/`OutputTensorSpec` (e.g., `SD_VAE_DECODER_CONFIG`, `SD_UNET_CONFIG`). Each component goes through the full Export → Optimize → Analyze pipeline independently.

4. **`model_index.json` fallback** for diffusers models where `AutoConfig.from_pretrained` fails. Creates synthetic HF configs with `_create_diffusers_placeholder_config()`.

5. **Cache isolation** via nested output directories (`cache_dir / model_id / subfolder`).

### How this maps to T5 encoder-decoder

| SD pattern | T5 encoder-decoder equivalent |
|---|---|
| `subfolder="text_encoder"` | Component name `"encoder"` / `"decoder"` |
| `StableDiffusionVaeDecoder` wrapper | `T5EncoderWrapper`, `T5DecoderWithStaticCache` wrappers |
| `SD_VAE_DECODER_CONFIG` | Per-component `WinMLBuildConfig` with KV cache I/O specs |
| `MODEL_CLASS_MAPPING[("autoencoderkl", "feature-extraction")]` | `MODEL_CLASS_MAPPING[("t5_encoder", "translation")]` etc. |
| `model_index.json` fallback | Not needed — T5 components are part of the same `T5ForConditionalGeneration` |
| Diffusers pipeline orchestration (external) | `transformers.pipeline` with ONNX-backed model (see below) |

**Key difference:** SD components live in separate subfolders on HuggingFace and are loaded independently via `from_pretrained(model_id, subfolder=...)`. T5 encoder and decoder are **submodules of a single model** — they share weights (embeddings) and must be extracted via wrapper modules, not loaded separately.

**What can be reused directly:**
- The per-component `WinMLBuildConfig` + `WinMLExportConfig` pattern
- `MODEL_CLASS_MAPPING` / `MODEL_BUILD_CONFIGS` registration
- Cache isolation via `cache_key` prefixing (already works — `_name()` in `build_hf_model`)
- The existing Export → Optimize → Analyze pipeline (called once per component)

**What needs new work:**
- Wrapper modules that extract encoder/decoder from a loaded `T5ForConditionalGeneration` and add static cache I/O
- A component discovery mechanism for encoder-decoder models (SD uses `model_index.json`; T5 needs `config.is_encoder_decoder` detection)
- An inference-time model that uses `transformers.pipeline` (SD uses diffusers pipeline; T5 needs a generate()-compatible wrapper)

## Inference Runtime: `transformers.pipeline` Feasibility

### How `transformers.pipeline('translation')` works

The `TranslationPipeline._forward()` calls `self.model.generate(**model_inputs)`. For encoder-decoder models, `GenerationMixin.generate()` does:

1. **Encoder pass (once):** `encoder_outputs = self.get_encoder()(input_ids, attention_mask)` → `BaseModelOutput`
2. **Decode loop (per token):**
   - `model_inputs = self.prepare_inputs_for_generation(input_ids, past_key_values, ...)`
   - `outputs = self(**model_inputs)` → `Seq2SeqLMOutput(logits, past_key_values)`
   - `past_key_values = outputs.past_key_values` (carried to next step)

### Required interface

The model object passed to `pipeline()` must provide:

| Attribute/Method | Purpose |
|---|---|
| `config.is_encoder_decoder = True` | Triggers encoder-decoder generate path |
| `config.decoder_start_token_id` | First decoder token |
| `generation_config` | Beam size, max length, etc. |
| `get_encoder()` → module with `forward(input_ids, attention_mask) → BaseModelOutput` | Runs encoder ONNX |
| `forward(decoder_input_ids, encoder_outputs, attention_mask, past_key_values, use_cache, ...) → Seq2SeqLMOutput` | Runs decoder ONNX with cache flatten/reconstruct |
| `prepare_inputs_for_generation(...)` | Inherited from `T5ForConditionalGeneration` |

### Verified feasibility

Monkey-patching `T5ForConditionalGeneration` with a swapped `get_encoder()` and `forward()` that delegate to simulated ONNX sessions **works end-to-end with `transformers.pipeline('translation')`**. The pipeline produces correct translations (verified: "Bonjour, comment êtes-vous ?" for "Hello, how are you?").

The recommended approach for ModelKit is:

1. **Load the real `T5ForConditionalGeneration`** (inherits `generate()`, `prepare_inputs_for_generation`, `config`, `generation_config`)
2. **Swap `get_encoder()`** to return a wrapper that runs the encoder ONNX via `WinMLSession`
3. **Swap `forward()`** to a method that:
   - Extracts `encoder_hidden_states` from `encoder_outputs`
   - Flattens `past_key_values` (`EncoderDecoderCache` → individual KV tensors) using the static ring-buffer
   - Runs decoder ONNX via `WinMLSession`
   - Reconstructs `EncoderDecoderCache` from output KV tensors
   - Returns `Seq2SeqLMOutput(logits=..., past_key_values=...)`
4. **Cache flatten/reconstruct is decoupled from the model** — it lives in a standalone `StaticKVCacheManager` that serializes/deserializes between `EncoderDecoderCache` and flat tensors

This keeps the ONNX models clean (pure tensor I/O, no cache objects), keeps the cache logic separate, and reuses transformers' full generate() machinery including beam search.

### Cross-attention KV optimization

Cross-attention KV is constant after step 1 (computed from encoder output). Options:

1. **Pass-through** (simplest): Include cross-attention KV as both input and output every step. ~50% wasted KV I/O but zero complexity.
2. **Cache externally**: The `StaticKVCacheManager` caches cross-attention KV after step 1 and supplies it to every subsequent step without re-reading from ONNX output.

Option 2 is recommended — the manager already mediates all cache I/O so this is trivial to add.

## Final Design: StaticWriteCache (append-only, scatter-write)

**Approach:** Append-only static buffer with `torch.scatter` write at `cache_position`. No shifting — `KV_index = sequence_position` always holds. T5's `memory_position = arange(key_length)` computes correct distances.

### Implementation Components

| Component | Location | Description |
|---|---|---|
| `StaticWriteLayer` | `models/cache.py` | Single layer KV buffer. `update()` writes at `cache_position` via `torch.scatter`. Returns full buffer. |
| `StaticWriteCache` | `models/cache.py` | Multi-layer cache using `StaticWriteLayer`. Implements `Cache` interface. |
| `StaticWriteEncoderDecoderCache` | `models/cache.py` | Wraps `StaticWriteCache` (self-attn) + empty `DynamicCache` (cross-attn). Forces `is_updated=False`. Tracks `fill_count` for mask construction. |
| `T5EncoderWrapper(nn.Module)` | `models/hf/t5.py` | Wraps `model.encoder` → `forward(input_ids, attention_mask) → encoder_hidden_states` |
| `T5DecoderWithStaticCache(nn.Module)` | `models/hf/t5.py` | Export wrapper. Takes positional args: `(decoder_input_ids, encoder_hidden_states, attention_mask, decoder_attention_mask, cache_position, past_0_key, past_0_value, ...)`. Constructs cache, calls model, extracts updated KV. |
| `make_t5_encoder_config()` / `make_t5_decoder_config()` | `models/hf/t5.py` | Config factories reading `num_layers`, `num_heads`, `d_kv`, `d_model` from HF config. |
| `_wrap_t5_component()` | `build/hf.py` | Post-load hook: loads full `T5ForConditionalGeneration`, wraps encoder or decoder based on `subfolder`. |
| `_accepts_var_positional()` | `export/htp/exporter.py` | Detects `*args` forward signature, switches `torch.onnx.export` to positional arg mode. |

### Verified Results

| Check | Result |
|---|---|
| PyTorch numerics (StaticWriteCache vs DynamicCache) | Step 0: diff=0.000000, Steps 1-2: diff~1e-5 |
| ONNX export | Encoder 134.8 MB, Decoder 221.6 MB |
| ORT vs PyTorch (different `cache_position`) | logits diff=4e-6, generalizes to different inputs |
| `wmk config --subfolder encoder` | 2 inputs, 1 output, dims from HF config |
| `wmk config --subfolder decoder` | 17 inputs (5 base + 12 KV), 13 outputs, dims from HF config |
| `wmk build` encoder | Success, artifacts in `encoder/` subfolder |
| `wmk build` decoder | Success, artifacts in `decoder/` subfolder |
| `transformers.pipeline('translation')` | **Exact match** vs PyTorch reference |

## Usage

### 1. Generate configs

```powershell
# Encoder config
./.venv/Scripts/activate.ps1
python -m winml.modelkit config -m google-t5/t5-small --task translation --subfolder encoder --device cpu -o t5_encoder_config.json

# Decoder config (reads num_layers, d_kv, etc. from model)
python -m winml.modelkit config -m google-t5/t5-small --task translation --subfolder decoder --device cpu -o t5_decoder_config.json
```

### 2. Build ONNX models

```powershell
# Build encoder (artifacts in .cache/winml/artifacts/google-t5_t5-small/encoder/)
python -m winml.modelkit build -c t5_encoder_config.json -m google-t5/t5-small --use-cache

# Build decoder (artifacts in .cache/winml/artifacts/google-t5_t5-small/decoder/)
python -m winml.modelkit build -c t5_decoder_config.json -m google-t5/t5-small --use-cache
```

### 3. Run translation pipeline with ONNX models

```python
import types
import torch
import onnxruntime as ort
from transformers import (
    T5ForConditionalGeneration, AutoTokenizer, AutoConfig, pipeline,
)
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from winml.modelkit.models.cache import StaticWriteCache, StaticWriteEncoderDecoderCache

# Config
config = AutoConfig.from_pretrained("google-t5/t5-small")
nl, nh, dk = config.num_layers, config.num_heads, config.d_kv
enc_seq, max_dec = 16, 32  # must match export config

# Load ONNX sessions
enc_sess = ort.InferenceSession("path/to/encoder/model.onnx")
dec_sess = ort.InferenceSession("path/to/decoder/model.onnx")
dec_input_names = [inp.name for inp in dec_sess.get_inputs()]

# Load real model for generate() machinery (config, generation_config, etc.)
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")


def _pad_to(t, tl, pv=0):
    s = t.shape[-1]
    if s == tl: return t
    if s > tl: return t[..., :tl]
    return torch.nn.functional.pad(t, (0, tl - s), value=pv)


# Swap encoder
class OnnxEncoderProxy(torch.nn.Module):
    def __init__(self, sess):
        super().__init__()
        self._sess = sess

    def forward(self, input_ids, attention_mask=None, **kw):
        ids = _pad_to(input_ids, enc_seq, tokenizer.pad_token_id or 0)
        mask = _pad_to(attention_mask, enc_seq, 0)
        out = self._sess.run(
            None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()}
        )
        return BaseModelOutput(last_hidden_state=torch.from_numpy(out[0]))


enc_proxy = OnnxEncoderProxy(enc_sess)
model.get_encoder = lambda: enc_proxy


# Swap forward
def onnx_forward(self, input_ids=None, attention_mask=None,
                 decoder_input_ids=None, encoder_outputs=None,
                 past_key_values=None, use_cache=None, **kw):
    enc_h = (
        encoder_outputs[0] if isinstance(encoder_outputs, tuple)
        else encoder_outputs.last_hidden_state
    ) if encoder_outputs is not None else enc_proxy(input_ids, attention_mask).last_hidden_state

    if not isinstance(past_key_values, StaticWriteEncoderDecoderCache):
        cache = StaticWriteEncoderDecoderCache.from_zeros(nl, nh, dk, max_dec)
    else:
        cache = past_key_values

    fc = cache.fill_count
    dec_mask = torch.zeros(1, max_dec, dtype=torch.int64)
    dec_mask[0, :fc + 1] = 1
    cp = torch.tensor([fc], dtype=torch.int64)

    feeds_list = [
        decoder_input_ids.numpy(),
        enc_h.detach().numpy(),
        _pad_to(attention_mask, enc_seq, 0).numpy(),
        dec_mask.numpy(),
        cp.numpy(),
    ]
    for i in range(nl):
        layer = cache.self_attention_cache.layers[i]
        feeds_list.append(layer.keys.detach().numpy())
        feeds_list.append(layer.values.detach().numpy())

    ort_out = dec_sess.run(None, dict(zip(dec_input_names, feeds_list)))
    logits = torch.from_numpy(ort_out[0])
    kv_pairs = [
        (torch.from_numpy(ort_out[1 + i * 2]), torch.from_numpy(ort_out[2 + i * 2]))
        for i in range(nl)
    ]
    new_cache = StaticWriteEncoderDecoderCache(
        StaticWriteCache.from_kv_pairs(kv_pairs),
        cache.cross_attention_cache,
        fill_count=fc + 1,
    )
    return Seq2SeqLMOutput(logits=logits, past_key_values=new_cache)


model.forward = types.MethodType(onnx_forward, model)
model.generation_config.num_beams = 1
model.generation_config.do_sample = False

# Run pipeline
pipe = pipeline(
    "translation_en_to_fr", model=model, tokenizer=tokenizer,
    device="cpu", max_length=max_dec,
)
result = pipe("Hello, how are you?", num_beams=1, truncation=True)
print(result[0]["translation_text"])
# Output: Bonjour, comment êtes-vous ?
```
