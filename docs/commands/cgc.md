# winml cgc

> Which D3D12 adapters are on this machine, and what patterns does an adapter's driver
> declare it can match.

## When to use this

Two questions get asked on every new machine and after every driver update: which
adapters are here, and what does this driver's MLIR-program implementation claim it can
match. `winml cgc adapters` answers the first and `winml cgc patterns` the second, both
from public Windows APIs through `ctypes`, so nothing has to be built.

Use it to check whether a GPU's driver implements D3D MLIR programs before relying on
the CGC path, to record what a driver declares after a driver update, and to compare
two drivers' matching surfaces.

## Synopsis

```bash
$ winml cgc adapters [options]           # which adapters are here
$ winml cgc patterns [options]           # what a driver declares it can match
```

`cgc` is a command group; run `winml cgc` or `winml cgc --help` to list its subcommands.
Both print help and exit 0.

## Requirements

Listing adapters needs nothing beyond Windows.

Querying a driver's MLIR support or patterns needs an **Agility SDK `D3D12Core.dll` of
SDK 720 or newer**, matching the Python interpreter's architecture, and a compatible
driver. The MLIR-program feature is a preview feature that the inbox D3D12 runtime does
not serve. See [Finding the redist](#finding-the-redist).

`winml cgc patterns --open` without `--dump` only opens the viewer or an existing JSON
dump. It needs a browser, but neither a GPU nor a redist, and does not query drivers.

## Flags

Shared by both subcommands:

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--d3d12-dir` | | path | `""` | Directory holding `D3D12Core.dll`. Overrides every other source. |
| `--verbose` | `-v` | count | `0` | Increase logging verbosity (`-v` for INFO, `-vv` for DEBUG). Any nonzero level also adds raw capability HRESULTs and, under `patterns`, a per-pattern listing. Accepted at the root or subcommand, so `winml -v cgc adapters` and `winml cgc adapters -v` behave the same. |
| `--quiet` | `-q` | flag | `false` | Errors only on the logging channel. Does not silence stdout or the directly printed probe details requested by `-v`. |
| `--help` | `-h` | flag | — | Show help and exit. |

`winml cgc patterns` adds:

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--adapter` | `-a` | string | `""` | The adapter to ask, by description substring (case-insensitive) or by index. A number that is not an index is tried as a substring, so `-a 5060` names an RTX 5060. Without it, every enumerated adapter is asked. Required by `--dump`. |
| `--dump` | | flag | `false` | Write a dump under `./patterns`. The files depend on the response: text, bytecode, empty, or unsupported; see [Dump](#dump). Needs `-a`. |
| `--overwrite/--no-overwrite` | | flag | `false` | Replace an existing dump directory. |
| `--open` | | flag or path | — | Open the [pattern atlas](#browsing-a-dump) in a browser. With `--dump`, show this run's dump when available. Given a `patterns.json`, show that file. With neither a file nor `--dump`, open an empty atlas. Without `--dump`, no driver is queried, so neither a redist nor a GPU is needed. |

## Usage scenarios

Common tasks, including viewing a dump without querying a driver.

| You want to know | Run |
|---|---|
| which adapters are on this machine, and whose driver implements MLIR programs | `winml cgc adapters` |
| *why* an adapter is marked `no` | `winml cgc adapters -v` |
| which Agility core a live query uses | the `redist:` line of a run that reaches redist resolution; viewer-only runs do not resolve one |
| how many patterns and rules this driver declares | `winml cgc patterns -a nvidia` |
| what every enumerated adapter's driver declares | `winml cgc patterns` |
| every pattern it declares, and what kind of claim each one is | `winml cgc patterns -a nvidia -v` |
| whether a non-NVIDIA adapter claims anything | `winml cgc patterns -a amd` |
| keep the declaration on disk, for a diff or a record | `winml cgc patterns -a nvidia --dump` |
| re-capture after a driver update | `winml cgc patterns -a nvidia --dump --overwrite` |
| dump a driver's patterns and browse them in the atlas | `winml cgc patterns -a nvidia --dump --open` |
| browse an existing dump without querying drivers | `winml cgc patterns --open <patterns.json>` |
| open an empty atlas to load a dump manually | `winml cgc patterns --open` |
| what a *different* redist makes the same driver say | `winml cgc patterns -a nvidia --d3d12-dir <dir>` |
| record that an adapter has *no* MLIR support, for the same baseline | `winml cgc patterns -a amd --dump` |
| which subcommands the group has | `winml cgc --help` |

The output examples below illustrate the format. Adapter names, versions, counts, and
HRESULTs depend on the machine, driver, and selected redist; they are not compatibility
guarantees.

### A machine with no redist

Listing adapters needs no redist, so this is the first thing to run on a new box.
Without a redist the capability question cannot be asked, and the column says so rather
than guessing:

```console
$ winml cgc adapters
redist: none -- MLIR support unknown (run with -v to see where cgc looked)
IDX ADAPTER                        DRIVER            TYPE       ATTRIBUTES  MLIR
--------------------------------------------------------------------------------
0   NVIDIA GeForce RTX 5060 Laptop 32.0.16.3004      hardware   ML CC GFX   ?
1   AMD Radeon(TM) 880M Graphics   32.0.31041.1004   integrated ML CC GFX   ?
2   Microsoft Basic Render Driver  10.0.26100.9278   software   ML CC GFX   ?
```

`?` is not `no`. The driver versions and the table are still correct and still useful.

### List adapters

The command enumerates DXCore adapters advertising generic ML support. If that list
is empty, it retries with core-compute support; it does not combine the two lists or
enumerate every graphics-only adapter. The list is sorted hardware-first, then by
high-performance preference.

```console
$ winml cgc adapters --d3d12-dir C:\path\to\D3D12
redist: C:\path\to\D3D12 (SDK 720, x64)
IDX ADAPTER                        DRIVER            TYPE       ATTRIBUTES  MLIR
--------------------------------------------------------------------------------
0   NVIDIA GeForce RTX 5060 Laptop 32.0.16.3004      hardware   ML CC GFX   yes
1   AMD Radeon(TM) 880M Graphics   32.0.31041.1004   integrated ML CC GFX   no
2   Microsoft Basic Render Driver  10.0.26100.9278   software   ML CC GFX   no
```

| Column | Meaning |
|---|---|
| `IDX` | Selection index accepted by `patterns -a`. |
| `ADAPTER` | DXCore description, truncated to 30 characters in this table only. |
| `DRIVER` | Driver version as four 16-bit components. |
| `TYPE` | `hardware`, `integrated`, or `software`. |
| `ATTRIBUTES` | `ML`: generic ML; `CC`: core compute; `GFX`: D3D12 graphics. These are not MLIR support indicators. |
| `MLIR` | `yes` or `no` from the capability probe, or `?` when it could not be asked through a usable redist. |

The `redist:` line precedes the table when redist resolution is reached. An explicitly
named directory rejected by the on-disk checks fails before that line; a DLL that
passes those checks but is rejected by D3D12 fails afterwards. See
[Driving it from a script](#driving-it-from-a-script) for output on failed runs.

`MLIR` is the column that matters: it is what the driver claims about the MLIR-program
exchange through the redist on the `redist:` line. A driver may report support through
a 721-shaped query but refuse the subsequent pattern exchange; see
[When something looks wrong](#when-something-looks-wrong). Fetching a declaration with
`patterns` checks that exchange, not whether a particular model will compile.

Probing support attempts to create a D3D12 device for each adapter and can take longer
than enumerating adapter metadata. Its cost depends on the machine and driver.

### Counts

```console
$ winml cgc patterns -a nvidia
redist: C:\path\to\D3D12 (SDK 720, x64)
NVIDIA GeForce RTX 5060 Laptop GPU: 51 patterns / 63 rules
```

`-a` answers for **one** adapter, so it reports that adapter rather
than printing the table. Add `-v` to list every pattern the driver declares, grouped by
what kind of claim it is:

```console
$ winml cgc patterns -a nvidia -v
redist: C:\path\to\D3D12 (SDK 720, x64)
NVIDIA GeForce RTX 5060 Laptop GPU: 51 patterns / 63 rules

kernel   12 patterns, 16 rules   declares a jitFunction -- the only kind that claims work
    38  gemm_input_major_expression_epilogue_cluster
    36  gemm_input_major  3 rules
   ...
fusion    4 patterns,  6 rules   collapses a marked cluster into one native op
   200  lora_no_views_output_major  3 rules
   ...
hint      5 patterns,  9 rules   anchors a cluster and sets the order clusters grow in
mark     25 patterns, 27 rules   tags an op so a fusion root can absorb it later
rewrite   5 patterns,  5 rules   graph surgery, so that other patterns can match
```

The five kinds are read off each pattern's body, never its name: a `jitFunction` makes
it a **kernel**, a `subgraph_rewrite_desc` without one a **fusion**, then
`cgc_add_pattern_cluster_hint` a **hint**, `cgc_mark_pattern_cluster_op` a **mark**, and
anything left a **rewrite**. Groups run from most claimed to least, patterns within a
group by descending benefit, then by full name for ties. A missing benefit is sorted
as zero and displayed as `-`. A per-pattern rule count appears only where a pattern
expands to more than one rule. A kind the driver declares nothing for is left out.

On an adapter whose driver has no MLIR support, this is a normal answer, not an error
(**exit 0**):

```console
$ winml cgc patterns -a amd
redist: C:\path\to\D3D12 (SDK 720, x64)
AMD Radeon(TM) 880M Graphics: driver does not implement MLIR programs
```

Without a redist the question cannot be asked at all, and that is a failure rather
than an answer (**exit 1**):

```console
$ winml cgc patterns -a nvidia
redist: none -- MLIR support unknown (run with -v to see where cgc looked)
Error: cgc patterns needs an Agility SDK redist; none was found (run with -v to see where cgc looked)
```

### Dump

`--dump` must name its adapter with `-a`:

```console
$ winml cgc patterns -a nvidia --dump
redist: C:\path\to\D3D12 (SDK 720, x64)
NVIDIA GeForce RTX 5060 Laptop GPU: 51 patterns / 63 rules
wrote patterns\nvidia-geforce-rtx-5060-laptop-gpu\32.0.16.3004\patterns.mlir (124753 bytes)
wrote patterns\nvidia-geforce-rtx-5060-laptop-gpu\32.0.16.3004\patterns.json
wrote patterns\nvidia-geforce-rtx-5060-laptop-gpu\32.0.16.3004\metadata.txt
```

For a non-empty text response, three files are written:

| file | what it is |
|---|---|
| `patterns.mlir` | the driver's declaration as text, with LF line endings and without the C string terminator the driver appends |
| `patterns.json` | the same patterns split one per record and grouped — see [patterns.json](#patternsjson) |
| `metadata.txt` | provenance: adapter, driver, redist, SDK, and the counts |

Other responses produce different files:

| Response | Files written by `--dump` |
|---|---|
| MLIR bytecode | `patterns.mlirbc` and `metadata.txt`; no counts or `patterns.json`. |
| Empty declaration after text normalization | `metadata.txt` with `status=empty`. |
| No MLIR support reported by the probe | `metadata.txt` with `status=unsupported`. |
| Redist or pattern exchange failure | No new dump is written; the command exits 1. |

Naming the adapter is required rather than defaulted, because a dump is filed under a
slug and a driver version: dumping whichever adapter happened to be first writes a
correct-looking directory for the wrong device. Omitting `-a` is a usage error:

```console
$ winml cgc patterns --dump
Usage: winml cgc patterns [OPTIONS]
Try 'winml cgc patterns --help' for help.

Error: --dump needs an adapter: name one with -a <substring|index>      # exit 2
```

The layout is `patterns/<adapter-slug>/<driver-version>/`, relative to the working
directory. It is keyed by the normalized description and driver version, not by a
unique device ID or SDK version. Identical descriptions, descriptions that normalize
to the same slug, and different redists used with the same driver can therefore share
a destination. Use separate working directories to keep those captures independently.
The root is always `patterns/`; there is no output-directory flag.

A dump is **never silently overwritten**:

```console
$ winml cgc patterns -a nvidia --dump    # second time
Error: Dump directory 'patterns\nvidia-geforce-rtx-5060-laptop-gpu\32.0.16.3004' already exists and is not empty. Re-run with --overwrite to replace its contents.
```

That message goes to stderr and the exit status is 1. After a successful query,
`--overwrite` removes the four known dump files (`patterns.mlir`, `patterns.mlirbc`,
`patterns.json`, and `metadata.txt`) before writing the replacement; unrelated files
are preserved. A query failure leaves the previous dump untouched, but a later
filesystem write failure can still leave an incomplete replacement. A non-empty
directory or an existing file at the destination blocks without `--overwrite`; an
empty directory is reused.

`metadata.txt` alongside the dump:

```ini
adapter=NVIDIA GeForce RTX 5060 Laptop GPU
driver_version=32.0.16.3004
status=dumped
received_encoding=text
text_file=patterns.mlir
bytecode_file=
abi=720
sdk_version=720
redist=C:\path\to\D3D12
patterns=51
rules=63
received_bytes=127117
line_endings=lf
```

The first six keys are the ones `dxcgc-dump-driver-patterns.exe` writes, unchanged so
existing readers keep working; the rest record which redist and ABI produced the dump.

| key | values | meaning |
|---|---|---|
| `status` | `dumped` \| `unsupported` \| `empty` | whether patterns were captured, the driver has no MLIR support, or it returned an empty declaration |
| `received_encoding` | `text` \| `bytecode` \| `none` | the stored payload's encoding; `none` for `unsupported` or `empty` |
| `text_file` / `bytecode_file` | filename or empty | only one is ever set |
| `abi` / `sdk_version` | e.g. `720` | both currently record the SDK version; SDK 720 uses the 720 exchange shape, SDK 721 or newer uses the 721 shape |
| `patterns` / `rules` | integers, or empty | empty when the counts could not be taken (`unsupported`, `empty`, or a bytecode answer) |
| `received_bytes` | integer, or empty | size returned by the exchange, including any terminator; an empty response records `0` (or the terminator's size), while an unsupported adapter leaves this empty |
| `line_endings` | `lf`, or empty | `lf` when a text payload was normalised to LF line endings and its terminator removed before it was stored |

#### Dumping an adapter with no MLIR support

This is a normal answer, not an error (**exit 0**). No `patterns.mlir` is written — just
a `metadata.txt` recording that this driver was asked and said no, so the baseline covers
every adapter rather than silently omitting the ones that declined:

```console
$ winml cgc patterns -a amd --dump
redist: C:\path\to\D3D12 (SDK 720, x64)
AMD Radeon(TM) 880M Graphics: driver does not implement MLIR programs
wrote patterns\amd-radeon-880m-graphics\32.0.31041.1004\metadata.txt
```

```ini
adapter=AMD Radeon(TM) 880M Graphics
driver_version=32.0.31041.1004
status=unsupported
received_encoding=none
text_file=
bytecode_file=
abi=720
sdk_version=720
redist=C:\path\to\D3D12
patterns=
rules=
received_bytes=
line_endings=
```

#### The adapter slug

The directory name is the description lowercased, with `(R)`, `(TM)` and `(C)` removed
and every run outside ASCII `a-z` and `0-9` collapsed to a single `-`, then any leading
or trailing `-` trimmed. If nothing remains, the slug is `adapter`:

| description | slug |
|---|---|
| `NVIDIA GeForce RTX 5060 Laptop GPU` | `nvidia-geforce-rtx-5060-laptop-gpu` |
| `AMD Radeon(TM) 880M Graphics` | `amd-radeon-880m-graphics` |
| `Intel(R) Graphics` | `intel-graphics` |

### After a driver update

Three things move: the version in the table, the counts, and the dump. In that order:

```bash
winml cgc adapters                  # current driver versions
winml cgc patterns -a nvidia        # current pattern and rule counts
winml cgc patterns -a nvidia -v     # inspect the current declarations
winml cgc patterns -a nvidia --dump # keep the current declaration
```

The dump lands under the new driver version, so the previous one is still there beside
it. Compare the two with `git diff --no-index OLD NEW`. Re-dumping the *same* version
is refused until you pass `--overwrite`.

Text dumps are stored with LF line endings and no trailing NULs; bytecode is preserved
unchanged. Older text dumps may retain CRLF endings and a trailing NUL. Compare
normalized copies: remove trailing NUL bytes and normalize line endings first.
`git diff --no-index --ignore-cr-at-eol OLD NEW` only ignores end-of-line CRs; it does
not remove a trailing NUL, which can make Git treat an older text dump as binary.

### Selecting an adapter

`-a` takes a description substring (case-insensitive) or a bare index:

```bash
winml cgc patterns -a nvidia
winml cgc patterns -a 0
```

With no `-a`, `winml cgc patterns` asks **every enumerated adapter** and prints one line
each. A driver that cannot answer -- one that claims the
exchange and then refuses it -- costs only its own line, on stderr; the rest are still
reported and the run then exits 1. `--dump` does not accept that default: it names
its adapter with `-a` or it exits 2.

A `-a` value is matched as an index when one exists, and otherwise as a description
substring, so `-a 5060` finds an RTX 5060 on a three-adapter machine. If several
descriptions match, the first in enumeration order wins; use `IDX` to distinguish
them. Matching uses the full description, not the table's truncated display.

Once adapters and a usable redist are available, a selector that matches nothing
exits 2. `winml cgc adapters` takes no `-a`: it always prints the enumerated list.
If enumeration finds no adapters after argument checks, either live-query subcommand prints
`no D3D12 adapters found` and exits 0 without resolving a redist.

### Driving it from a script

`stdout` carries the `redist:` line, table, counts, verbose pattern listing, and
`wrote`/`opening` notices. `stderr` carries logging and diagnostics such as the `-v`
probes and redist failure report. This is human-readable output, not a JSON stream:

```powershell
winml cgc patterns -a nvidia > counts.txt
if ($LASTEXITCODE -ne 0) { "command failed; output may be partial" }
```

The `redist:` line is written after redist resolution succeeds or concludes that no
implicit candidate is available. It remains in stdout if a later step fails. Parsing
errors and an explicitly named redist rejected during discovery fail before that
line. Viewer-only runs do not print it.

**A failed multi-adapter run can contain partial results.** Successful adapters are
still printed when another adapter's exchange fails, and the command exits 1 after
the sweep. A selected-adapter failure can leave just the `redist:` line. Always check
the exit code; neither non-empty output nor a count line proves the whole run
succeeded.

Read `patterns=` and `rules=` from a successfully written dump's `metadata.txt`, or
the structured `patterns.json`, rather than re-parsing stdout. There is no JSON-output
flag for these subcommands.

## patterns.json

A text dump also writes `patterns.json`: every pattern in the declaration, split out
one per record and grouped, so a reader never has to parse MLIR. The
[CGC Pattern Atlas](#browsing-a-dump) page is built on it.

```json
{
  "format": "winml-cgc-patterns/1",
  "meta": {
    "adapter": "NVIDIA GeForce RTX 5060 Laptop GPU",
    "driver_version": "32.0.16.3004",
    "sdk_version": 720,
    "redist": "C:\\path\\to\\D3D12",
    "encoding": "text",
    "bytes": 124753
  },
  "summary": {
    "patterns": 51, "rules": 63, "kernel_patterns": 12,
    "distinct_kernels": 9, "sources": 4, "top_benefit": 200
  },
  "kinds": { "kernel": 12, "fusion": 4, "hint": 5, "mark": 25, "rewrite": 5 },
  "sources": [
    { "name": "nvidia_conv_central.pdll", "patterns": [1, 2, 3] }
  ],
  "patterns": [
    {
      "index": 28,
      "name": "nvidia.gemm_input_major_expression_epilogue_cluster",
      "short_name": "gemm_input_major_expression_epilogue_cluster",
      "kind": "kernel",
      "benefit": 38,
      "rules": 1,
      "source": "nvidia_handwritten.mlir",
      "kernel": "GemmCluster",
      "lines": 92,
      "mlir": "cgc_pattern.pattern @nvidia.gemm_input_major_expression_epilogue_cluster ..."
    }
  ]
}
```

(`sources` and `patterns` are abbreviated here.)

| field | derived from |
|---|---|
| `meta.bytes` | length of the normalized text payload, not the original `received_bytes` in `metadata.txt` |
| `index`, `short_name` | one-based declaration order, and the portion of `name` after its final `.` |
| `kind`, `benefit`, `rules` | the same parser that prints the counts; `benefit` is `null` if absent |
| `source` | the preceding `// from <file>.pdll` / `// from <file>.mlir` marker, or `null` if none exists; prose beginning with "from" does not open a section |
| `kernel` | the value matched from `jitFunction = #cgc.string<"Name">` or a plain string; `null` if no kernel name is parsed |
| `lines`, `mlir` | line count and text from the declaration through its closing brace, including comments; invalid UTF-8 is decoded with replacement characters |
| `summary.kernel_patterns`, `summary.distinct_kernels` | records with a parsed kernel name, and the number of distinct names |
| `sources`, `summary.sources` | groups of pattern indices by source; records without a source share the `(unattributed)` group, which is included in the count |

`format` names the layout, so a reader can refuse a file it does not understand. A
bytecode dump gets no `patterns.json` — splitting it needs the real MLIR parser — and
neither does an unsupported adapter or an empty declaration. Non-empty text with no
recognized pattern declarations does produce JSON with an empty `patterns` array.

### Browsing a dump

The **CGC Pattern Atlas** is a single self-contained HTML page, opened by `--open`:

```bash
winml cgc patterns -a nvidia --dump --open # dump this driver, then show it
winml cgc patterns --open patterns.json   # show a dump taken earlier
winml cgc patterns --open                 # open an empty page; no driver query
```

Without `--dump`, `--open` bypasses adapter and redist selection. Use the file form
without `--dump`: when `--dump` is present, the viewer uses this run's output rather
than any filename passed to `--open`. If no `patterns.json` was written (bytecode,
empty declaration, or unsupported adapter), the CLI warns and opens an empty atlas.

An absent, unreadable, non-JSON, or wrong-format file is a usage error (exit 2). The
CLI checks the format tag and metadata object; the browser validates the pattern
array separately. The atlas currently requires at least one pattern, so an empty or
malformed array produces a browser error rather than a CLI exit-code failure.

It shows the summary, filters by name, kernel and kind, groups by source file or by
kind, and expands each pattern's MLIR on demand. Nothing is uploaded; the file is read
in the browser. The page ships with the package, at
`winml/modelkit/commands/assets/cgc-pattern-atlas.html`, so `--open` finds it in an
installed wheel as well as in a checkout; open that file directly if you would rather
not go through the command.

## Patterns and rules are not the same number

A **pattern** is one `cgc_pattern.pattern` declaration. A **rule** is one flat match
alternative after `any_of` expansion: `any_of { all_of {…} all_of {…} }` contributes one
rule per branch, and several such blocks in one pattern multiply.

They can diverge. For example, the following recorded counts show why both are useful:

| driver | patterns | rules |
|---|---:|---:|
| 32.0.16.2009 | 13 | 13 |
| 32.0.16.2035 | 60 | 60 |
| 32.0.16.3004 | 51 | **63** |

Reading only the declaration count across the 2035 → 3004 examples shows 60 → 51,
while the rule count increases from 60 to 63 because of `any_of` alternatives.
Neither number alone proves broader model coverage or better performance: these are
counts of declarations, not a compilation or execution test.

Beware `apply_native_constraint "cgc_is_any_of"`, which is an op-family whitelist and
not a rule multiplier at all. A substring search for `any_of` conflates the two.

## Finding the redist

For a run that reaches redist resolution, an **explicitly named** redist -- the flag,
or the first non-empty environment variable -- is the only candidate. If it cannot
be used, the run exits 1 rather than answering about a different runtime. Only the
implicit `bin/` roots are tried in sequence, stopping at the first candidate that
passes the on-disk checks:

| # | Source | Where |
|---|---|---|
| 1 | `--d3d12-dir PATH` | the flag; when given it is the **only** candidate |
| 2 | `$WINML_D3D12_DIR` | environment |
| 3 | `$D3D12_DIR` | environment |
| 4 | `<venv-parent>/bin/` | the directory holding the virtual environment — beside `.venv` |
| 5 | `<worktree>/bin/` | the directory holding `pyproject.toml`, for a source checkout |
| 6 | `<package>/bin/` | the installed `winml` package |
| 7 | `<site-packages>/bin/` | the directory holding that package |
| 8 | `<prefix>/bin/` | the environment root itself, e.g. `<venv>/bin/` |

An explicit `--d3d12-dir` short-circuits the rest, so a wrong path fails loudly instead
of silently falling through to some other core.

In PowerShell, set the environment variable with `$env:`:

```powershell
$env:WINML_D3D12_DIR = 'C:\path\to\D3D12'
winml cgc adapters
```

### The `bin/` default

A redist dropped in a `bin/` folder at the root of either a development worktree or a
wheel installation is used with no flag and no environment variable, from any working
directory.

**Development environment** — the `bin/` folder sits **beside `.venv`**:

```
my-project/
  .venv/
  bin/D3D12/D3D12Core.dll     <- dropped here
```

This is the rule that matters in practice, and it is checked first. It is derived from
`sys.prefix`, not from the source tree, so it keeps working once winml-cli is installed
from a wheel into a project's venv — at that point there is no `pyproject.toml` to walk
up to from `site-packages`, but `.venv`'s parent is still the project.

A source checkout is also located by its `pyproject.toml`, which covers a venv kept
somewhere other than the worktree. In the usual case the two are the same directory and
collapse to one set of candidates.

```
winml-cli/
  .venv/
  bin/D3D12/D3D12Core.dll     <- dropped here
  pyproject.toml
  src/winml/...
```

**Wheel installation** — these additional locations are searched for a locally
supplied redist:

```
<venv>/
  bin/D3D12/D3D12Core.dll                          <- <prefix>
  Lib/site-packages/
    bin/D3D12/D3D12Core.dll                        <- <site-packages>
    winml/
      bin/D3D12/D3D12Core.dll                      <- <package>
```

These are discovery locations, not a promise that the wheel contains an Agility SDK.
winml-cli does not download a redist; supply one separately.

Each root is checked twice — `bin/D3D12/` first, then `bin/` holding `D3D12Core.dll`
directly. The `D3D12` subfolder is preferred because it is the Agility SDK's own
convention, mirroring the `D3D12SDKPath` an application exports, but a flat `bin/` works.

Duplicate paths are checked once. Automatic worktree and virtual-environment-parent
roots at a filesystem root are excluded.

Without a separately supplied redist, the `MLIR` column reads `?`. Which candidate was
chosen is on the `redist:` line whether or not `-v` was passed. `-v` adds each
adapter's raw capability HRESULT on stderr, and the negotiated IR version when
nonzero:

```text
  NVIDIA GeForce RTX 5060 Laptop GPU: feature 70 -> 0x00000000 (S_OK)
  AMD Radeon(TM) 880M Graphics: feature 70 -> 0x80004001 (E_NOTIMPL)
  Microsoft Basic Render Driver: feature 70 -> 0x8000FFFF (E_UNEXPECTED)
```

A candidate directory is accepted only after its `D3D12Core.dll` is found, its PE
machine type matches this interpreter, and its exported `D3D12SDKVersion` is read — by
parsing the export table, never by loading the DLL, which would pin a runtime before the
choice has been made. `isdir()` is not enough on its own.

Passing these checks does not guarantee D3D12 can load the redist. If runtime loading
then fails, the command does not retry another candidate: an `adapters` listing with
an automatically discovered redist warns and shows `?`, while a named redist or a
`patterns` query fails with exit 1.

### The SDK 720 floor

A core older than SDK 720 is **never used**. It is *named* in the failure report when
no usable redist is found at all; when a later candidate does work, the rejection is not
reported -- the `redist:` line names the core that answered, and that is the one the
result describes:

```console
$ winml cgc adapters --d3d12-dir C:\Windows\System32
cgc: no usable D3D12 Agility SDK redist found (need a directory containing
     D3D12Core.dll of SDK 720 or newer, built for x64).

Looked in, in order:
  1. --d3d12-dir                C:\Windows\System32
     SDK 616 is older than 720 and does not serve D3D MLIR programs
...
Error: --d3d12-dir named a redist that cannot be used; refusing to answer about a different one.
```

This exits **1**.

This matters because an old core does not fail loudly — it answers `E_INVALIDARG` for
the feature query, which would report a perfectly capable driver as `MLIR no`. Falling
back to one would create a healthy-looking device with the MLIR path silently absent.
The CLI does not fall back to the inbox core. Candidate rejection is based on SDK
version and architecture, not on the directory name alone.

The SDK version also selects the **wire ABI** used by this CLI: SDK 720 selects the
exchange by GUID; SDK 721 and newer use a `Type` enum plus a negotiated IR version.
There is no `--abi` flag: point `--d3d12-dir` at the redist you want and the ABI follows.
Use separate invocations to compare redists.

## Exit codes

| code | meaning |
|---|---|
| 0 | the CLI completed: includes an unsupported adapter, an empty declaration, no adapters found, `adapters` with support unknown, help, or a viewer launch request |
| 1 | a live query could not complete, a dump was refused or could not be written, or the installed atlas asset is missing; an `adapters` listing through an unusable auto-discovered redist instead warns, shows `?`, and exits 0 |
| 2 | bad arguments: an unknown flag, `--dump` without `-a`, an unmatched selector after redist resolution, or a file rejected by `--open` validation |

## When something looks wrong

**`MLIR` shows `?` for every adapter.** Either no redist was found, or D3D12 refused the
one that was (the reason is then on stderr, e.g. a preview redist with Developer Mode
off). Either way the question could not be asked, and the listing itself is still
correct. When no redist was found, `-v` shows every place cgc looked.

**`MLIR` shows `no` on a GPU you expect to support it.** Check `-v` for a device-creation
failure or the capability-query result. `S_OK` alone does not mean support: the
returned support value must also be nonzero. For SDK 721 or newer, a zero negotiated
IR version is reported as `no` even if the HRESULT is `S_OK`.

**The exchange fails after the driver claimed support.** The declaration was not
retrieved, so the run exits 1 rather than reporting the adapter as unsupported.
One possible cause is a driver/redist ABI mismatch: a driver can advertise capability
through a 721-shaped query but reject the exchange with `DXGI_ERROR_UNSUPPORTED`.
Check the HRESULT and try a redist compatible with that driver.

**A driver answers in bytecode.** The CLI prints its byte count rather than pattern
counts. With `--dump`, it writes `patterns.mlirbc` and `metadata.txt`, but no
`patterns.json`. Rendering bytecode as text needs an MLIR-aware tool such as `cgc-opt`.

**`--open` reports a missing atlas asset.** The installed package must contain the HTML
page. Reinstall a package that includes it. If `--dump` already completed, the files
remain available even though opening the viewer failed.
