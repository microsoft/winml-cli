# Model accuracy report

This folder hosts the **model accuracy report** published on the winml-cli
documentation site, plus the script used to refresh it.

| File | Purpose |
| --- | --- |
| `model_accuracy_report.html` | The self-contained report page (all data embedded inline). |
| `download_report.py` | Fetches the latest report from the private artifacts repo. |
| `README.md` | This file. |

> These files live in `docs/reports/` on `main`. mkdocs publishes
> `model_accuracy_report.html` as a static asset under each docs version
> (e.g. `.../latest/reports/model_accuracy_report.html`); `download_report.py`
> and this README are excluded from the built site via `exclude_docs` in
> `mkdocs.yml`. Refreshing the report is a normal docs change to `main`.

## Source

The report is generated in the private `gim-home/ModelKitArtifacts` repo and
lives at `e2e_model_coverage_result/model_accuracy_report.html` on its
`site-src` branch. The page embeds all of its data inline, so only this one
file needs to be published — no JSON or other assets are fetched at runtime.

## Refreshing the report

### Prerequisites

- [GitHub CLI](https://cli.github.com) (`gh`) authenticated with an account that
  has access to the `gim-home` org.

### Fetch the latest report

Work on a branch off `main`. Run the script from this `docs/reports/` folder.
It uses only the Python standard library — no project dependencies or `uv`
required. By default it overwrites `model_accuracy_report.html` next to the
script:

```powershell
python docs/reports/download_report.py --account <your_gim-home_account>
```

This sparse-clones the `site-src` branch of the private artifacts repo and copies
the single report file. Use `--out <path>` to write elsewhere. The script only
fetches — it does not commit or push.

### Publish

Commit the refreshed report on your branch and open a PR to `main`:

```powershell
git add docs/reports/model_accuracy_report.html
git commit -m "Update model accuracy report"
```

Once merged, the **Build & Publish Docs** workflow rebuilds the site and the
report is available at `.../<version>/reports/model_accuracy_report.html`
(e.g. `.../latest/reports/...`). To refresh an already-released version, run
that workflow via `workflow_dispatch` with the matching version label.
