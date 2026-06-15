# Model compatibility report

This folder hosts the **model compatibility report** published to the public
winml-cli GitHub Pages site, plus the script used to refresh it.

| File | Purpose |
| --- | --- |
| `model_compatibility_report.html` | The self-contained report page (all data embedded inline). |
| `download_report.py` | Fetches the latest report from the private artifacts repo. |
| `README.md` | This file. |

> This is a **staging location**. The files are placed here directly on the
> `gh-pages` branch (which also hosts the MkDocs/mike documentation site) so a
> maintainer can later integrate the report into the site navigation. Nothing
> here is committed to `main`.

## Source

The report is generated in the private `gim-home/ModelKitArtifacts` repo and
lives at `e2e_model_coverage_result/model_compatibility_report.html` on its
`site-src` branch. The page embeds all of its data inline, so only this one
file needs to be published — no JSON or other assets are fetched at runtime.

## Refreshing the report

### Prerequisites

- [GitHub CLI](https://cli.github.com) (`gh`) authenticated with an account that
  has access to the `gim-home` org.

### Fetch the latest report

These files live on the `gh-pages` branch. Check out that branch — directly, or
via a worktree if you don't want to switch your current branch:

```powershell
# optional worktree (run from a winml-cli checkout)
git fetch origin gh-pages
git worktree add ../wmlcli-ghpages gh-pages
cd ../wmlcli-ghpages/reports
```

Then run the script from this `reports/` folder. It uses only the Python
standard library — no project dependencies or `uv` required. By default it
overwrites `model_compatibility_report.html` next to the script:

```powershell
python download_report.py --account <your_gim-home_account>
```

This sparse-clones the `site-src` branch of the private artifacts repo and copies
the single report file. Use `--out <path>` to write elsewhere. The script only
fetches — it does not commit or push.

### Publish

Commit and push the refreshed report on `gh-pages`:

```powershell
git add reports/model_compatibility_report.html
git commit -m "Update model compatibility report"
git push origin gh-pages
```

The report is then available at `.../reports/model_compatibility_report.html`.
GitHub Pages redeploys automatically within a minute or two of the push.

If you used a worktree, clean it up afterwards:

```powershell
git worktree remove ../wmlcli-ghpages
```
