"""
Self-describing provenance for a run's output directory.

outputs_fixed/*/manifest.json answers "what produced this?" without relying
on the directory name or the author's memory -- both of which have already
proven unreliable this project (three near-identically-named red_l/green_l
truncated output dirs in a row, one of them silently built from a
pre-bugfix beta_df.csv, caught only by comparing file mtimes by hand).

Large per-run data (npz/parquet/figures) stays out of git (see .gitignore)
because it's regeneratable from raw session data; this manifest is the
small, git-trackable record of the exact code + parameters that produced
it, so "go back to an earlier parameter set" means reading a JSON file and
checking out a commit, not reverse-engineering file timestamps.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path


class RunWouldOverwriteError(RuntimeError):
    """Raised when out_dir already holds a manifest.json for a DIFFERENT
    parameter set and overwrite=True wasn't passed -- see write_run_manifest.
    """


def write_run_manifest(out_dir, params, script=None, overwrite=False):
    """Write out_dir/manifest.json recording the git commit this run was
    produced under, whether the working tree had uncommitted changes at run
    time (dirty=True means this run's exact code isn't fully recoverable
    from git history alone -- flag it rather than silently imply otherwise),
    a timestamp, and the caller-supplied params dict (whatever parameters
    actually distinguish this run -- e.g. truncate_at_side_out, mice,
    model_names, min_retained_frac).

    script : optional label for which script/function produced this run
    (e.g. "run_model_series_comparison_sm_red_l.main_red_l") -- defaults to
    None since the caller usually knows this better than any introspection
    here would.

    overwrite : if out_dir already has a manifest.json recording a DIFFERENT
    params dict, raise RunWouldOverwriteError instead of proceeding --
    directory-name-based "versioning" only works if every new parameter set
    actually gets a new directory, and nothing enforced that before this
    guard existed. Re-running with the IDENTICAL params (e.g. regenerating
    after a crash) is always allowed without overwrite=True, since that's
    not a loss of any distinct prior result. Raised BEFORE the caller does
    any real work (this is called at the top of main_red_l, before
    run_comparison/plot_pooled), so the expensive computation never even
    starts if it would clobber something.
    """
    out_dir = Path(out_dir)
    existing_path = out_dir / "manifest.json"
    if existing_path.exists() and not overwrite:
        try:
            existing = json.loads(existing_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = None
        if existing is not None and existing.get("params") != params:
            raise RunWouldOverwriteError(
                f"{existing_path} already records a DIFFERENT parameter set:\n"
                f"  existing: {json.dumps(existing.get('params'), default=str)}\n"
                f"  new:      {json.dumps(params, default=str)}\n"
                "Pick a new out_dir for this run, or pass overwrite=True to "
                "write_run_manifest if you really mean to replace it."
            )

    repo_dir = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True, check=True,
        ).stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        commit, dirty = None, None
        print(f"write_run_manifest: could not read git state ({exc}) -- commit/dirty left null")

    manifest = {
        "script": script,
        "git_commit": commit,
        "git_dirty": dirty,
        "timestamp": datetime.now().isoformat(),
        "params": params,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    if dirty:
        print(f"write_run_manifest: WARNING -- working tree had uncommitted changes when this "
              f"run was produced; {manifest_path} records git_commit={commit} but that alone "
              "won't reproduce this exact run")
    print(f"Wrote {manifest_path}")
    return manifest
