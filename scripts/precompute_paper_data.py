"""Pre-compute and cache all analysis rows for identifiability_paper.qmd.

Run this once before rendering the Quarto paper so that the render is fast:

    conda run -n matfit1d python scripts/precompute_paper_data.py

The cache is written to:

    reviews/quarto_reports/paper_rows_cache.pkl

Subsequent ``quarto render identifiability_paper.qmd`` calls load the cache
(< 1 second) instead of rerunning the full covariance analysis (~10 min).

To force a fresh recomputation, delete the cache file and re-run this script.
"""

import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _analytical_runtime import default_analytical_run_paths
from analyze_beta_identifiability import collect_rows, load_sections

CACHE_FILE = PROJECT_ROOT / "reviews" / "quarto_reports" / "paper_rows_cache.pkl"


def main() -> None:
    h5_path, xlsx_path, _ = default_analytical_run_paths(PROJECT_ROOT)

    if not h5_path.exists():
        sys.exit(f"ERROR: HDF5 data file not found: {h5_path}")
    if not xlsx_path.exists():
        sys.exit(f"ERROR: XLSX results file not found: {xlsx_path}")

    print(f"Input files:")
    print(f"  HDF5 : {h5_path}")
    print(f"  XLSX : {xlsx_path}")
    print()

    t0 = time.perf_counter()
    print("Loading analytical sections …")
    sections = load_sections(h5_path=h5_path, xlsx_path=xlsx_path)
    print(f"  {len(sections)} sections loaded  ({time.perf_counter() - t0:.1f}s)")

    t1 = time.perf_counter()
    print("Computing covariance analysis for each section …")
    all_rows = collect_rows(sections)
    print(f"  {len(all_rows)} rows computed  ({time.perf_counter() - t1:.1f}s)")

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as fh:
        pickle.dump(all_rows, fh)

    elapsed = time.perf_counter() - t0
    print()
    print(f"Cache saved → {CACHE_FILE}  ({CACHE_FILE.stat().st_size / 1024:.0f} KB)")
    print(f"Total time  : {elapsed:.1f}s")
    print()
    print("You can now render the paper quickly:")
    print("  quarto render scripts/quarto/identifiability_paper.qmd")


if __name__ == "__main__":
    main()