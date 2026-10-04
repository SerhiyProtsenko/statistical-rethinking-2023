"""Execute lecture notebooks without overwriting their saved outputs.

Smoke mode reduces MCMC/prior draws only; it tests code execution, not inference
quality. Full mode preserves the lectures' original sampling settings.
"""

import argparse
import concurrent.futures
import json
from pathlib import Path
import time

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
EXECUTION_SETUP = """
from pathlib import Path as _Path
import utils as _utils
_figure_dir = _Path({figure_dir})
_figure_dir.mkdir(parents=True, exist_ok=True)
def _check_savefig(filename):
    _utils.plt.savefig(_figure_dir / filename, dpi=300, bbox_inches='tight')
_utils.savefig = _check_savefig
"""
SMOKE_SETUP = """
import pymc as _pm
_original_sample = _pm.sample
_original_prior = _pm.sample_prior_predictive
def _smoke_sample(*args, **kwargs):
    args = list(args)
    if args:
        args[0] = {draws}
    else:
        kwargs['draws'] = {draws}
    kwargs.update(tune={draws}, chains=2, cores=1, progressbar=False,
                  compute_convergence_checks=False, random_seed=42)
    return _original_sample(*args, **kwargs)
def _smoke_prior(*args, **kwargs):
    args = list(args)
    if args:
        args[0] = {draws}
    else:
        kwargs['draws'] = {draws}
    kwargs.pop('samples', None)
    kwargs['random_seed'] = 42
    return _original_prior(*args, **kwargs)
_pm.sample = _smoke_sample
_pm.sample_prior_predictive = _smoke_prior
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", help="Notebook filenames; default: all lectures")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--draws", type=int, default=100, help="Draws and tuning steps in smoke mode")
    parser.add_argument("--timeout", type=int, default=900, help="Timeout per cell, in seconds")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.draws < 20 or args.jobs < 1:
        parser.error("Use at least 20 draws and at least one job")
    paths = [ROOT / name for name in args.notebooks] if args.notebooks else sorted(ROOT.glob("Lecture*.ipynb"))
    output_dir = ROOT / ".cache" / ("smoke" if args.smoke else "full")
    output_dir.mkdir(parents=True, exist_ok=True)

    def execute(path):
        started = time.monotonic()
        notebook = nbformat.read(path, as_version=4)
        setup = EXECUTION_SETUP.format(figure_dir=repr(str(output_dir / "images")))
        if args.smoke:
            setup += SMOKE_SETUP.format(draws=args.draws)
        notebook.cells.insert(0, nbformat.v4.new_code_cell(setup))
        state = {"notebook": path.name, "mode": "smoke" if args.smoke else "full", "status": "running"}
        print(f"START {path.name}", flush=True)

        def before_cell(cell, cell_index):
            state["cell"] = cell_index - 1

        client = NotebookClient(
            notebook, timeout=args.timeout,
            kernel_name="statistical-rethinking-2023-uv",
            resources={"metadata": {"path": str(ROOT)}},
            on_cell_start=before_cell,
        )
        try:
            client.execute()
            state["status"] = "passed"
        except Exception as error:
            state.update(status="failed", error=str(error))
        finally:
            state["seconds"] = round(time.monotonic() - started, 1)
            notebook.cells.pop(0)
            nbformat.write(notebook, output_dir / path.name)
            (output_dir / f"{path.stem}.json").write_text(json.dumps(state, indent=2))
        print(f"{state['status'].upper()} {path.name} cell={state.get('cell')} {state['seconds']}s", flush=True)
        if "error" in state:
            print(state["error"][-2500:], flush=True)
        return state

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(execute, paths))
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    failures = sum(result["status"] != "passed" for result in results)
    print(f"{len(results) - failures}/{len(results)} notebooks passed", flush=True)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
