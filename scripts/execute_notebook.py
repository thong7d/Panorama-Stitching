#!/usr/bin/env python3
"""Execute this project's pure-Python notebook cells and persist outputs.

Authors: Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369),
         Dao Quoc Tuan (23120392)
External sources: nbformat schema documentation and Python standard-library
APIs; no external implementation code was copied.

The generated midterm notebook deliberately contains no IPython magics, so a
small deterministic executor is sufficient when Jupyter/nbclient is not
installed. Subprocess output is captured by the notebook's run_checked helper.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import nbformat


def execute_notebook(
    input_path: Path,
    output_path: Path,
    workdir: Path,
    start_cell: int = 0,
    only_cells: set[int] | None = None,
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    notebook = nbformat.read(input_path, as_version=4)
    def run_checked(command):
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        completed.check_returncode()
        return completed

    namespace: dict = {
        "__name__": "__main__",
        "Path": Path,
        "json": json,
        "os": os,
        "subprocess": subprocess,
        "sys": sys,
        "run_checked": run_checked,
    }
    count_scope = notebook.cells if only_cells is not None else notebook.cells[:start_cell]
    existing_counts = [
        cell.get("execution_count") or 0
        for cell in count_scope
        if cell.cell_type == "code"
    ]
    execution_count = max(existing_counts, default=0)
    original_cwd = Path.cwd()
    workdir = workdir.resolve()

    try:
        os.chdir(workdir)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            if only_cells is not None:
                if index not in only_cells:
                    continue
            elif index < start_cell:
                continue
            execution_count += 1
            cell.execution_count = execution_count
            cell.outputs = []
            stdout = io.StringIO()
            stderr = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exec(compile(cell.source, f"<notebook-cell-{index}>", "exec"), namespace)
            except BaseException as error:
                if stdout.getvalue():
                    cell.outputs.append(
                        nbformat.v4.new_output("stream", name="stdout", text=stdout.getvalue())
                    )
                if stderr.getvalue():
                    cell.outputs.append(
                        nbformat.v4.new_output("stream", name="stderr", text=stderr.getvalue())
                    )
                cell.outputs.append(
                    nbformat.v4.new_output(
                        "error",
                        ename=type(error).__name__,
                        evalue=str(error),
                        traceback=traceback.format_exception(error),
                    )
                )
                nbformat.write(notebook, output_path)
                raise

            if stdout.getvalue():
                cell.outputs.append(
                    nbformat.v4.new_output("stream", name="stdout", text=stdout.getvalue())
                )
            if stderr.getvalue():
                cell.outputs.append(
                    nbformat.v4.new_output("stream", name="stderr", text=stderr.getvalue())
                )

            # Persist Matplotlib figures just as an inline notebook backend would.
            try:
                import matplotlib.pyplot as plt

                for figure_number in plt.get_fignums():
                    figure = plt.figure(figure_number)
                    buffer = io.BytesIO()
                    figure.savefig(buffer, format="png", bbox_inches="tight", dpi=120)
                    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                    cell.outputs.append(
                        nbformat.v4.new_output(
                            "display_data",
                            data={
                                "image/png": encoded,
                                "text/plain": f"<Figure {figure_number}>",
                            },
                            metadata={},
                        )
                    )
                plt.close("all")
            except ImportError:
                pass
            # Checkpoint after every successful cell so later packaging cells
            # inspect the already-updated notebook rather than stale outputs.
            nbformat.write(notebook, output_path)
    finally:
        os.chdir(original_cwd)

    nbformat.write(notebook, output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument("--start-cell", type=int, default=0)
    parser.add_argument(
        "--cells",
        help="Comma-separated zero-based cell indices to execute selectively.",
    )
    args = parser.parse_args()
    only_cells = None
    if args.cells is not None:
        try:
            only_cells = {int(value.strip()) for value in args.cells.split(",")}
        except ValueError as error:
            parser.error(f"--cells must contain integers: {error}")
        if not only_cells or min(only_cells) < 0:
            parser.error("--cells must contain at least one non-negative index")
    output = args.output or args.notebook
    execute_notebook(
        args.notebook.resolve(),
        output.resolve(),
        args.workdir,
        args.start_cell,
        only_cells,
    )
    print(f"[OK] Executed {args.notebook} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
