"""
Component 15 -- LaTeX table export.

Every table this project produces (`results_df` from `pareto_sweep.py` /
`model_comparison.py` / `ablation.py`, `decision_matrix_df`,
`sensitivity_summary_df`, the statistical-significance table from
`statistics_tests.compare_models_statistically`, ...) is a plain
`pd.DataFrame`. Turning each one into a properly formatted table for a
written report currently means manually retyping it -- tedious and a
common source of copy-paste transcription errors between a notebook and a
thesis/paper. `dataframe_to_latex` / `export_all_results_to_latex` close
that gap directly.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Union

import pandas as pd

# This project's `"<mean> +/- <std>"` formatted strings render better in a
# typeset LaTeX table as a proper plus-minus glyph.
_PM_PATTERN = re.compile(r"\s*\+/-\s*")


def _prettify_pm(latex_source: str) -> str:
    """Post-processes RENDERED LaTeX source, replacing every literal
    `"X +/- Y"` occurrence with `"X $\\pm$ Y"` for a properly typeset
    plus-minus glyph.

    This runs AFTER `DataFrame.to_latex(escape=True)`, not before: doing
    the substitution on the raw DataFrame values first and letting
    `escape=True` process the result afterward would escape the `$` and
    `\\` characters just inserted (`escape=True` has no way to know they
    are intentional LaTeX markup rather than literal text to protect),
    corrupting `"$\\pm$"` into visible `"\\$\\textbackslash pm\\$"`. Since
    plain ASCII `"+"`, `"/"`, `"-"` need no LaTeX escaping themselves, the
    literal `"+/-"` substring survives `to_latex(escape=True)` unchanged
    and can be safely replaced afterward, on the final rendered text.
    """
    return _PM_PATTERN.sub(r" $\\pm$ ", latex_source)


def dataframe_to_latex(df: pd.DataFrame, caption: str | None = None, label: str | None = None,
                        column_format: str | None = None, float_format: str = "%.4f",
                        prettify_pm: bool = True, index: bool = False) -> str:
    """
    Renders `df` as a LaTeX `table`/`tabular` environment string
    (booktabs-style: relies on `\\usepackage{booktabs}` for `\\toprule` /
    `\\midrule` / `\\bottomrule`, the standard for publication-quality
    tables -- add that package to the document preamble).

    Parameters
    ----------
    caption, label : str, optional
        Passed through to `\\caption{}` / `\\label{}` inside the
        surrounding `table` float. If both are `None`, only the bare
        `tabular` environment is emitted (no `table` float wrapper) --
        useful when the table will be embedded inside an existing float
        or a `longtable` elsewhere.
    column_format : str, optional
        Explicit LaTeX column spec (e.g. `"lrrrr"`); if omitted, pandas
        infers one (left-aligned strings, right-aligned numbers).
    prettify_pm : bool
        If `True` (default), every `"X +/- Y"` string in `df` (this
        project's mean+/-std display format) is rewritten to `"X $\\pm$
        Y"` before rendering, for a properly typeset plus-minus sign
        rather than a literal ASCII `"+/-"`.
    index : bool
        Whether to include the DataFrame's row index as a column
        (default `False`: this project's tables are never meaningfully
        indexed by row number).

    Returns
    -------
    str : the LaTeX source. Values are escaped by default (column names
    or cell values containing `%`, `_`, `&`, etc. -- common in this
    project's `"QPU Yield (%)"`-style headers -- render correctly rather
    than breaking LaTeX compilation).
    """
    kwargs = dict(index=index, escape=True, float_format=float_format, na_rep="--")
    if column_format is not None:
        kwargs["column_format"] = column_format
    if caption is not None:
        kwargs["caption"] = caption
    if label is not None:
        kwargs["label"] = label

    try:
        latex_source = df.to_latex(**kwargs)
    except TypeError:
        # Older pandas versions (<1.0) don't accept caption/label kwargs
        # directly on `to_latex`; fall back to manually wrapping a plain
        # tabular in a table float when either was requested.
        kwargs.pop("caption", None)
        kwargs.pop("label", None)
        tabular = df.to_latex(**kwargs)
        if caption is None and label is None:
            latex_source = tabular
        else:
            parts = ["\\begin{table}[htbp]", "\\centering", tabular]
            if caption is not None:
                parts.append(f"\\caption{{{caption}}}")
            if label is not None:
                parts.append(f"\\label{{{label}}}")
            parts.append("\\end{table}")
            latex_source = "\n".join(parts)

    return _prettify_pm(latex_source) if prettify_pm else latex_source


def export_all_results_to_latex(output_dir: Union[str, Path], **named_dfs: pd.DataFrame) -> Dict[str, Path]:
    """
    Writes one `.tex` file per keyword argument into `output_dir`
    (created if missing), each via `dataframe_to_latex` with an
    auto-generated `caption` (the argument name, underscores replaced by
    spaces and title-cased) and `label` (`"tab:<name>"`).

    Example
    -------
        export_all_results_to_latex(
            "latex_tables",
            pareto_frontier=results_df,
            decision_matrix=decision_matrix_df,
            significance_vs_edgelstm=significance_df,
        )
        # -> latex_tables/pareto_frontier.tex, latex_tables/decision_matrix.tex,
        #    latex_tables/significance_vs_edgelstm.tex

    Returns a `{name: Path}` dict of every file written, so a caller (or
    `experiment_tracking.ExperimentRun`) can register them without
    re-deriving the filenames.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    for name, df in named_dfs.items():
        caption = name.replace("_", " ").title()
        label = f"tab:{name}"
        latex_source = dataframe_to_latex(df, caption=caption, label=label)
        path = output_dir / f"{name}.tex"
        with open(path, "w") as f:
            f.write(latex_source)
        written[name] = path

    return written
