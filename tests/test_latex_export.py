import pandas as pd
import pytest

from quantum_twin.latex_export import dataframe_to_latex, export_all_results_to_latex


def test_dataframe_to_latex_contains_tabular_environment():
    df = pd.DataFrame({"Model": ["A", "B"], "MAE": [0.01, 0.02]})
    latex = dataframe_to_latex(df)
    assert "\\begin{tabular}" in latex
    assert "\\end{tabular}" in latex


def test_dataframe_to_latex_caption_and_label():
    df = pd.DataFrame({"Model": ["A"], "MAE": [0.01]})
    latex = dataframe_to_latex(df, caption="My Table", label="tab:mine")
    assert "My Table" in latex
    assert "tab:mine" in latex
    assert "\\begin{table}" in latex


def test_dataframe_to_latex_without_caption_label_has_no_float_wrapper():
    df = pd.DataFrame({"Model": ["A"], "MAE": [0.01]})
    latex = dataframe_to_latex(df)
    assert "\\caption" not in latex
    assert "\\label" not in latex


def test_dataframe_to_latex_prettifies_mean_std_strings():
    df = pd.DataFrame({"Model": ["A"], "QPU Yield (%)": ["85.32 +/- 1.20"]})
    latex = dataframe_to_latex(df, prettify_pm=True)
    assert "$\\pm$" in latex
    assert "+/-" not in latex
    assert "\\textbackslash" not in latex  # no double-escaping artifact


def test_dataframe_to_latex_prettify_pm_false_keeps_raw_text():
    df = pd.DataFrame({"Model": ["A"], "QPU Yield (%)": ["85.32 +/- 1.20"]})
    latex = dataframe_to_latex(df, prettify_pm=False)
    assert "+/-" in latex
    assert "$\\pm$" not in latex


def test_dataframe_to_latex_escapes_percent_in_column_names():
    df = pd.DataFrame({"QPU Yield (%)": [50.0, 60.0]})
    latex = dataframe_to_latex(df)
    assert "\\%" in latex


def test_dataframe_to_latex_nan_uses_na_rep():
    df = pd.DataFrame({"A": [1.0, float("nan")]})
    latex = dataframe_to_latex(df)
    assert "--" in latex


def test_dataframe_to_latex_index_excluded_by_default():
    df = pd.DataFrame({"A": [1, 2, 3]}, index=["x", "y", "z"])
    latex = dataframe_to_latex(df)
    assert "x &" not in latex


def test_export_all_results_to_latex_writes_one_file_per_dataframe(tmp_path):
    df1 = pd.DataFrame({"Model": ["A"], "MAE": [0.01]})
    df2 = pd.DataFrame({"Model": ["B"], "MAE": [0.02]})

    written = export_all_results_to_latex(tmp_path, table_one=df1, table_two=df2)

    assert set(written.keys()) == {"table_one", "table_two"}
    for name, path in written.items():
        assert path.exists()
        assert path.suffix == ".tex"
        content = path.read_text()
        assert "\\begin{tabular}" in content


def test_export_all_results_to_latex_creates_output_dir(tmp_path):
    output_dir = tmp_path / "nested" / "latex_tables"
    assert not output_dir.exists()
    df = pd.DataFrame({"Model": ["A"], "MAE": [0.01]})
    export_all_results_to_latex(output_dir, my_table=df)
    assert output_dir.exists()


def test_export_all_results_to_latex_uses_derived_captions_and_labels(tmp_path):
    df = pd.DataFrame({"Model": ["A"], "MAE": [0.01]})
    written = export_all_results_to_latex(tmp_path, pareto_frontier=df)
    content = written["pareto_frontier"].read_text()
    assert "Pareto Frontier" in content
    assert "tab:pareto_frontier" in content
