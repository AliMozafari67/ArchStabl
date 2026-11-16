"""
5-GenerateProjectReport.py
--------------------------------------------------
Reads every output already produced by scripts 0-4 for a given project and
writes ONE clean, human-readable TEXT report per project, containing:

  1. Main model results          (comparison_table.csv - ALL models)
  2. Train-only Q80 threshold    (train_only_threshold_info.json)
  3. RQ1 Ablation                (rq1_ablation_<p>_<primary_model>_summary.txt / _summary_ci.csv)
  4. Repeated Stratified CV      (repeated_cv_<primary_model>_<p>_summary.csv)
  5. LOPO                        (lopo_cross_project_<model>.csv for EACH of the
                                   top-3 models, EXPORT_DIR-level)
  6. Alpha Sensitivity           (alpha_sensitivity_propagation_distribution.csv,
                                   alpha_sensitivity_ranking_stability.csv,
                                   alpha_sensitivity_ml_performance_<p>_<primary_model>.csv,
                                   alpha_sensitivity_ml_ranking_stability_<p>_<primary_model>.csv)
  7. Feature Importance          (feature_importance.csv, feature_importance_grouped_by_view.csv)
  8. Calibration / McNemar / Bootstrap
                                  (calibration_summary.csv, mcnemar_pairwise.csv,
                                   bootstrap_ci_balanced_accuracy.csv)
  9. Cross-project summary       (cross_project_summary.csv)
 10. Size / past-change baselines (baselines_<p>_<primary_model>_summary.txt / _summary_ci.csv,
                                   written by run_baseline_comparison() in 5-MLPipeline.py)

This script does NOT re-run any training - it only reads .csv/.json/.txt
files that 0-CoChange.py, 1-DependencyGraph.py, 3-ExtractHistoricalChanges.py
and 4-MLPipeline.py already write to each project's folder. If a given file
is missing, the corresponding section says so explicitly instead of silently
skipping it or crashing.

MODEL SELECTION (sections 3, 4, 6): 4-MLPipeline.py no longer assumes
RandomForest for RQ1 Ablation / Repeated CV / Alpha Sensitivity. It picks a
data-driven "primary model" - whichever model has the best mean Balanced
Accuracy across all 3 projects' comparison_table.csv results (see
select_primary_model() in 4-MLPipeline.py) - and reuses that SAME model for
every project so the ablation/CV/alpha results stay comparable across
projects. This script reads that choice from
'<EXPORT_DIR>/primary_model_selection.json' (written by 4-MLPipeline.py) and
uses it to find the right files, instead of hardcoding a model name.

MODEL SELECTION (section 5 - LOPO): runs for the TOP-3 models by the same
ranking (also recorded in primary_model_selection.json), so cross-project
generalization is shown for more than one algorithm.

How to run: just run this after 4-MLPipeline.py has finished for all
projects (same EXPORT_DIR / PROJECTS as the other scripts).

Output: "Project_Report_<project>.txt" inside each project's folder
(ExportFiles/<project>/), plus a combined
"Project_Report_ALL_PROJECTS.txt" in ExportFiles/ that concatenates all of
them one after another for convenience.
"""

import os
import json
import pandas as pd

from temporal_config import (verify_cutoff_consistency, read_cutoff_marker,
                             TemporalCutoffMismatch)

# ====================== Settings ======================
EXPORT_DIR = "ExportFiles"
PROJECTS = ["spring-framework", "elasticsearch", "hadoop"]
# ========================================================

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


class MissingModelSelectionError(RuntimeError):
    """Raised when '<EXPORT_DIR>/primary_model_selection.json' doesn't exist yet."""
    pass


def _load_model_selection():
    """Reads '<EXPORT_DIR>/primary_model_selection.json' (written by
    select_primary_model() in 4-MLPipeline.py) to find out which model was
    empirically chosen as primary (for RQ1/CV/Alpha) and which top-3 models
    were used for multi-model LOPO.

    Hard-fails instead of guessing a model if this file doesn't exist -
    e.g. because 4-MLPipeline.py hasn't finished Pass 1 yet, or crashed
    before reaching select_primary_model(). A report built around a guessed
    model name would misrepresent which model the RQ1/CV/Alpha/LOPO numbers
    actually belong to, so this script refuses to produce one rather than
    silently falling back to an assumption.
    """
    path = os.path.join(EXPORT_DIR, "primary_model_selection.json")
    if not os.path.exists(path):
        raise MissingModelSelectionError(
            f"'{path}' not found. This file is written by select_primary_model() at the end of "
            f"4-MLPipeline.py's Pass 1 (after run_pipeline() has run for every project). "
            f"Run 4-MLPipeline.py to completion for all of {PROJECTS} first, then re-run this script."
        )
    with open(path, "r", encoding="utf-8") as f:
        selection = json.load(f)
    primary = selection.get("primary_model")
    top3 = selection.get("top3_models_for_lopo")
    if not primary or not top3:
        raise MissingModelSelectionError(
            f"'{path}' exists but is missing 'primary_model' or 'top3_models_for_lopo'. "
            f"It may be corrupted or from an older version of 4-MLPipeline.py - re-run script 4."
        )
    return primary, top3, selection


PRIMARY_MODEL, TOP3_MODELS_FOR_LOPO, MODEL_SELECTION_INFO = _load_model_selection()


def _project_dir(project_name):
    return os.path.join(EXPORT_DIR, project_name)


def _section(title):
    bar = "=" * len(title)
    return f"\n{title}\n{bar}\n"


def _missing(path):
    return f"  [MISSING] '{path}' not found - re-run the script that produces it.\n"


def _read_csv_or_none(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        return f"  [ERROR] Could not read '{path}': {e}\n"


def section_1_main_results(project_name, pdir):
    out = _section("1) MAIN MODEL RESULTS (Balanced Accuracy / PR-AUC / F1 / ROC-AUC per model)")
    path = os.path.join(pdir, "comparison_table.csv")
    df = _read_csv_or_none(path)
    if df is None:
        out += _missing(path)
        out += "  (Produced by display_comparison_table() inside run_pipeline() in 4-MLPipeline.py.)\n"
        return out
    if isinstance(df, str):
        return out + df
    cols = ['Model', 'Balanced Accuracy', 'PR AUC', 'F1 Macro', 'AUC-ROC',
            'Change-Count RMSE', 'Change-Count Spearman',
            'Mean Architecture Size (Unstable)', 'Mean Architecture Cost (Unstable)']
    cols = [c for c in cols if c in df.columns]
    out += df[cols].to_string(index=False) + "\n"
    if 'Mean Architecture Cost (Unstable)' in df.columns and df['Mean Architecture Cost (Unstable)'].isna().all():
        out += ("  NOTE: 'Mean Architecture Cost (Unstable)' is NaN for every model/project - this metric is "
               "never actually computed by the pipeline yet, so it should not be reported as a real number.\n")
    return out


def section_2_threshold(project_name, pdir):
    out = _section("2) TRAIN-ONLY Q80 THRESHOLD FOR THIS SPLIT")
    path = os.path.join(pdir, "train_only_threshold_info.json")
    if not os.path.exists(path):
        out += _missing(path)
        out += "  (Produced right after the 'Train-only Q80 threshold used for this split' log line in run_pipeline().)\n"
        return out
    with open(path, "r", encoding="utf-8") as f:
        info = json.load(f)
    out += f"  Threshold (changes)      : {info.get('train_only_q80_threshold_changes'):.0f}\n"
    out += (f"  Train set                : n={info.get('train_n')}, "
            f"positives={info.get('train_positives')} "
            f"({100 * info.get('train_positive_rate', 0):.1f}%)\n")
    out += (f"  Test set                 : n={info.get('test_n')}, "
            f"positives={info.get('test_positives')} "
            f"({100 * info.get('test_positive_rate', 0):.1f}%)\n")
    return out


def section_3_rq1_ablation(project_name, pdir):
    out = _section(f"3) RQ1 ABLATION (Full vs No-Static / No-Network / No-Propagation + paired t-test) "
                    f"— model: {PRIMARY_MODEL}")
    txt_path = os.path.join(pdir, f"rq1_ablation_{project_name}_{PRIMARY_MODEL}_summary.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            out += f.read() + "\n"
        return out
    csv_path = os.path.join(pdir, f"rq1_ablation_{project_name}_{PRIMARY_MODEL}_summary_ci.csv")
    df = _read_csv_or_none(csv_path)
    if df is None:
        out += _missing(txt_path)
        out += "  (Produced by run_rq1_ablation() inside run_pipeline_followups() in 4-MLPipeline.py.)\n"
        return out
    if isinstance(df, str):
        return out + df
    out += df.to_string(index=False) + "\n"
    return out


def section_4_repeated_cv(project_name, pdir):
    out = _section(f"4) REPEATED STRATIFIED CV ({PRIMARY_MODEL}, mean +/- 95% CI over repeats)")
    path = os.path.join(pdir, f"repeated_cv_{PRIMARY_MODEL}_{project_name}_summary.csv")
    df = _read_csv_or_none(path)
    if df is None:
        out += _missing(path)
        out += "  (Produced by run_repeated_stratified_cv() inside run_pipeline_followups() in 4-MLPipeline.py.)\n"
        return out
    if isinstance(df, str):
        return out + df
    for _, row in df.iterrows():
        out += (f"  {row['Metric']:<18}: mean={row['Mean']:.4f}  "
                f"+/-95%CI={row['CI95']:.4f}  std={row['Std']:.4f}  "
                f"n_folds={int(row['N_Folds'])}\n")
    per_fold_path = os.path.join(pdir, f"repeated_cv_{PRIMARY_MODEL}_{project_name}.csv")
    if os.path.exists(per_fold_path):
        out += f"  (Per-fold detail: '{per_fold_path}')\n"
    return out


def section_5_lopo(project_name, pdir):
    out = _section(f"5) LEAVE-ONE-PROJECT-OUT (LOPO) CROSS-PROJECT EVALUATION "
                    f"— top-{len(TOP3_MODELS_FOR_LOPO)} models: {', '.join(TOP3_MODELS_FOR_LOPO)}")
    any_found = False
    for lopo_model in TOP3_MODELS_FOR_LOPO:
        out += f"\n  -- Model: {lopo_model} --\n"
        path = os.path.join(EXPORT_DIR, f"lopo_cross_project_{lopo_model}.csv")
        df = _read_csv_or_none(path)
        if df is None:
            out += _missing(path)
            out += ("  (Produced once for ALL projects by run_leave_one_project_out_pipeline() "
                     "at the end of 4-MLPipeline.py's __main__.)\n")
            continue
        if isinstance(df, str):
            out += df
            continue
        any_found = True
        row = df[df["Held_Out_Project"] == project_name]
        if row.empty:
            out += f"  No LOPO fold found for '{project_name}' in '{path}'.\n"
        else:
            r = row.iloc[0]
            out += f"  Held out             : {project_name}\n"
            out += f"  Trained on           : {r['Trained_On']}\n"
            out += f"  N_Train / N_Test     : {int(r['N_Train'])} / {int(r['N_Test'])}\n"
            out += f"  Test Balanced Acc.   : {r['Test_Balanced_Accuracy']:.4f}\n"
            out += f"  Test F1 Macro        : {r['Test_F1_Macro']:.4f}\n"
            out += f"  Test PR-AUC          : {r['Test_PR_AUC']:.4f}\n"
            out += f"  Test AUC-ROC         : {r['Test_AUC_ROC']:.4f}\n"
            if 'Q80_Threshold' in r:
                out += (f"  Pooled Q80 threshold : {r['Q80_Threshold']:.0f} changes; unstable share train pool "
                        f"{r['Train_Unstable_Fraction']:.1%}, held-out project {r['Test_Unstable_Fraction']:.1%}\n")
            if 'OwnThr_Test_Balanced_Accuracy' in r and pd.notna(r['OwnThr_Test_Balanced_Accuracy']):
                out += (f"  Own-threshold variant: threshold {r['OwnThr_Q80_Threshold']:.0f} changes (held-out project's Q80, "
                        f"diagnostic only); unstable share train {r['OwnThr_Train_Unstable_Fraction']:.1%} / test "
                        f"{r['OwnThr_Test_Unstable_Fraction']:.1%}\n")
                out += (f"      Balanced Acc. {r['OwnThr_Test_Balanced_Accuracy']:.4f}  PR-AUC {r['OwnThr_Test_PR_AUC']:.4f}  "
                        f"AUC-ROC {r['OwnThr_Test_AUC_ROC']:.4f}\n")
        ba = df["Test_Balanced_Accuracy"].dropna()
        if len(ba) > 0:
            out += f"  Across all {len(ba)} LOPO fold(s) (all projects), {lopo_model}: "
            out += f"mean Balanced Acc. = {ba.mean():.4f} +/- {ba.std():.4f} (std)\n"
    if not any_found:
        out += "\n  (No LOPO results found for any of the top-3 models yet.)\n"
    return out


def section_6_alpha_sensitivity(project_name, pdir):
    out = _section(f"6) ALPHA (ALPHA_STRUCTURAL) SENSITIVITY — ML side uses model: {PRIMARY_MODEL}")

    out += "  -- Raw propagation distribution across alpha (script 1) --\n"
    path = os.path.join(pdir, "alpha_sensitivity_propagation_distribution.csv")
    df = _read_csv_or_none(path)
    if df is None:
        out += _missing(path)
    elif isinstance(df, str):
        out += df
    else:
        out += "  " + df.to_string(index=False).replace("\n", "\n  ") + "\n"

    out += "\n  -- Class-ranking stability across alpha, raw propagation (script 1) --\n"
    path = os.path.join(pdir, "alpha_sensitivity_ranking_stability.csv")
    df = _read_csv_or_none(path)
    if df is None:
        out += _missing(path)
    elif isinstance(df, str):
        out += df
    else:
        out += "  " + df.to_string(index=False).replace("\n", "\n  ") + "\n"

    out += f"\n  -- ML performance (Balanced Accuracy etc.) vs alpha, model={PRIMARY_MODEL} (script 4) --\n"
    path = os.path.join(pdir, f"alpha_sensitivity_ml_performance_{project_name}_{PRIMARY_MODEL}.csv")
    df = _read_csv_or_none(path)
    if df is None:
        out += _missing(path)
        out += ("  (Requires RUN_ALPHA_SENSITIVITY=True in script 2 to have produced "
                 "propagation_frequencies_alpha_<a>.pkl first, then "
                 "run_alpha_sensitivity_ml_analysis() in script 4's run_pipeline_followups().)\n")
    elif isinstance(df, str):
        out += df
    else:
        out += "  " + df.to_string(index=False).replace("\n", "\n  ") + "\n"

    out += f"\n  -- Predicted-ranking stability vs alpha, ML side, model={PRIMARY_MODEL} (script 4) --\n"
    path = os.path.join(pdir, f"alpha_sensitivity_ml_ranking_stability_{project_name}_{PRIMARY_MODEL}.csv")
    df = _read_csv_or_none(path)
    if df is None:
        out += _missing(path)
    elif isinstance(df, str):
        out += df
    else:
        out += "  " + df.to_string(index=False).replace("\n", "\n  ") + "\n"

    return out


def section_7_feature_importance(project_name, pdir):
    out = _section("7) FEATURE IMPORTANCE (permutation, primary ranking)")
    path = os.path.join(pdir, "feature_importance.csv")
    df = _read_csv_or_none(path)
    if df is None:
        out += _missing(path)
    elif isinstance(df, str):
        out += df
    else:
        top = df.sort_values("Permutation_Importance", ascending=False).head(15)
        out += top.to_string(index=False) + "\n"
        if len(df) > 15:
            out += f"  ... ({len(df) - 15} more features in '{path}')\n"

    out += "\n  -- Grouped by view (Static / Network / Propagation) --\n"
    path = os.path.join(pdir, "feature_importance_grouped_by_view.csv")
    df = _read_csv_or_none(path)
    if df is None:
        out += _missing(path)
    elif isinstance(df, str):
        out += df
    else:
        out += "  " + df.to_string().replace("\n", "\n  ") + "\n"
    return out


def section_8_calibration_significance(project_name, pdir):
    out = _section("8) CALIBRATION / McNEMAR / BOOTSTRAP")

    out += "  -- Calibration (Brier score, ECE) --\n"
    path = os.path.join(pdir, "calibration_summary.csv")
    df = _read_csv_or_none(path)
    if df is None:
        out += _missing(path)
    elif isinstance(df, str):
        out += df
    else:
        out += "  " + df.to_string(index=False).replace("\n", "\n  ") + "\n"

    out += "\n  -- Pairwise McNemar's test (same held-out test set) --\n"
    path = os.path.join(pdir, "mcnemar_pairwise.csv")
    df = _read_csv_or_none(path)
    if df is None:
        out += _missing(path)
    elif isinstance(df, str):
        out += df
    else:
        out += "  " + df.to_string(index=False).replace("\n", "\n  ") + "\n"

    out += "\n  -- Bootstrap 95% CI for Balanced Accuracy --\n"
    path = os.path.join(pdir, "bootstrap_ci_balanced_accuracy.csv")
    df = _read_csv_or_none(path)
    if df is None:
        out += _missing(path)
    elif isinstance(df, str):
        out += df
    else:
        out += "  " + df.to_string(index=False).replace("\n", "\n  ") + "\n"

    return out


def section_9_cross_project_summary(project_name, pdir):
    out = _section("9) CROSS-PROJECT SUMMARY (mean +/- std across all projects, per model)")
    path = os.path.join(EXPORT_DIR, "cross_project_summary.csv")
    df = _read_csv_or_none(path)
    if df is None:
        out += _missing(path)
        out += "  (Produced once for ALL projects by build_cross_project_summary_table() in 4-MLPipeline.py's __main__.)\n"
        return out
    if isinstance(df, str):
        return out + df
    display_cols = [
        ("Balanced Accuracy", "Balanced_Accuracy"), ("PR AUC", "PR_AUC"), ("F1 Macro", "F1_Macro"),
        ("AUC-ROC", "AUC_ROC"), ("Change-Count RMSE", "RMSE"), ("Change-Count Spearman", "Spearman"),
    ]
    for _, row in df.iterrows():
        parts = [f"  {row['Model']:<20}"]
        for label, short in display_cols:
            mean_col, std_col = f"{label}_mean", f"{label}_std"
            if mean_col in df.columns:
                parts.append(f"{short}={row[mean_col]:.4f}±{row[std_col]:.4f}")
        parts.append(f"n={int(row['N_Projects'])}")
        out += "  ".join(parts) + "\n"
    cost_mean_col = "Mean Architecture Cost (Unstable)_mean"
    if cost_mean_col in df.columns and df[cost_mean_col].isna().all():
        out += ("  NOTE: 'Mean Architecture Cost (Unstable)' is NaN for every model - never actually computed "
               "by the pipeline yet, so it is omitted above and should not be reported as a real number.\n")
    return out


def section_10_baselines(project_name, pdir):
    out = _section(f"10) SIZE / PAST-CHANGE BASELINES vs FULL REPRESENTATION — model: {PRIMARY_MODEL}")
    txt_path = os.path.join(pdir, f"baselines_{project_name}_{PRIMARY_MODEL}_summary.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return out + f.read() + "\n"
    csv_path = os.path.join(pdir, f"baselines_{project_name}_{PRIMARY_MODEL}_summary_ci.csv")
    df = _read_csv_or_none(csv_path)
    if df is None:
        out += _missing(txt_path)
        out += "  (Produced by run_baseline_comparison() inside run_pipeline_followups() in 5-MLPipeline.py.)\n"
        return out
    if isinstance(df, str):
        return out + df
    return out + df.to_string(index=False) + "\n"


def _txt_section(project_name, pdir, title, file_name, producer):
    out = _section(title)
    path = os.path.join(pdir, file_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return out + f.read() + "\n"
    return out + _missing(path) + f"  (Produced by {producer} in 5-MLPipeline.py.)\n"


def section_11_view_combinations(project_name, pdir):
    return _txt_section(project_name, pdir,
                        f"11) ALL SEVEN VIEW COMBINATIONS (S, N, P, S+N, S+P, N+P, Full) — model: {PRIMARY_MODEL}",
                        f"viewcombos_{project_name}_{PRIMARY_MODEL}_summary.txt", "run_view_combinations()")


def section_12_group_split(project_name, pdir):
    return _txt_section(project_name, pdir,
                        f"12) MODULE-GROUPED SPLIT (whole source modules held out) — model: {PRIMARY_MODEL}",
                        f"groupsplit_{project_name}_{PRIMARY_MODEL}_summary.txt", "run_group_split_analysis()")


def section_13_population(project_name, pdir):
    import json as _json
    out = _section("13) MODELLING POPULATION, PARSING EXCLUSIONS AND EXTENDED-POPULATION SENSITIVITY")
    cov_path = os.path.join(pdir, f"population_coverage_{project_name}.json")
    if os.path.exists(cov_path):
        with open(cov_path, "r", encoding="utf-8") as f:
            cov = _json.load(f)
        out += "  -- Coverage of the modelling population (counts of classes) --\n"
        for k, v in cov.items():
            if isinstance(v, dict):
                out += "  snapshot population from git ls-tree HEAD (script 4):\n"
                for k2, v2 in v.items():
                    out += f"      {k2:<45}{v2}\n"
            else:
                out += f"    {k:<70}{v}\n"
    else:
        out += _missing(cov_path)
    # Parsing statistics (script 3)
    ps_path = os.path.join(pdir, "parsing_stats_code_metrics.pkl")
    out += "\n  -- Java files excluded by the metric extraction (script 3) --\n"
    if os.path.exists(ps_path):
        import pickle as _pickle
        with open(ps_path, "rb") as f:
            ps = _pickle.load(f)
        for k, v in ps.items():
            out += f"    {k:<45}{v if not isinstance(v, float) else round(v, 2)}\n"
    else:
        out += _missing(ps_path)
    ex_path = os.path.join(pdir, "parsing_excluded_files.csv")
    ex = _read_csv_or_none(ex_path)
    if ex is None:
        out += "    (parsing_excluded_files.csv not found - re-run script 3 to obtain the per-file exclusion list.)\n"
    elif not isinstance(ex, str) and len(ex):
        out += "    Excluded files by reason:\n" + ex.groupby("reason").size().to_string() + "\n"
        sz = ex["size_bytes"].dropna()
        if len(sz):
            out += f"    Size of excluded files (bytes): median={sz.median():.0f}  mean={sz.mean():.0f}  max={sz.max():.0f}\n"
        out += "    Top modules among excluded files:\n" + ex.groupby("top_module").size().sort_values(ascending=False).head(8).to_string() + "\n"
    out += "\n"
    path = os.path.join(pdir, f"population_extended_{project_name}_{PRIMARY_MODEL}_summary.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            out += "  -- Extended population (unchanged surviving classes added with change count 0) --\n" + f.read() + "\n"
    else:
        out += _missing(path) + "  (Needs snapshot_population_<project>.json from script 4.)\n"
    path = os.path.join(pdir, f"population_extended_graphonly_{project_name}_{PRIMARY_MODEL}_summary.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            out += ("  -- Extended population, graph-only (only unchanged classes with complete graph/propagation features;"
                    " no zero-filled features, no re-admitted changed classes) --\n" + f.read() + "\n")
    else:
        out += _missing(path) + "  (Produced by run_population_analysis() in 5-MLPipeline.py.)\n"
    return out


def section_16_head_and_mass_commits(project_name, pdir):
    out = _txt_section(project_name, pdir,
                       f"16a) MAIN POPULATION WITHOUT CLASSES ABSENT AT HEAD — model: {PRIMARY_MODEL}",
                       f"population_headonly_{project_name}_{PRIMARY_MODEL}_summary.txt", "run_population_analysis()")
    out += _txt_section(project_name, pdir,
                        f"16b) LABELS WITHOUT MASS COMMITS (> 30 Java files per commit) — model: {PRIMARY_MODEL}",
                        f"population_nomass_{project_name}_{PRIMARY_MODEL}_summary.txt", "run_population_analysis()")
    return out


def section_15_size_matched(project_name, pdir):
    return _txt_section(project_name, pdir,
                        f"15) SIZE-MATCHED ANALYSIS (random subsample of classes per repeat) — model: {PRIMARY_MODEL}",
                        f"sizematched_{project_name}_{PRIMARY_MODEL}_summary.txt", "run_size_matched_analysis()")


def section_14_corrected_tests(project_name, pdir):
    out = _section("14) NADEAU-BENGIO CORRECTED TESTS AND HOLM CORRECTION (per hypothesis family, across projects)")
    path = os.path.join(EXPORT_DIR, f"stat_tests_corrected_{PRIMARY_MODEL}.csv")
    df = _read_csv_or_none(path)
    if df is None:
        return out + _missing(path)
    if isinstance(df, str):
        return out + df
    sub = df[df["Project"] == project_name]
    if sub.empty:
        return out + "  (no comparisons for this project)\n"
    cols = ["Family", "Comparison", "Mean_diff_Full_minus_other", "CI95_low", "CI95_high", "Cohens_d",
            "p_paired_unadjusted", "p_NadeauBengio", "p_Holm_within_family", "Family_size",
            "p_TOST_equiv", "p_TOST_Holm", "Equiv_margin"]
    cols = [c for c in cols if c in sub.columns]
    out += ("  Mean_diff > 0 means the Full representation is better. CI95 = Nadeau-Bengio corrected 95% interval of the\n"
            "  balanced-accuracy difference. Holm is applied within each family across all projects.\n"
            "  Families: ablation (3 reduced views, pre-specified), viewcombos (6 subsets; the single-view comparisons form the\n"
            "  exploratory 'viewsingles' family), baselines, groupsplit, population_extended, population_extended_graphonly,\n"
            "  sizematched. viewcombos contains the same S+N, S+P, N+P comparisons as the ablation; report one family per question.\n"
            "  p_TOST_equiv = two one-sided tests of |difference| < Equiv_margin (small = equivalence supported); p_TOST_Holm is\n"
            "  adjusted within the family. Non-significance of the ordinary test is NOT evidence of equivalence.\n\n")
    out += sub[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n"
    meta = _read_csv_or_none(os.path.join(EXPORT_DIR, f"meta_analysis_{PRIMARY_MODEL}.csv"))
    if meta is not None and not isinstance(meta, str):
        out += ("\n  -- Random-effects meta-analysis across the three projects (DerSimonian-Laird, Nadeau-Bengio variances;"
                " t-based interval with k-1 df; same table in every project report) --\n")
        out += meta.to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n"
    figs = os.path.join(EXPORT_DIR, "paper_figures")
    if os.path.isdir(figs):
        out += "\n  Publication figures (generated by 5-MLPipeline.py): " + ", ".join(sorted(os.listdir(figs))) + "\n"
    return out


SECTION_FUNCS = [
    section_1_main_results,
    section_2_threshold,
    section_3_rq1_ablation,
    section_4_repeated_cv,
    section_5_lopo,
    section_6_alpha_sensitivity,
    section_7_feature_importance,
    section_8_calibration_significance,
    section_9_cross_project_summary,
    section_10_baselines,
    section_11_view_combinations,
    section_12_group_split,
    section_13_population,
    section_14_corrected_tests,
    section_15_size_matched,
    section_16_head_and_mass_commits,
]


def section_0_temporal_protocol(project_name, pdir):
    """Documents the temporal protocol the results were produced with, and refuses to
    produce a report if scripts 0-3 were not run with one consistent cutoff (so a stale or
    whole-history (leaky) run can never be reported by accident)."""
    cutoff = verify_cutoff_consistency(pdir, project_name=project_name)   # raises on problems
    out = _section("0) TEMPORAL PROTOCOL (verified from the cutoff markers of scripts 0-3)")
    out += f"  Cutoff datetime          : {cutoff}\n"
    out += f"  Snapshot / anchor commit : {read_cutoff_marker(pdir, 'historical').get('anchor_commit')}\n"
    out += ("  Features   : static metrics + dependency graph from the anchor-commit snapshot; "
            "co-change from commits reachable from the anchor.\n")
    out += "  Target     : class change counts in anchor..HEAD (post-cutoff), merges excluded.\n"
    return out


def generate_report_for_project(project_name):
    pdir = _project_dir(project_name)
    os.makedirs(pdir, exist_ok=True)

    header = (
        f"PROJECT REPORT: {project_name}\n"
        f"{'#' * (18 + len(project_name))}\n"
        f"Generated from: {pdir}\n"
        f"Primary model (RQ1/Repeated-CV/Alpha, chosen empirically - see section 1's footer): {PRIMARY_MODEL}\n"
        f"Top-{len(TOP3_MODELS_FOR_LOPO)} models used for LOPO: {', '.join(TOP3_MODELS_FOR_LOPO)}\n"
    )
    body = section_0_temporal_protocol(project_name, pdir) + "".join(fn(project_name, pdir) for fn in SECTION_FUNCS)
    footer = _section("MODEL SELECTION DETAILS (how PRIMARY_MODEL / top-3 were chosen)")
    footer += f"  Selection metric: {MODEL_SELECTION_INFO.get('selection_metric', 'mean test Balanced Accuracy (older run)')}\n"
    footer += f"  Selection rule: {MODEL_SELECTION_INFO.get('selection_rule')}\n"
    footer += f"  Candidate models considered: {MODEL_SELECTION_INFO.get('candidate_models')}\n"
    footer += "  Ranking (mean of the selection metric across projects):\n"
    for row in MODEL_SELECTION_INFO.get("ranking", []):
        footer += (f"    {row.get('Model'):<20} mean_BA={row.get('Mean_Balanced_Accuracy'):.4f}  "
                   f"n_projects={row.get('N_Projects')}  mean_rank={row.get('Mean_Rank'):.2f}\n")
    missing = MODEL_SELECTION_INFO.get("missing_projects") or []
    if missing:
        footer += f"  NOTE: comparison_table.csv was missing for: {missing} (excluded from ranking).\n"
    report_text = header + body + footer

    out_path = os.path.join(pdir, f"Project_Report_{project_name}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[{project_name}] Report saved to: {out_path}")
    return report_text


if __name__ == "__main__":
    all_reports = []
    failed = []
    for project_name in PROJECTS:
        try:
            all_reports.append(generate_report_for_project(project_name))
        except Exception as e:
            print(f"[{project_name}] Failed to generate report: {e}")
            failed.append(project_name)

    combined_path = os.path.join(EXPORT_DIR, "Project_Report_ALL_PROJECTS.txt")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(("\n\n" + "=" * 100 + "\n\n").join(all_reports))
    print(f"\nCombined report for all projects saved to: {combined_path}")
    if failed:
        raise SystemExit(f"Report generation FAILED for: {failed}. The combined report above is incomplete - "
                         f"fix the errors (e.g. cutoff markers) and re-run.")
