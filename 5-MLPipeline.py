import os
import io
import json
import networkx as nx
import pandas as pd
import numpy as np
import pickle
import logging
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict, \
    RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, \
    balanced_accuracy_score, precision_recall_curve, auc as calculate_auc, f1_score, roc_auc_score, \
    mean_squared_error, mean_absolute_error, r2_score, brier_score_loss
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE, RFECV
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from scipy import stats as scipy_stats

from temporal_config import (verify_cutoff_consistency, TemporalCutoffMismatch,
                             verified_anchor_commit, ALLOW_WHOLE_HISTORY)

# Logging
class InfoFilter(logging.Filter):
    def filter(self, record): return record.levelno == logging.INFO


logger = logging.getLogger('stability_calculator')
logger.setLevel(logging.INFO)
logger.propagate = False
EXPORT_DIR = "ExportFiles"

# Set per project by configure_logging()
PROJECT_NAME = None
project_dir = None

# Mirrors console output; flushed to 'Result (<project>) Log.txt' after each project
console_log_buffer = None

# Projects already logged in this run: the first call overwrites the log, later calls append
_projects_logged_this_run = set()


def configure_logging(project_name):
    """(Re)configure the logger and globals for a project; attaches file and in-memory console handlers."""
    global PROJECT_NAME, project_dir, console_log_buffer

    PROJECT_NAME = project_name
    project_dir = os.path.join(EXPORT_DIR, PROJECT_NAME)
    os.makedirs(project_dir, exist_ok=True)

    is_first_call_this_run = project_name not in _projects_logged_this_run
    file_mode = 'w' if is_first_call_this_run else 'a'

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(InfoFilter())
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Overwrite on the first call for a project in this run, append afterwards
    file_handler = logging.FileHandler(os.path.join(project_dir, 'git_stability_results.log'), mode=file_mode, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Captures console output for the per-project log dump; reset on every call
    console_log_buffer = io.StringIO()
    buffer_handler = logging.StreamHandler(console_log_buffer)
    buffer_handler.setLevel(logging.INFO)
    buffer_handler.addFilter(InfoFilter())
    buffer_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    if logger.hasHandlers(): logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(buffer_handler)


def log_progress(prefix, current, total, start_time):
    """Log a 'N/total (pct%) | elapsed | ETA' line via `logger` so it reaches the console and all log files."""
    elapsed = time.time() - start_time
    pct = 100.0 * current / total if total else 0.0
    rate = elapsed / current if current else 0.0
    eta = rate * (total - current) if current else float('nan')
    logger.info(f"[progress] {prefix}: {current}/{total} ({pct:5.1f}%) | "
                f"elapsed {elapsed:7.1f}s | ETA {eta:7.1f}s")


def save_console_log_to_file():
    """Write the captured console output to 'Result (<project>) Log.txt' (overwrite on the first call per run, append afterwards)."""
    global _projects_logged_this_run
    log_path = os.path.join(project_dir, f"Result ({PROJECT_NAME}) Log.txt")
    is_first_write_this_run = PROJECT_NAME not in _projects_logged_this_run
    mode = 'w' if is_first_write_this_run else 'a'
    with open(log_path, mode, encoding="utf-8") as f:
        f.write(console_log_buffer.getvalue())
    _projects_logged_this_run.add(PROJECT_NAME)
    logger.info(f"Console log saved to: {log_path}")


# Feature groups and data preparation
FEATURE_GROUPS = {
    'static': ['loc', 'cc_class', 'cbo_approx', 'lcom4_approx',
               'num_methods', 'total_method_cc', 'ccn', 'architecture_size'],
    'network': ['degree_centrality', 'betweenness_centrality', 'pagerank',
                'clustering', 'eigenvector_centrality'],
    'propagation': ['propagation_frequency'],
}

VIEW_SETS = {
    'Full (Static+Network+Propagation)': (FEATURE_GROUPS['static'] +
                                          FEATURE_GROUPS['network'] +
                                          FEATURE_GROUPS['propagation']),
    'No-Static (Network+Propagation)': FEATURE_GROUPS['network'] + FEATURE_GROUPS['propagation'],
    'No-Network (Static+Propagation)': FEATURE_GROUPS['static'] + FEATURE_GROUPS['propagation'],
    'No-Propagation (Static+Network)': FEATURE_GROUPS['static'] + FEATURE_GROUPS['network'],
}

RF_PARAM_GRID = {'n_estimators': [100, 200], 'max_depth': [10, 15], 'min_samples_split': [5, 10]}

# Module-level so follow-up analyses and LOPO can look up any model's grid by name
MODEL_PARAM_GRIDS = {
    'RandomForest': {'n_estimators': [100, 200], 'max_depth': [10, 15], 'min_samples_split': [5, 10]},
    'GaussianNB': {'var_smoothing': [1e-9, 1e-8]},
    'XGBoost': {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]},
    'SVM': {'C': [0.1, 1], 'kernel': ['rbf']},
    'DummyMostFrequent': {},
    'DummyStratified': {},
    'LogisticRegression': {'C': [0.01, 0.1, 1, 10]},
}
# Candidates for primary model / multi-model LOPO (Dummy baselines excluded)
CANDIDATE_MODEL_NAMES = ['RandomForest', 'GaussianNB', 'XGBoost', 'SVM', 'LogisticRegression']

# Fixed RandomForest config (middle of RF_PARAM_GRID), used by the ablation, repeated-CV and
# alpha analyses instead of re-running GridSearchCV in every iteration.
# Nested tuning: if =1,ARCHSTABL_NESTED_TUNING hyperparameters are re-tuned by GridSearchCV inside the training
# partition of every repeat/fold/alpha (no reuse of seed-42 hyperparameters on overlapping test sets). Slower.
EQUIV_MARGIN = 0.02        # pre-specified smallest effect of interest (Balanced Accuracy) for the equivalence tests (TOST)
SIZE_MATCH_N = 5000        # class count used by the size-matched subsampling analysis (about the size of Spring Framework)
NESTED_TUNING = os.environ.get('ARCHSTABL_NESTED_TUNING', '0') == '1'
RF_FIXED_PARAMS = {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5}

ALPHA_SENSITIVITY_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Raw metrics architecture_size is derived from; recomputed from train-only statistics after each split
SIZE_FACTORS = ['loc', 'num_methods', 'cbo_approx', 'lcom4_approx', 'total_method_cc', 'cc_class']


def add_train_test_architecture_size(X_train, X_test, size_factors=SIZE_FACTORS):
    """Add 'architecture_size' to copies of X_train/X_test using train-only min/max of each raw size factor.

    Returns (X_train, X_test).
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    norm_cols = []
    for factor in size_factors:
        min_val = X_train[factor].min()
        max_val = X_train[factor].max()
        denom = (max_val - min_val) if (max_val - min_val) > 0 else 1.0
        norm_col = f'_norm_{factor}'
        X_train[norm_col] = (X_train[factor] - min_val) / denom
        X_test[norm_col] = (X_test[factor] - min_val) / denom  # same train min/max, never refit on test
        norm_cols.append(norm_col)
    X_train['architecture_size'] = X_train[norm_cols].mean(axis=1)
    X_test['architecture_size'] = X_test[norm_cols].mean(axis=1)
    X_train.drop(columns=norm_cols, inplace=True)
    X_test.drop(columns=norm_cols, inplace=True)
    return X_train, X_test


# Q80 threshold estimated from training rows only
QUANTILE_THRESHOLD = 0.80


def make_stability_labels(change_count, train_index, quantile=QUANTILE_THRESHOLD):
    """Label rows as unstable using a threshold estimated from the train_index rows only.

    change_count: Series (class_name -> raw change count) for all rows.
    train_index: labels selecting the training rows.
    Returns (labels 0/1 Series, train-only threshold). The threshold varies across splits/folds; report it.
    """
    train_counts = change_count.loc[train_index]
    threshold = train_counts.quantile(quantile)
    if threshold == 0:
        threshold = 1
    labels = (change_count >= threshold).astype(int)
    return labels, threshold


def make_stratification_proxy(change_count, n_bins=5):
    """Quintile bins of change_count (by rank), used only to stratify splits before the real Q80 label exists."""
    ranks = change_count.rank(method="first")
    return pd.qcut(ranks, q=n_bins, labels=False)


def compute_scale_pos_weight(y_train):
    """XGBoost scale_pos_weight = N_negative / N_positive, computed from training labels only."""
    y_train_arr = np.asarray(y_train)
    n_pos = int(np.sum(y_train_arr == 1))
    n_neg = int(np.sum(y_train_arr == 0))
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


N_ABLATION_REPEATS = 10


def run_rq1_ablation(X, change_count, project_name, project_dir, train_classification_model, evaluate_classification_model,
                     n_repeats=N_ABLATION_REPEATS, model_name='RandomForest', fixed_params=None):
    """Leave-one-view-out ablation over n_repeats stratified splits.

    X: full feature matrix.
    change_count: raw change counts (the Q80 label is built per repeat from train rows).
    train_classification_model / evaluate_classification_model: the pipeline's training/evaluation functions.
    model_name / fixed_params: model and held-constant hyperparameters (default RandomForest + RF_FIXED_PARAMS).
    Reports mean +/- 95% CI per view, plus a paired t-test and Cohen's d of each ablated view vs Full.
    """
    if fixed_params is None:
        fixed_params = RF_FIXED_PARAMS if model_name == 'RandomForest' else {}
    param_grid = MODEL_PARAM_GRIDS.get(model_name, {})
    logger.info(f"[{project_name}] RQ1 ablation will use model={model_name}, fixed_params={fixed_params}")

    per_repeat_scores = {view_name: [] for view_name in VIEW_SETS}
    rows = []
    ablation_start_time = time.time()
    strat_proxy = make_stratification_proxy(change_count)

    for repeat_idx in range(n_repeats):
        seed = 42 + repeat_idx
        X_train_full, X_test_full, cc_train, cc_test = train_test_split(
            X, change_count, test_size=0.2, stratify=strat_proxy, random_state=seed
        )
        # Threshold from this repeat's train rows only
        y_train, repeat_threshold = make_stability_labels(change_count, X_train_full.index)
        y_train = y_train.loc[X_train_full.index]
        y_test = (change_count.loc[X_test_full.index] >= repeat_threshold).astype(int)

        # Rebuild architecture_size from this split's train-only statistics
        X_train_full, X_test_full = add_train_test_architecture_size(X_train_full, X_test_full)

        for view_name, cols in VIEW_SETS.items():
            cols_present = [c for c in cols if c in X_train_full.columns]
            if not cols_present:
                continue
            X_train_view = X_train_full[cols_present]
            X_test_view = X_test_full[cols_present]

            scaler = MinMaxScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train_view),
                index=X_train_view.index, columns=X_train_view.columns
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test_view),
                index=X_test_view.index, columns=X_test_view.columns
            )

            # Fixed config instead of GridSearchCV per (repeat, view)
            model, _, _ = train_classification_model(
                X_train_scaled, y_train, model_name, param_grid,
                skip_grid_search=not NESTED_TUNING, fixed_params=(None if NESTED_TUNING else fixed_params)
            )
            if model is None:
                continue
            metrics = evaluate_classification_model(
                model, X_test_scaled, y_test, model_name=f'{model_name}_{view_name}_rep{repeat_idx}'
            )
            bal_acc = metrics.get('Balanced Accuracy') if metrics else np.nan
            per_repeat_scores[view_name].append(bal_acc)
            rows.append({
                'Project': project_name, 'View': view_name, 'Repeat': repeat_idx,
                'N_Features': len(cols_present),
                'Balanced_Accuracy': bal_acc,
                'F1_Macro': metrics.get('F1 Macro') if metrics else np.nan,
                'PR_AUC': metrics.get('PR AUC') if metrics else np.nan,
                'AUC_ROC': metrics.get('AUC-ROC') if metrics else np.nan,
            })

        log_progress(f"[{project_name}] RQ1 ablation repeats", repeat_idx + 1, n_repeats, ablation_start_time)

    per_repeat_df = pd.DataFrame(rows)
    csv_path = f"{project_dir}/rq1_ablation_{project_name}_{model_name}_per_repeat.csv"
    per_repeat_df.to_csv(csv_path, index=False)

    # --- Aggregate mean +/- 95% CI per view ---
    summary_rows = []
    full_scores = np.array([s for s in per_repeat_scores.get('Full (Static+Network+Propagation)', []) if not np.isnan(s)])
    for view_name, scores in per_repeat_scores.items():
        arr = np.array([s for s in scores if not np.isnan(s)])
        if arr.size == 0:
            continue
        mean = arr.mean()
        sem = arr.std(ddof=1) / np.sqrt(arr.size) if arr.size > 1 else 0.0
        ci95 = 1.96 * sem
        entry = {
            'View': view_name, 'N_Repeats': arr.size,
            'Mean_Balanced_Accuracy': mean, 'CI95': ci95,
            'Std': arr.std(ddof=1) if arr.size > 1 else 0.0,
        }
        # Paired test vs Full view, with Cohen's d
        if view_name != 'Full (Static+Network+Propagation)' and full_scores.size == arr.size and arr.size > 1:
            try:
                t_stat, p_val = scipy_stats.ttest_rel(full_scores, arr)
                diff = full_scores - arr
                cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
                entry['Paired_t_vs_Full'] = t_stat
                entry['p_value_vs_Full'] = p_val
                entry['Cohens_d_vs_Full'] = cohens_d
                entry['Significant_at_0.05'] = bool(p_val < 0.05) if not np.isnan(p_val) else False
            except Exception as e:
                logger.warning(f"Paired t-test failed for view '{view_name}': {e}")
        summary_rows.append(entry)

    result_df = pd.DataFrame(summary_rows).sort_values('Mean_Balanced_Accuracy', ascending=False)
    summary_csv_path = f"{project_dir}/rq1_ablation_{project_name}_{model_name}_summary_ci.csv"
    result_df.to_csv(summary_csv_path, index=False)

    # --- Human-readable summary .txt ---
    txt_path = f"{project_dir}/rq1_ablation_{project_name}_{model_name}_summary.txt"
    lines = []
    lines.append(f"RQ1 Ablation Summary — Project: {project_name}  ({n_repeats} repeated stratified splits)")
    lines.append(f"Model (held constant across views/repeats): {model_name}  |  fixed_params={fixed_params}")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"{'View':<38}{'N':>3}  {'MeanBalAcc':>11}  {'+/-95%CI':>9}  {'p vs Full':>10}  {'Cohens d':>9}  {'Sig?':>5}")
    lines.append("-" * 90)
    for _, r in result_df.iterrows():
        p_str = f"{r.get('p_value_vs_Full', np.nan):.4f}" if not pd.isna(r.get('p_value_vs_Full', np.nan)) else "  n/a"
        d_str = f"{r.get('Cohens_d_vs_Full', np.nan):+.3f}" if not pd.isna(r.get('Cohens_d_vs_Full', np.nan)) else "  n/a"
        sig_str = "YES" if r.get('Significant_at_0.05', False) else ("--" if r['View'] == 'Full (Static+Network+Propagation)' else "no")
        lines.append(
            f"{r['View']:<38}{int(r['N_Repeats']):>3}  "
            f"{r['Mean_Balanced_Accuracy']:>11.4f}  +/-{r['CI95']:>6.4f}  "
            f"{p_str:>10}  {d_str:>9}  {sig_str:>5}"
        )
    lines.append("")
    lines.append("Sig? = paired t-test (paired on repeat index) vs the Full view at alpha=0.05.")
    lines.append("A view's contribution should only be claimed as 'significant' when Sig?=YES,")
    lines.append("not merely because its point-estimate mean is lower than Full's (item #15 fix -")
    lines.append("a single-split gap like 0.84 vs 0.79 could easily be sampling noise; this repeats")
    lines.append("the split and tests it explicitly).")

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"\n--- RQ1 Ablation Results ({project_name}) ---")
    print("\n".join(lines))
    return result_df


# ---------------------------------------------------------------------------------------------
# Size / past-change baselines (supplementary analysis, Section "Comparison with baselines")
# ---------------------------------------------------------------------------------------------
RUN_BASELINE_COMPARISON = True
BASELINE_FULL_VIEW = 'Full (Static+Network+Propagation)'


def load_prior_change_counts(project_dir, index):
    """Number of non-merge commits touching each class in the PRE-cutoff window (available at prediction time).

    Reads class_change_frequency_before.pkl (written by the historical-changes script). Classes that exist in
    the snapshot but have no pre-cutoff change get 0. Returns None if the file does not exist.
    """
    path = os.path.join(project_dir, "class_change_frequency_before.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        before = dict(pickle.load(f))
    return pd.Series(before, dtype=float).reindex(index).fillna(0.0)


def run_baseline_comparison(X, change_count, project_name, project_dir, train_classification_model,
                            evaluate_classification_model, n_repeats=N_ABLATION_REPEATS,
                            model_name='RandomForest', fixed_params=None):
    """Compare the Full ArchStabl representation with LOC-only / past-change baselines.

    Same protocol as run_rq1_ablation (same seeds, train-only Q80 label, train-only architecture_size,
    min-max scaling, fixed hyperparameters, no RFECV), so the 'Full' row is numerically identical to the
    ablation's Full row (checked and reported). Feature sets:
        LOC-only | PriorChanges-only | LOC+PriorChanges | Full | Full+PriorChanges
    Writes baselines_<project>_<model>_per_repeat.csv / _summary_ci.csv / _summary.txt in project_dir.
    """
    if fixed_params is None:
        fixed_params = RF_FIXED_PARAMS if model_name == 'RandomForest' else {}
    param_grid = MODEL_PARAM_GRIDS.get(model_name, {})
    prior = load_prior_change_counts(project_dir, X.index)
    if prior is None:
        logger.warning(f"[{project_name}] class_change_frequency_before.pkl not found in '{project_dir}'; "
                       f"skipping baseline comparison.")
        return None
    X = X.copy()
    X['prior_changes'] = prior.values
    logger.info(f"[{project_name}] Baseline comparison: model={model_name}, fixed_params={fixed_params}; "
                f"{(prior > 0).mean():.1%} of classes have prior_changes > 0")

    full_cols = list(VIEW_SETS[BASELINE_FULL_VIEW])
    feature_sets = {
        'LOC-only': ['loc'],
        'PriorChanges-only': ['prior_changes'],
        'LOC+PriorChanges': ['loc', 'prior_changes'],
        BASELINE_FULL_VIEW: full_cols,
        'Full+PriorChanges': full_cols + ['prior_changes'],
    }
    per_repeat_scores = {name: [] for name in feature_sets}
    rows = []
    start_time = time.time()
    strat_proxy = make_stratification_proxy(change_count)
    for repeat_idx in range(n_repeats):
        seed = 42 + repeat_idx
        X_train_full, X_test_full, _, _ = train_test_split(
            X, change_count, test_size=0.2, stratify=strat_proxy, random_state=seed)
        y_train, repeat_threshold = make_stability_labels(change_count, X_train_full.index)
        y_train = y_train.loc[X_train_full.index]
        y_test = (change_count.loc[X_test_full.index] >= repeat_threshold).astype(int)
        X_train_full, X_test_full = add_train_test_architecture_size(X_train_full, X_test_full)
        for name, cols in feature_sets.items():
            cols_present = [c for c in cols if c in X_train_full.columns]
            if not cols_present:
                continue
            scaler = MinMaxScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_full[cols_present]),
                                          index=X_train_full.index, columns=cols_present)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test_full[cols_present]),
                                         index=X_test_full.index, columns=cols_present)
            model, _, _ = train_classification_model(X_train_scaled, y_train, model_name, param_grid,
                                                     skip_grid_search=not NESTED_TUNING, fixed_params=(None if NESTED_TUNING else fixed_params))
            if model is None:
                continue
            metrics = evaluate_classification_model(model, X_test_scaled, y_test,
                                                    model_name=f'{model_name}_baseline_{name}_rep{repeat_idx}')
            bal_acc = metrics.get('Balanced Accuracy') if metrics else np.nan
            per_repeat_scores[name].append(bal_acc)
            rows.append({'Project': project_name, 'FeatureSet': name, 'Repeat': repeat_idx,
                         'N_Features': len(cols_present), 'Threshold': repeat_threshold,
                         'Balanced_Accuracy': bal_acc,
                         'F1_Macro': metrics.get('F1 Macro') if metrics else np.nan,
                         'PR_AUC': metrics.get('PR AUC') if metrics else np.nan,
                         'AUC_ROC': metrics.get('AUC-ROC') if metrics else np.nan})
        log_progress(f"[{project_name}] baseline comparison repeats", repeat_idx + 1, n_repeats, start_time)

    per_repeat_df = pd.DataFrame(rows)
    per_repeat_df.to_csv(f"{project_dir}/baselines_{project_name}_{model_name}_per_repeat.csv", index=False)

    full_scores = np.array(per_repeat_scores[BASELINE_FULL_VIEW], dtype=float)
    summary_rows = []
    for name in feature_sets:
        arr = np.array(per_repeat_scores[name], dtype=float)
        if arr.size == 0:
            continue
        sub = per_repeat_df[per_repeat_df.FeatureSet == name]
        sem = arr.std(ddof=1) / np.sqrt(arr.size) if arr.size > 1 else 0.0
        entry = {'FeatureSet': name, 'N_Repeats': arr.size, 'N_Features': int(sub.N_Features.iloc[0]),
                 'Mean_Balanced_Accuracy': arr.mean(), 'CI95': 1.96 * sem,
                 'Std': arr.std(ddof=1) if arr.size > 1 else 0.0,
                 'Mean_PR_AUC': sub.PR_AUC.mean(), 'Mean_AUC_ROC': sub.AUC_ROC.mean(),
                 'Full_minus_this': (full_scores.mean() - arr.mean()) if full_scores.size else np.nan}
        if name != BASELINE_FULL_VIEW and full_scores.size == arr.size and arr.size > 1:
            diff = full_scores - arr
            try:
                entry['p_value_vs_Full'] = scipy_stats.ttest_rel(full_scores, arr).pvalue
            except Exception:
                entry['p_value_vs_Full'] = np.nan
            entry['Cohens_d_Full_minus_this'] = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
        summary_rows.append(entry)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{project_dir}/baselines_{project_name}_{model_name}_summary_ci.csv", index=False)

    # Consistency check against the RQ1 ablation Full row (same code path => must be identical)
    consistency = "not checked (ablation per-repeat file not found)"
    abl_path = f"{project_dir}/rq1_ablation_{project_name}_{model_name}_per_repeat.csv"
    if os.path.exists(abl_path):
        try:
            abl = pd.read_csv(abl_path)
            abl_full = abl[abl.View == BASELINE_FULL_VIEW].sort_values('Repeat')['Balanced_Accuracy'].values
            if abl_full.size == full_scores.size and full_scores.size:
                max_diff = float(np.max(np.abs(abl_full - full_scores)))
                consistency = (f"Full row identical to the RQ1 ablation Full row (max |diff| = {max_diff:.2e})"
                               if max_diff < 1e-9 else
                               f"WARNING: Full row DIFFERS from the RQ1 ablation Full row (max |diff| = {max_diff:.4f}) "
                               f"- the run is not deterministic; investigate before reporting.")
        except Exception as e:
            consistency = f"consistency check failed: {e}"
    logger.info(f"[{project_name}] Baseline consistency check: {consistency}")

    lines = [f"Baseline comparison — Project: {project_name}  ({n_repeats} repeated stratified splits)",
             f"Model (held constant): {model_name}  |  fixed_params={fixed_params}",
             "prior_changes = non-merge commits touching the class BEFORE the cutoff (0 if none)",
             "=" * 118, "",
             f"{'Feature set':<36}{'N':>3}{'F':>3}  {'MeanBalAcc':>11}  {'+/-95%CI':>9}  {'PR-AUC':>7}  {'ROC-AUC':>8}"
             f"  {'Full-this':>9}  {'p vs Full':>10}  {'d(Full-this)':>12}", "-" * 118]
    for _, r in summary_df.iterrows():
        p = r.get('p_value_vs_Full', np.nan)
        d = r.get('Cohens_d_Full_minus_this', np.nan)
        lines.append(f"{r['FeatureSet']:<36}{int(r['N_Repeats']):>3}{int(r['N_Features']):>3}  "
                     f"{r['Mean_Balanced_Accuracy']:>11.4f}  +/-{r['CI95']:>6.4f}  {r['Mean_PR_AUC']:>7.3f}  "
                     f"{r['Mean_AUC_ROC']:>8.3f}  {r['Full_minus_this']:>+9.4f}  "
                     f"{(f'{p:.4f}' if not pd.isna(p) else 'n/a'):>10}  {(f'{d:+.3f}' if not pd.isna(d) else 'n/a'):>12}")
    lines += ["", "Full-this = mean BA of the Full representation minus this row (positive = Full is better).",
              "p = paired t-test on the repeat index (unadjusted); d = Cohen's d of the paired differences.",
              f"Consistency: {consistency}"]
    with open(f"{project_dir}/baselines_{project_name}_{model_name}_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n--- Baseline comparison ({project_name}) ---")
    print("\n".join(lines))
    return summary_df



# =====================================================================================================
# Supplementary analyses added after the external review of the paper (all use the ablation protocol)
#   1. VIEW COMBINATIONS : all 7 non-empty subsets of the three views (S, N, P, S+N, S+P, N+P, Full)
#   2. GROUP SPLIT       : train/test split by top-level source module instead of random classes
#   3. POPULATION        : coverage of the modelling population + an extended population that also contains
#                          snapshot classes that still exist at HEAD but never changed (change count 0)
#   4. CORRECTED TESTS   : Nadeau-Bengio corrected resampled t-test + Holm correction per hypothesis family
# =====================================================================================================
RUN_VIEW_COMBINATIONS = True
RUN_GROUP_SPLIT_ANALYSIS = True
RUN_POPULATION_ANALYSIS = True
GROUP_DEPTH = 2  # number of leading path segments (e.g. 'spring-core.src') that define a source module
SELECT_PRIMARY_BY_TRAIN_CV = True  # primary model = best mean TRAIN-ONLY CV balanced accuracy (never test results)

_FULL_NAME = 'Full (Static+Network+Propagation)'
VIEW_COMBINATIONS = {
    'S (Static only)': FEATURE_GROUPS['static'],
    'N (Network only)': FEATURE_GROUPS['network'],
    'P (Propagation only)': FEATURE_GROUPS['propagation'],
    'S+N (Static+Network)': FEATURE_GROUPS['static'] + FEATURE_GROUPS['network'],
    'S+P (Static+Propagation)': FEATURE_GROUPS['static'] + FEATURE_GROUPS['propagation'],
    'N+P (Network+Propagation)': FEATURE_GROUPS['network'] + FEATURE_GROUPS['propagation'],
    _FULL_NAME: FEATURE_GROUPS['static'] + FEATURE_GROUPS['network'] + FEATURE_GROUPS['propagation'],
}


def nadeau_bengio_test(diff, test_train_ratio=0.25):
    """Corrected resampled t-test (Nadeau & Bengio 2003) for k overlapping train/test repeats.

    diff: per-repeat paired differences (reference minus other). Variance is inflated by (1/k + n_test/n_train).
    Returns dict(mean, sd, se, k, t, p, ci_low, ci_high, cohens_d).
    """
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    k = diff.size
    if k < 2:
        return dict(mean=np.nan, sd=np.nan, se=np.nan, k=k, t=np.nan, p=np.nan, ci_low=np.nan, ci_high=np.nan, cohens_d=np.nan)
    mean, sd = diff.mean(), diff.std(ddof=1)
    se = np.sqrt((1.0 / k + test_train_ratio) * sd ** 2)
    if se == 0:
        return dict(mean=mean, sd=sd, se=0.0, k=k, t=np.nan, p=np.nan, ci_low=mean, ci_high=mean, cohens_d=np.nan)
    t = mean / se
    p = 2 * scipy_stats.t.sf(abs(t), k - 1)
    tcrit = scipy_stats.t.ppf(0.975, k - 1)
    return dict(mean=mean, sd=sd, se=se, k=k, t=t, p=p, ci_low=mean - tcrit * se, ci_high=mean + tcrit * se,
                cohens_d=mean / sd if sd > 0 else np.nan)


def run_feature_set_experiment(X, change_count, project_name, project_dir, train_fn, eval_fn, feature_sets, tag,
                               reference_name, model_name, fixed_params, n_repeats=N_ABLATION_REPEATS,
                               split='stratified', groups=None, title='', notes=(), subsample_n=None):
    """Generic repeated-split experiment with the SAME protocol as run_rq1_ablation (seeds 42+rep, train-only Q80
    label, train-only architecture_size, min-max scaling, fixed hyperparameters, no RFECV).

    split='stratified': stratified 80/20 on the change-count proxy (identical to the ablation).
    split='group'     : GroupShuffleSplit 80/20 by `groups` (Series indexed like X), so classes of one source module
                        never appear in both train and test.
    Writes <tag>_<project>_<model>_per_repeat.csv / _summary_ci.csv / _summary.txt in project_dir.
    """
    from sklearn.model_selection import GroupShuffleSplit
    if fixed_params is None:
        fixed_params = RF_FIXED_PARAMS if model_name == 'RandomForest' else {}
    param_grid = MODEL_PARAM_GRIDS.get(model_name, {})
    rows, ratios = [], []
    strat_proxy = make_stratification_proxy(change_count) if split == 'stratified' else None
    start_time = time.time()
    skipped_repeats = 0
    X_full_, cc_full_ = X, change_count
    for rep in range(n_repeats):
        seed = 42 + rep
        X, change_count = X_full_, cc_full_
        if subsample_n is not None and len(X) > subsample_n:
            # size-matched analysis: a fresh random subset of subsample_n classes in every repeat
            sel = np.random.RandomState(seed).choice(len(X), size=subsample_n, replace=False)
            X, change_count = X_full_.iloc[np.sort(sel)], cc_full_.iloc[np.sort(sel)]
            strat_proxy = make_stratification_proxy(change_count) if split == 'stratified' else None
        if split == 'group':
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            tr_idx, te_idx = next(gss.split(X, groups=groups.reindex(X.index).values))
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        else:
            X_tr, X_te, _, _ = train_test_split(X, change_count, test_size=0.2, stratify=strat_proxy,
                                                random_state=seed)
        y_tr, thr = make_stability_labels(change_count, X_tr.index)
        y_tr = y_tr.loc[X_tr.index]
        y_te = (change_count.loc[X_te.index] >= thr).astype(int)
        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            skipped_repeats += 1
            logger.warning(f"[{project_name}] {tag}: repeat {rep} skipped (a split has a single class).")
            continue
        ratios.append(len(X_te) / max(1, len(X_tr)))
        X_tr, X_te = add_train_test_architecture_size(X_tr, X_te)
        for name, cols in feature_sets.items():
            cols_present = [c for c in cols if c in X_tr.columns]
            if not cols_present:
                continue
            scaler = MinMaxScaler()
            Xs_tr = pd.DataFrame(scaler.fit_transform(X_tr[cols_present]), index=X_tr.index, columns=cols_present)
            Xs_te = pd.DataFrame(scaler.transform(X_te[cols_present]), index=X_te.index, columns=cols_present)
            model, _, _ = train_fn(Xs_tr, y_tr, model_name, param_grid, skip_grid_search=True,
                                   fixed_params=fixed_params)
            if model is None:
                continue
            m = eval_fn(model, Xs_te, y_te, model_name=f'{model_name}_{tag}_{name}_rep{rep}')
            rows.append({'Project': project_name, 'FeatureSet': name, 'Repeat': rep, 'N_Features': len(cols_present),
                         'Threshold': thr, 'N_Train': len(X_tr), 'N_Test': len(X_te),
                         'Test_Unstable_Share': float(y_te.mean()),
                         'Balanced_Accuracy': m.get('Balanced Accuracy') if m else np.nan,
                         'F1_Macro': m.get('F1 Macro') if m else np.nan,
                         'PR_AUC': m.get('PR AUC') if m else np.nan,
                         'AUC_ROC': m.get('AUC-ROC') if m else np.nan})
        log_progress(f"[{project_name}] {tag} repeats", rep + 1, n_repeats, start_time)
    per = pd.DataFrame(rows)
    if per.empty:
        logger.error(f"[{project_name}] {tag}: no results produced.")
        return None
    base = f"{project_dir}/{tag}_{project_name}_{model_name}"
    per.to_csv(f"{base}_per_repeat.csv", index=False)
    ratio = float(np.mean(ratios)) if ratios else 0.25

    ref = per[per.FeatureSet == reference_name].sort_values('Repeat')
    summary = []
    for name in feature_sets:
        sub = per[per.FeatureSet == name].sort_values('Repeat')
        if sub.empty:
            continue
        arr = sub.Balanced_Accuracy.values.astype(float)
        sem = arr.std(ddof=1) / np.sqrt(arr.size) if arr.size > 1 else 0.0
        e = {'FeatureSet': name, 'N_Repeats': arr.size, 'N_Features': int(sub.N_Features.iloc[0]),
             'Mean_Balanced_Accuracy': arr.mean(), 'CI95': 1.96 * sem,
             'Mean_PR_AUC': sub.PR_AUC.mean(), 'Mean_AUC_ROC': sub.AUC_ROC.mean(),
             'Mean_Threshold': sub.Threshold.mean(), 'Mean_Test_Unstable_Share': sub.Test_Unstable_Share.mean(),
             'Ref_minus_this': (ref.Balanced_Accuracy.mean() - arr.mean()) if len(ref) else np.nan}
        if name != reference_name and len(ref) == len(sub) and len(sub) > 1:
            d = ref.Balanced_Accuracy.values.astype(float) - arr
            e['p_paired_unadjusted'] = scipy_stats.ttest_rel(ref.Balanced_Accuracy.values, arr).pvalue
            nb = nadeau_bengio_test(d, ratio)
            e['p_NadeauBengio'] = nb['p']
            e['diff_CI95_low'] = nb['ci_low']
            e['diff_CI95_high'] = nb['ci_high']
            e['Cohens_d_ref_minus_this'] = nb['cohens_d']
        summary.append(e)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(f"{base}_summary_ci.csv", index=False)

    lines = [f"{title or tag} — Project: {project_name}  ({n_repeats} repeated {split} 80/20 splits"
             f"{', ' + str(skipped_repeats) + ' skipped' if skipped_repeats else ''})",
             f"Model (held constant): {model_name}  |  fixed_params={fixed_params}"] + list(notes) + \
            ["=" * 150, "",
             f"{'Feature set':<38}{'N':>3}{'F':>3}  {'MeanBalAcc':>11}  {'+/-95%CI':>9}  {'PR-AUC':>7}  {'ROC-AUC':>8}"
             f"  {'Ref-this':>9}  {'p (paired)':>10}  {'p (N-B)':>8}  {'95% CI of diff (N-B)':>24}  {'d':>7}", "-" * 150]
    for _, r in sdf.iterrows():
        pu, pn = r.get('p_paired_unadjusted', np.nan), r.get('p_NadeauBengio', np.nan)
        lo, hi, d = r.get('diff_CI95_low', np.nan), r.get('diff_CI95_high', np.nan), r.get('Cohens_d_ref_minus_this', np.nan)
        fmt = lambda v, f: (format(v, f) if not pd.isna(v) else 'n/a')
        lines.append(f"{r['FeatureSet']:<38}{int(r['N_Repeats']):>3}{int(r['N_Features']):>3}  "
                     f"{r['Mean_Balanced_Accuracy']:>11.4f}  +/-{r['CI95']:>6.4f}  {r['Mean_PR_AUC']:>7.3f}  "
                     f"{r['Mean_AUC_ROC']:>8.3f}  {r['Ref_minus_this']:>+9.4f}  {fmt(pu, '.4f'):>10}  {fmt(pn, '.4f'):>8}  "
                     f"{('[' + fmt(lo, '+.4f') + ', ' + fmt(hi, '+.4f') + ']'):>24}  {fmt(d, '+.3f'):>7}")
    lines += ["", f"Reference row = '{reference_name}'. Ref-this > 0 means the reference is better.",
              "p (paired) = ordinary paired t-test (ignores the overlap of the repeats; anti-conservative).",
              f"p (N-B)    = Nadeau-Bengio corrected resampled t-test (variance x (1/k + n_test/n_train), "
              f"n_test/n_train = {ratio:.3f}); CI = corrected 95% interval of the BA difference; d = Cohen's d of the paired differences.",
              "No multiplicity correction is applied here; see stat_tests_corrected_<model>.csv for Holm-adjusted values."]
    with open(f"{base}_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n--- {title or tag} ({project_name}) ---")
    print("\n".join(lines))
    return sdf


def run_view_combinations(X, change_count, project_name, project_dir, train_fn, eval_fn, model_name, fixed_params):
    """All seven view subsets with identical splits/settings; the Full row must equal the ablation Full row."""
    sdf = run_feature_set_experiment(
        X, change_count, project_name, project_dir, train_fn, eval_fn, VIEW_COMBINATIONS, 'viewcombos', _FULL_NAME,
        model_name, fixed_params, title='View combinations (S, N, P, pairs, Full)')
    # Consistency with the ablation Full row
    try:
        abl = pd.read_csv(f"{project_dir}/rq1_ablation_{project_name}_{model_name}_per_repeat.csv")
        vc = pd.read_csv(f"{project_dir}/viewcombos_{project_name}_{model_name}_per_repeat.csv")
        a = abl[abl.View == _FULL_NAME].sort_values('Repeat')['Balanced_Accuracy'].values
        b = vc[vc.FeatureSet == _FULL_NAME].sort_values('Repeat')['Balanced_Accuracy'].values
        if a.size == b.size and a.size:
            logger.info(f"[{project_name}] view-combination consistency: Full row vs ablation Full row max |diff| = "
                        f"{np.max(np.abs(a - b)):.2e}")
    except Exception as e:
        logger.warning(f"[{project_name}] view-combination consistency check failed: {e}")
    return sdf


def run_size_matched_analysis(X, change_count, project_name, project_dir, train_fn, eval_fn, model_name, fixed_params):
    """Full vs LOC-only (and Static-only) when every project is subsampled to SIZE_MATCH_N classes in each repeat.
    Separates a sample-size effect (larger projects give more statistical power) from a project effect."""
    if len(X) < SIZE_MATCH_N:
        logger.info(f"[{project_name}] size-matched analysis: population {len(X)} < {SIZE_MATCH_N}; skipped.")
        return None
    return run_feature_set_experiment(
        X, change_count, project_name, project_dir, train_fn, eval_fn,
        {'LOC-only': ['loc'], 'S (Static only)': VIEW_COMBINATIONS['S (Static only)'],
         _FULL_NAME: VIEW_COMBINATIONS[_FULL_NAME]}, 'sizematched', _FULL_NAME, model_name, fixed_params,
        subsample_n=SIZE_MATCH_N, title=f'Size-matched analysis ({SIZE_MATCH_N} random classes per repeat)',
        notes=[f"Each repeat draws {SIZE_MATCH_N} of {len(X)} classes at random (seed = 42 + repeat) before the split."])


def run_group_split_analysis(X, change_count, project_name, project_dir, train_fn, eval_fn, model_name, fixed_params):
    """Full vs LOC-only when whole source modules are held out (classes of a module never straddle train/test)."""
    groups = pd.Series(['.'.join(str(c).split('.')[:GROUP_DEPTH]) for c in X.index], index=X.index)
    n_groups = groups.nunique()
    logger.info(f"[{project_name}] group split: {n_groups} source modules (first {GROUP_DEPTH} path segments)")
    return run_feature_set_experiment(
        X, change_count, project_name, project_dir, train_fn, eval_fn,
        {'LOC-only': ['loc'], _FULL_NAME: VIEW_COMBINATIONS[_FULL_NAME]}, 'groupsplit', _FULL_NAME,
        model_name, fixed_params, split='group', groups=groups,
        title='Module-grouped split (Full vs LOC-only)',
        notes=[f"Groups = first {GROUP_DEPTH} path segments of the class name ({n_groups} modules)."])


def run_population_analysis(project_name, project_dir, features_graph, train_fn, eval_fn, model_name, fixed_params):
    """Coverage of the modelling population + extended-population sensitivity analysis.

    Always writes population_coverage_<project>.json (stage counts from the pickles). If script 4 wrote
    snapshot_population_<project>.json (git ls-tree at HEAD), snapshot classes that still exist at HEAD but
    never changed are added with change count 0 (classes deleted/renamed after the anchor are NOT added, because
    their 0 would be a tracking artefact). Isolated classes keep zero network features / zero propagation.
    NOTE: the MAIN population is not restricted to classes present at HEAD (changed-then-deleted classes are in it);
    the 'population_headonly' analysis below removes them as a sensitivity analysis.
    """
    def _load(name):
        p = os.path.join(project_dir, name)
        if not os.path.exists(p):
            return None
        with open(p, 'rb') as f:
            return pickle.load(f)
    code_metrics = _load("code_metrics.pkl")
    propagation = _load("propagation_frequencies.pkl")
    history = _load("class_change_frequency_after.pkl")
    if code_metrics is None or propagation is None or history is None or features_graph is None:
        logger.warning(f"[{project_name}] population analysis skipped: missing pickles.")
        return None
    m_nodes, g_nodes, p_nodes, h_nodes = set(code_metrics), set(features_graph), set(propagation), set(dict(history))
    final = g_nodes & m_nodes & p_nodes & h_nodes
    cov = {'classes_with_static_metrics': len(m_nodes), 'classes_in_dependency_graph(non_isolated)': len(g_nodes),
           'classes_with_propagation_frequency': len(p_nodes), 'classes_changed_after_anchor(history)': len(h_nodes),
           'metrics_and_graph_and_propagation(snapshot_feature_complete)': len(m_nodes & g_nodes & p_nodes),
           'modelling_population(all_sources_and_changed)': len(final),
           'snapshot_classes_without_graph_node(isolated_or_unresolved)': len(m_nodes - g_nodes),
           'changed_classes_not_in_snapshot(new_or_renamed_after_anchor)': len(h_nodes - m_nodes),
           'changed_snapshot_classes_dropped_for_missing_feature_source': len((h_nodes & m_nodes) - final)}
    pop = None
    pop_path = os.path.join(project_dir, f"snapshot_population_{project_name}.json")
    if os.path.exists(pop_path):
        with open(pop_path, "r", encoding="utf-8") as f:
            pop = json.load(f)
        cov['snapshot_population_from_git'] = {k: v for k, v in pop.items() if not isinstance(v, list)}
    with open(os.path.join(project_dir, f"population_coverage_{project_name}.json"), "w", encoding="utf-8") as f:
        json.dump(cov, f, indent=2, default=str)
    logger.info(f"[{project_name}] population coverage: {json.dumps(cov, default=str)}")

    if pop is None or 'unchanged_present_at_head' not in pop:
        logger.warning(f"[{project_name}] snapshot_population_{project_name}.json not found - run script 4 again to "
                       f"enable the extended-population analysis.")
        return None
    extra = set(pop['unchanged_present_at_head']) & m_nodes
    X2, cc2, _, _, _ = prepare_data_combined(features_graph, code_metrics, propagation, project_dir,
                                             extra_zero_classes=extra, relax_graph=True)
    if X2 is None or len(X2) == 0:
        return None
    logger.info(f"[{project_name}] extended population: {len(X2)} classes "
                f"({int((cc2 == 0).sum())} with change count 0) vs main population {len(final)}")
    # Composition of the relaxed population: it re-admits changed classes that lack a graph node / propagation value
    # and zero-fills their network and propagation features; the unchanged classes may be graph-isolated as well.
    iso_unchanged = len(extra - g_nodes)
    iso_changed = len(((h_nodes & m_nodes) - final) - g_nodes)
    logger.info(f"[{project_name}] extended population composition: main={len(final)}, "
                f"changed re-admitted without complete features={len((h_nodes & m_nodes) - final)} "
                f"(without graph node: {iso_changed}), unchanged added={len(extra)} "
                f"(without graph node: {iso_unchanged}, share {iso_unchanged / max(1, len(extra)):.1%}); "
                f"graph-isolated share in the main population = 0 by construction.")
    res_relaxed = run_feature_set_experiment(
        X2, cc2, project_name, project_dir, train_fn, eval_fn,
        {'LOC-only': ['loc'], _FULL_NAME: VIEW_COMBINATIONS[_FULL_NAME]}, 'population_extended', _FULL_NAME,
        model_name, fixed_params, title='Extended population (relaxed: unchanged classes added with count 0; '
                                        'classes without graph node kept with zero-filled network/propagation features)',
        notes=[f"Population = {len(X2)} classes; {int((cc2 == 0).sum())} unchanged (count 0); "
               f"main population = {len(final)}; {len((h_nodes & m_nodes) - final)} changed classes without complete "
               f"features re-admitted. Deleted/renamed classes excluded."])
    # Graph-only variant: add only unchanged classes that have complete features (no zero-filled features, no
    # re-admitted changed classes), so that graph isolation cannot act as a proxy for 'unchanged'.
    extra_g = extra & g_nodes & p_nodes
    X3, cc3, _, _, _ = prepare_data_combined(features_graph, code_metrics, propagation, project_dir,
                                             extra_zero_classes=extra_g, relax_graph=False)
    if X3 is not None and len(X3) > 0:
        logger.info(f"[{project_name}] graph-only extended population: {len(X3)} classes "
                    f"({int((cc3 == 0).sum())} with change count 0) vs main population {len(final)}")
        run_feature_set_experiment(
            X3, cc3, project_name, project_dir, train_fn, eval_fn,
            {'LOC-only': ['loc'], _FULL_NAME: VIEW_COMBINATIONS[_FULL_NAME]}, 'population_extended_graphonly',
            _FULL_NAME, model_name, fixed_params,
            title='Extended population, graph-only (unchanged classes with complete features added, count 0)',
            notes=[f"Population = {len(X3)} classes; {int((cc3 == 0).sum())} unchanged (count 0); "
                   f"main population = {len(final)}. No zero-filled features; no re-admitted changed classes."])
    # Head-only variant of the MAIN population: the main population is not filtered by 'present at HEAD', so classes
    # that were changed and later deleted/moved (their last change is the deleting commit) are part of it. Their labels
    # reflect short lifetime, not change intensity; this analysis drops them and repeats Full vs LOC-only.
    try:
        head_set = set(pop.get('changed_present_at_head', []))
        X_main, cc_main, _, _, _ = prepare_data_combined(features_graph, code_metrics, propagation, project_dir)
        if X_main is not None and head_set:
            keep = [c for c in X_main.index if c in head_set]
            n_drop = len(X_main) - len(keep)
            logger.info(f"[{project_name}] main population classes absent at HEAD (deleted/moved after the anchor): "
                        f"{n_drop} of {len(X_main)} ({n_drop / max(1, len(X_main)):.1%}); head-only population = {len(keep)}")
            if len(keep) > 100:
                run_feature_set_experiment(
                    X_main.loc[keep], cc_main.loc[keep], project_name, project_dir, train_fn, eval_fn,
                    {'LOC-only': ['loc'], _FULL_NAME: VIEW_COMBINATIONS[_FULL_NAME]}, 'population_headonly',
                    _FULL_NAME, model_name, fixed_params,
                    title='Main population restricted to classes still present at HEAD (deleted/moved classes dropped)',
                    notes=[f"Population = {len(keep)} classes; {n_drop} classes absent at HEAD removed from the main population "
                           f"of {len(X_main)}."])
    except Exception as e:
        logger.error(f"[{project_name}] head-only population analysis failed: {e}")
    # Mass-commit sensitivity: same main population, but the change counts (and therefore the labels) ignore commits that
    # touch more than MAX_FILES_PER_COMMIT Java files, as the co-change extraction does (script 4 writes the file).
    try:
        nomass_path = os.path.join(project_dir, "class_change_frequency_after_nomass.pkl")
        if os.path.exists(nomass_path):
            with open(nomass_path, "rb") as f:
                nomass = dict(pickle.load(f))
            X_m, cc_m, _, _, _ = prepare_data_combined(features_graph, code_metrics, propagation, project_dir)
            if X_m is not None:
                cc_nm = pd.Series({c: nomass.get(c, 0) for c in X_m.index}, dtype=float)
                logger.info(f"[{project_name}] mass-commit sensitivity: {int((cc_nm == 0).sum())} of {len(cc_nm)} population "
                            f"classes have count 0 once mass commits are ignored; Spearman rho of counts = "
                            f"{cc_m.astype(float).corr(cc_nm, method='spearman'):.3f}")
                run_feature_set_experiment(
                    X_m, cc_nm, project_name, project_dir, train_fn, eval_fn,
                    {'LOC-only': ['loc'], _FULL_NAME: VIEW_COMBINATIONS[_FULL_NAME]}, 'population_nomass',
                    _FULL_NAME, model_name, fixed_params,
                    title='Labels from change counts without mass commits (> 30 Java files per commit)',
                    notes=["Same classes as the main population; change counts exclude commits touching more than "
                           "30 Java files."])
        else:
            logger.warning(f"[{project_name}] class_change_frequency_after_nomass.pkl missing: rerun script 4 for the "
                           f"mass-commit sensitivity analysis.")
    except Exception as e:
        logger.error(f"[{project_name}] mass-commit sensitivity analysis failed: {e}")
    return res_relaxed


_PROJ_COLORS = {'spring-framework': '#0072B2', 'elasticsearch': '#D55E00', 'hadoop': '#009E73'}
_PROJ_LABELS = {'spring-framework': 'Spring Framework', 'elasticsearch': 'Elasticsearch', 'hadoop': 'Hadoop'}


def _radar_axes(fig, n_panels):
    return [fig.add_subplot(1, n_panels, i + 1, projection='polar') for i in range(n_panels)]


def _draw_radar(ax, labels, series, title, colors=None):
    """series: {name: [values per axis]}; radial range is fitted to the data of the panel."""
    n = len(labels)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    allv = np.array([v for vals in series.values() for v in vals], dtype=float)
    lo, hi = np.nanmin(allv), np.nanmax(allv)
    pad = max(0.02, (hi - lo) * 0.15)
    lo, hi = max(0.0, lo - pad), min(1.0, hi + pad)
    cmap = plt.get_cmap('tab10')
    for i, (name, vals) in enumerate(series.items()):
        col = (colors or {}).get(name, cmap(i % 10))
        v = list(vals) + [vals[0]]
        ax.plot(ang + [ang[0]], v, color=col, lw=1.6, label=name)
        ax.fill(ang + [ang[0]], v, color=col, alpha=0.08)
    ax.set_xticks(ang); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(lo, hi); ax.set_yticks(np.linspace(lo, hi, 4)); ax.set_yticklabels([f"{t:.2f}" for t in np.linspace(lo, hi, 4)], fontsize=6)
    ax.set_title(title, fontsize=10, pad=14)


def generate_paper_figures(projects, export_dir, model_name):
    """Publication figures built ONLY from the pipeline outputs (no hard-coded numbers); written to
    <export_dir>/paper_figures/ as 300-dpi PNG:
      fig_view_forest      Full minus each view subset, corrected 95% intervals (stat_tests_corrected_<model>.csv)
      fig_baselines        BA of size-only / past-change / Full feature sets (baselines_*_per_repeat.csv)
      fig_view_importance  permutation importance summed by view (feature_importance_grouped_by_view.csv)
      fig_radar_models     radar of the five classifiers on BA / PR-AUC / Macro-F1 / ROC-AUC (comparison_table.csv)
      fig_radar_ablation   radar of Full and the leave-one-view-out models (rq1_ablation_*_per_repeat.csv)
    A figure is skipped (with a warning) when its input files are missing."""
    out_dir = os.path.join(export_dir, 'paper_figures')
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    present = [p for p in projects if os.path.isdir(os.path.join(export_dir, p))]

    def save(fig, name):
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, name + '.png'), dpi=300); plt.close(fig)
        logger.info(f"Figure saved: {os.path.join(out_dir, name + '.png')}")

    # 1. Forest plot of Full minus each view subset (exploratory singles and pairs)
    try:
        st = pd.read_csv(os.path.join(export_dir, f"stat_tests_corrected_{model_name}.csv"))
        vc = st[st.Family == 'viewcombos'].copy()
        vc['Subset'] = vc['Comparison'].str.replace('Full vs ', '', regex=False).str.split(' ').str[0]
        order = ['S', 'N', 'P', 'S+N', 'S+P', 'N+P']
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        projs = [p for p in present if p in set(vc.Project)]
        for j, p in enumerate(projs):
            for i, s_ in enumerate(order):
                r = vc[(vc.Project == p) & (vc.Subset == s_)]
                if r.empty:
                    continue
                r = r.iloc[0]; y = len(order) - i + (1 - j) * 0.24
                ax.plot([r.CI95_low, r.CI95_high], [y, y], color=_PROJ_COLORS.get(p, 'k'), lw=1.8)
                ax.plot(r.Mean_diff_Full_minus_other, y, 'o', color=_PROJ_COLORS.get(p, 'k'), ms=4.5,
                        label=_PROJ_LABELS.get(p, p) if i == 0 else None)
        ax.axvspan(-EQUIV_MARGIN, EQUIV_MARGIN, color='#999999', alpha=0.15, lw=0); ax.axvline(0, color='k', lw=0.8)
        ax.set_yticks(range(len(order), 0, -1)); ax.set_yticklabels(order)
        ax.set_xlabel(f'Balanced Accuracy: Full - subset (corrected 95% CI)\npositive = Full better; grey band = equivalence margin +/-{EQUIV_MARGIN}')
        ax.legend(frameon=False, loc='upper right'); save(fig, 'fig_view_forest')
    except Exception as e:
        logger.warning(f"fig_view_forest skipped: {e}")

    # 2. Baselines
    try:
        order = ['LOC-only', 'PriorChanges-only', 'LOC+PriorChanges', _FULL_NAME, 'Full+PriorChanges']
        fig, ax = plt.subplots(figsize=(6.6, 3.8)); w = 0.26; x = np.arange(len(order)); drew = False
        for j, p in enumerate(present):
            f = os.path.join(export_dir, p, f"baselines_{p}_{model_name}_per_repeat.csv")
            if not os.path.exists(f):
                continue
            df = pd.read_csv(f); means, errs, labs = [], [], []
            names = list(df.FeatureSet.unique())
            for nm in names:
                a = df[df.FeatureSet == nm].Balanced_Accuracy.dropna().values
                means.append(a.mean()); errs.append(1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0)
            xs = np.arange(len(names))
            ax.bar(xs + (j - 1) * w, means, w, yerr=errs, color=_PROJ_COLORS.get(p, 'gray'), capsize=2,
                   label=_PROJ_LABELS.get(p, p), error_kw={'lw': 0.9}); drew = True
            ax.set_xticks(xs); ax.set_xticklabels([n.replace(' (Static+Network+Propagation)', '') for n in names], rotation=15, ha='right')
        if drew:
            lo = ax.get_ylim()[0]; ax.set_ylim(max(0.5, lo - 0.0), None)
            ax.set_ylabel('Balanced Accuracy (mean +/- 95% half-width)'); ax.legend(frameon=False, ncol=3, loc='upper left')
            save(fig, 'fig_baselines')
        else:
            plt.close(fig)
    except Exception as e:
        logger.warning(f"fig_baselines skipped: {e}")

    # 3. Importance by view
    try:
        fig, ax = plt.subplots(figsize=(6.0, 3.4)); w = 0.26; views = ['static', 'network', 'propagation']; drew = False
        for j, p in enumerate(present):
            f = os.path.join(export_dir, p, 'feature_importance_grouped_by_view.csv')
            if not os.path.exists(f):
                continue
            g = pd.read_csv(f, index_col=0)
            vals = [float(g.loc[v, 'Sum_Importance']) if v in g.index else 0.0 for v in views]
            ax.bar(np.arange(3) + (j - 1) * w, vals, w, color=_PROJ_COLORS.get(p, 'gray'), label=_PROJ_LABELS.get(p, p)); drew = True
        if drew:
            ax.axhline(0, color='k', lw=0.8); ax.set_xticks(range(3)); ax.set_xticklabels(['Static', 'Network', 'Propagation'])
            ax.set_ylabel('Summed permutation importance\n(decrease in test Balanced Accuracy)'); ax.legend(frameon=False)
            save(fig, 'fig_view_importance')
        else:
            plt.close(fig)
    except Exception as e:
        logger.warning(f"fig_view_importance skipped: {e}")

    # 4. Radar of the classifiers (main split, per project)
    try:
        metrics = [('Balanced Accuracy', 'Balanced Acc.'), ('PR AUC', 'PR-AUC'), ('F1 Macro', 'Macro-F1'), ('AUC-ROC', 'ROC-AUC')]
        tabs = {p: pd.read_csv(os.path.join(export_dir, p, 'comparison_table.csv')) for p in present
                if os.path.exists(os.path.join(export_dir, p, 'comparison_table.csv'))}
        if tabs:
            fig = plt.figure(figsize=(4.0 * len(tabs), 4.4)); axes = _radar_axes(fig, len(tabs))
            for ax, (p, t) in zip(axes, tabs.items()):
                t = t[~t['Model'].str.contains('Dummy', case=False)]
                series = {r['Model']: [float(r[m]) for m, _ in metrics] for _, r in t.iterrows()}
                _draw_radar(ax, [l for _, l in metrics], series, _PROJ_LABELS.get(p, p))
            axes[-1].legend(loc='upper left', bbox_to_anchor=(1.05, 1.05), frameon=False, fontsize=8)
            save(fig, 'fig_radar_models')
    except Exception as e:
        logger.warning(f"fig_radar_models skipped: {e}")

    # 5. Radar of Full vs leave-one-view-out (repeated splits)
    try:
        cols = [('Balanced_Accuracy', 'Balanced Acc.'), ('PR_AUC', 'PR-AUC'), ('F1_Macro', 'Macro-F1'), ('AUC_ROC', 'ROC-AUC')]
        abl = {p: pd.read_csv(os.path.join(export_dir, p, f"rq1_ablation_{p}_{model_name}_per_repeat.csv")) for p in present
               if os.path.exists(os.path.join(export_dir, p, f"rq1_ablation_{p}_{model_name}_per_repeat.csv"))}
        if abl:
            fig = plt.figure(figsize=(4.0 * len(abl), 4.4)); axes = _radar_axes(fig, len(abl))
            for ax, (p, t) in zip(axes, abl.items()):
                series = {v: [float(g[c].mean()) for c, _ in cols] for v, g in t.groupby('View', sort=False)}
                _draw_radar(ax, [l for _, l in cols], series, _PROJ_LABELS.get(p, p))
            axes[-1].legend(loc='upper left', bbox_to_anchor=(1.05, 1.05), frameon=False, fontsize=8)
            save(fig, 'fig_radar_ablation')
    except Exception as e:
        logger.warning(f"fig_radar_ablation skipped: {e}")


def build_corrected_stat_tests(projects, export_dir, model_name):
    """Nadeau-Bengio corrected tests + Holm correction, per hypothesis family, from the per-repeat CSVs.

    Families (Holm applied within each family, across all projects): ablation (3 reduced views x projects),
    viewcombos (6 subsets x projects), baselines (4 baseline sets x projects), groupsplit (Full vs LOC-only under
    module-grouped splits x projects), population_extended (Full vs LOC-only with unchanged classes added x projects)
    and viewsingles (exploratory: the three single-view models x projects). The Nadeau-Bengio test/train ratio is
    taken from the recorded split sizes (N_Test / N_Train) when available, otherwise 0.25.
    """
    families = {
        'ablation': ('rq1_ablation_{p}_{m}_per_repeat.csv', 'View', _FULL_NAME),
        'viewcombos': ('viewcombos_{p}_{m}_per_repeat.csv', 'FeatureSet', _FULL_NAME),
        'baselines': ('baselines_{p}_{m}_per_repeat.csv', 'FeatureSet', _FULL_NAME),
        'groupsplit': ('groupsplit_{p}_{m}_per_repeat.csv', 'FeatureSet', _FULL_NAME),
        'population_extended': ('population_extended_{p}_{m}_per_repeat.csv', 'FeatureSet', _FULL_NAME),
        'sizematched': ('sizematched_{p}_{m}_per_repeat.csv', 'FeatureSet', _FULL_NAME),
        'population_headonly': ('population_headonly_{p}_{m}_per_repeat.csv', 'FeatureSet', _FULL_NAME),
        'population_nomass': ('population_nomass_{p}_{m}_per_repeat.csv', 'FeatureSet', _FULL_NAME),
        'population_extended_graphonly': ('population_extended_graphonly_{p}_{m}_per_repeat.csv', 'FeatureSet', _FULL_NAME),
        # exploratory single-view contrasts (a separate family of nine; the pre-specified RQ1 family is 'ablation')
        'viewsingles': ('viewcombos_{p}_{m}_per_repeat.csv', 'FeatureSet', _FULL_NAME,
                        ('S (Static only)', 'N (Network only)', 'P (Propagation only)')),
    }
    rows = []
    for fam, spec in families.items():
        pattern, col, ref_name = spec[:3]
        only = spec[3] if len(spec) > 3 else None
        fam_rows = []
        for p in projects:
            path = os.path.join(export_dir, p, pattern.format(p=p, m=model_name))
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            ref = df[df[col] == ref_name].sort_values('Repeat')['Balanced_Accuracy'].values
            # test/train ratio of the actual splits (0.25 for stratified 80/20; 0.268/0.356/0.427 for the module-grouped splits)
            ratio = 0.25
            if {'N_Train', 'N_Test'}.issubset(df.columns):
                rr = (df['N_Test'] / df['N_Train'].clip(lower=1)).dropna()
                if len(rr):
                    ratio = float(rr.mean())
            for name in df[col].unique():
                if name == ref_name or (only is not None and name not in only):
                    continue
                oth = df[df[col] == name].sort_values('Repeat')['Balanced_Accuracy'].values
                if oth.size != ref.size or ref.size < 2:
                    continue
                nb = nadeau_bengio_test(ref - oth, ratio)
                fam_rows.append({'Family': fam, 'Project': p, 'Comparison': f'Full vs {name}',
                                 'Mean_diff_Full_minus_other': nb['mean'], 'CI95_low': nb['ci_low'],
                                 'CI95_high': nb['ci_high'], 'Cohens_d': nb['cohens_d'],
                                 'p_paired_unadjusted': scipy_stats.ttest_rel(ref, oth).pvalue,
                                 'p_NadeauBengio': nb['p'], 'SE_NadeauBengio': nb['se'], 'k': nb['k'],
                                 'p_TOST_equiv': tost_equivalence_p(nb['mean'], nb['se'], nb['k'], EQUIV_MARGIN),
                                 'Equiv_margin': EQUIV_MARGIN})
        if fam_rows:
            fdf = pd.DataFrame(fam_rows)
            order = fdf['p_NadeauBengio'].rank(method='first').astype(int).sort_values().index
            m = len(fdf)
            holm, prev = {}, 0.0
            for i, idx in enumerate(fdf.loc[order].index):
                prev = max(prev, min(1.0, (m - i) * fdf.loc[idx, 'p_NadeauBengio']))
                holm[idx] = prev
            fdf['p_Holm_within_family'] = pd.Series(holm)
            # Holm across the TOST p-values of the same family (equivalence claims need the larger of the two one-sided p)
            order_t = fdf['p_TOST_equiv'].rank(method='first').astype(int).sort_values().index
            holm_t, prev = {}, 0.0
            for i, idx in enumerate(fdf.loc[order_t].index):
                prev = max(prev, min(1.0, (m - i) * fdf.loc[idx, 'p_TOST_equiv']))
                holm_t[idx] = prev
            fdf['p_TOST_Holm'] = pd.Series(holm_t)
            fdf['Family_size'] = m
            rows.append(fdf)
    if not rows:
        logger.warning("build_corrected_stat_tests: no per-repeat files found.")
        return None
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(export_dir, f"stat_tests_corrected_{model_name}.csv"), index=False)
    logger.info(f"Corrected statistical tests saved ({len(out)} comparisons).")
    try:
        build_meta_analysis(out, export_dir, model_name)
    except Exception as e:
        logger.error(f"Meta-analysis failed: {e}")
    return out


def tost_equivalence_p(mean_diff, se, k, margin):
    """Two one-sided tests (TOST) for |true difference| < margin, using the Nadeau-Bengio standard error.
    Returns max(p_lower, p_upper); a small value supports equivalence within +/- margin."""
    if se is None or not np.isfinite(se) or se <= 0 or not np.isfinite(mean_diff):
        return np.nan
    df = max(int(k) - 1, 1)
    p_lower = scipy_stats.t.sf((mean_diff + margin) / se, df)   # H0: diff <= -margin
    p_upper = scipy_stats.t.cdf((mean_diff - margin) / se, df)  # H0: diff >= +margin
    return float(max(p_lower, p_upper))


def build_meta_analysis(stat_df, export_dir, model_name):
    """Random-effects (DerSimonian-Laird) pooling of the per-project BA differences (Full minus other) with their
    Nadeau-Bengio variances; reported per family and comparison across the projects. Only 3 projects: complementary
    to, not a replacement for, the per-project tests; the interval uses t(k-1) quantiles."""
    rows = []
    for (fam, comp), g in stat_df.groupby(['Family', 'Comparison']):
        g = g.dropna(subset=['Mean_diff_Full_minus_other', 'SE_NadeauBengio'])
        g = g[g['SE_NadeauBengio'] > 0]
        K = len(g)
        if K < 2:
            continue
        y = g['Mean_diff_Full_minus_other'].values.astype(float)
        v = g['SE_NadeauBengio'].values.astype(float) ** 2
        w = 1.0 / v
        fe = np.sum(w * y) / np.sum(w)
        Q = float(np.sum(w * (y - fe) ** 2))
        c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
        tau2 = max(0.0, (Q - (K - 1)) / c) if c > 0 else 0.0
        ws = 1.0 / (v + tau2)
        mu = float(np.sum(ws * y) / np.sum(ws))
        se = float(np.sqrt(1.0 / np.sum(ws)))
        tcrit = scipy_stats.t.ppf(0.975, K - 1)
        p = float(2 * scipy_stats.t.sf(abs(mu / se), K - 1))
        I2 = max(0.0, (Q - (K - 1)) / Q) * 100 if Q > 0 else 0.0
        rows.append({'Family': fam, 'Comparison': comp, 'K_projects': K, 'Pooled_diff': mu, 'SE': se,
                     'CI95_low': mu - tcrit * se, 'CI95_high': mu + tcrit * se, 'p_random_effects': p,
                     'tau2': tau2, 'I2_percent': I2, 'Cochran_Q': Q,
                     'p_TOST_equiv': tost_equivalence_p(mu, se, K, EQUIV_MARGIN), 'Equiv_margin': EQUIV_MARGIN})
    if not rows:
        return None
    mdf = pd.DataFrame(rows)
    mdf.to_csv(os.path.join(export_dir, f"meta_analysis_{model_name}.csv"), index=False)
    logger.info("=" * 100)
    logger.info(f"RANDOM-EFFECTS META-ANALYSIS of Full-minus-other BA differences across projects ({model_name})")
    logger.info("\n" + mdf.round(4).to_string(index=False))
    return mdf


def compute_graph_features(graph):
    logger.info("Starting computation of graph features...")
    start_time = time.time()
    try:
        degree_centrality = nx.degree_centrality(graph)
    except:
        degree_centrality = {n: 0 for n in graph.nodes()}
    try:
        betweenness_centrality = nx.betweenness_centrality(graph, normalized=True)
    except:
        betweenness_centrality = {n: 0 for n in graph.nodes()}
    try:
        pagerank = nx.pagerank(graph, max_iter=1000, tol=1e-08)
    except Exception as e:
        logger.warning(f"Error in PageRank calculation: {e}. Replacing with zeros."); pagerank = {n: 0 for n in
                                                                                                  graph.nodes()}
    try:
        G_undirected = graph.to_undirected() if graph.is_directed() else graph; clustering = nx.clustering(G_undirected)
    except:
        clustering = {n: 0 for n in graph.nodes()}
    try:
        eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000, tol=1e-08)
    except Exception as e:
        logger.warning(
            f"Error in eigenvector centrality calculation: {e}. Replacing with zeros."); eigenvector_centrality = {n: 0
                                                                                                                   for n
                                                                                                                   in
                                                                                                                   graph.nodes()}

    features = {node: {
        "degree_centrality": degree_centrality.get(node, 0),
        "betweenness_centrality": betweenness_centrality.get(node, 0),
        "pagerank": pagerank.get(node, 0),
        "clustering": clustering.get(node, 0),
        "eigenvector_centrality": eigenvector_centrality.get(node, 0)
    } for node in graph.nodes()}
    logger.info(f"Graph features computed for {len(features)} nodes in {time.time() - start_time:.2f} seconds.")
    return features


def prepare_data_combined(graph_features, code_metrics, propagation_frequencies, project_dir,
                          extra_zero_classes=None, relax_graph=False):
    """extra_zero_classes / relax_graph are used ONLY by the supplementary population analysis: relax_graph=True keeps
    classes without a graph node or propagation value (their network features / propagation are 0) and
    extra_zero_classes adds snapshot classes with no post-cutoff change (change count 0)."""
    logger.info("--- Preparing dataset (Using Git History as Ground Truth) ---")

    graph_nodes = set(graph_features.keys())
    metrics_nodes = set(code_metrics.keys())
    propagation_nodes = set(propagation_frequencies.keys())

    # 1. Load change history; the after-cutoff window is the prediction target
    historical_file_after = os.path.join(project_dir, "class_change_frequency_after.pkl")
    historical_file_whole = os.path.join(project_dir, "class_change_frequency.pkl")
    project_name_for_log = os.path.basename(os.path.normpath(project_dir))
    # Whole-history fallback only if allowed in temporal_config.py (leaky; never for the paper)
    if os.path.exists(historical_file_after):
        # Hard stop unless the upstream cutoff markers all exist and agree on cutoff and anchor commit
        cutoff_used = verify_cutoff_consistency(project_dir, project_name=project_name_for_log)
        historical_file = historical_file_after
        logger.info("Using temporally-separated 'after-cutoff' change counts as the target "
                    "(class_change_frequency_after.pkl). Cutoff markers of scripts 0-3 confirmed to match "
                    f"(cutoff={cutoff_used}, snapshot commit={verified_anchor_commit(project_dir)}).")
    elif ALLOW_WHOLE_HISTORY:
        historical_file = historical_file_whole
        logger.info("WARNING: LEAKY CONFIGURATION - whole-history target (class_change_frequency.pkl); "
                    "ALLOW_WHOLE_HISTORY=True. Do NOT report these results.")
    else:
        raise TemporalCutoffMismatch(
            f"[{project_name_for_log}] class_change_frequency_after.pkl not found. Set a cutoff in "
            f"temporal_config.py and re-run scripts 0, 1, 2 and 3 before running 4-MLPipeline.py.")
    try:
        with open(historical_file, "rb") as f:
            historical_changes_raw = pickle.load(f)

        # All inputs use fully-qualified, path-based class names; do not collapse them to simple names
        historical_changes = dict(historical_changes_raw)
        history_nodes = set(historical_changes.keys())
        logger.info(
            f"Loaded Git history: {len(history_nodes)} classes (fully-qualified identity, no collapsing).")

    except FileNotFoundError:
        logger.error(f"History file '{historical_file}' not found! Run script 3 first.")
        return pd.DataFrame(), pd.Series(), pd.Series(), None, None

    # 2. Find Common Classes
    # IMPORTANT: sorted() makes the row order of X independent of Python's per-process string-hash seed.
    # A plain set has an arbitrary iteration order, which silently changed the train/test splits between
    # runs even with identical random_state values.
    if relax_graph:
        common_classes = sorted(metrics_nodes & (history_nodes | set(extra_zero_classes or [])))
    else:
        # extra_zero_classes (unchanged classes with count 0) are admitted only when they have a graph node and
        # a propagation value, i.e. complete features (used by the 'graph-only' extended-population analysis)
        common_classes = sorted(graph_nodes & metrics_nodes & propagation_nodes &
                                (history_nodes | set(extra_zero_classes or [])))
    logger.info(f"Common classes with full data: {len(common_classes)}")

    if not common_classes:
        logger.error("No common classes found! (Check naming consistency between Graph and Git History)");
        return pd.DataFrame(), pd.Series(), pd.Series(), None, None

    # 3. Calculate Architecture Size (for normalization and cost proxy)
    size_factors = ['loc', 'num_methods', 'cbo_approx', 'lcom4_approx', 'total_method_cc', 'cc_class']
    data_dict = {}
    for cls in common_classes:
        data_dict[cls] = {factor: code_metrics.get(cls, {}).get(factor, 0) for factor in size_factors}

    temp_data = pd.DataFrame.from_dict(data_dict, orient='index')

    for factor in size_factors:
        min_val = temp_data[factor].min()
        max_val = temp_data[factor].max()
        # Prevent division by zero
        denom = (max_val - min_val) if (max_val - min_val) > 0 else 1.0
        temp_data[f'norm_{factor}'] = (temp_data[factor] - min_val) / denom

    # Architecture size is the mean of normalized size factors
    norm_cols = [c for c in temp_data.columns if c.startswith('norm_')]
    temp_data['architecture_size'] = temp_data[norm_cols].mean(axis=1)

    # 4. Combine all features
    combined_features_list = []
    for cls in common_classes:
        features = graph_features.get(cls, {}).copy()
        features.update(code_metrics.get(cls, {}))
        features['class_name'] = cls
        features['propagation_frequency'] = propagation_frequencies.get(cls, 0)
        features['architecture_size'] = temp_data.loc[cls, 'architecture_size'] if cls in temp_data.index else 0
        features['real_change_count'] = historical_changes.get(cls, 0)
        combined_features_list.append(features)

    df = pd.DataFrame(combined_features_list).set_index('class_name')
    df = df.sort_index()  # belt-and-braces: deterministic row order
    df.fillna(0, inplace=True)

    # 5. Descriptive whole-dataset Q80 threshold (reporting only).
    # Model labels come from make_stability_labels() using train rows only.
    descriptive_threshold_quantile = df['real_change_count'].quantile(QUANTILE_THRESHOLD)
    if descriptive_threshold_quantile == 0:
        descriptive_threshold_quantile = 1
    logger.info(
        f"[descriptive only, NOT the model target] whole-dataset {int(QUANTILE_THRESHOLD*100)}th "
        f"percentile of changes: {descriptive_threshold_quantile:.0f}. Actual training/evaluation "
        f"labels are computed per-split from TRAIN rows only via make_stability_labels()."
    )
    # Descriptive only; never used as a train/test label
    df['stability_label'] = (df['real_change_count'] >= descriptive_threshold_quantile).astype(int)
    counts_preview = df['stability_label'].value_counts()
    logger.info(f"[descriptive only] Class distribution at whole-dataset threshold: "
                f"Stable(0): {counts_preview.get(0, 0)} | Unstable(1): {counts_preview.get(1, 0)}")

    # 6. Regression target: log1p(change count), a variance-stabilizing transform for right-skewed counts.
    # git_based_stability is kept only for descriptive tables.
    scaler = MinMaxScaler()
    changes_reshaped = df['real_change_count'].values.reshape(-1, 1)
    df['norm_change_count'] = scaler.fit_transform(changes_reshaped)
    df['git_based_stability'] = 1.0 - df['norm_change_count']  # descriptive/display only - NOT the model target
    df['log_future_change_count'] = np.log1p(df['real_change_count'])  # regression target

    # 7. Cost proxy (descriptive only)
    df['cost'] = df['architecture_size'] * (1 - df['git_based_stability'])

    # 8. Final X, y. 'architecture_size' is excluded as a model feature (whole-dataset min/max would leak);
    # it is rebuilt from train-only statistics after each split.
    drop_cols = ['stability_label', 'real_change_count', 'git_based_stability', 'norm_change_count',
                 'log_future_change_count', 'cost', 'architecture_size']
    # Keep numeric columns as features, remove target columns
    feature_cols = [col for col in df.columns if col not in drop_cols]

    # Remove possible non-numeric columns that may have been added
    X = df[feature_cols].select_dtypes(include=np.number)
    # Second value is the raw continuous change_count; callers derive a train-only label via make_stability_labels()
    change_count = df['real_change_count']
    # y_stability holds log1p(future change count), the regression target (name kept for existing call sites)
    y_stability = df['log_future_change_count']

    logger.info(f"Raw features before train-only architecture_size is added back ({len(X.columns)}): "
                f"{X.columns.tolist()[:5]}...")

    # For reporting functions like display_architecture_factors_quantile_table
    df_for_reporting = df.copy()

    return X, change_count, y_stability, descriptive_threshold_quantile, df_for_reporting

# Reporting functions

def select_features(X, y, n_features=15, min_features_to_select=5, cv_folds=5):
    # RFECV (5-fold stratified, balanced accuracy) picks the feature count; X, y must be the training split only.
    # n_features is only the fallback if RFECV fails.
    if X.empty or y.empty: logger.error("Cannot perform feature selection on empty data."); return pd.DataFrame(), []
    logger.info(f"Starting feature selection with RFECV (cross-validated RFE, {cv_folds}-fold CV, "
                f"min_features_to_select={min_features_to_select})...")
    start_time = time.time()
    estimator = RandomForestClassifier(random_state=42, n_estimators=50, class_weight='balanced')
    try:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        rfecv = RFECV(estimator=estimator, step=0.1, min_features_to_select=min_features_to_select,
                      cv=cv, scoring='balanced_accuracy', n_jobs=-1)
        rfecv.fit(X, y)
        selected_features = X.columns[rfecv.support_].tolist()
        X_selected = X[selected_features]
        logger.info(f"RFECV selected {len(selected_features)}/{X.shape[1]} features "
                    f"as optimal via cross-validation: {selected_features}")
    except Exception as e:
        logger.error(f"Error in RFECV: {e}. Falling back to fixed-count RFE (n_features={n_features}).")
        n_features_fixed = min(n_features, X.shape[1]) if n_features > 0 else X.shape[1]
        try:
            rfe = RFE(estimator=estimator, n_features_to_select=n_features_fixed, step=0.1)
            rfe.fit(X, y)
            selected_features = X.columns[rfe.support_].tolist()
            X_selected = X[selected_features]
        except Exception as e2:
            logger.error(f"Error in fallback RFE: {e2}. Returning all features.")
            selected_features = X.columns.tolist()
            X_selected = X
    logger.info(f"Feature selection completed in {time.time() - start_time:.2f} seconds.")
    return X_selected, selected_features


def train_classification_model(X_train, y_train, model_name, param_grid,
                                skip_grid_search=False, fixed_params=None):
    """Train a classifier. With skip_grid_search=True, fit once with fixed_params instead of GridSearchCV(cv=5)."""
    # Returns (best_model, best_cv_score, best_params). The train-only CV score is used for model selection,
    # never test performance. best_params is the grid-search result, or fixed_params if the search was skipped.
    # XGBoost gets scale_pos_weight from y_train, for parity with the class-weighted models.
    logger.info(f"Starting training of {model_name} (classification)...")
    start_time = time.time()
    model = {'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
             'GaussianNB': GaussianNB(), 'BernoulliNB': BernoulliNB(), 'MultinomialNB': MultinomialNB(),
             'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss',
                                          scale_pos_weight=compute_scale_pos_weight(y_train)),
             'SVM': SVC(random_state=42, probability=True, class_weight='balanced'),
             # Baselines: trivial and simple linear references
             'DummyMostFrequent': DummyClassifier(strategy='most_frequent', random_state=42),
             'DummyStratified': DummyClassifier(strategy='stratified', random_state=42),
             'LogisticRegression': LogisticRegression(random_state=42, class_weight='balanced',
                                                      max_iter=2000)}.get(model_name)
    if model is None: logger.error(f"Unknown model name: {model_name}"); return None, np.nan, {}

    if skip_grid_search:
        if fixed_params:
            try:
                model.set_params(**fixed_params)
            except Exception as e:
                logger.warning(f"Could not set fixed_params {fixed_params} on {model_name}: {e}")
        try:
            best_model = model.fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Error fitting {model_name} with fixed params: {e}")
            return None, np.nan, {}
        logger.info(f"Training of {model_name} completed in {time.time() - start_time:.2f}s "
                    f"(fixed params {fixed_params}, no grid search).")
        return best_model, np.nan, dict(fixed_params or {})

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1,
                               verbose=0)
    cv_score = np.nan
    best_params = {}
    try:
        grid_search.fit(X_train, y_train)
        if hasattr(grid_search, 'best_params_'):
            best_model = grid_search.best_estimator_
            cv_score = grid_search.best_score_
            best_params = dict(grid_search.best_params_)
        else:
            best_model = model.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"Error in GridSearchCV for {model_name}: {e}. Using default model.")
        try:
            best_model = model.fit(X_train, y_train)
        except Exception as fit_e:
            logger.error(f"Error fitting default model {model_name}: {fit_e}"); return None, np.nan, {}
    logger.info(f"Training of {model_name} completed in {time.time() - start_time:.2f} seconds.")
    if not np.isnan(cv_score):
        logger.info(f"[{model_name}] Mean 5-fold CV Balanced Accuracy (training data only): {cv_score:.4f}")
    return best_model, cv_score, best_params


def train_stability_model(X_train, y_train, y_stability_train, class_model, model_name_prefix):
    # Out-of-fold classifier probabilities feed the regressor (avoids in-sample leakage)
    logger.info(f"Starting training of stability model for {model_name_prefix} (RandomForestRegressor)...")
    start_time = time.time()

    X_train_values = X_train.values if isinstance(X_train, pd.DataFrame) else X_train

    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_probs = cross_val_predict(
            clone(class_model), X_train_values, y_train, cv=cv, method='predict_proba', n_jobs=-1
        )
        train_probs_positive = oof_probs[:, 1].reshape(-1, 1)
        logger.info(f"Using out-of-fold (5-fold CV) probabilities from {model_name_prefix} as the regressor's "
                    f"stacking feature (no in-sample leakage).")
    except Exception as e:
        logger.error(f"Could not compute OOF probabilities for {model_name_prefix}: {e}. "
                     f"Falling back to in-sample predict_proba on X_train (less rigorous).")
        try:
            train_probs_positive = class_model.predict_proba(X_train)[:, 1].reshape(-1, 1) if hasattr(class_model,
                                                                                                      "predict_proba") else class_model.predict(
                X_train).reshape(-1, 1)
        except Exception as e2:
            logger.error(f"Could not get probabilities from {model_name_prefix}: {e2}. Using class predictions.")
            train_probs_positive = class_model.predict(X_train).reshape(-1, 1)

    try:
        X_train_augmented = np.hstack((X_train_values, train_probs_positive))
    except ValueError as e:
        logger.error(f"Error in stacking features and probabilities for {model_name_prefix}: {e}"); return None

    stability_regressor = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10, n_jobs=-1)
    try:
        stability_regressor.fit(X_train_augmented, y_stability_train)
    except Exception as e:
        logger.error(f"Error training stability regressor for {model_name_prefix}: {e}"); return None
    logger.info(
        f"Training of stability model for {model_name_prefix} completed in {time.time() - start_time:.2f} seconds.")
    return stability_regressor


def evaluate_classification_model(model, X_test, y_test, model_name="Model"):
    logger.info(f"--- Evaluating {model_name} (classification) ---")
    results = {'Accuracy': np.nan, 'Balanced Accuracy': np.nan, 'F1 Macro': np.nan, 'AUC-ROC': np.nan, 'PR AUC': np.nan}
    try:
        y_pred = model.predict(X_test)
        # ROC/PR-AUC use a ranking score: for SVC the decision function (the internal Platt-scaled predict_proba is
        # fitted by an internal CV and need not rank classes like the decision function); Brier/ECE use probabilities.
        if isinstance(model, SVC) and hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(
                X_test) if hasattr(model, "decision_function") else None

        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        report = classification_report(y_test, y_pred, target_names=['Stable (0)', 'Unstable (1)'], zero_division=0)

        auc_roc = np.nan
        pr_auc = np.nan

        if y_prob is not None:
            if y_prob.ndim == 1 and y_prob.min() != y_prob.max():
                y_prob_norm = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
            else:
                y_prob_norm = y_prob

            try:
                auc_roc = roc_auc_score(y_test, y_prob_norm)
            except:
                pass

            try:
                precision, recall, _ = precision_recall_curve(y_test, y_prob_norm)
                pr_auc = calculate_auc(recall, precision)
                # Plot PR curve
                plt.plot(recall, precision, marker='.', label=f'{model_name} (PR AUC = {pr_auc:.2f})')
                plt.xlabel('Recall'); plt.ylabel('Precision')
                plt.title(f'Precision-Recall Curve ({model_name})')
                plt.legend(); plt.grid(True)
                plt.savefig(os.path.join(project_dir, f'precision_recall_{model_name}.png')); plt.close()
            except:
                pass

        logger.info(f"[{model_name}] Balanced Accuracy: {balanced_acc:.4f}")
        logger.info(f"[{model_name}] F1 Score (Macro): {f1_macro:.4f}")
        logger.info(
            f"[{model_name}] AUC-ROC: {auc_roc:.4f}" if not np.isnan(auc_roc) else f"[{model_name}] AUC-ROC: N/A")
        logger.info(f"[{model_name}] Precision-Recall AUC: {pr_auc:.4f}" if not np.isnan(
            pr_auc) else f"[{model_name}] Precision-Recall AUC: N/A")
        logger.info(f"[{model_name}] Classification Report:\n{report}")

        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stable (0)', 'Unstable (1)'])
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title(f'Confusion Matrix ({model_name})')
        plt.savefig(os.path.join(project_dir, f'confusion_matrix_{model_name}.png')); plt.close()

        results = {'Accuracy': accuracy, 'Balanced Accuracy': balanced_acc, 'F1 Macro': f1_macro, 'AUC-ROC': auc_roc,
                   'PR AUC': pr_auc}
    except Exception as e:
        logger.error(f"Error evaluating classification model {model_name}: {e}")
    return results


def evaluate_stability_model(stability_regressor, X_test, y_stability_test, class_model, model_name_prefix):
    """Evaluate the regressor on the log1p(change count) scale: MAE, RMSE, R^2, Spearman rho, Pearson r."""
    logger.info(f"--- Evaluating {model_name_prefix} (future change-count regression) ---")
    results = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'Spearman': np.nan, 'Pearson': np.nan,
               'Predicted_Mean_Log_Change_Count': np.nan}
    if stability_regressor is None: logger.error(
        f"Stability regressor for {model_name_prefix} is None. Cannot evaluate."); return results
    try:
        test_probs_positive = class_model.predict_proba(X_test)[:, 1].reshape(-1, 1) if hasattr(class_model,
                                                                                                "predict_proba") else class_model.predict(
            X_test).reshape(-1, 1)
        X_test_values = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        X_test_augmented = np.hstack((X_test_values, test_probs_positive))
        y_pred = stability_regressor.predict(X_test_augmented)

        mae = mean_absolute_error(y_stability_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_stability_test, y_pred)))
        r2 = r2_score(y_stability_test, y_pred)
        try:
            spearman_rho, _ = scipy_stats.spearmanr(y_stability_test, y_pred)
        except Exception:
            spearman_rho = np.nan
        try:
            pearson_r, _ = scipy_stats.pearsonr(y_stability_test, y_pred)
        except Exception:
            pearson_r = np.nan
        mean_pred = float(np.mean(y_pred))

        logger.info(f"[{model_name_prefix}] MAE (log1p change-count scale): {mae:.4f}")
        logger.info(f"[{model_name_prefix}] RMSE (log1p change-count scale): {rmse:.4f}")
        logger.info(f"[{model_name_prefix}] R^2: {r2:.4f}")
        logger.info(f"[{model_name_prefix}] Spearman rho (predicted vs actual ranking): {spearman_rho:.4f}")
        logger.info(f"[{model_name_prefix}] Pearson r: {pearson_r:.4f}")

        results = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Spearman': spearman_rho, 'Pearson': pearson_r,
                   'Predicted_Mean_Log_Change_Count': mean_pred}
    except Exception as e:
        logger.error(f"Error evaluating stability model for {model_name_prefix}: {e}")
    return results


def plot_stability_comparison(stability_results, filename='git_stability_comparison.png'):
    """Plot RMSE and mean predicted future-change activity per model."""
    logger.info("Generating stability comparison plot...")

    # Filter invalid results
    valid_results = {
        k: v for k, v in stability_results.items()
        if v is not None and not np.isnan(v.get('Predicted_Mean_Log_Change_Count', np.nan))
    }

    if not valid_results:
        logger.warning("No valid stability results available for plotting.")
        return

    model_names = list(valid_results.keys())
    predicted_activity = [valid_results[model].get('Predicted_Mean_Log_Change_Count', np.nan) for model in model_names]
    rmses = [valid_results[model].get('RMSE', np.nan) for model in model_names]


    try:
        fig, ax1 = plt.subplots(figsize=(12, 7))

        ax1.set_xlabel('Models')
        ax1.set_ylabel('Predicted Mean log1p(Future Change Count)', color='skyblue')

        ax1.bar(model_names, predicted_activity, color='skyblue', edgecolor='black',
                label='Predicted Mean log1p(Future Change Count)')
        ax1.tick_params(axis='y', labelcolor='skyblue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('RMSE (log1p change-count scale)', color='tab:red')

        ax2.plot(model_names, rmses, color='tab:red', linestyle=':', marker='o', label='RMSE')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        valid_rmses = [m for m in rmses if not np.isnan(m)]
        if valid_rmses:
            min_rmse = np.min(valid_rmses)
            max_rmse = np.max(valid_rmses)
            ax2.set_ylim(bottom=max(0, min_rmse * 0.9), top=max_rmse * 1.1 if max_rmse > 0 else 1.0)

        fig.suptitle('Comparison of Predicted Future Change Activity and RMSE Across Models', fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.savefig(os.path.join(project_dir, filename))
        plt.close()
        logger.info(f"Stability comparison plot saved at {os.path.join(project_dir, filename)}")

    except Exception as e:
        logger.error(f"Critical error during plotting stability comparison: {e}")

def display_comparison_table(class_results, stability_results, X_test, y_test, trained_class_models, selected_features):
    logger.info("\n--- Comparison Table for All Models (Sorted by Balanced Accuracy) ---")
    header = "| {:<15} | {:<19} | {:<15} | {:<15} | {:<15} | {:<23} | {:<20} | {:<25} | {:<25} |".format("Model",
                                                                                                         "Balanced Accuracy",
                                                                                                         "PR AUC",
                                                                                                         "F1 Macro",
                                                                                                         "AUC-ROC",
                                                                                                         "Change-Count RMSE",
                                                                                                         "Change-Count Spearman",
                                                                                                         "Mean Architecture Size (Unstable)",
                                                                                                         "Mean Architecture Cost (Unstable)")
    logger.info(header)
    logger.info("-" * len(header))
    rows = []
    for name in class_results:
        class_metrics = class_results.get(name) or {}
        stab_metrics = stability_results.get(name) or {}
        model = trained_class_models.get(name)
        if model is None: continue

        # Predict using classification model
        y_pred = model.predict(X_test[selected_features])

        # Filter classes predicted as unstable (1)
        unstable_classes = X_test.loc[y_pred == 1]

        # Calculate mean size and cost for predicted unstable classes
        mean_architecture_size_unstable = unstable_classes[
            'architecture_size'].mean() if 'architecture_size' in unstable_classes.columns else np.nan
        mean_architecture_cost_unstable = unstable_classes[
            'cost'].mean() if 'cost' in unstable_classes.columns else np.nan

        rows.append({
            'Model': name,
            'Balanced Accuracy': class_metrics.get('Balanced Accuracy', np.nan),
            'PR AUC': class_metrics.get('PR AUC', np.nan),
            'F1 Macro': class_metrics.get('F1 Macro', np.nan),
            'AUC-ROC': class_metrics.get('AUC-ROC', np.nan),
            'Change-Count RMSE': stab_metrics.get('RMSE', np.nan),
            'Change-Count Spearman': stab_metrics.get('Spearman', np.nan),
            'Mean Architecture Size (Unstable)': mean_architecture_size_unstable,
            'Mean Architecture Cost (Unstable)': mean_architecture_cost_unstable
        })
    if not rows: logger.warning("No results available for display."); return
    df = pd.DataFrame(rows).sort_values(by='Balanced Accuracy', ascending=False)
    for _, row in df.iterrows():
        row_str = f"| {row['Model']:<15} | {row['Balanced Accuracy']:<19.4f} | {row['PR AUC']:<15.4f} | {row['F1 Macro']:<15.4f} | {row['AUC-ROC']:<15.4f} | {row['Change-Count RMSE']:<23.4f} | {row['Change-Count Spearman']:<20.4f} | {row['Mean Architecture Size (Unstable)']:<25.2f} | {row['Mean Architecture Cost (Unstable)']:<25.2f} |"
        logger.info(row_str)

    # Also saved as CSV for the downstream report script
    comparison_csv_path = os.path.join(project_dir, "comparison_table.csv")
    df.to_csv(comparison_csv_path, index=False)
    logger.info(f"Comparison table saved to '{comparison_csv_path}'")


def display_architecture_factors_quantile_table(data):
    logger.info("\n--- Summary Table for Architecture Size Factors by Quantile ---")

    # To prevent errors if columns are missing in DataFrame
    required_cols = ['architecture_size', 'stability_label', 'real_change_count', 'loc', 'num_methods', 'cbo_approx',
                     'lcom4_approx', 'total_method_cc', 'cc_class', 'cost', 'git_based_stability']


    header = "| {:<15} | {:<10} | {:<10} | {:<12} | {:<10} | {:<12} | {:<15} | {:<10} | {:<15} | {:<15} | {:<10} | {:<12} | {:<15} |".format(
        "Range", "Count", "loc", "num_methods", "cbo_approx", "lcom4_approx", "total_method_cc", "cc_class",
        "Arch Size", "Arch Cost", "Stability", "Unstable %", "Avg Changes"
    )
    logger.info(header)
    logger.info("-" * len(header))

    quantiles = data['architecture_size'].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
    labels = ['Low (0-25%)', 'Low-Mid (25-50%)', 'High-Mid (50-75%)', 'High (75-100%)']

    data['quantile'] = pd.cut(data['architecture_size'], bins=quantiles, labels=labels, include_lowest=True,
                              duplicates='drop')

    grouped = data.groupby('quantile', observed=True).agg({
        'architecture_size': ['count', 'mean'],
        'loc': 'mean',
        'num_methods': 'mean',
        'cbo_approx': 'mean',
        'lcom4_approx': 'mean',
        'total_method_cc': 'mean',
        'cc_class': 'mean',
        'cost': 'mean',
        'git_based_stability': 'mean',
        'stability_label': lambda x: 100 * np.mean(x),
        'real_change_count': 'mean'
    }).reset_index()

    grouped.columns = ['Range', 'Count', 'Arch Size', 'loc', 'num_methods', 'cbo_approx', 'lcom4_approx',
                       'total_method_cc', 'cc_class', 'Arch Cost', 'Stability', 'Unstable (%)', 'Avg Changes']

    grouped.to_csv(os.path.join(project_dir, 'architecture_factors_quantile_table.csv'), index=False)

    for _, row in grouped.iterrows():
        row_str = f"| {str(row['Range']):<15} | {int(row['Count']):<10} | {row['loc']:<10.0f} | {row['num_methods']:<12.0f} | {row['cbo_approx']:<10.0f} | {row['lcom4_approx']:<12.0f} | {row['total_method_cc']:<15.0f} | {row['cc_class']:<10.0f} | {row['Arch Size']:<15.2f} | {row['Arch Cost']:<15.2f} | {row['Stability']:<10.2f} | {row['Unstable (%)']:<12.1f} | {row['Avg Changes']:<15.1f} |"
        logger.info(row_str)


def run_repeated_stratified_cv(X, change_count, project_name, project_dir, model_name='RandomForest', param_grid=None,
                               fixed_params=None, n_splits=5, n_repeats=10, n_features=15):
    """Repeated stratified K-fold CV; feature selection, scaling and the model are fitted on each training fold.

    Uses the full X and raw change_count; architecture_size and the Q80 label are rebuilt per fold from train rows only.
    model_name / fixed_params: held-constant model; if fixed_params is given, grid search is skipped.
    Reports mean +/- 95% CI.
    """
    if param_grid is None:
        param_grid = MODEL_PARAM_GRIDS.get(model_name, RF_PARAM_GRID)
    if fixed_params is None and model_name == 'RandomForest':
        fixed_params = RF_FIXED_PARAMS
    skip_grid_search = fixed_params is not None
    strat_proxy = make_stratification_proxy(change_count)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    fold_rows = []
    total_folds = n_splits * n_repeats
    rcv_start_time = time.time()
    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, strat_proxy)):
        X_train_full, X_test_full = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        # Threshold from this fold's train rows only
        y_train, fold_threshold = make_stability_labels(change_count, X_train_full.index)
        y_train = y_train.loc[X_train_full.index]
        y_test = (change_count.loc[X_test_full.index] >= fold_threshold).astype(int)
        if y_train.nunique() < 2 or y_test.nunique() < 1:
            continue

        X_train_full, X_test_full = add_train_test_architecture_size(X_train_full, X_test_full)
        X_train, selected_features = select_features(X_train_full, y_train, n_features=n_features)
        X_test = X_test_full[selected_features]

        scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

        # Fixed config instead of GridSearchCV per fold
        model, _, _ = train_classification_model(X_train_scaled, y_train, model_name, param_grid,
                                                  skip_grid_search=skip_grid_search and not NESTED_TUNING,
                                                  fixed_params=(None if NESTED_TUNING else fixed_params))
        if model is None:
            continue
        metrics = evaluate_classification_model(model, X_test_scaled, y_test,
                                                 model_name=f'{model_name}_RCV_fold{fold_idx}')
        fold_rows.append({
            'Fold': fold_idx,
            'Balanced_Accuracy': metrics.get('Balanced Accuracy') if metrics else np.nan,
            'F1_Macro': metrics.get('F1 Macro') if metrics else np.nan,
            'PR_AUC': metrics.get('PR AUC') if metrics else np.nan,
            'AUC_ROC': metrics.get('AUC-ROC') if metrics else np.nan,
        })

        log_progress(f"[{project_name}] Repeated-CV folds ({model_name})", fold_idx + 1, total_folds, rcv_start_time)

    fold_df = pd.DataFrame(fold_rows)
    if fold_df.empty:
        logger.error(f"[{project_name}] Repeated stratified CV produced no valid folds.")
        return fold_df

    csv_path = os.path.join(project_dir, f"repeated_cv_{model_name}_{project_name}.csv")
    fold_df.to_csv(csv_path, index=False)

    logger.info(f"--- Repeated Stratified {n_splits}-Fold x {n_repeats} CV summary "
                f"({len(fold_df)} folds, {model_name}, {project_name}) ---")
    summary_rows = []
    for c in ['Balanced_Accuracy', 'F1_Macro', 'PR_AUC', 'AUC_ROC']:
        vals = fold_df[c].dropna()
        if len(vals) > 1:
            sem = vals.std(ddof=1) / np.sqrt(len(vals))
            mean, ci95, std = vals.mean(), 1.96 * sem, vals.std(ddof=1)
            logger.info(f"  {c}: mean={mean:.4f}  +/-95%CI={ci95:.4f}  std={std:.4f}  n={len(vals)}")
            summary_rows.append({"Metric": c, "Mean": mean, "CI95": ci95, "Std": std, "N_Folds": len(vals)})
    logger.info(f"Per-fold results saved to '{csv_path}'")

    # Also saved as an aggregate summary (mean +/- 95% CI per metric)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv_path = os.path.join(project_dir, f"repeated_cv_{model_name}_{project_name}_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        logger.info(f"Repeated-CV summary (mean +/- 95% CI) saved to '{summary_csv_path}'")

    return fold_df


def run_alpha_sensitivity_ml_analysis(project_name, alpha_grid=None, n_features=15,
                                      model_name='RandomForest', fixed_params=None,
                                      features_graph=None):
    """Measure how ALPHA_STRUCTURAL affects classification performance and the predicted class ranking.

    Requires propagation_frequencies_alpha_<a>.pkl files (from the dependency-graph script's alpha sweep).
    model_name / fixed_params: held-constant model and hyperparameters.
    features_graph: precomputed compute_graph_features() output, reused if given; its 'pagerank' value (the only alpha-dependent
    network feature) is replaced per alpha from pagerank_alpha_<a>.pkl written by script 1's alpha sweep.
    """
    if alpha_grid is None:
        alpha_grid = ALPHA_SENSITIVITY_GRID
    if fixed_params is None and model_name == 'RandomForest':
        fixed_params = RF_FIXED_PARAMS
    param_grid = MODEL_PARAM_GRIDS.get(model_name, RF_PARAM_GRID)

    proj_dir = os.path.join(EXPORT_DIR, project_name)
    metrics_file = os.path.join(proj_dir, "code_metrics.pkl")
    try:
        with open(metrics_file, "rb") as f:
            code_metrics = pickle.load(f)
        if features_graph is None:
            graph_file = os.path.join(proj_dir, "dependency_graph.pkl")
            with open(graph_file, "rb") as f:
                graph = pickle.load(f)
            features_graph = compute_graph_features(graph)
    except FileNotFoundError as e:
        logger.error(f"[{project_name}] Cannot run alpha ML sensitivity analysis: {e}")
        return pd.DataFrame()

    rows = []
    predicted_rankings = {}  # alpha -> pd.Series indexed by class, predicted prob of instability
    alpha_start_time = time.time()
    for alpha_i, alpha in enumerate(alpha_grid):
        prop_file = os.path.join(proj_dir, f"propagation_frequencies_alpha_{alpha:.1f}.pkl")
        if not os.path.exists(prop_file):
            logger.warning(f"[{project_name}] Missing '{prop_file}' - run script 2's "
                           f"run_alpha_sensitivity_sweep with RUN_ALPHA_SENSITIVITY=True first. Skipping alpha={alpha}.")
            continue
        with open(prop_file, "rb") as f:
            propagation_frequencies = pickle.load(f)

        # Recompute the alpha-dependent network feature: PageRank uses the edge weights (betweenness, eigenvector,
        # clustering and degree centrality are unweighted), so it is replaced by the PageRank of this alpha's graph.
        features_graph_alpha = features_graph
        pr_file = os.path.join(proj_dir, f"pagerank_alpha_{alpha:.1f}.pkl")
        if os.path.exists(pr_file):
            with open(pr_file, "rb") as f:
                pr_alpha = pickle.load(f)
            features_graph_alpha = {n: dict(v) for n, v in features_graph.items()}
            for n_, v_ in features_graph_alpha.items():
                if n_ in pr_alpha:
                    v_['pagerank'] = pr_alpha[n_]
        else:
            logger.warning(f"[{project_name}] '{pr_file}' missing (rerun script 1's alpha sweep): PageRank is NOT "
                           f"recomputed for alpha={alpha}; only the propagation feature varies.")

        X, change_count, y_stability, threshold, df = prepare_data_combined(
            features_graph_alpha, code_metrics, propagation_frequencies, proj_dir
        )
        if X is None or X.empty:
            continue

        # Stratify by a proxy, then derive the Q80 label from this split's train rows only
        strat_proxy = make_stratification_proxy(change_count)
        X_train_full, X_test_full, cc_train, cc_test = train_test_split(
            X, change_count, test_size=0.2, stratify=strat_proxy, random_state=42
        )
        y_train, alpha_threshold = make_stability_labels(change_count, X_train_full.index)
        y_train = y_train.loc[X_train_full.index]
        y_test = (change_count.loc[X_test_full.index] >= alpha_threshold).astype(int)

        X_train_full, X_test_full = add_train_test_architecture_size(X_train_full, X_test_full)
        X_train, selected_features = select_features(X_train_full, y_train, n_features=n_features)
        X_test = X_test_full[selected_features]

        scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

        # Fixed config instead of GridSearchCV per alpha
        model, cv_score, _ = train_classification_model(X_train_scaled, y_train, model_name, param_grid,
                                                         skip_grid_search=not NESTED_TUNING, fixed_params=(None if NESTED_TUNING else fixed_params))
        if model is None:
            continue
        metrics = evaluate_classification_model(model, X_test_scaled, y_test, model_name=f'{model_name}_alpha_{alpha:.1f}')

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test_scaled)[:, 1]
            predicted_rankings[alpha] = pd.Series(probs, index=X_test_scaled.index)

        rows.append({
            'alpha': alpha,
            'Train_CV_Balanced_Accuracy': cv_score,
            'Test_Balanced_Accuracy': metrics.get('Balanced Accuracy') if metrics else np.nan,
            'Test_F1_Macro': metrics.get('F1 Macro') if metrics else np.nan,
            'Test_PR_AUC': metrics.get('PR AUC') if metrics else np.nan,
        })

        log_progress(f"[{project_name}] Alpha sensitivity sweep", alpha_i + 1, len(alpha_grid), alpha_start_time)

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        logger.error(f"[{project_name}] Alpha ML sensitivity analysis produced no results.")
        return result_df

    csv_path = os.path.join(proj_dir, f"alpha_sensitivity_ml_performance_{project_name}_{model_name}.csv")
    result_df.to_csv(csv_path, index=False)
    logger.info(f"[{project_name}] Alpha ML-performance sensitivity ({model_name}) saved to '{csv_path}'")

    # Ranking stability: Spearman rho vs the reference alpha=0.6 ranking
    if 0.6 in predicted_rankings:
        ref_ranking = predicted_rankings[0.6]
        stability_rows = []
        for alpha, ranking in predicted_rankings.items():
            common = ranking.index.intersection(ref_ranking.index)
            if len(common) > 2:
                rho, _ = scipy_stats.spearmanr(ranking.loc[common], ref_ranking.loc[common])
            else:
                rho = np.nan
            stability_rows.append({'alpha': alpha, 'spearman_rho_vs_alpha_0.6_predictions': rho})
        stability_df = pd.DataFrame(stability_rows)
        stability_csv = os.path.join(proj_dir, f"alpha_sensitivity_ml_ranking_stability_{project_name}_{model_name}.csv")
        stability_df.to_csv(stability_csv, index=False)
        logger.info(f"[{project_name}] Alpha ML-prediction ranking-stability ({model_name}) saved to '{stability_csv}'")

    logger.info(f"[{project_name}] Balanced Accuracy range across alpha grid: "
                f"{result_df['Test_Balanced_Accuracy'].min():.4f} - {result_df['Test_Balanced_Accuracy'].max():.4f} "
                f"(span={result_df['Test_Balanced_Accuracy'].max() - result_df['Test_Balanced_Accuracy'].min():.4f})")
    return result_df


def load_project_dataset(project_name):
    """Load and prepare (X, change_count, y_stability, df) for one project, without splitting or training.

    change_count is the raw continuous count; callers derive their own train-only Q80 label.
    Returns None if a required artifact is missing.
    """
    proj_dir = os.path.join(EXPORT_DIR, project_name)
    graph_file = os.path.join(proj_dir, "dependency_graph.pkl")
    prop_file = os.path.join(proj_dir, "propagation_frequencies.pkl")
    metrics_file = os.path.join(proj_dir, "code_metrics.pkl")
    try:
        with open(graph_file, "rb") as f:
            graph = pickle.load(f)
        with open(prop_file, "rb") as f:
            propagation_frequencies = pickle.load(f)
        with open(metrics_file, "rb") as f:
            code_metrics = pickle.load(f)
    except FileNotFoundError as e:
        logger.error(f"[{project_name}] Cannot load data for LOPO evaluation: {e}")
        return None

    features_graph = compute_graph_features(graph)
    # Second value is the raw change_count; LOPO builds its label from the pooled training projects only
    X, change_count, y_stability, threshold, df = prepare_data_combined(
        features_graph, code_metrics, propagation_frequencies, proj_dir
    )
    if X is None or X.empty:
        logger.error(f"[{project_name}] Data preparation failed or returned empty DataFrame for LOPO evaluation.")
        return None
    return X, change_count, y_stability, df


def run_leave_one_project_out_pipeline(projects, model_name='RandomForest',
                                       param_grid=None, n_features=15, preloaded_datasets=None):
    """Leave-One-Project-Out evaluation: for each project, train on the others and test on it.

    Reported alongside the within-project results; with only 3 projects it is limited evidence of
    generalization (state this in Threats to Validity).
    preloaded_datasets: optional {project: (X, change_count, y_stability, df)} reused instead of load_project_dataset().
    """
    if param_grid is None:
        # Use the grid of the requested model_name
        param_grid = MODEL_PARAM_GRIDS.get(model_name, RF_PARAM_GRID)

    logger.info("=" * 70)
    logger.info("Leave-One-Project-Out (LOPO) cross-project evaluation")
    logger.info(f"Projects: {projects}  |  Model: {model_name}")
    logger.info("=" * 70)

    datasets = {}
    for p in projects:
        if preloaded_datasets is not None and p in preloaded_datasets:
            datasets[p] = preloaded_datasets[p]
            continue
        loaded = load_project_dataset(p)
        if loaded is not None:
            datasets[p] = loaded
        else:
            logger.warning(f"[{p}] Skipped from LOPO evaluation (missing/invalid data).")

    if len(datasets) < 2:
        logger.error("LOPO evaluation needs at least 2 projects with valid data. Aborting.")
        return pd.DataFrame()

    fold_rows = []
    for test_project in datasets:
        train_projects = [p for p in datasets if p != test_project]
        logger.info(f"--- LOPO fold: train={train_projects}  test={test_project} ---")

        X_train_full = pd.concat([datasets[p][0] for p in train_projects], axis=0, ignore_index=False)
        cc_train_full = pd.concat([datasets[p][1] for p in train_projects], axis=0, ignore_index=False)
        X_test_full, cc_test_full = datasets[test_project][0], datasets[test_project][1]

        # Deduplicate the (project, class_name) index so same-named classes from different projects don't collide
        X_train_full = X_train_full.reset_index(drop=True)
        cc_train_full = cc_train_full.reset_index(drop=True)
        X_test_full = X_test_full.reset_index(drop=True)
        cc_test_full = cc_test_full.reset_index(drop=True)

        # The pooled training data are concatenated project by project; shuffle the rows (fixed seed) so that the
        # unshuffled 5-fold CV inside GridSearchCV/RFECV does not form project-blocks as validation folds.
        _perm = np.random.RandomState(42).permutation(len(X_train_full))
        X_train_full = X_train_full.iloc[_perm].reset_index(drop=True)
        cc_train_full = cc_train_full.iloc[_perm].reset_index(drop=True)

        # Q80 threshold from the pooled training projects only, applied unchanged to the held-out project
        y_train, lopo_threshold = make_stability_labels(cc_train_full, cc_train_full.index)
        y_test = (cc_test_full >= lopo_threshold).astype(int)
        logger.info(f"[{test_project}] LOPO train-pool-only Q80 threshold: {lopo_threshold:.0f} changes "
                    f"(train n={len(y_train)}, positives={int(y_train.sum())}; "
                    f"test n={len(y_test)}, positives={int(y_test.sum())}).")

        if y_train.nunique() < 2:
            logger.warning(f"[{test_project}] Training pool has a single class after pooling; skipping fold.")
            continue

        # architecture_size from train-projects-only statistics
        X_train_full, X_test_full = add_train_test_architecture_size(X_train_full, X_test_full)

        X_train, selected_features = select_features(X_train_full, y_train, n_features=n_features)
        X_test = X_test_full[selected_features]

        scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

        model, cv_score, _ = train_classification_model(X_train_scaled, y_train, model_name, param_grid)
        if model is None:
            logger.error(f"[{test_project}] Model training failed for this fold; skipping.")
            continue

        metrics = evaluate_classification_model(model, X_test_scaled, y_test,
                                                 model_name=f'{model_name}_LOPO_test={test_project}')
        # Diagnostic variant (not a deployable protocol): the same 'changes' threshold is used for the training pool and
        # the held-out project, namely the held-out project's own Q80 (computed from its change counts). It removes the
        # prevalence shift caused by the pooled threshold, so the drop in Balanced Accuracy can be attributed to
        # threshold shift versus lost ranking ability. RFECV subset and scaler are reused from the pooled-threshold fit.
        own = {}
        try:
            _, own_thr = make_stability_labels(cc_test_full, cc_test_full.index)
            y_train_own = (cc_train_full >= own_thr).astype(int)
            y_test_own = (cc_test_full >= own_thr).astype(int)
            if y_train_own.nunique() == 2 and y_test_own.nunique() == 2:
                model_own, _, _ = train_classification_model(X_train_scaled, y_train_own, model_name, param_grid)
                m_own = evaluate_classification_model(model_own, X_test_scaled, y_test_own,
                                                      model_name=f'{model_name}_LOPO_ownthr_test={test_project}') \
                    if model_own is not None else None
                own = {'OwnThr_Q80_Threshold': float(own_thr),
                       'OwnThr_Train_Unstable_Fraction': float(y_train_own.mean()),
                       'OwnThr_Test_Unstable_Fraction': float(y_test_own.mean()),
                       'OwnThr_Test_Balanced_Accuracy': m_own.get('Balanced Accuracy') if m_own else np.nan,
                       'OwnThr_Test_PR_AUC': m_own.get('PR AUC') if m_own else np.nan,
                       'OwnThr_Test_AUC_ROC': m_own.get('AUC-ROC') if m_own else np.nan}
                logger.info(f"[{test_project}] LOPO with the held-out project's own Q80 threshold ({own_thr:.0f} changes): "
                            f"BA={own['OwnThr_Test_Balanced_Accuracy']:.4f} (pooled-threshold BA="
                            f"{(metrics or {}).get('Balanced Accuracy', float('nan')):.4f}); "
                            f"test unstable share {y_test_own.mean():.1%} vs {y_test.mean():.1%} with the pooled threshold.")
        except Exception as e_own:
            logger.warning(f"[{test_project}] own-threshold LOPO variant failed: {e_own}")
        fold_rows.append({
            **own,
            'Held_Out_Project': test_project,
            'Trained_On': ",".join(train_projects),
            'N_Train': len(X_train_scaled),
            'N_Test': len(X_test_scaled),
            'N_Features': len(selected_features),
            # Q80 threshold learned from the pooled training projects and applied unchanged to the held-out project
            'Q80_Threshold': float(lopo_threshold),
            'Train_N_Unstable': int(y_train.sum()),
            'Train_Unstable_Fraction': float(y_train.mean()),
            'Test_N_Unstable': int(y_test.sum()),
            'Test_Unstable_Fraction': float(y_test.mean()),
            # RFECV-selected features for this fold (training pool only), ';'-separated
            'Selected_Features': ";".join(selected_features),
            'Train_CV_Balanced_Accuracy': cv_score,
            'Test_Balanced_Accuracy': metrics.get('Balanced Accuracy') if metrics else np.nan,
            'Test_F1_Macro': metrics.get('F1 Macro') if metrics else np.nan,
            'Test_PR_AUC': metrics.get('PR AUC') if metrics else np.nan,
            'Test_AUC_ROC': metrics.get('AUC-ROC') if metrics else np.nan,
        })

    result_df = pd.DataFrame(fold_rows)
    if result_df.empty:
        logger.error("LOPO evaluation produced no valid folds.")
        return result_df

    os.makedirs(EXPORT_DIR, exist_ok=True)
    csv_path = os.path.join(EXPORT_DIR, f"lopo_cross_project_{model_name}.csv")
    result_df.to_csv(csv_path, index=False)

    summary_cols = ['Test_Balanced_Accuracy', 'Test_F1_Macro', 'Test_PR_AUC', 'Test_AUC_ROC']
    logger.info("-" * 70)
    logger.info(f"LOPO summary across {len(result_df)} fold(s) (mean +/- std):")
    for c in summary_cols:
        vals = result_df[c].dropna()
        if len(vals) > 0:
            logger.info(f"  {c}: {vals.mean():.4f} +/- {vals.std():.4f}")
    logger.info(f"Per-fold results saved to '{csv_path}'")
    logger.info(
        "Compare these numbers to the within-project (random split) results from run_pipeline: "
        "a large gap (LOPO much worse) indicates the model is leaning on project-specific "
        "patterns rather than transferable structural/architectural signal."
    )
    logger.info("-" * 70)
    return result_df


# Calibration (Brier/ECE), pairwise McNemar and bootstrap CI helpers

def _get_positive_class_probability(model, X):
    """Positive-class probability via predict_proba/decision_function fallback (shared by ROC/PR-AUC, Brier and ECE)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.max() != scores.min():
            return (scores - scores.min()) / (scores.max() - scores.min())
        return scores
    raise ValueError("Model has neither predict_proba nor decision_function.")


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Equal-width-bin ECE: weighted mean of |mean predicted probability - observed frequency| per bin."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_table = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi) if i == n_bins - 1 else (y_prob >= lo) & (y_prob < hi)
        n_in_bin = int(mask.sum())
        if n_in_bin == 0:
            continue
        avg_confidence = float(y_prob[mask].mean())
        observed_freq = float(y_true[mask].mean())
        weight = n_in_bin / len(y_prob)
        ece += weight * abs(avg_confidence - observed_freq)
        bin_table.append({"bin_low": lo, "bin_high": hi, "n": n_in_bin,
                          "avg_confidence": avg_confidence, "observed_frequency": observed_freq})
    return ece, pd.DataFrame(bin_table)


def compute_calibration_table(trained_class_models, X_test, y_test, selected_features,
                              project_dir, project_name, logger, n_bins=10):
    rows = []
    for model_name, model in trained_class_models.items():
        if model is None:
            continue
        try:
            y_prob = _get_positive_class_probability(model, X_test[selected_features])
            brier = brier_score_loss(y_test, y_prob)
            ece, bin_df = expected_calibration_error(y_test, y_prob, n_bins=n_bins)
            bin_df.to_csv(os.path.join(project_dir, f"calibration_bins_{model_name}.csv"), index=False)
            rows.append({"Model": model_name, "Brier_Score": brier, "ECE": ece})
            logger.info(f"[{project_name}][{model_name}] Brier={brier:.4f}  ECE={ece:.4f}")
        except Exception as e:
            logger.warning(f"[{project_name}][{model_name}] calibration computation failed: {e}")

    calib_df = pd.DataFrame(rows).sort_values("Brier_Score") if rows else pd.DataFrame()
    calib_path = os.path.join(project_dir, "calibration_summary.csv")
    calib_df.to_csv(calib_path, index=False)
    logger.info(f"[{project_name}] Calibration summary saved to '{calib_path}'")
    return calib_df


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """Exact McNemar test. n01 = A wrong/B right, n10 = A right/B wrong. Returns (n01, n10, p_value)."""
    y_true = np.asarray(y_true)
    correct_a = (np.asarray(y_pred_a) == y_true)
    correct_b = (np.asarray(y_pred_b) == y_true)
    n01 = int(np.sum(~correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    n_discordant = n01 + n10
    if n_discordant == 0:
        return n01, n10, 1.0
    p_value = scipy_stats.binomtest(min(n01, n10), n_discordant, 0.5, alternative="two-sided").pvalue
    return n01, n10, p_value


def compute_pairwise_mcnemar(trained_class_models, X_test, y_test, selected_features,
                             project_dir, project_name, logger):
    """McNemar test for every model pair on the same held-out test set."""
    names = [n for n, m in trained_class_models.items() if m is not None]
    predictions = {name: trained_class_models[name].predict(X_test[selected_features]) for name in names}

    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            n01, n10, p = mcnemar_test(y_test, predictions[a], predictions[b])
            rows.append({"Model_A": a, "Model_B": b, "n_A_wrong_B_right": n01,
                        "n_A_right_B_wrong": n10, "p_value": p, "Significant_at_0.05": bool(p < 0.05)})
            logger.info(f"[{project_name}] McNemar {a} vs {b}: n01={n01} n10={n10} p={p:.4f} "
                        f"{'SIGNIFICANT' if p < 0.05 else 'not significant'}")

    mcnemar_df = pd.DataFrame(rows)
    mcnemar_path = os.path.join(project_dir, "mcnemar_pairwise.csv")
    mcnemar_df.to_csv(mcnemar_path, index=False)
    logger.info(f"[{project_name}] Pairwise McNemar results saved to '{mcnemar_path}'")
    return mcnemar_df


def bootstrap_balanced_accuracy_ci(y_true, y_pred, n_boot=2000, ci=0.95, random_state=42):
    """Bootstrap CI of balanced accuracy, resampling the test set with replacement."""
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    scores = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        scores[b] = balanced_accuracy_score(y_true[idx], y_pred[idx])
    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return float(np.mean(scores)), float(lo), float(hi)


def compute_bootstrap_ci_table(trained_class_models, X_test, y_test, selected_features,
                               project_dir, project_name, logger, n_boot=2000):
    rows = []
    for model_name, model in trained_class_models.items():
        if model is None:
            continue
        y_pred = model.predict(X_test[selected_features])
        mean_ba, lo, hi = bootstrap_balanced_accuracy_ci(y_test, y_pred, n_boot=n_boot)
        rows.append({"Model": model_name, "Mean_Balanced_Accuracy": mean_ba,
                    "CI_Lower_2.5%": lo, "CI_Upper_97.5%": hi})
        logger.info(f"[{project_name}][{model_name}] Bootstrap BA={mean_ba:.4f} "
                    f"95% CI=[{lo:.4f}, {hi:.4f}] (n_boot={n_boot})")
    boot_df = pd.DataFrame(rows).sort_values("Mean_Balanced_Accuracy", ascending=False) if rows else pd.DataFrame()
    boot_path = os.path.join(project_dir, "bootstrap_ci_balanced_accuracy.csv")
    boot_df.to_csv(boot_path, index=False)
    logger.info(f"[{project_name}] Bootstrap CI table saved to '{boot_path}'")
    return boot_df


def run_calibration_and_significance_analysis(trained_class_models, X_test, y_test, selected_features,
                                               project_dir, project_name, logger):
    calib_df = compute_calibration_table(trained_class_models, X_test, y_test, selected_features,
                                         project_dir, project_name, logger)
    mcnemar_df = compute_pairwise_mcnemar(trained_class_models, X_test, y_test, selected_features,
                                          project_dir, project_name, logger)
    boot_df = compute_bootstrap_ci_table(trained_class_models, X_test, y_test, selected_features,
                                         project_dir, project_name, logger)
    return calib_df, mcnemar_df, boot_df


# Main pipeline
def run_pipeline(project_name):
    """Run the main pipeline (steps 1-10) for one project.

    Ablation, repeated CV and alpha sensitivity run later in run_pipeline_followups(), once the primary model is known.
    Returns (X, change_count, y_stability, df, project_dir, features_graph) on success, None on failure.
    """
    configure_logging(project_name)

    N_FEATURES = 15  # Fallback cap if RFECV errors
    logger.info("--- Starting Classification and Regression using Git History (Step 4 Hybrid Stability) ---")
    logger.info(f"--- Project: {project_name} ---")
    main_start_time = time.time()

    # File paths
    graph_file = os.path.join(project_dir, "dependency_graph.pkl")
    prop_file = os.path.join(project_dir, "propagation_frequencies.pkl")
    metrics_file = os.path.join(project_dir, "code_metrics.pkl")
    best_model_file = os.path.join(project_dir, f"best_model_git_based.pkl")

    # 1. Load data
    try:
        with open(graph_file, "rb") as f:
            graph = pickle.load(f)
        with open(prop_file, "rb") as f:
            propagation_frequencies = pickle.load(f)
        with open(metrics_file, "rb") as f:
            code_metrics = pickle.load(f)
        logger.info("Data loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Error loading data: File not found - {e}"); save_console_log_to_file(); return None

    # 2. Compute graph features and prepare data (using Git History)
    features_graph = compute_graph_features(graph)

    # change_count is the raw continuous series; the Q80 label is computed after the split from train rows only.
    # `threshold` is a descriptive whole-dataset value (quantile table only).
    X, change_count, y_stability, threshold, full_df_for_reporting = prepare_data_combined(
        features_graph, code_metrics, propagation_frequencies, project_dir
    )

    if X is None or X.empty:
        logger.error("Data preparation failed or returned empty DataFrames."); save_console_log_to_file(); return None

    # 3. Split first (RFE must never see test data); stratify by a change_count quintile proxy
    try:
        strat_proxy = make_stratification_proxy(change_count)
        X_train_full, X_test_full, cc_train, cc_test, y_stability_train, y_stability_test = train_test_split(
            X, change_count, y_stability, test_size=0.2, stratify=strat_proxy, random_state=42
        )
    except Exception as e:
        logger.error(f"Error in data splitting: {e}"); save_console_log_to_file(); return None

    # Q80 threshold from train rows only, applied unchanged to test rows
    y_train, threshold = make_stability_labels(change_count, X_train_full.index)
    y_train = y_train.loc[X_train_full.index]
    y_test = (change_count.loc[X_test_full.index] >= threshold).astype(int)
    logger.info(f"[{project_name}] Train-only Q80 threshold used for this split: {threshold:.0f} changes "
                f"(train n={len(y_train)}, positives={int(y_train.sum())}; "
                f"test n={len(y_test)}, positives={int(y_test.sum())}).")

    # Also saved as a structured file for the downstream report script
    threshold_info = {
        "project": project_name,
        "train_only_q80_threshold_changes": float(threshold),
        "train_n": int(len(y_train)),
        "train_positives": int(y_train.sum()),
        "train_positive_rate": float(y_train.mean()),
        "test_n": int(len(y_test)),
        "test_positives": int(y_test.sum()),
        "test_positive_rate": float(y_test.mean()),
    }
    threshold_info_path = os.path.join(project_dir, "train_only_threshold_info.json")
    try:
        with open(threshold_info_path, "w", encoding="utf-8") as f:
            json.dump(threshold_info, f, indent=2)
    except Exception as e:
        logger.warning(f"[{project_name}] Could not save train_only_threshold_info.json: {e}")

    # 3.5 Rebuild architecture_size from train-only min/max
    X_train_full, X_test_full = add_train_test_architecture_size(X_train_full, X_test_full)

    # 4. Feature selection (fit on the training split only)
    X_train, selected_features = select_features(X_train_full, y_train, n_features=N_FEATURES)
    X_test = X_test_full[selected_features]

    # 5. Normalization
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

    # 6. Define parameters and models (grids from MODEL_PARAM_GRIDS)
    param_grids = MODEL_PARAM_GRIDS
    model_names = ['RandomForest', 'GaussianNB', 'XGBoost', 'SVM',
                   'DummyMostFrequent', 'DummyStratified', 'LogisticRegression']
    # MultinomialNB and BernoulliNB may have issues with normalized data

    class_results = {}
    stability_results = {}
    trained_class_models = {}
    trained_stability_models = {}
    class_model_cv_scores = {}  # mean train-only 5-fold CV balanced accuracy (model selection)
    class_model_best_params = {}  # tuned hyperparameters per model, reused by the follow-up analyses

    # 7. Train, evaluate and save results
    models_start_time = time.time()
    for model_idx, model_name in enumerate(model_names):
        logger.info(f"--- Processing model: {model_name} ---")
        param_grid = param_grids.get(model_name, {})

        # a) Train classification model (predict Git label 0/1)
        class_model, class_model_cv_score, class_model_best_params_this = train_classification_model(
            X_train_df, y_train, model_name, param_grid)
        if class_model is None: continue
        trained_class_models[model_name] = class_model
        class_model_cv_scores[model_name] = class_model_cv_score
        class_model_best_params[model_name] = class_model_best_params_this

        # b) Evaluate classification model
        class_eval = evaluate_classification_model(class_model, X_test_df, y_test, model_name)
        class_results[model_name] = class_eval if class_eval else None

        # c) Train regression model (log1p future change count)
        stability_model = train_stability_model(X_train_df, y_train, y_stability_train, class_model, model_name)
        if stability_model is None: continue
        trained_stability_models[model_name] = stability_model

        # d) Evaluate regression model
        stability_eval = evaluate_stability_model(stability_model, X_test_df, y_stability_test, class_model, model_name)
        stability_results[model_name] = stability_eval if stability_eval else None

        log_progress(f"[{project_name}] Model training", model_idx + 1, len(model_names), models_start_time)

    # Save each model's tuned hyperparameters for reuse by the follow-up analyses
    tuned_hyperparams_path = os.path.join(project_dir, f"tuned_hyperparams_{project_name}.json")
    try:
        with open(tuned_hyperparams_path, "w", encoding="utf-8") as f:
            json.dump(class_model_best_params, f, indent=2, default=str)
        logger.info(f"Tuned hyperparameters (per model, from the main comparison's GridSearchCV) "
                    f"saved to '{tuned_hyperparams_path}'")
    except Exception as e:
        logger.warning(f"Could not save tuned_hyperparams_{project_name}.json: {e}")

    # 8. Generate final tables and charts
    plot_stability_comparison(stability_results)
    display_comparison_table(class_results, stability_results, X_test_df, y_test, trained_class_models,
                             selected_features)

    # Use full data for quantile table
    display_architecture_factors_quantile_table(full_df_for_reporting)

    # 9. Save best model, selected by mean train-CV balanced accuracy (never by test performance)
    valid_cv_scores = {k: v for k, v in class_model_cv_scores.items() if v is not None and not pd.isna(v)}
    best_model_name = max(valid_cv_scores, key=lambda k: valid_cv_scores[k], default=None)
    try:  # train-only CV scores, used by select_primary_model (no test-set information)
        pd.DataFrame([{'Model': k, 'Train_CV_Balanced_Accuracy': v} for k, v in class_model_cv_scores.items()]
                     ).to_csv(os.path.join(project_dir, 'train_cv_scores.csv'), index=False)
    except Exception as e:
        logger.warning(f"Could not save train_cv_scores.csv: {e}")

    if best_model_name:
        best_class_model = trained_class_models.get(best_model_name)
        best_stability_model = trained_stability_models.get(best_model_name)

        if best_class_model:
            model_data_to_save = {
                'classification_model': best_class_model,
                'stability_regressor': best_stability_model,
                'scaler': scaler,
                'selected_features': selected_features,
                'instability_threshold_changes': threshold,
                'model_name': best_model_name
            }
            try:
                with open(best_model_file, 'wb') as f:
                    pickle.dump(model_data_to_save, f)
                logger.info(
                    f"\n--- Best model ({best_model_name}) based on Git history saved at: {best_model_file} ---")
            except Exception as e:
                logger.error(f"Error saving best model: {e}")

            # 10. Feature importance for the best model: permutation importance on the test set is the primary ranking;
            # native importance (if available) is a secondary column. A grouped-by-view table is also produced.
            feature_names = list(X_train_df.columns)

            perm = permutation_importance(
                estimator=best_class_model,
                X=X_test_df,
                y=y_test,
                scoring="balanced_accuracy",
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )
            perm_importances = perm.importances_mean
            perm_std = perm.importances_std

            if hasattr(best_class_model, "feature_importances_"):
                native_importances = best_class_model.feature_importances_
            else:
                native_importances = np.full(len(feature_names), np.nan)

            feature_to_view = {}
            for view, cols in FEATURE_GROUPS.items():
                for c in cols:
                    feature_to_view[c] = view

            feature_importance_df = (
                pd.DataFrame({
                    "Feature": feature_names,
                    "Permutation_Importance": perm_importances,
                    "Permutation_Std": perm_std,
                    "Native_Impurity_Importance": native_importances,
                    "View": [feature_to_view.get(f, "unmapped") for f in feature_names],
                })
                .sort_values("Permutation_Importance", ascending=False)
                .reset_index(drop=True)
            )

            unmapped = feature_importance_df.loc[feature_importance_df["View"] == "unmapped", "Feature"].tolist()
            if unmapped:
                logger.warning(f"[{PROJECT_NAME}] Feature(s) not found in FEATURE_GROUPS, excluded from the "
                               f"grouped/view-level importance table: {unmapped}.")

            csv_path = os.path.join(project_dir, "feature_importance.csv")
            feature_importance_df.to_csv(csv_path, index=False)

            # ---------- Save TXT (feature-level, primary ranking) ----------
            txt_path = os.path.join(project_dir, "feature_importance.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"Project : {PROJECT_NAME}\n")
                f.write(f"Best Model : {best_model_name}\n")
                f.write(f"Importance Method : Permutation (primary, on held-out test set); "
                        f"native impurity importance shown for reference only where available.\n\n")
                for i, row in feature_importance_df.iterrows():
                    line = (
                        f"{i + 1:2d}. {row['Feature']:<28} "
                        f"perm={row['Permutation_Importance']:.6f} (+/-{row['Permutation_Std']:.4f})  "
                        f"native={row['Native_Impurity_Importance']:.6f}  view={row['View']}"
                    )
                    logger.info(line)
                    f.write(line + "\n")
            logger.info(f"Feature importance CSV/TXT saved to '{csv_path}' / '{txt_path}'")

            # ---------- Grouped (view-level) importance ----------
            grouped_importance_df = (
                feature_importance_df[feature_importance_df["View"] != "unmapped"]
                .groupby("View", observed=True)["Permutation_Importance"]
                .agg(["sum", "mean", "count"])
                .rename(columns={"sum": "Sum_Importance", "mean": "Mean_Importance", "count": "N_Features"})
                .sort_values("Sum_Importance", ascending=False)
            )
            grouped_csv_path = os.path.join(project_dir, "feature_importance_grouped_by_view.csv")
            grouped_importance_df.to_csv(grouped_csv_path)
            logger.info(f"[{PROJECT_NAME}] Grouped (view-level) importance:\n{grouped_importance_df}")
            logger.info(f"Grouped importance saved to '{grouped_csv_path}'")

            # ---------- Calibration (Brier/ECE), pairwise McNemar, bootstrap CI ----------
            try:
                run_calibration_and_significance_analysis(
                    trained_class_models, X_test_df, y_test, selected_features,
                    project_dir, PROJECT_NAME, logger
                )
            except Exception as e:
                logger.error(f"[{PROJECT_NAME}] Calibration/significance analysis failed: {e}")
        else:
            logger.error("Could not find trained classification model to save as the best model.")
    else:
        logger.warning("No valid models found to select as the best model.")

    logger.info(f"\n--- Process completed in {time.time() - main_start_time:.2f} seconds. ---")

    # Save the console output to 'Result (<project>) Log.txt'
    save_console_log_to_file()

    # Returned for reuse by the follow-up analyses and LOPO (avoids recomputing graph features)
    return X, change_count, y_stability, full_df_for_reporting, project_dir, features_graph


def run_pipeline_followups(project_name, X, change_count, project_dir, primary_model_name, primary_fixed_params,
                           features_graph=None):
    """Run RQ1 ablation, baseline comparison, repeated stratified CV and ML-side alpha sensitivity for one project.

    Uses the primary model with this project's tuned hyperparameters. X, change_count and project_dir
    come from run_pipeline(). features_graph is reused by the alpha analysis to avoid recomputation.
    """
    configure_logging(project_name)  # re-point PROJECT_NAME/project_dir/log buffer at this project
    logger.info("=" * 70)
    logger.info(f"[{project_name}] Follow-up analyses (RQ1 ablation / repeated CV / alpha sensitivity)")
    logger.info(f"[{project_name}] Primary model = {primary_model_name}  |  fixed_params = {primary_fixed_params}")
    logger.info("=" * 70)
    followups_start_time = time.time()

    # RQ1 ablation (leave-one-view-out)
    run_rq1_ablation(X, change_count, project_name, project_dir,
                     train_classification_model, evaluate_classification_model,
                     model_name=primary_model_name, fixed_params=primary_fixed_params)

    # Size / past-change baselines under the same protocol as the ablation (supplementary analysis)
    if RUN_BASELINE_COMPARISON:
        try:
            run_baseline_comparison(X, change_count, project_name, project_dir,
                                    train_classification_model, evaluate_classification_model,
                                    model_name=primary_model_name, fixed_params=primary_fixed_params)
        except Exception as e:
            logger.error(f"Baseline comparison failed for '{project_name}': {e}")

    # Supplementary analyses requested by the review (view combinations, module-grouped split, population)
    if RUN_VIEW_COMBINATIONS:
        try:
            run_view_combinations(X, change_count, project_name, project_dir, train_classification_model,
                                  evaluate_classification_model, primary_model_name, primary_fixed_params)
        except Exception as e:
            logger.error(f"View-combination analysis failed for '{project_name}': {e}")
    if RUN_GROUP_SPLIT_ANALYSIS:
        try:
            run_group_split_analysis(X, change_count, project_name, project_dir, train_classification_model,
                                     evaluate_classification_model, primary_model_name, primary_fixed_params)
        except Exception as e:
            logger.error(f"Group-split analysis failed for '{project_name}': {e}")
    try:
        run_size_matched_analysis(X, change_count, project_name, project_dir, train_classification_model,
                                  evaluate_classification_model, primary_model_name, primary_fixed_params)
    except Exception as e:
        logger.error(f"Size-matched analysis failed for '{project_name}': {e}")
    if RUN_POPULATION_ANALYSIS:
        try:
            run_population_analysis(project_name, project_dir, features_graph, train_classification_model,
                                    evaluate_classification_model, primary_model_name, primary_fixed_params)
        except Exception as e:
            logger.error(f"Population analysis failed for '{project_name}': {e}")

    # Repeated stratified K-fold, reported alongside the single split
    try:
        run_repeated_stratified_cv(X, change_count, project_name, project_dir,
                                   model_name=primary_model_name, fixed_params=primary_fixed_params,
                                   n_features=15)
    except Exception as e:
        logger.error(f"Repeated stratified CV failed for '{project_name}': {e}")

    # ML-side alpha sensitivity; needs propagation_frequencies_alpha_<a>.pkl from the dependency-graph
    # script (missing files are skipped with a warning).
    try:
        run_alpha_sensitivity_ml_analysis(project_name, n_features=15,
                                          model_name=primary_model_name, fixed_params=primary_fixed_params,
                                          features_graph=features_graph)
    except Exception as e:
        logger.error(f"Alpha sensitivity ML analysis failed for '{project_name}': {e}")

    logger.info(f"[{project_name}] Follow-up analyses completed in "
                f"{time.time() - followups_start_time:.2f}s.")

    # Appends to the same 'Result (<project>) Log.txt' written by run_pipeline
    save_console_log_to_file()


def build_cross_project_summary_table(projects, export_dir):
    """Aggregate each project's comparison_table.csv into a cross-project summary (mean +/- std per model, Dummy baselines included).

    Writes '<export_dir>/cross_project_summary.csv' and returns it, sorted by mean Balanced Accuracy (descending).
    """
    metric_cols = [
        'Balanced Accuracy', 'PR AUC', 'F1 Macro', 'AUC-ROC',
        'Change-Count RMSE', 'Change-Count Spearman',
        'Mean Architecture Size (Unstable)', 'Mean Architecture Cost (Unstable)',
    ]

    per_project_rows = []
    missing_projects = []
    for project_name in projects:
        csv_path = os.path.join(export_dir, project_name, "comparison_table.csv")
        if not os.path.exists(csv_path):
            missing_projects.append(project_name)
            continue
        df = pd.read_csv(csv_path)
        df = df.copy()
        df["Project"] = project_name
        per_project_rows.append(df[["Project", "Model"] + metric_cols])

    if not per_project_rows:
        logger.warning(f"build_cross_project_summary_table: no comparison_table.csv found for any of "
                       f"{projects} in '{export_dir}' - skipping cross-project summary.")
        return None

    all_df = pd.concat(per_project_rows, ignore_index=True)

    agg = all_df.groupby("Model")[metric_cols].agg(['mean', 'std', 'count'])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg = agg.reset_index().rename(columns={"Balanced Accuracy_count": "N_Projects"})
    # Drop the redundant per-metric count columns; keep one
    agg = agg.drop(columns=[c for c in agg.columns if c.endswith("_count") and c != "N_Projects"])
    agg = agg.sort_values("Balanced Accuracy_mean", ascending=False).reset_index(drop=True)

    logger.info("=" * 70)
    logger.info(f"CROSS-PROJECT SUMMARY (mean +/- std across projects {projects})")
    logger.info("\n" + agg.to_string(index=False))
    if missing_projects:
        logger.warning(f"build_cross_project_summary_table: comparison_table.csv missing for "
                       f"{missing_projects} - excluded from this summary.")
    logger.info("=" * 70)

    out_path = os.path.join(export_dir, "cross_project_summary.csv")
    try:
        agg.to_csv(out_path, index=False)
        logger.info(f"Cross-project summary saved to '{out_path}'")
    except Exception as e:
        logger.warning(f"Could not save cross_project_summary.csv: {e}")

    return agg


def select_primary_model(projects, export_dir, candidate_models=None):
    """Choose the primary model held constant across ablation, repeated CV and alpha sensitivity, plus a top-3 list for LOPO.

    Rule: highest mean Balanced Accuracy across the projects' comparison_table.csv (Dummy baselines excluded);
    ties broken by mean within-project rank, then alphabetically.
    Returns a dict (primary_model, top3_models_for_lopo, ranking, ...) and writes '<export_dir>/primary_model_selection.json'.
    """
    if candidate_models is None:
        candidate_models = CANDIDATE_MODEL_NAMES

    per_project_rows = []
    missing_projects = []
    metric_used = set()
    for project_name in projects:
        csv_path = os.path.join(export_dir, project_name, "comparison_table.csv")
        cv_path = os.path.join(export_dir, project_name, "train_cv_scores.csv")
        if SELECT_PRIMARY_BY_TRAIN_CV and os.path.exists(cv_path):
            df = pd.read_csv(cv_path).rename(columns={"Train_CV_Balanced_Accuracy": "Balanced Accuracy"})
            metric_used.add("train_cv")
        elif os.path.exists(csv_path):
            if SELECT_PRIMARY_BY_TRAIN_CV:
                logger.warning(f"select_primary_model: '{cv_path}' missing - falling back to TEST balanced accuracy "
                               f"for {project_name} (this reintroduces test-set model selection).")
            df = pd.read_csv(csv_path)
            metric_used.add("test")
        else:
            missing_projects.append(project_name)
            continue
        df = df[df["Model"].isin(candidate_models)].copy()
        df["Project"] = project_name
        per_project_rows.append(df[["Project", "Model", "Balanced Accuracy"]])

    if not per_project_rows:
        raise RuntimeError(
            f"select_primary_model: no comparison_table.csv found for any of {projects} in '{export_dir}'. "
            f"Run run_pipeline() for every project first."
        )

    all_df = pd.concat(per_project_rows, ignore_index=True)

    # Mean Balanced Accuracy per model across whichever projects it has a row in.
    mean_ba = all_df.groupby("Model")["Balanced Accuracy"].mean().sort_values(ascending=False)
    project_counts = all_df.groupby("Model")["Project"].nunique()

    # Mean within-project rank (1 = best), used only to break ties
    all_df["Rank_In_Project"] = all_df.groupby("Project")["Balanced Accuracy"].rank(ascending=False, method="min")
    mean_rank = all_df.groupby("Model")["Rank_In_Project"].mean()

    ranking = (
        pd.DataFrame({"Mean_Balanced_Accuracy": mean_ba, "N_Projects": project_counts, "Mean_Rank": mean_rank})
        .sort_values(["Mean_Balanced_Accuracy", "Mean_Rank"], ascending=[False, True])
    )
    ranking = ranking.sort_index().sort_values(["Mean_Balanced_Accuracy", "Mean_Rank"], ascending=[False, True])

    ranked_models = list(ranking.index)
    if not ranked_models:
        raise RuntimeError("select_primary_model: no candidate models found in any comparison_table.csv.")

    primary_model = ranked_models[0]
    top3_models = ranked_models[:3]

    logger.info("=" * 70)
    logger.info("PRIMARY MODEL SELECTION (for RQ1 Ablation / Repeated CV / Alpha Sensitivity)")
    logger.info(f"Ranking by mean Balanced Accuracy across projects {projects}:")
    logger.info("\n" + ranking.to_string())
    logger.info(f"--> PRIMARY MODEL = {primary_model}")
    logger.info(f"--> TOP-3 MODELS FOR MULTI-MODEL LOPO = {top3_models}")
    if missing_projects:
        logger.warning(f"select_primary_model: comparison_table.csv missing for {missing_projects} "
                       f"- these projects were excluded from the ranking above.")
    logger.info("=" * 70)

    result = {
        "selection_metric": ("mean 5-fold CV balanced accuracy on the TRAINING partition (train_cv_scores.csv)"
                             if metric_used == {"train_cv"} else
                             f"MIXED/TEST balanced accuracy ({sorted(metric_used)}) - not leakage-free"),
        "selection_rule": "argmax of the mean (across projects) selection metric, "
                          "ties broken by mean within-project rank, then alphabetically",
        "candidate_models": candidate_models,
        "primary_model": primary_model,
        "top3_models_for_lopo": top3_models,
        "ranking": ranking.reset_index().rename(columns={"index": "Model"}).to_dict(orient="records"),
        "missing_projects": missing_projects,
    }

    out_path = os.path.join(export_dir, "primary_model_selection.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Primary model selection saved to '{out_path}'")
    except Exception as e:
        logger.warning(f"Could not save primary_model_selection.json: {e}")

    return result


# Outputs written by THIS script (and by script 6). Removed at the start of a run so no stale file survives.
# Inputs of scripts 0-4 (code_metrics.pkl, dependency_graph.pkl, propagation_frequencies*.pkl,
# class_change_frequency*.pkl, alpha_sensitivity_propagation_distribution.csv,
# alpha_sensitivity_ranking_stability.csv) are never matched by these patterns.
CLEAN_ON_START = True
_OUTPUT_PATTERNS_PROJECT = [
    "*.png", "calibration_bins_*.csv", "calibration_summary.csv", "comparison_table.csv",
    "architecture_factors_quantile_table.csv", "baselines_*", "rq1_ablation_*", "repeated_cv_*",
    "alpha_sensitivity_ml_*", "feature_importance*", "mcnemar_pairwise.csv",
    "bootstrap_ci_balanced_accuracy.csv", "train_only_threshold_info.json",
    "tuned_hyperparams_*.json", "best_model_git_based.pkl", "git_stability_results.log",
    "Result (*) Log.txt", "Project_Report_*.txt", "viewcombos_*", "groupsplit_*", "population_*", "sizematched_*",
    "train_cv_scores.csv",
]
_OUTPUT_PATTERNS_ROOT = ["lopo_cross_project_*.csv", "cross_project_summary.csv",
                         "primary_model_selection.json", "Project_Report_ALL_PROJECTS.txt", "stat_tests_corrected_*.csv",
                         "meta_analysis_*.csv", "paper_figures/*.png"]


def clean_previous_outputs(projects, export_dir):
    import glob
    removed = 0
    targets = [(export_dir, _OUTPUT_PATTERNS_ROOT)] + \
              [(os.path.join(export_dir, p), _OUTPUT_PATTERNS_PROJECT) for p in projects]
    for base, patterns in targets:
        for pat in patterns:
            for f in glob.glob(os.path.join(glob.escape(base), pat)):
                if os.path.isfile(f):
                    try:
                        os.remove(f)
                        removed += 1
                    except OSError as e:
                        print(f"Could not remove {f}: {e}")
    print(f"Cleaned {removed} output files from previous runs (input pickles/alpha files kept).")


if __name__ == "__main__":
    # Settings: project names must match the folders created by the previous scripts
    PROJECTS = ["spring-framework", "elasticsearch", "hadoop"]

    if CLEAN_ON_START:
        clean_previous_outputs(PROJECTS, EXPORT_DIR)

    overall_start_time = time.time()

    # Pass 1: main model comparison for every project
    pipeline_cache = {}  # project_name -> (X, change_count, y_stability, full_df_for_reporting, project_dir, features_graph)
    for proj_idx, project_name in enumerate(PROJECTS):
        print(f"\n=== [{proj_idx + 1}/{len(PROJECTS)}] Starting project: {project_name} (main comparison) ===")
        try:
            result = run_pipeline(project_name)
            if result is not None:
                pipeline_cache[project_name] = result
        except Exception as e:
            # Continue with the next project if one fails
            logger.error(f"Pipeline failed for project '{project_name}': {e}")
            if console_log_buffer is not None and project_dir is not None:
                save_console_log_to_file()
        elapsed = time.time() - overall_start_time
        done = proj_idx + 1
        eta = (elapsed / done) * (len(PROJECTS) - done)
        print(f"=== [{done}/{len(PROJECTS)}] Finished project: {project_name} | "
              f"elapsed {elapsed:.1f}s | ETA for remaining projects: {eta:.1f}s ===")

    # Reuse the Pass 1 datasets for LOPO (avoids recomputing graph features)
    lopo_preloaded_datasets = {
        project_name: (X, change_count, y_stability, full_df_for_reporting)
        for project_name, (X, change_count, y_stability, full_df_for_reporting, _cached_project_dir, _cached_features_graph)
        in pipeline_cache.items()
    }

    # Choose the primary model (best mean balanced accuracy across projects)
    primary_model_name = None
    top3_models_for_lopo = ["RandomForest"]
    try:
        selection = select_primary_model(PROJECTS, EXPORT_DIR)
        primary_model_name = selection["primary_model"]
        top3_models_for_lopo = selection["top3_models_for_lopo"]
    except Exception as e:
        logger.error(f"select_primary_model failed, falling back to RandomForest: {e}")
        primary_model_name = "RandomForest"
        top3_models_for_lopo = ["RandomForest"]

    # Cross-project summary table (reporting only)
    try:
        build_cross_project_summary_table(PROJECTS, EXPORT_DIR)
    except Exception as e:
        logger.error(f"build_cross_project_summary_table failed: {e}")

    # Pass 2: follow-up analyses with the primary model
    for project_name, (X, change_count, y_stability, full_df_for_reporting, cached_project_dir, cached_features_graph) in pipeline_cache.items():
        print(f"\n=== Follow-up analyses for: {project_name} (model={primary_model_name}) ===")
        # Reuse this project's tuned hyperparameters for the primary model; fall back to library defaults
        primary_fixed_params = {}
        tuned_path = os.path.join(cached_project_dir, f"tuned_hyperparams_{project_name}.json")
        try:
            with open(tuned_path, "r", encoding="utf-8") as f:
                tuned = json.load(f)
            primary_fixed_params = tuned.get(primary_model_name, {}) or {}
        except Exception as e:
            logger.warning(f"[{project_name}] Could not load '{tuned_path}' for primary model "
                           f"'{primary_model_name}', using library defaults: {e}")
        try:
            run_pipeline_followups(project_name, X, change_count, cached_project_dir,
                                   primary_model_name, primary_fixed_params,
                                   features_graph=cached_features_graph)
        except Exception as e:
            logger.error(f"Follow-up analyses failed for project '{project_name}': {e}")
            if console_log_buffer is not None:
                save_console_log_to_file()

    # Nadeau-Bengio corrected tests + Holm per family (needs the per-repeat files of all projects)
    try:
        build_corrected_stat_tests(PROJECTS, EXPORT_DIR, primary_model_name)
    except Exception as e:
        logger.error(f"build_corrected_stat_tests failed: {e}")

    # Publication figures (forest, baselines, view importance, two radar charts) from the files written above
    try:
        generate_paper_figures(PROJECTS, EXPORT_DIR, primary_model_name)
    except Exception as e:
        logger.error(f"generate_paper_figures failed: {e}")

    # Leave-One-Project-Out evaluation for the top-3 models, reported alongside the within-project results
    for lopo_model_name in top3_models_for_lopo:
        print(f"\n=== LOPO cross-project evaluation (model={lopo_model_name}) ===")
        try:
            # Reuse the Pass 1 datasets instead of recomputing graph features
            run_leave_one_project_out_pipeline(PROJECTS, model_name=lopo_model_name,
                                               preloaded_datasets=lopo_preloaded_datasets)
        except Exception as e:
            logger.error(f"LOPO cross-project evaluation failed for model '{lopo_model_name}': {e}")
