"""DECAT decision tree ablation study.

Evaluates how each metric contributes to the classification decision tree
by systematically ablating individual components and measuring accuracy
degradation.
"""

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from decision.scenario_classifier import MetricBundle, classify_scenario


_MAIN_DIR = Path(os.environ.get('DECAT_RESULTS_DIR', REPO_ROOT / 'data/main_experiment'))
RESULTS_DIR = _MAIN_DIR / 'results'
OUTPUT_DIR = _MAIN_DIR / 'ablation'
OUTPUT_CSV = OUTPUT_DIR / 'ablation_table.csv'

PURE_PREFIXES = ('s1_', 's2b_', 's3_', 's4h_', 's4r_')

# Ground-truth scenario mapping from alpha_config_id prefix
PREFIX_TO_SCENARIO = {
    's1_': 'scenario_1',
    's2b_': 'scenario_2',
    's3_': 'scenario_3',
    's4h_': 'scenario_4',
    's4r_': 'scenario_4',
}

# Model family mapping
FACTORIZED_MODELS = {'jive_K50'}  # dssl_* also factorized, handled below

ABLATION_CONDITIONS = [
    'Full',
    '-D_quantile',
    '-Delta_shared',
    '-A_norm',
    '-P_transfer',
]


def is_factorized(model_name: str) -> bool:
    """Determine if a model is factorized (provides shared/specific split)."""
    return model_name == 'jive_K50' or model_name.startswith('dssl_')


def get_model_family(model_name: str) -> str:
    """Map model name to family label."""
    if model_name == 'cca_K50':
        return 'CCA'
    elif model_name == 'clip_K50':
        return 'CLIP'
    elif model_name == 'jive_K50':
        return 'JIVE'
    elif model_name.startswith('dssl_'):
        return 'DSSL'
    else:
        return 'OTHER'


def get_ground_truth(alpha_config_id: str) -> str:
    """Derive ground-truth scenario from alpha_config_id prefix."""
    for prefix, scenario in PREFIX_TO_SCENARIO.items():
        if alpha_config_id.startswith(prefix):
            return scenario
    raise ValueError(f"Unknown alpha_config_id prefix: {alpha_config_id}")


def parse_is_proxy(filename: str) -> bool:
    """Determine if a file is a proxy run from its filename."""
    return filename.startswith('proxy')


def _parse_regime(filename: str) -> str:
    """Extract measurement_regime from result filename."""
    parts = filename.split('_')
    return '_'.join(parts[1:-1])


def load_all_records(n_eval: int = 1000,
                     regime: str = 'baseline') -> list[dict]:
    """Load result JSON files, filtering to a specific regime and N_eval."""
    records = []
    for fpath in sorted(RESULTS_DIR.glob('*.json')):
        fname = fpath.stem
        if parse_is_proxy(fname):
            continue
        if _parse_regime(fname) != regime:
            continue
        with open(fpath) as f:
            data = json.load(f)
        for rec in data:
            if rec['n_eval'] != n_eval:
                continue
            if not rec['alpha_config_id'].startswith(PURE_PREFIXES):
                continue
            rec['_source_file'] = fname
            records.append(rec)
    return records


def build_metric_bundle(rec: dict) -> MetricBundle:
    """Reconstruct a MetricBundle from a result record."""
    model = rec['model']
    factorized = is_factorized(model)
    return MetricBundle(
        anorm=rec['anorm'],
        anorm_null_upper=rec['anorm_null_upper'],
        signal_present=rec['signal_present'],
        transfer_ci_lower=rec['transfer_ci_lower'],
        transfer_null_upper=rec['transfer_null_upper'],
        dquantile_task=rec['dquantile_task'],
        dquantile_task_null_upper=rec['dquantile_task_null_upper'],
        is_factorized=factorized,
        delta_ci_lower=rec.get('delta_ci_lower'),
        delta_ci_upper=rec.get('delta_ci_upper'),
        delta_null_lower=rec.get('delta_null_lower'),
        delta_null_upper=rec.get('delta_null_upper'),
    )


def apply_ablation(bundle: MetricBundle, condition: str) -> MetricBundle:
    """Return a modified MetricBundle for the given ablation condition."""
    if condition == 'Full':
        return bundle

    # Deep copy to avoid mutating original
    from dataclasses import asdict
    fields = asdict(bundle)

    if condition == '-D_quantile':
        # Force dquantile_task = 0.0 → always within null (since 0 < null_upper)
        fields['dquantile_task'] = 0.0

    elif condition == '-Delta_shared':
        # Force all models to be non-factorized (removes localization step)
        fields['is_factorized'] = False

    elif condition == '-A_norm':
        # Force anorm to never exceed null: removes the alignment confirmation
        # gate so S1 can never be reached via this condition
        fields['anorm'] = 0.0

    elif condition == '-P_transfer':
        # Force transfer_ci_lower = 0.0 → always fails (since 0 < transfer_null_upper)
        fields['transfer_ci_lower'] = 0.0

    else:
        raise ValueError(f"Unknown ablation condition: {condition}")

    return MetricBundle(**fields)


def run_ablation_analysis(records: list[dict]) -> tuple[dict, dict]:
    """Run all ablation conditions and compute per-scenario and per-family accuracy."""

    # Structure: {condition: {ground_truth: {'correct': n, 'total': n}}}
    results_by_condition = {}
    # Per model family breakdown
    results_by_family = {}

    for condition in ABLATION_CONDITIONS:
        results_by_condition[condition] = defaultdict(lambda: {'correct': 0, 'total': 0})
        results_by_family[condition] = defaultdict(
            lambda: defaultdict(lambda: {'correct': 0, 'total': 0})
        )

    for rec in records:
        ground_truth = get_ground_truth(rec['alpha_config_id'])
        model_family = get_model_family(rec['model'])
        bundle = build_metric_bundle(rec)

        for condition in ABLATION_CONDITIONS:
            ablated_bundle = apply_ablation(bundle, condition)
            pred = classify_scenario(ablated_bundle)
            correct = (pred == ground_truth)

            results_by_condition[condition][ground_truth]['total'] += 1
            if correct:
                results_by_condition[condition][ground_truth]['correct'] += 1

            results_by_family[condition][model_family][ground_truth]['total'] += 1
            if correct:
                results_by_family[condition][model_family][ground_truth]['correct'] += 1

    return results_by_condition, results_by_family


def compute_fscr(records: list[dict], condition: str) -> float:
    """Compute FSCR: fraction of non-S1 ground-truth cases classified as scenario_1."""
    non_s1_total = 0
    classified_s1 = 0
    for rec in records:
        ground_truth = get_ground_truth(rec['alpha_config_id'])
        if ground_truth == 'scenario_1':
            continue
        non_s1_total += 1
        bundle = build_metric_bundle(rec)
        ablated = apply_ablation(bundle, condition)
        pred = classify_scenario(ablated)
        if pred == 'scenario_1':
            classified_s1 += 1
    return classified_s1 / non_s1_total if non_s1_total > 0 else 0.0


def compute_indeterminate_rate(records: list[dict], condition: str) -> float:
    """Compute indeterminate rate across all records."""
    indet_count = 0
    for rec in records:
        bundle = build_metric_bundle(rec)
        ablated = apply_ablation(bundle, condition)
        pred = classify_scenario(ablated)
        if pred == 'indeterminate':
            indet_count += 1
    return indet_count / len(records) if records else 0.0


def sanity_check(records: list[dict]) -> int:
    """Verify that classify_scenario reproduces stored predicted_scenario."""
    mismatches = 0
    for rec in records:
        bundle = build_metric_bundle(rec)
        pred = classify_scenario(bundle)
        stored = rec['predicted_scenario']
        if pred != stored:
            mismatches += 1
    return mismatches


def print_results(results_by_condition: dict, results_by_family: dict,
                  records: list[dict]) -> list[dict]:
    """Print formatted tables to stdout."""

    scenarios = ['scenario_1', 'scenario_2', 'scenario_3', 'scenario_4']
    scenario_labels = {'scenario_1': 'S1', 'scenario_2': 'S2B',
                       'scenario_3': 'S3', 'scenario_4': 'S4'}

    print("=" * 80)
    print("DECAT DECISION TREE ABLATION STUDY")
    print("=" * 80)
    print(f"\nTotal records: {len(records)}")
    print(f"N_eval=1000, baseline (non-proxy), pure scenarios only")
    print()

    # Main accuracy table
    print("-" * 80)
    print("PER-SCENARIO STRICT ACCURACY")
    print("-" * 80)
    header = f"{'Condition':<16}" + "".join(f"{'Acc_' + scenario_labels[s]:<12}" for s in scenarios) + f"{'Overall':<12}{'FSCR':<10}{'Indet%':<10}"
    print(header)
    print("-" * len(header))

    rows_for_csv = []

    for condition in ABLATION_CONDITIONS:
        cond_data = results_by_condition[condition]
        accs = []
        row = {'condition': condition}
        for s in scenarios:
            d = cond_data[s]
            acc = d['correct'] / d['total'] if d['total'] > 0 else 0.0
            accs.append(acc)
            row[f'acc_{scenario_labels[s]}'] = acc

        overall_correct = sum(cond_data[s]['correct'] for s in scenarios)
        overall_total = sum(cond_data[s]['total'] for s in scenarios)
        overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0

        fscr = compute_fscr(records, condition)
        indet = compute_indeterminate_rate(records, condition)

        row['overall_acc'] = overall_acc
        row['fscr'] = fscr
        row['indeterminate_rate'] = indet

        line = f"{condition:<16}" + "".join(f"{a:<12.3f}" for a in accs) + f"{overall_acc:<12.3f}{fscr:<10.3f}{indet:<10.3f}"
        print(line)
        rows_for_csv.append(row)

    print()

    # Per model family breakdown
    print("-" * 80)
    print("PER MODEL FAMILY BREAKDOWN (Overall Accuracy)")
    print("-" * 80)
    families = ['CCA', 'CLIP', 'JIVE', 'DSSL']
    header2 = f"{'Condition':<16}" + "".join(f"{fam:<10}" for fam in families)
    print(header2)
    print("-" * len(header2))

    for condition in ABLATION_CONDITIONS:
        parts = [f"{condition:<16}"]
        for fam in families:
            fam_data = results_by_family[condition][fam]
            correct = sum(fam_data[s]['correct'] for s in scenarios)
            total = sum(fam_data[s]['total'] for s in scenarios)
            acc = correct / total if total > 0 else 0.0
            parts.append(f"{acc:<10.3f}")

            # Add to CSV row
            for r in rows_for_csv:
                if r['condition'] == condition:
                    r[f'acc_{fam}'] = acc
        print("".join(parts))

    print()

    # Detailed per-family per-scenario table
    print("-" * 80)
    print("DETAILED PER FAMILY x SCENARIO (Full condition)")
    print("-" * 80)
    header3 = f"{'Family':<10}" + "".join(f"{'Acc_' + scenario_labels[s]:<12}" for s in scenarios) + f"{'Overall':<10}"
    print(header3)
    print("-" * len(header3))
    for fam in families:
        fam_data = results_by_family['Full'][fam]
        parts = [f"{fam:<10}"]
        for s in scenarios:
            d = fam_data[s]
            acc = d['correct'] / d['total'] if d['total'] > 0 else 0.0
            parts.append(f"{acc:<12.3f}")
        correct = sum(fam_data[s]['correct'] for s in scenarios)
        total = sum(fam_data[s]['total'] for s in scenarios)
        overall = correct / total if total > 0 else 0.0
        parts.append(f"{overall:<10.3f}")
        print("".join(parts))

    print()
    return rows_for_csv


def save_csv(rows: list[dict], output_path: Path) -> None:
    """Save results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    # Ensure all keys from all rows are captured
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for k, v in row.items():
                if isinstance(v, float):
                    formatted[k] = f"{v:.4f}"
                else:
                    formatted[k] = v
            writer.writerow(formatted)
    print(f"Results saved to: {output_path}")


def main() -> None:
    """Run the full ablation study and save results."""
    print("Loading data...")
    records = load_all_records()
    print(f"Loaded {len(records)} records from {RESULTS_DIR}")
    print(f"  (N_eval=1000, regime=baseline, pure scenarios)")
    print()

    # Sanity check
    print("Running sanity check: classify_scenario vs stored predicted_scenario...")
    mismatches = sanity_check(records)
    print(f"  Mismatches: {mismatches} / {len(records)}")
    if mismatches > 0:
        print("  WARNING: Some predictions do not match stored values!")
    else:
        print("  All predictions match. Decision tree is consistent.")
    print()

    # Run ablation
    print("Running ablation analysis...")
    results_by_condition, results_by_family = run_ablation_analysis(records)
    print()

    # Print and save
    rows = print_results(results_by_condition, results_by_family, records)
    save_csv(rows, OUTPUT_CSV)

    # Stratified ablation: non-factorized vs factorized (all scenarios)
    eval_scenarios = ['scenario_1', 'scenario_2', 'scenario_3', 'scenario_4']
    print("\nSTRATIFIED ABLATION")
    strat_rows = []
    for model_group, group_label in [
        (['cca_K50', 'clip_K50'], 'non_factorized'),
        (['jive_K50'] + [m for m in set(r['model'] for r in records)
                         if m.startswith('dssl_')], 'factorized'),
    ]:
        subset = [r for r in records
                  if r['model'] in model_group
                  and get_ground_truth(r['alpha_config_id']) in eval_scenarios]
        nice_label = 'Non-factorized (CCA, CLIP)' if 'non' in group_label else 'Factorized (JIVE, DSSL)'
        print(f"\n  {nice_label} — {len(subset)} records")
        print(f"  {'Condition':<16}{'Overall':<10}{'S1':<10}{'S2':<10}{'S3':<10}{'S4':<10}{'FSCR':<10}")
        for condition in ABLATION_CONDITIONS:
            counts = {s: [0, 0] for s in eval_scenarios}
            non_s1_total = 0
            classified_s1 = 0
            for rec in subset:
                gt = get_ground_truth(rec['alpha_config_id'])
                bundle = build_metric_bundle(rec)
                ablated = apply_ablation(bundle, condition)
                pred = classify_scenario(ablated)
                counts[gt][1] += 1
                if pred == gt:
                    counts[gt][0] += 1
                if gt != 'scenario_1':
                    non_s1_total += 1
                    if pred == 'scenario_1':
                        classified_s1 += 1
            overall_c = sum(v[0] for v in counts.values())
            overall_t = sum(v[1] for v in counts.values())
            overall = overall_c / overall_t if overall_t > 0 else 0.0
            fscr = classified_s1 / non_s1_total if non_s1_total > 0 else 0.0
            parts = f"  {condition:<16}{overall:<10.3f}"
            for s in eval_scenarios:
                acc = counts[s][0] / counts[s][1] if counts[s][1] > 0 else 0.0
                parts += f"{acc:<10.3f}"
            parts += f"{fscr:<10.3f}"
            print(parts)

            row = {'model_group': group_label, 'condition': condition,
                   'overall_acc': overall, 'fscr': fscr}
            for s in eval_scenarios:
                label = s.replace('scenario_', 'acc_S')
                row[label] = counts[s][0] / counts[s][1] if counts[s][1] > 0 else 0.0
            strat_rows.append(row)

    strat_csv = OUTPUT_DIR / 'ablation_stratified.csv'
    save_csv(strat_rows, strat_csv)

    # A_norm sensitivity across cohort sizes
    print("\nA_norm ablation impact across cohort sizes:")
    print(f"  {'N_eval':<8}{'Full_acc':<12}{'-A_norm_acc':<12}{'Delta':<10}")
    anorm_rows = []
    for n in [50, 100, 200, 500, 1000]:
        recs_n = load_all_records(n_eval=n)
        if not recs_n:
            continue
        full_correct = 0
        abl_correct = 0
        for rec in recs_n:
            gt = get_ground_truth(rec['alpha_config_id'])
            bundle = build_metric_bundle(rec)
            if classify_scenario(bundle) == gt:
                full_correct += 1
            ablated = apply_ablation(bundle, '-A_norm')
            if classify_scenario(ablated) == gt:
                abl_correct += 1
        if not recs_n:
            continue
        full_acc = full_correct / len(recs_n)
        abl_acc = abl_correct / len(recs_n)
        delta = abl_acc - full_acc
        print(f"  {n:<8}{full_acc:<12.3f}{abl_acc:<12.3f}{delta:<+10.3f}")
        anorm_rows.append({'n_eval': n, 'full_acc': full_acc,
                           'anorm_ablated_acc': abl_acc, 'delta': delta})

    # Save A_norm cohort-size table
    anorm_csv = OUTPUT_DIR / 'ablation_anorm_by_n.csv'
    if anorm_rows:
        save_csv(anorm_rows, anorm_csv)


if __name__ == '__main__':
    main()
