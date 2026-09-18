"""Leak-free C1/C2/C3 data splitting protocol.

This module intentionally depends only on NumPy and pandas so the complete
split protocol can be validated before model or GPU initialization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


GENE_A = 'Gene.A'
GENE_B = 'Gene.B'
LABEL = 'label'
_A = '__protocol_gene_a'
_B = '__protocol_gene_b'
_PAIR = '__protocol_pair'


@dataclass(frozen=True)
class SplitReport:
    scenario: str
    train_size: int
    val_size: int
    test_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    train_gene_count: int
    val_new_gene_count: int
    test_new_gene_count: int


def _validate_input(data_df: pd.DataFrame) -> None:
    required = {GENE_A, GENE_B, LABEL}
    missing = required.difference(data_df.columns)
    if missing:
        raise ValueError(f"Split input is missing columns: {sorted(missing)}")
    if data_df.empty:
        raise ValueError("Split input is empty")
    if data_df[[GENE_A, GENE_B, LABEL]].isna().any().any():
        raise ValueError("Split input contains missing gene IDs or labels")
    invalid_labels = set(data_df[LABEL].unique()).difference({0, 1, 0.0, 1.0})
    if invalid_labels:
        raise ValueError(f"Split input contains non-binary labels: {invalid_labels}")


def _prepare(data_df: pd.DataFrame) -> pd.DataFrame:
    _validate_input(data_df)
    prepared = data_df.copy().reset_index(drop=True)
    prepared[_A] = prepared[GENE_A].astype(str).str.strip()
    prepared[_B] = prepared[GENE_B].astype(str).str.strip()
    prepared[_PAIR] = [
        tuple(sorted((gene_a, gene_b)))
        for gene_a, gene_b in zip(prepared[_A], prepared[_B])
    ]
    return prepared


def _clean(data_df: pd.DataFrame) -> pd.DataFrame:
    return data_df.drop(columns=[_A, _B, _PAIR], errors='ignore').reset_index(drop=True)


def _random_equal_buckets(items, n_folds: int, seed: int):
    ordered = sorted(items, key=str)
    rng = np.random.RandomState(seed)
    rng.shuffle(ordered)
    buckets = [set() for _ in range(n_folds)]
    for index, item in enumerate(ordered):
        buckets[index % n_folds].add(item)
    return buckets


def _balanced_pair_buckets(prepared: pd.DataFrame, n_folds: int, seed: int):
    pair_weights = prepared.groupby(_PAIR, sort=False).size().to_dict()
    pairs = list(pair_weights)
    rng = np.random.RandomState(seed)
    rng.shuffle(pairs)
    pairs.sort(key=lambda pair: pair_weights[pair], reverse=True)

    buckets = [set() for _ in range(n_folds)]
    bucket_rows = [0] * n_folds
    for pair in pairs:
        target_bucket = min(range(n_folds), key=lambda idx: bucket_rows[idx])
        buckets[target_bucket].add(pair)
        bucket_rows[target_bucket] += int(pair_weights[pair])
    return buckets


def _allocate_stratified_counts(data_df: pd.DataFrame, sample_size: int):
    counts = data_df[LABEL].value_counts(sort=False).to_dict()
    labels = sorted(counts, key=str)
    raw = {
        label: sample_size * counts[label] / len(data_df)
        for label in labels
    }
    allocation = {
        label: min(counts[label], int(np.floor(raw[label])))
        for label in labels
    }

    if sample_size >= len(labels):
        for label in labels:
            if allocation[label] == 0 and counts[label] > 0:
                allocation[label] = 1

    while sum(allocation.values()) > sample_size:
        removable = [
            label for label in labels
            if allocation[label] > (1 if sample_size >= len(labels) else 0)
        ]
        if not removable:
            break
        label = max(removable, key=lambda item: allocation[item] - raw[item])
        allocation[label] -= 1

    while sum(allocation.values()) < sample_size:
        expandable = [
            label for label in labels if allocation[label] < counts[label]
        ]
        if not expandable:
            break
        label = max(expandable, key=lambda item: raw[item] - allocation[item])
        allocation[label] += 1

    if sum(allocation.values()) != sample_size:
        raise RuntimeError("Could not allocate an exact stratified sample")
    return allocation


def _stratified_sample(data_df: pd.DataFrame, sample_size: int, seed: int):
    if sample_size <= 0:
        raise ValueError("Requested split sample size must be positive")
    if sample_size > len(data_df):
        raise ValueError(
            f"Requested {sample_size} rows from a pool of {len(data_df)}"
        )
    if sample_size == len(data_df):
        return data_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    allocation = _allocate_stratified_counts(data_df, sample_size)
    sampled_parts = []
    for offset, label in enumerate(sorted(allocation, key=str)):
        count = allocation[label]
        if count:
            sampled_parts.append(
                data_df[data_df[LABEL] == label].sample(
                    n=count,
                    replace=False,
                    random_state=seed + 1009 * (offset + 1)
                )
            )
    return pd.concat(sampled_parts, ignore_index=True).sample(
        frac=1, random_state=seed + 7919
    ).reset_index(drop=True)


def _sample_train_with_coverage(
    train_pool: pd.DataFrame,
    sample_size: int,
    required_genes,
    seed: int
):
    if sample_size >= len(train_pool):
        return train_pool.sample(frac=1, random_state=seed).reset_index(drop=True)

    working = train_pool.sample(frac=1, random_state=seed).reset_index(drop=True)
    uncovered = {str(gene_id) for gene_id in required_genes}
    selected_positions = []
    for position, row in working.iterrows():
        endpoints = {row[_A], row[_B]}
        if endpoints & uncovered:
            selected_positions.append(position)
            uncovered.difference_update(endpoints)
            if not uncovered or len(selected_positions) == sample_size:
                break

    selected = working.iloc[selected_positions]
    remaining = working.drop(index=selected_positions)
    fill_size = sample_size - len(selected)
    if fill_size > 0:
        fill = _stratified_sample(remaining, fill_size, seed + 1543)
        selected = pd.concat([selected, fill], ignore_index=True)

    if len(selected) != sample_size:
        raise RuntimeError("Training coverage sampling did not return the target size")
    return selected.sample(frac=1, random_state=seed + 3571).reset_index(drop=True)


def _observed_genes(data_df: pd.DataFrame):
    return set(data_df[_A]).union(set(data_df[_B]))


def _filter_c2_edges(
    data_df: pd.DataFrame,
    observed_train_genes,
    heldout_genes
):
    a_seen = data_df[_A].isin(observed_train_genes)
    b_seen = data_df[_B].isin(observed_train_genes)
    a_new = data_df[_A].isin(heldout_genes)
    b_new = data_df[_B].isin(heldout_genes)
    return data_df[(a_seen & b_new) | (b_seen & a_new)].copy()


def _inductive_pools(
    prepared: pd.DataFrame,
    train_genes,
    val_genes,
    test_genes,
    scenario: str
):
    train_mask = prepared[_A].isin(train_genes) & prepared[_B].isin(train_genes)
    train_pool = prepared[train_mask].copy()
    observed_train_genes = _observed_genes(train_pool)

    if scenario == 'C2':
        val_pool = _filter_c2_edges(
            prepared, observed_train_genes, val_genes
        )
        test_pool = _filter_c2_edges(
            prepared, observed_train_genes, test_genes
        )
    elif scenario == 'C3':
        val_pool = prepared[
            prepared[_A].isin(val_genes) & prepared[_B].isin(val_genes)
        ].copy()
        test_pool = prepared[
            prepared[_A].isin(test_genes) & prepared[_B].isin(test_genes)
        ].copy()
    else:
        raise ValueError(f"Unsupported inductive scenario: {scenario}")

    return train_pool, val_pool, test_pool


def _balance_inductive_pools(
    train_pool: pd.DataFrame,
    val_pool: pd.DataFrame,
    test_pool: pd.DataFrame,
    val_genes,
    test_genes,
    scenario: str,
    seed: int
):
    unit_size = min(len(train_pool) // 8, len(val_pool), len(test_pool))
    if unit_size <= 0:
        raise ValueError(
            f"{scenario} has insufficient eligible edges: "
            f"train={len(train_pool)}, val={len(val_pool)}, test={len(test_pool)}"
        )

    base_val_pool = val_pool
    base_test_pool = test_pool
    for attempt in range(20):
        required_genes = set()
        if scenario == 'C2':
            candidate_genes = _observed_genes(base_val_pool).union(
                _observed_genes(base_test_pool)
            )
            required_genes = candidate_genes.difference(val_genes).difference(
                test_genes
            )

        train_df = _sample_train_with_coverage(
            train_pool,
            sample_size=8 * unit_size,
            required_genes=required_genes,
            seed=seed + attempt * 10007
        )
        observed_train_genes = _observed_genes(train_df)

        if scenario == 'C2':
            eligible_val = _filter_c2_edges(
                base_val_pool, observed_train_genes, val_genes
            )
            eligible_test = _filter_c2_edges(
                base_test_pool, observed_train_genes, test_genes
            )
        else:
            eligible_val = base_val_pool
            eligible_test = base_test_pool

        next_unit_size = min(
            len(train_df) // 8, len(eligible_val), len(eligible_test)
        )
        if next_unit_size <= 0:
            raise ValueError(
                f"{scenario} lost all eligible validation/test edges after "
                "enforcing observed training anchors"
            )
        if next_unit_size < unit_size:
            unit_size = next_unit_size
            continue

        val_df = _stratified_sample(eligible_val, unit_size, seed + 17)
        test_df = _stratified_sample(eligible_test, unit_size, seed + 29)
        return train_df, val_df, test_df

    raise RuntimeError(f"{scenario} 8:1:1 balancing did not converge")


def _build_inductive_fold(
    prepared: pd.DataFrame,
    train_genes,
    val_genes,
    test_genes,
    scenario: str,
    seed: int
):
    train_pool, val_pool, test_pool = _inductive_pools(
        prepared, train_genes, val_genes, test_genes, scenario
    )
    return _balance_inductive_pools(
        train_pool,
        val_pool,
        test_pool,
        val_genes,
        test_genes,
        scenario,
        seed
    )


def validate_protocol_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scenario: str
) -> SplitReport:
    train = _prepare(train_df)
    val = _prepare(val_df)
    test = _prepare(test_df)

    train_pairs = set(train[_PAIR])
    val_pairs = set(val[_PAIR])
    test_pairs = set(test[_PAIR])
    overlap = (
        (train_pairs & val_pairs)
        | (train_pairs & test_pairs)
        | (val_pairs & test_pairs)
    )
    if overlap:
        raise ValueError(f"Gene pairs overlap across splits: {sorted(overlap)[:5]}")

    train_genes = _observed_genes(train)
    val_genes = _observed_genes(val)
    test_genes = _observed_genes(test)
    val_new_genes = val_genes - train_genes
    test_new_genes = test_genes - train_genes

    if scenario in {'C2', 'C3'}:
        if len(train) != 8 * len(val) or len(val) != len(test):
            raise ValueError(
                f"{scenario} split is not exact 8:1:1: "
                f"{len(train)}:{len(val)}:{len(test)}"
            )
        heldout_overlap = val_new_genes & test_new_genes
        if heldout_overlap:
            raise ValueError(
                "Validation and test share held-out genes: "
                f"{sorted(heldout_overlap)[:5]}"
            )

        expected_seen_count = 1 if scenario == 'C2' else 0
        for split_name, split in (('validation', val), ('test', test)):
            seen_count = (
                split[_A].isin(train_genes).astype(int)
                + split[_B].isin(train_genes).astype(int)
            )
            if not (seen_count == expected_seen_count).all():
                invalid = split.loc[
                    seen_count != expected_seen_count, [GENE_A, GENE_B]
                ].head()
                raise ValueError(
                    f"{split_name} violates {scenario}: "
                    f"{invalid.to_dict('records')}"
                )
    elif scenario == 'C1':
        total = len(train) + len(val) + len(test)
        ratios = np.array([len(train), len(val), len(test)]) / total
        if np.max(np.abs(ratios - np.array([0.8, 0.1, 0.1]))) > 0.02:
            raise ValueError(
                f"C1 row ratio deviates from 8:1:1: "
                f"{len(train)}:{len(val)}:{len(test)}"
            )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    for split_name, split in (('train', train), ('validation', val), ('test', test)):
        if set(split[LABEL].unique()) != {0, 1}:
            raise ValueError(
                f"{split_name} split must contain both labels; "
                f"found {sorted(split[LABEL].unique())}"
            )

    total = len(train) + len(val) + len(test)
    return SplitReport(
        scenario=scenario,
        train_size=len(train),
        val_size=len(val),
        test_size=len(test),
        train_ratio=len(train) / total,
        val_ratio=len(val) / total,
        test_ratio=len(test) / total,
        train_gene_count=len(train_genes),
        val_new_gene_count=len(val_new_genes),
        test_new_gene_count=len(test_new_genes)
    )


def build_cross_validation_splits(
    data_df: pd.DataFrame,
    scenario: str,
    seed: int = 42,
    n_folds: int = 10
):
    if n_folds != 10:
        raise ValueError("The 8:1:1 rotating protocol requires exactly 10 folds")
    prepared = _prepare(data_df)
    fold_splits = []

    if scenario == 'C1':
        buckets = _balanced_pair_buckets(prepared, n_folds, seed)
        for fold_index in range(n_folds):
            test_pairs = buckets[fold_index]
            val_pairs = buckets[(fold_index + 1) % n_folds]
            train_pairs = set().union(*[
                bucket for index, bucket in enumerate(buckets)
                if index not in {fold_index, (fold_index + 1) % n_folds}
            ])
            train_df = prepared[prepared[_PAIR].isin(train_pairs)].copy()
            val_df = prepared[prepared[_PAIR].isin(val_pairs)].copy()
            test_df = prepared[prepared[_PAIR].isin(test_pairs)].copy()
            split = tuple(map(_clean, (train_df, val_df, test_df)))
            validate_protocol_split(*split, scenario)
            fold_splits.append(split)
        return fold_splits

    if scenario not in {'C2', 'C3'}:
        raise ValueError(f"Unknown scenario: {scenario}")

    all_genes = set(prepared[_A]).union(set(prepared[_B]))
    gene_buckets = _random_equal_buckets(all_genes, n_folds, seed)
    for fold_index in range(n_folds):
        test_genes = gene_buckets[fold_index]
        val_genes = gene_buckets[(fold_index + 1) % n_folds]
        train_genes = set().union(*[
            bucket for index, bucket in enumerate(gene_buckets)
            if index not in {fold_index, (fold_index + 1) % n_folds}
        ])
        split = _build_inductive_fold(
            prepared,
            train_genes,
            val_genes,
            test_genes,
            scenario,
            seed + fold_index * 100003
        )
        split = tuple(map(_clean, split))
        validate_protocol_split(*split, scenario)
        fold_splits.append(split)
    return fold_splits


def build_single_split(
    data_df: pd.DataFrame,
    scenario: str,
    seed: int = 42
):
    return build_cross_validation_splits(
        data_df, scenario=scenario, seed=seed, n_folds=10
    )[0]
