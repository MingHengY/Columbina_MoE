"""Reproducible checkpoint and deployment artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path


PATH_CONFIG_KEYS = {
    'BASE_DIR',
    'DATA_DIR',
    'OUTPUT_DIR',
    'LOCAL_MODEL_PATH',
    'SL_PAIRS_FILE',
    'NON_SL_PAIRS_FILE',
    'KG_EDGES_FILE',
    'KG_NODES_FILE',
    'STRING_FILE',
    'GENE_ID_MAPPING_FILE',
    'GENE_SYNONYM_FILE',
    'CANCER_CELL_LINE_MAPPING_FILE',
    'FEATURE_FILES',
}


def _json_safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def serialize_config(config):
    """Return every public uppercase Config value in a JSON-safe form."""
    values = {}
    for name in dir(config):
        if not name.isupper() or name.startswith('_'):
            continue
        value = getattr(config, name)
        if callable(value):
            continue
        values[name] = _json_safe(value)
    return values


def apply_model_config(config, snapshot):
    """Apply checkpoint settings while preserving local data/output paths."""
    if not isinstance(snapshot, dict):
        return config
    for name, value in snapshot.items():
        if name in PATH_CONFIG_KEYS or not name.isupper() or not hasattr(config, name):
            continue
        current = getattr(config, name)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(config, name, value)
    return config


def canonical_config_hash(config_snapshot):
    payload = json.dumps(
        _json_safe(config_snapshot), sort_keys=True, separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_config_metadata(config):
    snapshot = serialize_config(config)
    return {
        'model_config': snapshot,
        'model_config_sha256': canonical_config_hash(snapshot),
    }


def write_model_config(output_dir, config):
    output_path = Path(output_dir) / 'model_config.json'
    snapshot = serialize_config(config)
    payload = {
        'schema_version': 1,
        'config': snapshot,
        'config_sha256': canonical_config_hash(snapshot),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return output_path


def write_deployment_manifest(
    output_dir,
    checkpoint_name,
    scenario,
    config,
    metrics=None,
):
    """Write a self-contained deployment manifest and reject incomplete runs."""
    output_dir = Path(output_dir).resolve()
    required_names = [
        checkpoint_name,
        'scaler.pkl',
        'preprocessing_info.pkl',
        'preprocessing_info.json',
        'gene_mapping.pkl',
        'gene_mapping.csv',
        'model_config.json',
        'calibration.json',
    ]
    if scenario in {'C2', 'C3'}:
        required_names.append('strict_inductive_manifest.json')

    missing = [name for name in required_names if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            'Deployment artifacts are incomplete: ' + ', '.join(sorted(missing))
        )

    optional_names = [
        'pca.pkl',
        'gene_id_to_symbol.csv',
        'train_gene_embeddings.pt',
        'sl_connectivity.pt',
        'test_metrics.csv',
    ]
    artifact_names = required_names + [
        name for name in optional_names if (output_dir / name).is_file()
    ]
    artifacts = {}
    for name in artifact_names:
        path = output_dir / name
        artifacts[name] = {
            'bytes': path.stat().st_size,
            'sha256': sha256_file(path),
            'required': name in required_names,
        }

    snapshot = serialize_config(config)
    payload = {
        'schema_version': 1,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'model_family': 'Columbina',
        'architecture_version': snapshot.get('ARCHITECTURE_VERSION', '4.0-MoE'),
        'scenario': scenario,
        'strict_inductive': scenario in {'C2', 'C3'},
        'checkpoint': checkpoint_name,
        'preprocessing_dir': '.',
        'config_sha256': canonical_config_hash(snapshot),
        'metrics': _json_safe(metrics or {}),
        'runtime': {
            'python': platform.python_version(),
            'platform': platform.platform(),
        },
        'artifacts': artifacts,
    }
    manifest_path = output_dir / 'deployment_manifest.json'
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return manifest_path


def validate_deployment_manifest(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    root = manifest_path.parent
    for name, metadata in payload.get('artifacts', {}).items():
        path = root / name
        if metadata.get('required') and not path.is_file():
            raise FileNotFoundError(f'Required deployment artifact is missing: {path}')
        if path.is_file() and sha256_file(path) != metadata.get('sha256'):
            raise ValueError(f'Deployment artifact hash mismatch: {path}')
    return payload
