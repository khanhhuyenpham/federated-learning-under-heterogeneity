from dataclasses import dataclass
from .benchmark_results import AlgorithmName
from .heterogeneity import HeterogeneityConfig
from .benchmark_suite import AlgorithmSpec
from pathlib import Path

import copy
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .benchmark_results import BenchmarkRunResult

_SAVED_DATAFRAME_FILES = {
    "validation_by_round":
        "validation_by_round.csv",

    "validation_by_client":
        "validation_by_client.csv",

    "client_update_diagnostics":
        "client_update_diagnostics.csv",

    "pairwise_update_diagnostics":
        "pairwise_update_diagnostics.csv",

    "round_diagnostics":
        "round_diagnostics.csv",

    "local_test_by_client":
        "local_test_by_client.csv",
}

def _series_matches_value(
    series: pd.Series,
    expected_value,
) -> bool:
    if expected_value is None:
        return bool(series.isna().all())

    if isinstance(expected_value, float):
        numeric_series = pd.to_numeric(
            series,
            errors="coerce",
        )

        return bool(
            np.isclose(
                numeric_series,
                expected_value,
                rtol=1e-12,
                atol=0.0,
            ).all()
        )

    return bool(
        (series == expected_value).all()
    )

def _add_or_validate_identity(
    *,
    dataframe: pd.DataFrame,
    column: str,
    value,
    source_path: Path,
) -> None:
    if column not in dataframe.columns:
        dataframe[column] = value
        return

    if not _series_matches_value(
        dataframe[column],
        value,
    ):
        raise ValueError(
            "Saved table identity does not match "
            f"its checkpoint for column {column}: "
            f"{source_path}"
        )

def _format_optional_float(
    value: float | None,
) -> str:
    if value is None:
        return "NA"

    return format(value, ".12g").replace(
        ".",
        "p",
    ).replace("-", "m")

def _to_json_safe(value):
    if value is None:
        return None

    if isinstance(
        value,
        (str, bool, int, float),
    ):
        return value

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(key): _to_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _to_json_safe(item)
            for item in value
        ]

    raise TypeError(
        "Value is not JSON serializable: "
        f"{type(value).__name__}"
    )


def _write_json(
    path: Path,
    payload,
) -> None:
    safe_payload = _to_json_safe(payload)

    path.write_text(
        json.dumps(
            safe_payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

def _snapshot_artifact(value):
    if torch.is_tensor(value):
        return (
            value.detach()
            .cpu()
            .clone()
        )

    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)

    if isinstance(value, dict):
        return {
            key: _snapshot_artifact(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _snapshot_artifact(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _snapshot_artifact(item)
            for item in value
        )

    return copy.deepcopy(value)

@dataclass(frozen=True)
class BenchmarkRunKey:
    mode: str
    seed: int
    num_clients: int
    alpha: float | None
    min_samples_per_client: int
    max_abs_angle: float
    algorithm: AlgorithmName
    mu: float | None

    
    def to_slug(self) -> str:
        return (
            f"{self.mode}"
            f"_seed{self.seed}"
            f"_clients{self.num_clients}"
            f"_alpha{_format_optional_float(self.alpha)}"
            f"_min{self.min_samples_per_client}"
            f"_angle{_format_optional_float(self.max_abs_angle)}"
            f"_{self.algorithm}"
            f"_mu{_format_optional_float(self.mu)}"
        )
    
def make_run_key(
    *,
    condition_config: HeterogeneityConfig,
    algorithm_spec: AlgorithmSpec,
) -> BenchmarkRunKey:
    return BenchmarkRunKey(
        mode=condition_config.mode,
        seed=condition_config.seed,
        num_clients=condition_config.num_clients,
        alpha=condition_config.alpha,
        min_samples_per_client=(
            condition_config
            .min_samples_per_client
        ),
        max_abs_angle=(
            condition_config.max_abs_angle
        ),
        algorithm=algorithm_spec.algorithm,
        mu=algorithm_spec.mu,
    )

def _make_run_key_from_result(
    result: BenchmarkRunResult,
) -> BenchmarkRunKey:
    metadata = result.condition_metadata

    return BenchmarkRunKey(
        mode=metadata.mode,
        seed=result.seed,
        num_clients=metadata.num_clients,
        alpha=metadata.alpha,
        min_samples_per_client=(
            metadata.min_samples_per_client
        ),
        max_abs_angle=metadata.max_abs_angle,
        algorithm=result.algorithm,
        mu=result.mu,
    )

def _run_key_mismatches(
    *,
    requested_key: BenchmarkRunKey,
    result_key: BenchmarkRunKey,
) -> dict:
    requested = asdict(requested_key)
    result = asdict(result_key)

    return {
        field: {
            "requested": requested[field],
            "result": result[field],
        }
        for field in requested
        if requested[field] != result[field]
    }

def _read_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )

class BenchmarkRunStore:
    def __init__(
        self,
        root_directory,
    ):
        self.root_directory = Path(root_directory)

    def run_directory(
        self,
        key: BenchmarkRunKey,
    ):
        return (
            self.root_directory / "runs" / key.to_slug()
        )

    def is_complete(
        self,
        key: BenchmarkRunKey,
    ) -> bool:
        completion_file = (
            self.run_directory(key) / "COMPLETED.json"
        )
        return completion_file.is_file()

    def save(
        self,
        *,
        key: BenchmarkRunKey,
        result: BenchmarkRunResult,
    ) -> Path:
        self._validate_key_matches_result(
            key=key,
            result=result,
        )

        final_directory = self.run_directory(key)

        if self.is_complete(key):
            raise FileExistsError(
                "Benchmark run is already complete: "
                f"{key.to_slug()}"
            )

        if final_directory.exists():
            raise FileExistsError(
                "An incomplete benchmark run directory "
                "already exists: "
                f"{final_directory}"
            )

        runs_directory = (
            self.root_directory / "runs"
        )

        runs_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_directory = Path(
            tempfile.mkdtemp(
                prefix=f".{key.to_slug()}.tmp-",
                dir=runs_directory,
            )
        )

        try:
            self._write_result_files(
                directory=temporary_directory,
                key=key,
                result=result,
            )

            temporary_directory.rename(
                final_directory
            )

        except Exception:
            if temporary_directory.exists():
                shutil.rmtree(
                    temporary_directory,
                    ignore_errors=True,
                )

            raise

        return final_directory

    def _write_result_files(
        self,
        *,
        directory: Path,
        key: BenchmarkRunKey,
        result: BenchmarkRunResult,
    ) -> None:
        _write_json(
            directory / "summary.json",
            result.to_summary_record(),
        )

        _write_json(
            directory / "condition_metadata.json",
            asdict(result.condition_metadata),
        )

        _write_json(
            directory / "local_test_summary.json",
            result.local_test_summary,
        )

        _write_json(
            directory / "global_test_summary.json",
            result.global_test_summary,
        )

        dataframes = {
            "validation_by_round.csv":
                result.validation_by_round,

            "validation_by_client.csv":
                result.validation_by_client,

            "client_update_diagnostics.csv":
                result.client_update_diagnostics,

            "pairwise_update_diagnostics.csv":
                result.pairwise_update_diagnostics,

            "round_diagnostics.csv":
                result.round_diagnostics,

            "local_test_by_client.csv":
                result.local_test_by_client,
        }

        for filename, dataframe in dataframes.items():
            dataframe.to_csv(
                directory / filename,
                index=False,
            )

        model_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor
            in result.final_model.state_dict().items()
        }

        torch.save(
            model_state,
            directory / "model_state.pt",
        )

        torch.save(
            _snapshot_artifact(
                result.extra_artifacts
            ),
            directory / "extra_artifacts.pt",
        )

        # This must be written last.
        _write_json(
            directory / "COMPLETED.json",
            {
                "schema_version": 1,
                "status": "completed",
                "run_key": key.to_slug(),
            },
        )

    def _validate_key_matches_result(
        self,
        *,
        key: BenchmarkRunKey,
        result: BenchmarkRunResult,
    ) -> None:
        result_key = _make_run_key_from_result(
            result
        )

        mismatches = _run_key_mismatches(
            requested_key=key,
            result_key=result_key,
        )

        if mismatches:
            raise ValueError(
                "Benchmark run key does not match "
                "the supplied result. "
                f"Mismatches: {mismatches}"
            )

    def is_spec_complete(
        self,
        condition_config: HeterogeneityConfig,
        algorithm_spec: AlgorithmSpec,
    ) -> bool:
        key = make_run_key(
            condition_config=condition_config,
            algorithm_spec=algorithm_spec,
        )

        return self.is_complete(key)

    def save_spec_result(
        self,
        condition_config: HeterogeneityConfig,
        algorithm_spec: AlgorithmSpec,
        result: BenchmarkRunResult,
    ) -> None:
        key = make_run_key(
            condition_config=condition_config,
            algorithm_spec=algorithm_spec,
        )

        self.save(
            key=key,
            result=result,
        )

    def load_summary(
        self,
        key: BenchmarkRunKey,
    ) -> dict:
        if not self.is_complete(key):
            raise FileNotFoundError(
                "Benchmark run is not complete: "
                f"{key.to_slug()}"
            )

        summary_path = (
            self.run_directory(key)
            / "summary.json"
        )

        summary = _read_json(summary_path)

        if not isinstance(summary, dict):
            raise ValueError(
                "Saved benchmark summary must "
                "contain a JSON object"
            )

        return summary

    def load_completed_summary(
        self,
    ) -> pd.DataFrame:
        runs_directory = (
            self.root_directory / "runs"
        )

        if not runs_directory.exists():
            return pd.DataFrame()

        records = []

        for run_directory in sorted(
            runs_directory.iterdir(),
            key=lambda path: path.name,
        ):
            if not run_directory.is_dir():
                continue

            completion_file = (
                run_directory / "COMPLETED.json"
            )

            if not completion_file.is_file():
                continue

            summary_path = (
                run_directory / "summary.json"
            )

            if not summary_path.is_file():
                raise FileNotFoundError(
                    "Completed benchmark run is "
                    "missing summary.json: "
                    f"{run_directory}"
                )

            summary = _read_json(summary_path)

            if not isinstance(summary, dict):
                raise ValueError(
                    "Saved benchmark summary must "
                    "contain a JSON object: "
                    f"{summary_path}"
                )

            records.append(summary)

        return pd.DataFrame(records)
    
    def load_completed_table(
        self,
        table_name: str,
    ) -> pd.DataFrame:
        if table_name not in _SAVED_DATAFRAME_FILES:
            raise ValueError(
                "Unknown saved benchmark table: "
                f"{table_name}"
            )

        runs_directory = (
            self.root_directory / "runs"
        )

        if not runs_directory.exists():
            return pd.DataFrame()

        filename = _SAVED_DATAFRAME_FILES[
            table_name
        ]

        frames = []

        for run_directory in sorted(
            runs_directory.iterdir(),
            key=lambda path: path.name,
        ):
            if not run_directory.is_dir():
                continue

            completion_path = (
                run_directory / "COMPLETED.json"
            )

            if not completion_path.is_file():
                continue

            table_path = run_directory / filename
            summary_path = (
                run_directory / "summary.json"
            )
            metadata_path = (
                run_directory
                / "condition_metadata.json"
            )

            for required_path in (
                table_path,
                summary_path,
                metadata_path,
            ):
                if not required_path.is_file():
                    raise FileNotFoundError(
                        "Completed benchmark run is "
                        "missing a required file: "
                        f"{required_path}"
                    )

            dataframe = pd.read_csv(table_path)
            summary = _read_json(summary_path)
            metadata = _read_json(metadata_path)

            identity = {
                "run_key": run_directory.name,
                "algorithm": summary["algorithm"],
                "seed": summary["seed"],
                "mu": summary["mu"],
                "mode": metadata["mode"],
                "alpha": metadata["alpha"],
                "num_clients": (
                    metadata["num_clients"]
                ),
                "min_samples_per_client": (
                    metadata[
                        "min_samples_per_client"
                    ]
                ),
                "max_abs_angle": (
                    metadata["max_abs_angle"]
                ),
            }

            for column, value in identity.items():
                _add_or_validate_identity(
                    dataframe=dataframe,
                    column=column,
                    value=value,
                    source_path=table_path,
                )

            identity_columns = list(identity)

            remaining_columns = [
                column
                for column in dataframe.columns
                if column not in identity_columns
            ]

            dataframe = dataframe[
                identity_columns
                + remaining_columns
            ]

            frames.append(dataframe)

        if not frames:
            return pd.DataFrame()

        return pd.concat(
            frames,
            ignore_index=True,
        )
