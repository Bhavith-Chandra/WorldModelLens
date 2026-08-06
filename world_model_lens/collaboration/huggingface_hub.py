"""HuggingFace Hub synchronization for models and trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

from world_model_lens.hub.model_hub import ModelHub as BaseModelHub
from world_model_lens.hub.trajectory_hub import TrajectoryHub as BaseTrajectoryHub
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from world_model_lens.core.dataset import TrajectoryDataset


class HuggingFaceModelHub(BaseModelHub):
    """ModelHub with HuggingFace Hub synchronization.

    Allows pushing/pulling models to/from HuggingFace Hub.

    Example:
        hub = HuggingFaceModelHub(repo_id="my_org/dreamer_atari")
        hub.push(model, "v1.0", commit_message="Add DreamerV3 on Atari")
        loaded = hub.pull("v1.0")
    """

    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
    ):
        self.repo_id = repo_id
        self.token = token
        self.private = private
        self._api = None

    def _get_api(self):
        if self._api is None:
            try:
                from huggingface_hub import HfApi
            except ImportError:
                raise ImportError("huggingface_hub required: pip install huggingface_hub")

            self._api = HfApi(token=self.token)
        return self._api

    def push(
        self,
        model: Any,
        path: str,
        version: str = "main",
        commit_message: Optional[str] = None,
    ) -> str:
        """Push model checkpoint to Hub.

        Args:
            model: Model to save (must be picklable or have save method).
            path: Local path or checkpoint to upload.
            version: Version tag.
            commit_message: Git commit message.

        Returns:
            URL to uploaded model.
        """
        api = self._get_api()

        try:
            api.create_repo(
                repo_id=self.repo_id,
                repo_type="model",
                private=self.private,
                exist_ok=True,
            )

            api.upload_folder(
                folder_path=path,
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=commit_message or f"Upload {version}",
            )

            return f"https://huggingface.co/{self.repo_id}/tree/{version}"

        except Exception as e:
            raise RuntimeError(f"Failed to push to hub: {e}")

    def pull(
        self,
        version: str = "main",
        output_dir: Optional[str] = None,
    ) -> str:
        """Pull model checkpoint from Hub.

        Args:
            version: Version to download.
            output_dir: Local directory to save to.

        Returns:
            Local path to downloaded model.
        """
        api = self._get_api()
        output_dir = output_dir or f"./downloads/{self.repo_id.replace('/', '_')}"

        import os

        os.makedirs(output_dir, exist_ok=True)

        try:
            api.hf_hub_download(
                repo_id=self.repo_id,
                filename="*.pt",
                local_dir=output_dir,
                repo_type="model",
            )
            return output_dir

        except Exception as e:
            raise RuntimeError(f"Failed to pull from hub: {e}")

    def list_versions(self) -> List[str]:
        """List all versions on the Hub."""
        api = self._get_api()

        try:
            model_info = api.repo_info(
                repo_id=self.repo_id,
                repo_type="model",
            )
            return [s.name for s in model_info.siblings]
        except Exception:
            return ["main"]

    def push_adapter(
        self,
        adapter: Any,
        adapter_name: str,
        commit_message: Optional[str] = None,
    ) -> str:
        """Push an adapter (e.g., LoRA weights) to Hub.

        Args:
            adapter: Adapter to upload.
            adapter_name: Name for the adapter.
            commit_message: Commit message.

        Returns:
            URL to adapter.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = f"{tmpdir}/{adapter_name}.pt"

            import torch

            torch.save(adapter.state_dict(), adapter_path)

            return self.push(adapter, adapter_path, commit_message=commit_message)


class HuggingFaceTrajectoryHub(BaseTrajectoryHub):
    """TrajectoryHub with HuggingFace Hub synchronization.

    Allows pushing/pulling trajectory datasets to/from HuggingFace Hub.

    Example:
        hub = HuggingFaceTrajectoryHub(repo_id="my_org/atari_trajectories")
        hub.push(dataset, "episode_1000", commit_message="Add trajectories")
        loaded = hub.pull("episode_1000")
    """

    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
    ):
        self.repo_id = repo_id
        self.token = token
        self.private = private
        self._api = None

    def _get_api(self):
        if self._api is None:
            try:
                from huggingface_hub import HfApi
            except ImportError:
                raise ImportError("huggingface_hub required: pip install huggingface_hub")

            self._api = HfApi(token=self.token)
        return self._api

    def push(
        self,
        dataset: "TrajectoryDataset",
        commit_message: Optional[str] = None,
    ) -> str:
        """Push trajectory dataset to Hub.

        Args:
            dataset: TrajectoryDataset to upload.
            commit_message: Commit message.

        Returns:
            URL to dataset.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            from world_model_lens.collaboration.serialization import serialize_dataset

            serialize_dataset(dataset, tmpdir, format="zarr")

            api = self._get_api()

            try:
                api.create_repo(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    private=self.private,
                    exist_ok=True,
                )

                api.upload_folder(
                    folder_path=tmpdir,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=commit_message or "Upload trajectories",
                )

                return f"https://huggingface.co/datasets/{self.repo_id}"

            except Exception as e:
                raise RuntimeError(f"Failed to push to hub: {e}")

    def pull(
        self,
        output_dir: Optional[str] = None,
    ) -> "TrajectoryDataset":
        """Pull trajectory dataset from Hub.

        Args:
            output_dir: Local directory to save to.

        Returns:
            Loaded TrajectoryDataset.
        """
        from world_model_lens.collaboration.serialization import deserialize_dataset

        output_dir = output_dir or f"./downloads/{self.repo_id.replace('/', '_')}"

        api = self._get_api()

        try:
            api.download(
                repo_id=self.repo_id,
                repo_type="dataset",
                local_dir=output_dir,
            )

            return deserialize_dataset(output_dir, format="zarr")

        except Exception as e:
            raise RuntimeError(f"Failed to pull from hub: {e}")


class BenchmarkHub:
    """Hub for sharing benchmark results and model cards."""

    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
    ):
        self.repo_id = repo_id
        self.token = token

    def push_benchmark_results(
        self,
        results: Dict[str, Any],
        commit_message: Optional[str] = None,
    ) -> str:
        """Push benchmark results to Hub.

        Args:
            results: Benchmark results dict.
            commit_message: Commit message.

        Returns:
            URL to results.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = f"{tmpdir}/benchmark_results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

            api = self._get_api()

            try:
                api.create_repo(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    exist_ok=True,
                )

                api.upload_file(
                    path_or_fileobj=results_path,
                    path_in_repo="benchmark_results.json",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=commit_message or "Upload benchmark results",
                )

                return f"https://huggingface.co/datasets/{self.repo_id}"

            except Exception as e:
                raise RuntimeError(f"Failed to push benchmarks: {e}")

    def pull_benchmark_results(self) -> Dict[str, Any]:
        """Pull benchmark results from Hub."""
        api = self._get_api()

        try:
            content = api.hf_hub_download(
                repo_id=self.repo_id,
                filename="benchmark_results.json",
                repo_type="dataset",
            )

            with open(content) as f:
                return json.load(f)

        except Exception as e:
            raise RuntimeError(f"Failed to pull benchmarks: {e}")

    def _get_api(self):
        if not hasattr(self, "_api") or self._api is None:
            try:
                from huggingface_hub import HfApi
            except ImportError:
                raise ImportError("huggingface_hub required: pip install huggingface_hub")

            self._api = HfApi(token=self.token)
        return self._api
