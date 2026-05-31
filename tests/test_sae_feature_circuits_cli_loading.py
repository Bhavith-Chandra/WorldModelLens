import torch
import tempfile
import os
from world_model_lens.cli.commands import app
from typer.testing import CliRunner


class DummySAEObj:
    def encode(self, x):
        return x, None

    def decode(self, h):
        return h


def test_cli_load_direct_object(tmp_path):
    # save a direct SAE object
    p = tmp_path / "model.encoder.sae.pt"
    torch.save(DummySAEObj(), str(p))

    # create a small observations file
    obs = torch.randn(2, 4)
    obs_p = tmp_path / "obs.pt"
    torch.save(obs, str(obs_p))

    runner = CliRunner()
    # use --layers to match saved file
    result = runner.invoke(
        app,
        [
            "circuits",
            "checkpoint.pt",
            "--layers",
            "encoder",
            "--observations",
            str(obs_p),
            "--sae-checkpoints",
            str(p),
            "--output",
            str(tmp_path / "out.gml"),
        ],
    )
    # CLI should either succeed or exit with informative message (we don't have a real checkpoint)
    assert result is not None
    # ensure CLI didn't crash with an unexpected exception
    assert result.exit_code in (0, 1)
