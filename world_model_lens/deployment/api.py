from __future__ import annotations
"""FastAPI server for WorldModelLens analysis."""

from __future__ import annotations

import json
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from world_model_lens.probing import LatentProber

app = FastAPI(
    title="WorldModelLens API",
    description="Interpretability toolkit for world models",
    version="0.1.0",
)


class AnalyzeRequest(BaseModel):
    observations: list[list[float]]
    actions: list[list[float]] | None = None
    component: str = "h"
    timestep: int | None = None


class ProbingRequest(BaseModel):
    component: str
    labels: list[float]
    concept_name: str
    probe_type: str = "ridge"


class PatchingRequest(BaseModel):
    clean_cache_path: str | None = None
    corrupted_cache_path: str | None = None
    component: str
    timestep: int
    metric_fn: str = "reward"


class AnalysisResponse(BaseModel):
    status: str
    result: dict[str, Any]


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    content = (
        "<html>"
        "<head>"
        "<title>WorldModelLens API</title>"
        "<style>"
        "body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }"
        "h1 { color: #333; }"
        " .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }"
        "code { background: #eee; padding: 2px 5px; }"
        "</style>"
        "</head>"
        "<body>"
        "<h1>WorldModelLens API</h1>"
        "<p>Interpretability toolkit for world models</p>"
        "<h2>Endpoints</h2>"
        '<div class="endpoint"><code>POST /analyze</code> - Analyze a trajectory</div>'
        '<div class="endpoint"><code>POST /probe</code> - Train a probe on activations</div>'
        '<div class="endpoint"><code>POST /patch</code> - Run patching experiment</div>'
        '<div class="endpoint"><code>GET /health</code> - Health check</div>'
        "</body>"
        "</html>"
    )
    return HTMLResponse(content=content)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "healthy", "version": "0.1.0"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_episode(request: AnalyzeRequest) -> AnalysisResponse:
    """Analyze a trajectory and return activations."""
    try:
        obs_tensor = torch.tensor(request.observations)

        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        actions_tensor = None
        if request.actions:
            actions_tensor = torch.tensor(request.actions)
            if actions_tensor.dim() == 1:
                actions_tensor = actions_tensor.unsqueeze(0)

        return AnalysisResponse(
            status="success",
            result={
                "message": "Analysis endpoint ready. Add HookedWorldModel to process.",
                "observations_shape": list(obs_tensor.shape),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/probe", response_model=AnalysisResponse)
async def train_probe(request: ProbingRequest) -> AnalysisResponse:
    """Train a probe on activations."""
    try:
        # instantiate prober when integrating with activation cache
        _ = LatentProber(seed=42)

        return AnalysisResponse(
            status="success",
            result={
                "message": "Probe training ready. Provide activation cache.",
                "component": request.component,
                "concept": request.concept_name,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/patch", response_model=AnalysisResponse)
async def run_patching(request: PatchingRequest) -> AnalysisResponse:
    """Run patching experiment."""
    try:
        return AnalysisResponse(
            status="success",
            result={
                "message": "Patching ready. Provide caches for experiment.",
                "component": request.component,
                "timestep": request.timestep,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/upload/trajectory", response_model=AnalysisResponse)
async def upload_trajectory(file: UploadFile | None = None) -> AnalysisResponse:
    """Upload trajectory JSON for analysis."""
    if file is None:
        raise HTTPException(status_code=400, detail="file is required")

    try:
        content = await file.read()
        data = json.loads(content)

        return AnalysisResponse(
            status="success",
            result={
                "message": "Trajectory loaded",
                "n_timesteps": len(data.get("observations", [])),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def generate_html_report(trajectory_data: dict[str, Any], analysis_results: dict[str, Any]) -> str:
    """Generate HTML report from analysis results.

    Build the HTML using small parts to avoid formatting conflicts with CSS
    braces when using str.format.
    """
    n_timesteps = trajectory_data.get("n_timesteps", 0)
    results_json = json.dumps(analysis_results, indent=2)

    parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "  <title>WorldModelLens Analysis Report</title>",
        "  <style>",
        "    body { font-family: Arial; margin: 40px; }",
        "    h1 { color: #2c3e50; }",
        "    .section { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; }",
        "    .metric { display: inline-block; margin: 10px 20px 10px 0; }",
        "    .metric-value { font-size: 24px; font-weight: bold; color: #3498db; }",
        "    table { border-collapse: collapse; width: 100%; }",
        "    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "    th { background-color: #3498db; color: white; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>WorldModelLens Analysis Report</h1>",
        '  <div class="section">',
        "    <h2>Trajectory Summary</h2>",
        f'    <div class="metric"><div>Timesteps</div><div class="metric-value">{n_timesteps}</div></div>',
        "  </div>",
        '  <div class="section">',
        "    <h2>Analysis Results</h2>",
        f"    <pre>{results_json}</pre>",
        "  </div>",
        "</body>",
        "</html>",
    ]

    return "\n".join(parts)


@app.get("/report/{analysis_id}")
async def get_report(analysis_id: str):
    """Get HTML report for a completed analysis."""
    return HTMLResponse(content=f"<h1>Analysis {analysis_id}</h1><p>Report placeholder</p>")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
