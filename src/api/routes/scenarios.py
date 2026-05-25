"""Scenario A/B runners — the assessment's evaluation flow."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.schemas import (
    ScenarioBIterationResult, ScenarioBRunRequest, ScenarioBRunResponse,
)
from src.config import settings
from src.kb.mastery import clear_all_mastery
from src.output.snapshot import export_questions, export_snapshot
from src.services.prep_service import run_prep_session


router = APIRouter(prefix="/scenarios", tags=["scenarios"])


SCENARIO_B_PLAN = [
    {"iter": 1, "sections": [5, 8]},
    {"iter": 2, "sections": [6, 8, 9]},
    {"iter": 3, "sections": [8]},
]


@router.post("/b/run", response_model=ScenarioBRunResponse)
def run_scenario_b(
    req: ScenarioBRunRequest,
    db: Session = Depends(get_db),
):
    """Run the assessment's 3-iteration Scenario B and write outputs.

    Writes per iteration:
        outputs/scenario_b_iter{N}/questions_iter{N}.json
        outputs/scenario_b_iter{N}/kb_snapshot_iter{N}.json
    """
    t0 = time.time()

    outputs_root = Path(settings.paths.outputs_dir)

    if req.reset_state:
        clear_all_mastery(db)

    iterations = []
    for plan in SCENARIO_B_PLAN:
        result = run_prep_session(
            session=db,
            section_ids=plan["sections"],
            questions_per_section=req.questions_per_section,
            simulate_strategy=req.simulate_strategy,
            seed=req.seed,
        )

        out_dir = outputs_root / f"scenario_b_iter{plan['iter']}"
        export_questions(
            db,
            result.session_id,
            out_dir / f"questions_iter{plan['iter']}.json",
        )
        export_snapshot(
            db,
            out_dir / f"kb_snapshot_iter{plan['iter']}.json",
            after_session_id=result.session_id,
        )

        iterations.append(ScenarioBIterationResult(
            iteration=plan["iter"],
            section_ids=plan["sections"],
            session_id=result.session_id,
            score_pct=result.score_report.score_pct,
            output_dir=str(out_dir),
        ))

    return ScenarioBRunResponse(
        iterations=iterations,
        total_elapsed_seconds=round(time.time() - t0, 2),
        output_root=str(outputs_root),
    )
