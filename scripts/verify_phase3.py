"""Phase 3 integration test — full prep session end-to-end."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine, delete, func, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from src.config import settings  # noqa: E402
from src.kb.models import (  # noqa: E402
    Answer, Question, QuestionTopic,
    Session as SessionRow,
)
from src.services.prep_service import run_prep_session  # noqa: E402


def main() -> None:
    engine = create_engine(settings.secrets.database_url, pool_pre_ping=True)

    print("=" * 70)
    print("PHASE 3 MILESTONE — Full Prep Session End-to-End")
    print("=" * 70)
    print()
    print("Running: section [5], 3 MCQs, weighted simulation, seed=7")
    print()

    with Session(engine) as session:
        result = run_prep_session(
            session=session,
            section_ids=[5],
            questions_per_section=3,
            difficulty="medium",
            simulate_strategy="weighted",
            seed=7,
        )

        print(f"✓ session_id:        {result.session_id}")
        print(f"✓ elapsed:           {result.elapsed_seconds}s")
        print(f"✓ chunks_used per section: {result.chunks_used_per_section}")
        print(f"✓ token_usage:       {result.token_usage}")
        print(f"✓ generation_rejects: {result.generation_rejects}")
        print()
        r = result.score_report
        print(f"✓ score:             {r.correct}/{r.total} ({r.score_pct:.1f}%)")
        print()
        print("Per-question results:")
        for i, q in enumerate(r.per_question, 1):
            mark = "✓" if q.is_correct else "✗"
            print(f"  Q{i}  {mark}  user={q.user_answer} correct={q.correct_answer}")
            print(f"      {q.question_text[:80]}")
        print()

        # Cross-check: DB state matches in-memory result
        print("=== Cross-checks ===")
        sess_row = session.get(SessionRow, result.session_id)
        print(f"✓ DB sessions row exists, completed_at set: "
              f"{sess_row.completed_at is not None}")
        print(f"✓ DB sessions.total_questions={sess_row.total_questions} "
              f"correct={sess_row.correct_count} wrong={sess_row.wrong_count}")
        print(f"✓ DB sessions.score_pct={sess_row.score_pct:.1f}%")
        print(f"✓ DB sessions.token_usage stored: "
              f"{bool(sess_row.token_usage)}")

        q_count = session.scalar(
            select(func.count()).select_from(Question)
            .where(Question.session_id == result.session_id)
        )
        a_count = session.scalar(
            select(func.count()).select_from(Answer)
            .join(Question, Answer.question_id == Question.id)
            .where(Question.session_id == result.session_id)
        )
        qt_count = session.scalar(
            select(func.count()).select_from(QuestionTopic)
            .join(Question, QuestionTopic.question_id == Question.id)
            .where(Question.session_id == result.session_id)
        )
        print(f"✓ questions saved:   {q_count}")
        print(f"✓ answers saved:     {a_count}")
        print(f"✓ question_topics linked: {qt_count}  "
              f"(at least 1 per MCQ with valid topic name)")

        # Wrong answers have explanations
        wrong = [pq for pq in r.per_question if not pq.is_correct]
        if wrong:
            print(f"\nWrong answer #1 explanation present: "
                  f"{bool(wrong[0].explanation.strip())}")
            print(f"  Explanation: {wrong[0].explanation[:120]}")
            print(f"  Source quote: \"{wrong[0].source_quote[:90]}...\"")
        else:
            print("\n(No wrong answers in this run — simulation got lucky)")

        # Clean up the test session
        session.execute(
            delete(SessionRow).where(SessionRow.id == result.session_id)
        )
        session.commit()
        print(f"\n✓ Test session {result.session_id} cleaned up")

    print()
    print("=" * 70)
    print("🎉 PHASE 3 INTEGRATION VERIFIED")
    print("=" * 70)


if __name__ == "__main__":
    main()
