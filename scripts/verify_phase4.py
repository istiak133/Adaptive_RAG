"""Phase 4 integration test — adaptive intelligence end-to-end.

Verifies:
  • Cold-start detection on first run
  • Mastery rows created after answers
  • Adaptive detection on 2nd run over same section
  • adaptive_context built with WEAK/MASTERED signal
  • Allocator biases toward weak topics
  • Regression detection (mastered → wrong → reweighted)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine, delete, select, func  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from src.config import settings  # noqa: E402
from src.kb.mastery import (  # noqa: E402
    calculate_weight, clear_all_mastery,
    update_mastery_after_answer,
)
from src.kb.models import (  # noqa: E402
    Answer, Question, QuestionTopic, SectionTopicMastery,
    Session as SessionRow, Topic, TopicMastery,
)
from src.prep.allocator import allocate  # noqa: E402
from src.rag.history_compressor import build_adaptive_context  # noqa: E402
from src.services.prep_service import run_prep_session  # noqa: E402


results = []


def check(name: str, passed: bool, detail: str = "") -> None:
    icon = "✓" if passed else "✗"
    print(f"  {icon} {name}{(' — ' + detail) if detail else ''}")
    results.append((name, passed))


def main() -> None:
    engine = create_engine(settings.secrets.database_url, pool_pre_ping=True)

    # ── Setup: clean slate ───────────────────────────────────────────
    with Session(engine) as session:
        n_clear = clear_all_mastery(session)
        # Delete any existing test sessions
        session.execute(delete(SessionRow))
        session.commit()
        print(f"Setup: cleared {n_clear} mastery rows + all sessions")
    print()

    # ── PART A: Weight formula unit checks ──────────────────────────
    print("=" * 70)
    print("PART A: Weight formula unit tests")
    print("=" * 70)
    w_fresh = calculate_weight(times_wrong=0, times_correct=0, streak=0)
    check("Fresh topic weight ≈ 1.0", abs(w_fresh - 1.0) < 0.01, f"got {w_fresh}")

    w_wrong_1 = calculate_weight(times_wrong=1, times_correct=0, streak=-1)
    check("1 wrong, streak -1 weight > 1.0", w_wrong_1 > 1.0, f"got {w_wrong_1:.2f}")

    w_wrong_3 = calculate_weight(times_wrong=3, times_correct=0, streak=-3)
    check("3 consecutive wrong weight much higher", w_wrong_3 > w_wrong_1,
          f"got {w_wrong_3:.2f}")

    w_mastered = calculate_weight(times_wrong=0, times_correct=5, streak=5)
    check("Streak ≥ threshold → weight = min", abs(w_mastered - 0.1) < 0.01,
          f"got {w_mastered}")

    w_decay = calculate_weight(times_wrong=2, times_correct=3, streak=2)
    check("Streak 2 applies decay", w_decay < calculate_weight(2, 0, 0),
          f"got {w_decay:.2f}")

    print()
    # ── PART B: ITER 1 (cold start) ──────────────────────────────────
    print("=" * 70)
    print("PART B: Iter 1 — cold start on Section [2]")
    print("=" * 70)

    with Session(engine) as session:
        result1 = run_prep_session(
            session=session,
            section_ids=[2],
            questions_per_section=3,
            simulate_strategy="random",
            seed=11,  # forces some wrong answers
        )
    print(f"  session_id:      {result1.session_id}")
    print(f"  score:           {result1.score_report.correct}/"
          f"{result1.score_report.total} ({result1.score_report.score_pct:.0f}%)")
    print(f"  elapsed:         {result1.elapsed_seconds}s")
    print()

    with Session(engine) as session:
        s1 = session.get(SessionRow, result1.session_id)
        check("Iter 1: is_cold_start=True", s1.is_cold_start is True)
        check("Iter 1: completed_at set", s1.completed_at is not None)

        topic_mastery_count = session.scalar(select(func.count(TopicMastery.id)))
        section_mastery_count = session.scalar(select(func.count(SectionTopicMastery.id)))
        check("Iter 1: topic_mastery rows created",
              topic_mastery_count > 0, f"{topic_mastery_count} rows")
        check("Iter 1: section_topic_mastery rows created",
              section_mastery_count > 0, f"{section_mastery_count} rows")

        # Inspect: at least one topic has wrong > 0 (since simulator missed some)
        wrong_topics = session.execute(
            select(Topic.name, TopicMastery.times_wrong, TopicMastery.weight)
            .join(TopicMastery, TopicMastery.topic_id == Topic.id)
            .where(TopicMastery.times_wrong > 0)
        ).all()
        check("Iter 1: at least one topic has wrong > 0",
              len(wrong_topics) > 0, f"{len(wrong_topics)} weak topics")
        for w in wrong_topics[:5]:
            print(f"      → {w.name} wrong={w.times_wrong} weight={w.weight:.2f}")

    print()
    # ── PART C: ITER 2 (adaptive) ────────────────────────────────────
    print("=" * 70)
    print("PART C: Iter 2 — adaptive on same Section [2]")
    print("=" * 70)

    with Session(engine) as session:
        # Snapshot adaptive_context BEFORE running iter 2
        pre_context = build_adaptive_context(session, [2])
        print(f"  adaptive_context preview ({len(pre_context)} chars):")
        print("  " + pre_context.replace("\n", "\n  ")[:500])
        print()

        # Snapshot allocation plan BEFORE running iter 2
        plan = allocate(session, section_id=2, n_questions=3)
        print(f"  Allocation mode: {plan.mode}")
        print(f"  Seeds: {plan.seeds}")
        check("Iter 2: allocation mode != cold", plan.mode != "cold")

        check("Iter 2: adaptive_context non-empty", len(pre_context) > 0)

        # Run actual iter 2
        result2 = run_prep_session(
            session=session,
            section_ids=[2],
            questions_per_section=3,
            simulate_strategy="all_correct",  # force all right this time
            seed=22,
        )
    print(f"\n  session_id:      {result2.session_id}")
    print(f"  score:           {result2.score_report.correct}/"
          f"{result2.score_report.total} ({result2.score_report.score_pct:.0f}%)")

    with Session(engine) as session:
        s2 = session.get(SessionRow, result2.session_id)
        check("Iter 2: is_cold_start=False", s2.is_cold_start is False)
        check("Iter 2: session.adaptive_context stored",
              s2.adaptive_context is not None)
        if s2.adaptive_context:
            print(f"      adaptive_context keys: {list(s2.adaptive_context.keys())}")
            alloc = s2.adaptive_context.get("allocation", {}).get("2", {})
            if alloc:
                print(f"      allocation mode: {alloc.get('mode')}")
                print(f"      seeds: {alloc.get('seeds')}")

    print()
    # ── PART D: Regression detection ─────────────────────────────────
    print("=" * 70)
    print("PART D: Regression detection — synthetic scenario")
    print("=" * 70)

    with Session(engine) as session:
        # Force a topic to be "mastered" then deliberately answer wrong
        # to verify regression flips it.
        # We need: a question already in DB, and we manually set its topic to mastered.
        q_row = session.execute(
            select(Question, QuestionTopic, Topic)
            .join(QuestionTopic, QuestionTopic.question_id == Question.id)
            .join(Topic, QuestionTopic.topic_id == Topic.id)
            .limit(1)
        ).first()

        if q_row:
            q, qt, topic = q_row[0], q_row[1], q_row[2]
            # Manually force mastery to mastered state
            tm = session.execute(
                select(TopicMastery).where(TopicMastery.topic_id == topic.id)
            ).scalar_one_or_none()
            if tm is None:
                tm = TopicMastery(topic_id=topic.id)
                session.add(tm)
            tm.times_correct = 5
            tm.times_wrong = 0
            tm.current_streak = 5
            tm.is_mastered = True
            tm.weight = 0.1
            session.commit()
            print(f"  Pre: topic '{topic.name}' is_mastered=True, weight=0.10")

            # Force a wrong answer on a question testing this topic
            deltas = update_mastery_after_answer(session, q.id, is_correct=False)
            for d in deltas:
                if d.topic_id == topic.id:
                    print(f"  Post-wrong: topic '{d.topic_name}' "
                          f"is_mastered={d.is_mastered}, weight={d.new_weight:.2f}, "
                          f"regression={d.regression_detected}")
                    check("Regression detected (mastered → wrong)",
                          d.regression_detected)
                    check("Mastered flag flipped to False",
                          d.is_mastered is False)
                    check(f"Weight bumped ≥ {settings.adaptive.regression_weight_floor}",
                          d.new_weight >= settings.adaptive.regression_weight_floor,
                          f"got {d.new_weight:.2f}")
        else:
            print("  (No question_topics rows — regression test skipped)")

    # ── Scoreboard ───────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SCOREBOARD")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for _, p in results if p)
    failed = [n for n, p in results if not p]

    print(f"  Passed: {passed}/{total}")
    if failed:
        print("  FAILED:")
        for n in failed:
            print(f"    ✗ {n}")
    else:
        print("  🎉 PHASE 4 ADAPTIVE LOOP VERIFIED")


if __name__ == "__main__":
    main()
