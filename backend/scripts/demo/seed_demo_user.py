"""Give the local demo student a realistic past week of practice, on top of the onboarding
already recorded, for the demo video:

    docker compose -f docker-compose.dev.yml exec backend python -m scripts.demo.seed_demo_user --week

Every graded answer goes through the Profiling Agent (ingest) and the practice-session
lifecycle with a past timestamp, so mastery, the learning profile, awards and the session
wrap-ups all agree. No LLM is called: evaluation summaries and session feedback are written
here. Only the named student is touched. A rerun first clears that student's practice
history (onboarding stays), so it always ends in the same state."""

import argparse
import re
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete

from app.core.database import SessionLocal
from app.core.topics import TOPIC_LABELS
from app.features.accounts.models import User
from app.features.accounts.service import AuthService
from app.features.gamification import service as gamification
from app.features.gamification.models import GamificationAward
from app.features.grading.models import (
    CodeEvaluation,
    CodeSubmission,
    EvalHistoryEntry,
    Language,
)
from app.features.hints.models import HintEvent
from app.features.hints.service import highest_level
from app.features.learner_model.agent.graph import ingest
from app.features.learner_model.logic.mastery import (
    SEED_KINDS,
    WEIGHTS,
    code_score,
    debug_score,
    updated_mastery,
)
from app.features.learner_model.models import LearningEvent, TopicMastery
from app.features.learner_model.service import EventInput, derive_profile, mastery_snapshot
from app.features.practice import service as lifecycle
from app.features.practice.agent.graph import next_item
from app.features.practice.logic.assessment import policy
from app.features.practice.models import (
    DebugAttempt,
    DebugChallenge,
    McqAttempt,
    McqQuestion,
    PracticeSession,
)
from app.features.problems.logic.listing import CURATED_SOURCES
from app.features.problems.models import ExecRole, ExecutionResult, Problem
from scripts._local_db import require_local_db

LANGUAGE = "python"
DIFFICULTY_RANK = {"easy": 0, "medium": 1, "hard": 2}

TARGETS = {
    "basics_io": 0.80, "functions": 0.68, "loops": 0.62, "sorting": 0.60, "arrays": 0.58,
    "searching": 0.57, "hashing_sets": 0.56, "strings": 0.55, "dynamic_programming": 0.55,
    "stacks_queues": 0.53, "conditionals": 0.46, "recursion": 0.38,
}

KEPT_FOR_DEMO = "Leap Year"

FEEDBACK = [
    {"summary": "A strong start on new topics: your first questions on Hashing & Sets, "
                "Recursion, Searching, Sorting, Stacks & Queues and Dynamic Programming were "
                "all correct. Conditionals was harder. In Largest of Three one comparison "
                "pointed the wrong way, but you found it and passed every test on the second "
                "try. One Loops question was missed on where a count should stop.",
     "work_on_next": ["Conditionals: before you submit, check each comparison sign against "
                      "one small example worked by hand",
                      "Loops: trace the first and the last pass of a loop to see where it "
                      "starts and stops"]},
    {"summary": "You solved the Recursion problem after two hints; your first attempt never "
                "stopped for the smallest input. Dynamic Programming moved up, though your "
                "table attempts passed only some tests, and one stopped a step early. You "
                "repaired two broken programs and solved a Basics and an Arrays problem first "
                "time. Two questions on Arrays and Strings were missed by one index.",
     "work_on_next": ["Recursion: before coding, write down the smallest input and what the "
                      "function must return for it",
                      "Arrays: check the first and the last index your loop touches"]},
    {"summary": "Conditionals took most of this session. Triangle Type and Electricity Bill "
                "both failed at first because a comparison was wrong right at the boundary; "
                "with hints you made both pass every test. The last Conditionals debugging "
                "challenge is not fixed yet. Recursion is now your weakest topic: a question "
                "and a debugging challenge both came down to a function with no stopping case.",
     "work_on_next": ["Recursion: write the stopping case first and test it on its own",
                      "Conditionals: test the exact boundary values, such as equal sides or "
                      "the first unit of a new price band"]},
    {"summary": "Basics & I/O is now mastered, your first mastered topic. Most coding "
                "problems today passed many tests but not all. In Loops, Arrays, Strings and "
                "Searching the misses came from a loop or an index one step off, and in "
                "Sorting from a comparison the wrong way round. Recursion is still below the "
                "line, so Dynamic Programming stays locked for now.",
     "work_on_next": ["Recursion: answer a few easy questions to rebuild the stopping-case "
                      "idea, then try a problem",
                      "Loops: run your loop on a tiny input and check the first and last "
                      "index before submitting"]},
]


class Week:
    def __init__(self, db, user: User):
        self.db = db
        self.user = user
        self.used_mcq: set[int] = set()
        self.used_debug: set[int] = set()
        self.used_problems: set[int] = {self._problem(KEPT_FOR_DEMO, "conditionals").id}
        self.last_eval: dict[int, int] = {}
        self.at: datetime | None = None

    def _problem(self, title: str, topic: str) -> Problem:
        return (self.db.query(Problem)
                .filter(Problem.source.in_(CURATED_SOURCES), Problem.topic == topic,
                        Problem.title == title).one())

    def _mastery(self, topic: str) -> float:
        row = self.db.get(TopicMastery, (self.user.id, topic, LANGUAGE))
        return row.mastery if row else 0.3

    def _finish_event(self, event: LearningEvent, item_type: str) -> None:
        event.created_at = self.at
        lifecycle.attach_event(self.db, event, item_type, self.at)
        self.db.commit()

    def _serve(self, item_type: str, item_id: int) -> None:
        lifecycle.record_served(self.db, self.user.id, item_type, item_id, LANGUAGE, self.at)
        self.db.commit()

    def mcq(self, topic: str, correct: bool, tag: str | None = None) -> None:
        questions = (self.db.query(McqQuestion)
                     .filter(McqQuestion.topic == topic, McqQuestion.reviewed.is_(True),
                             McqQuestion.is_diagnostic.is_(False),
                             McqQuestion.language.in_((LANGUAGE, "any")),
                             McqQuestion.id.notin_(self.used_mcq or {-1})).all())
        questions.sort(key=lambda q: (DIFFICULTY_RANK[q.difficulty], q.id))
        choice = None
        for q in questions:
            if correct:
                choice = (q, q.answer_index)
                break
            wrong = [i for i, o in enumerate(q.options) if i != q.answer_index
                     and (tag is None or o.get("misconception") == tag)]
            if wrong:
                choice = (q, wrong[0])
                break
        if choice is None:
            raise SystemExit(f"no multiple-choice question fits {topic} {correct} {tag}")
        q, chosen = choice
        self.used_mcq.add(q.id)
        self._serve("mcq", q.id)
        attempt = McqAttempt(user_id=self.user.id, question_id=q.id, chosen_index=chosen,
                             correct=chosen == q.answer_index, created_at=self.at)
        self.db.add(attempt)
        self.db.flush()
        misconception = None if attempt.correct else q.options[chosen].get("misconception")
        event = ingest(self.db, EventInput(
            user_id=self.user.id, kind="mcq", score=1.0 if attempt.correct else 0.0,
            raw_topic=q.topic, item_id=q.id,
            misconceptions=[misconception] if misconception else [],
            source_ref=f"mcq_attempt:{attempt.id}",
            language=None if q.language == "any" else q.language))
        self._finish_event(event, "mcq")

    def _challenge(self, topic: str, tag: str | None) -> DebugChallenge:
        pool = (self.db.query(DebugChallenge)
                .filter(DebugChallenge.topic == topic, DebugChallenge.language == LANGUAGE,
                        DebugChallenge.reviewed.is_(True),
                        DebugChallenge.id.notin_(self.used_debug or {-1})).all())
        pool.sort(key=lambda c: (c.bug_type != tag, DIFFICULTY_RANK.get(c.difficulty, 3), c.id))
        if not pool:
            raise SystemExit(f"no debugging challenge left for {topic}")
        return pool[0]

    def debug(self, topic: str, fixed: bool, tag: str | None = None, aim: bool = False) -> None:
        challenge = self._challenge(topic, tag)
        n = len(self.db.get(Problem, challenge.problem_id).test_cases)
        passed = n if fixed else 0
        if aim:
            start, target = self._mastery(topic), TARGETS[topic]
            passed = min(range(n), key=lambda k: abs(
                updated_mastery(start, WEIGHTS["debug"], debug_score(False, k, n)) - target))
        self.used_debug.add(challenge.id)
        self._serve("debug", challenge.id)
        done = passed == n
        attempt = DebugAttempt(user_id=self.user.id, challenge_id=challenge.id,
                               submitted_code=challenge.fixed_code if done else challenge.buggy_code,
                               tests_passed=passed, tests_total=n, fixed=done,
                               created_at=self.at)
        self.db.add(attempt)
        self.db.flush()
        event = ingest(self.db, EventInput(
            user_id=self.user.id, kind="debug", score=debug_score(done, passed, n),
            raw_topic=challenge.topic, item_id=challenge.id,
            misconceptions=[] if done else [challenge.bug_type],
            source_ref=f"debug_attempt:{attempt.id}", language=LANGUAGE))
        self._finish_event(event, "debug")

    def _pick_problem(self, topic: str) -> Problem:
        pool = (self.db.query(Problem)
                .filter(Problem.source.in_(CURATED_SOURCES), Problem.topic == topic,
                        Problem.test_cases.any(),
                        Problem.id.notin_(self.used_problems)).all())
        pool = [p for p in pool if _reference(p)]
        pool.sort(key=lambda p: (DIFFICULTY_RANK[p.difficulty.value], p.id))
        if not pool:
            raise SystemExit(f"no coding problem left for {topic}")
        return pool[0]

    def _aimed_problem(self, topic: str) -> tuple[Problem, int]:
        start, target = self._mastery(topic), TARGETS[topic]
        best = None
        for p in self.db.query(Problem).filter(
                Problem.source.in_(CURATED_SOURCES), Problem.topic == topic,
                Problem.test_cases.any(), Problem.id.notin_(self.used_problems)):
            if not _reference(p):
                continue
            n = len(p.test_cases)
            for k in range(1, n + 1):
                miss = abs(updated_mastery(start, WEIGHTS["code"], code_score(k, n)) - target)
                key = (round(miss, 4), DIFFICULTY_RANK[p.difficulty.value], p.id)
                if best is None or key < best[0]:
                    best = (key, p, k)
        if best is None:
            raise SystemExit(f"no coding problem left for {topic}")
        return best[1], best[2]

    def code(self, topic: str, *, title: str | None = None, share: float = 1.0,
             tags: tuple[str, ...] = (), aim: bool = False, retry: Problem | None = None
             ) -> Problem:
        if retry is not None:
            problem = retry
            passed = round(share * len(problem.test_cases))
        elif aim:
            problem, passed = self._aimed_problem(topic)
        else:
            problem = self._problem(title, topic) if title else self._pick_problem(topic)
            n = len(problem.test_cases)
            passed = n if share >= 1 else max(1, min(n - 1, round(share * n)))
        self.used_problems.add(problem.id)
        if retry is None:
            self._serve("coding", problem.id)
        evaluation = self._evaluation(problem, passed, tags)
        hint_level = highest_level(self.db, self.user.id, problem.id, before=evaluation.id,
                                   sources=("practice",))
        n = len(problem.test_cases)
        event = ingest(self.db, EventInput(
            user_id=self.user.id, kind="code", score=code_score(passed, n, hint_level),
            raw_topic=problem.topic, item_id=problem.id, hint_level=hint_level,
            misconceptions=list(tags) if passed < n else [],
            source_ref=f"evaluation:{evaluation.id}", language=LANGUAGE))
        self._finish_event(event, "coding")
        return problem

    def hints(self, problem: Problem, levels: list[int], minutes_apart: int = 2) -> None:
        for level in levels:
            self.at += timedelta(minutes=minutes_apart)
            hint = HintEvent(user_id=self.user.id, problem_id=problem.id,
                             evaluation_id=self.last_eval[problem.id], level=level,
                             was_blocked=False, source="practice", created_at=self.at)
            self.db.add(hint)
            self.db.flush()
            lifecycle.attach_hint(self.db, hint, self.at)
            self.db.commit()

    def _evaluation(self, problem: Problem, passed: int, tags: tuple[str, ...]) -> CodeEvaluation:
        cases = list(problem.test_cases)
        n = len(cases)
        source = _reference(problem)
        code = source if passed == n else _broken(source, tags[0] if tags else None)
        submission = CodeSubmission(user_id=self.user.id, title=problem.title,
                                    language=Language.PYTHON, code=code, problem_id=problem.id,
                                    from_bank=True, created_at=self.at)
        self.db.add(submission)
        self.db.flush()
        correctness = round(100.0 * passed / n, 1)
        complexity, efficiency, style = 90.0, 85.0, 88.0
        overall = round(0.55 * correctness + 0.2 * complexity + 0.15 * efficiency
                        + 0.1 * style, 1)
        first_fail = next((i for i in range(n) if i >= passed), None)
        summary = (f"All {n} tests passed." if passed == n else
                   f"{passed} of {n} tests passed. Test {first_fail + 1} gives a different "
                   "answer from the one expected.")
        evaluation = CodeEvaluation(
            submission_id=submission.id, user_id=self.user.id, overall_score=overall,
            correctness_score=correctness, complexity_score=complexity,
            efficiency_score=efficiency, style_score=style, mode="verified",
            verification_status="verified", tests_total=n, tests_passed=passed,
            status="done", attempt_number=1, issues=[],
            lines_of_code=len([line for line in code.splitlines() if line.strip()]),
            summary=summary, key_improvements=[], learning_points=[],
            best_practices_used=[], anti_patterns=[], suggested_resources=[],
            agent_steps=[], static_tool_used="lizard", created_at=self.at)
        self.db.add(evaluation)
        self.db.flush()
        for index, case in enumerate(cases):
            ok = index < passed
            self.db.add(ExecutionResult(
                evaluation_id=evaluation.id, test_case_id=case.id, role=ExecRole.STUDENT,
                status="ok" if ok else "wrong_answer",
                stdout=case.expected_stdout if ok else _wrong(case.expected_stdout),
                stderr="", time_ms=0, created_at=self.at))
        self.db.add(EvalHistoryEntry(user_id=self.user.id, submission_id=submission.id,
                                     evaluation_id=evaluation.id, overall_score=overall,
                                     language=Language.PYTHON, eval_number=1,
                                     created_at=self.at))
        self.db.commit()
        self.last_eval[problem.id] = evaluation.id
        return evaluation

    def finish(self, feedback: dict) -> None:
        session = lifecycle.open_session(self.db, self.user.id, self.at)
        session = lifecycle.finish(self.db, self.user.id, session.id, self.at)
        session.feedback = feedback
        session.feedback_status = "ready"
        session.wrap_up_seen_at = self.at
        self.db.commit()


def _reference(problem: Problem) -> str | None:
    for ref in problem.references:
        if ref.language.value == LANGUAGE and ref.status.value == "validated":
            return ref.source
    return None


_FLIP = {">=": ">", "<=": "<", ">": ">=", "<": "<="}


def _broken(source: str, tag: str | None) -> str:
    """The reference with the student's mistake put back in, for the failing attempts."""
    if tag == "missing_base_case":
        lines = source.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("if ") and "return" in (
                    line + (lines[i + 1] if i + 1 < len(lines) else "")):
                body = 1 if line.rstrip().endswith(":") else 0
                return "\n".join(lines[:i] + lines[i + 1 + body:]) + "\n"
    if tag == "off_by_one":
        changed = re.sub(r"range\(([^()]*?) \+ 1\)", r"range(\1)", source, count=1)
        if changed == source:
            changed = re.sub(r"range\(([^(),]+)\)", r"range(\1 - 1)", source, count=1)
        if changed != source:
            return changed
    return re.sub(r" (>=|<=|>|<) ", lambda m: f" {_FLIP[m.group(1)]} ", source, count=1)


def _wrong(expected: str) -> str:
    text = expected.strip()
    swaps = {"Yes": "No", "No": "Yes", "YES": "NO", "NO": "YES", "True": "False",
             "False": "True"}
    if text in swaps:
        return swaps[text] + "\n"
    try:
        return f"{int(text) + 1}\n"
    except ValueError:
        return ""


def clear_week(db, user_id: int) -> None:
    """Remove every graded answer after onboarding and put mastery back to the
    onboarding readings."""
    events = (db.query(LearningEvent)
              .filter(LearningEvent.user_id == user_id, LearningEvent.kind.notin_(SEED_KINDS))
              .all())
    refs: dict[str, list[int]] = {"mcq_attempt": [], "debug_attempt": [], "evaluation": []}
    for e in events:
        kind, _, ref_id = (e.source_ref or "").partition(":")
        if kind in refs and ref_id.isdigit():
            refs[kind].append(int(ref_id))
    submissions = [sid for (sid,) in db.query(CodeEvaluation.submission_id)
                   .filter(CodeEvaluation.id.in_(refs["evaluation"] or [-1]))]
    db.execute(delete(GamificationAward).where(GamificationAward.user_id == user_id))
    db.execute(delete(HintEvent).where(HintEvent.user_id == user_id,
                                       HintEvent.evaluation_id.in_(refs["evaluation"] or [-1])))
    db.execute(delete(LearningEvent).where(LearningEvent.id.in_([e.id for e in events] or [-1])))
    db.execute(delete(McqAttempt).where(McqAttempt.id.in_(refs["mcq_attempt"] or [-1])))
    db.execute(delete(DebugAttempt).where(DebugAttempt.id.in_(refs["debug_attempt"] or [-1])))
    db.execute(delete(CodeSubmission).where(CodeSubmission.id.in_(submissions or [-1])))
    db.execute(delete(PracticeSession).where(PracticeSession.user_id == user_id))
    db.execute(delete(TopicMastery).where(TopicMastery.user_id == user_id))
    db.flush()
    for seed in (db.query(LearningEvent)
                 .filter(LearningEvent.user_id == user_id, LearningEvent.kind.in_(SEED_KINDS),
                         LearningEvent.topic.isnot(None))
                 .order_by(LearningEvent.id)):
        row = db.get(TopicMastery, (user_id, seed.topic, seed.language))
        if row is None:
            row = TopicMastery(user_id=user_id, topic=seed.topic, language=seed.language,
                               evidence_count=0)
            db.add(row)
        row.mastery = seed.mastery_after if seed.mastery_after is not None else seed.score
        row.evidence_count = max(row.evidence_count or 0, 1)
        row.last_event_at = seed.created_at
        db.flush()
    gamification.rebuild(db, user_id)
    derive_profile(db, user_id)
    db.commit()


def play_week(db, user: User) -> None:
    week = Week(db, user)
    now = datetime.now(timezone.utc)

    def session(days_ago: int) -> None:
        week.at = (now - timedelta(days=days_ago)).replace(hour=13, minute=30, second=0,
                                                         microsecond=0)

    def later(minutes: int) -> None:
        week.at += timedelta(minutes=minutes)

    session(6)
    for topic in ("hashing_sets", "recursion", "searching", "sorting", "stacks_queues",
                  "dynamic_programming"):
        week.mcq(topic, correct=True)
        later(2)
    week.mcq("conditionals", correct=False, tag="wrong_comparison_operator")
    later(3)
    largest = week.code("conditionals", title="Largest of Three", share=0.5,
                        tags=("wrong_comparison_operator",))
    later(6)
    week.code("conditionals", retry=largest)
    later(3)
    week.mcq("loops", correct=False, tag="off_by_one")
    later(2)
    week.finish(FEEDBACK[0])

    session(4)
    recursive = week.code("recursion", share=0.5, tags=("missing_base_case",))
    week.hints(recursive, [1, 2])
    later(4)
    week.code("recursion", retry=recursive)
    later(3)
    week.mcq("dynamic_programming", correct=True)
    later(3)
    week.code("dynamic_programming", share=4 / 11, tags=("off_by_one",))
    later(6)
    week.code("dynamic_programming", aim=True, tags=("off_by_one",))
    later(5)
    week.debug("sorting", fixed=True)
    later(3)
    week.code("basics_io")
    later(4)
    week.debug("basics_io", fixed=True)
    later(3)
    week.code("arrays")
    later(4)
    week.mcq("arrays", correct=False, tag="off_by_one")
    later(2)
    week.mcq("strings", correct=False, tag="off_by_one")
    later(2)
    week.finish(FEEDBACK[1])

    session(2)
    week.mcq("recursion", correct=False, tag="missing_base_case")
    later(3)
    week.debug("recursion", fixed=False, tag="missing_base_case")
    later(6)
    triangle = week.code("conditionals", title="Triangle Type", share=5 / 11,
                         tags=("wrong_comparison_operator",))
    week.hints(triangle, [1, 2, 3], minutes_apart=3)
    later(5)
    week.code("conditionals", retry=triangle)
    later(4)
    bill = week.code("conditionals", title="Electricity Bill", share=4 / 11,
                     tags=("wrong_comparison_operator",))
    week.hints(bill, [1, 2])
    later(5)
    week.code("conditionals", retry=bill)
    later(4)
    week.debug("conditionals", fixed=False, tag="wrong_comparison_operator", aim=True)
    later(4)
    week.code("searching")
    later(3)
    week.mcq("searching", correct=False, tag="off_by_one")
    later(3)
    week.debug("stacks_queues", fixed=True)
    later(4)
    week.code("hashing_sets", aim=True, tags=("off_by_one",))
    later(2)
    week.finish(FEEDBACK[2])

    session(1)
    week.mcq("basics_io", correct=True)
    later(3)
    week.code("basics_io", aim=True)
    later(5)
    week.mcq("functions", correct=False)
    later(3)
    week.code("functions", aim=True, tags=("return_vs_print",))
    later(5)
    week.code("loops")
    later(5)
    for topic, tag in (("loops", "off_by_one"), ("arrays", "off_by_one"),
                       ("strings", "off_by_one"), ("sorting", "wrong_comparison_operator"),
                       ("searching", "off_by_one"), ("stacks_queues", "off_by_one"),
                       ("recursion", "missing_base_case")):
        week.code(topic, aim=True, tags=(tag,))
        later(5)
    week.finish(FEEDBACK[3])

    last_seen: dict[str, datetime] = {}
    for e in (db.query(LearningEvent).filter(LearningEvent.user_id == user.id,
                                             LearningEvent.language == LANGUAGE)):
        if e.topic and (e.topic not in last_seen or e.created_at > last_seen[e.topic]):
            last_seen[e.topic] = e.created_at
    for row in db.query(TopicMastery).filter(TopicMastery.user_id == user.id,
                                             TopicMastery.language == LANGUAGE):
        row.last_event_at = last_seen.get(row.topic, row.last_event_at)
    db.commit()
    gamification.rebuild(db, user.id)
    derive_profile(db, user.id)
    db.commit()


def _describe(item: policy.NextItem) -> str:
    payload = item.item
    name = payload.get("title") or payload.get("stem") or payload.get("statement") or ""
    return (f"{item.item_type} | {TOPIC_LABELS[item.topic]} | {item.difficulty} | "
            f"{name.splitlines()[0][:70]}")


def _after_correct(db, user_id: int, first: policy.NextItem) -> policy.NextItem | None:
    """What the Assessment Agent serves once `first` is answered correctly, worked out on
    a copy of the profile so nothing is written."""
    plan = policy.load_profile(db, user_id)
    state = plan.topics[first.topic]
    state.mastery = updated_mastery(state.mastery, WEIGHTS["mcq"], 1.0)
    state.evidence += 1
    state.last_event_at = datetime.now(timezone.utc)
    for topic in policy.ranked_topics(plan):
        plan.topic = topic
        plan.item_type = policy.choose_type(db, plan, topic)
        plan.difficulty = policy.choose_difficulty(plan.topics[topic].mastery)
        plan.notes = []
        found = policy.pick_item(db, plan)
        if found is not None:
            return found
    return None


def report(db, user_id: int) -> bool:
    snap = mastery_snapshot(db, user_id)
    print("\nTopic levels (python):")
    for t in sorted(snap["topics"], key=lambda t: -t["mastery"]):
        lock = "" if t["unlocked"] else "  (locked)"
        print(f"  {t['label']:22} {round(t['mastery'] * 100):3d}%  {t['band']:10} "
              f"{t['evidence_count']} answers{lock}")
    print("Top mistakes:", ", ".join(f"{m['tag']} x{m['count']}"
                                     for m in snap["top_misconceptions"][:4]))
    user = db.get(User, user_id)
    sessions = (db.query(PracticeSession).filter(PracticeSession.user_id == user_id)
                .order_by(PracticeSession.id).all())
    graded = sum(s.graded_count for s in sessions)
    print(f"Sessions: {len(sessions)} finished, {graded} graded answers | points {user.points} "
          f"| badges {', '.join(user.badges or []) or 'none'}")

    first = next_item(db, user_id)
    second = _after_correct(db, user_id, first)
    print("\nNext item:            ", _describe(first))
    print("After a correct answer:", _describe(second) if second else "nothing")
    ok = (first.item_type == "mcq" and first.topic == "recursion" and first.difficulty == "easy"
          and second is not None and second.item_type == "coding"
          and second.item.get("title") == KEPT_FOR_DEMO)
    print("Demo path:", "OK" if ok else "NOT as planned")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", action="store_true", required=True,
                        help="replace the student's practice history with the demo week")
    parser.add_argument("--email", default=AuthService.LOCAL_EMAIL)
    args = parser.parse_args()

    require_local_db()
    policy.generate_coding = lambda *a, **k: (None, None)
    with SessionLocal() as db:
        user = db.query(User).filter(User.email == args.email).first()
        if user is None:
            raise SystemExit(f"No account {args.email}; open PET once to create it.")
        if user.preferred_language != LANGUAGE:
            raise SystemExit(f"{args.email} studies {user.preferred_language}, not {LANGUAGE}.")
        if not db.query(LearningEvent.id).filter(LearningEvent.user_id == user.id,
                                                 LearningEvent.kind == "diagnostic_seed",
                                                 LearningEvent.language == LANGUAGE).first():
            raise SystemExit("Finish onboarding first: the week builds on its readings.")
        clear_week(db, user.id)
        play_week(db, user)
        if not report(db, user.id):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
