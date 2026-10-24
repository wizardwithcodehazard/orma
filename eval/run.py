"""
run.py - The eval harness. Every case gets a fresh memory: its sessions are fed in as user turns (one
minute apart) and slept on with the real model, then the graph is scored (score.py) and the questions
are asked with memory, several samples each, and once more without memory as a baseline.

    python -m eval.run --upstream http://127.0.0.1:9931/v1 --model MODEL --label baseline
    python -m eval.run ... --set enable_fact_quotes=false --label no-quotes     # an ablation
    python -m eval.run ... --cases trek-date,quit-coffee --samples 1            # a quick check
    python -m eval.run --compare eval/runs/a.json eval/runs/b.json              # side by side

Each run is saved as JSON under eval/runs/ with every check, answer and timing, so runs can be compared
after the fact. Times in the cases are local to the cases file's timezone.
"""

import argparse
import dataclasses
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from orma import clock
from orma.config import SmrutiConfig
from orma.embed import FastEmbedder
from orma.llm import LLMClient
from orma.memory import Memory

from eval.score import CaseScore, Check, answer_passes, finds, score_memory

HERE = Path(__file__).parent
SPACE = "default"
TURN_GAP = 60.0  # seconds between the turns of a session
SLEEP_AFTER = 600.0  # a session is slept on this long after its last turn
ASK_AFTER = 3600.0  # questions come this long after the last session


def load_cases(path: Path, only: set[str] | None = None) -> tuple[str, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = [c for c in data["cases"] if not only or c["id"] in only]
    missing = (only or set()) - {c["id"] for c in cases}
    if missing:
        raise SystemExit(f"unknown case ids: {', '.join(sorted(missing))}")
    return data.get("timezone", ""), cases


def overrides(pairs: list[str]) -> dict:
    """--set key=value pairs, typed by SmrutiConfig's field types."""
    defaults = SmrutiConfig(data_dir=Path("."))
    out = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if not hasattr(defaults, key) or key == "data_dir":
            raise SystemExit(f"--set: SmrutiConfig has no settable field {key!r}")
        kind = type(getattr(defaults, key))
        out[key] = raw.strip().lower() in ("1", "true", "yes", "on") if kind is bool else kind(raw)
    return out


def add(memory: Memory, turn, at: float) -> None:
    """A turn is a user message (a string) or {"role": ..., "text": ...}: chat roles go through observe like the
    proxy, notes and documents through add (a document's speaker is its "speaker", else "document")."""
    if isinstance(turn, str):
        memory.observe("user", turn, SPACE, now=at)
    elif turn["role"] in ("user", "assistant", "tool"):
        memory.observe(turn["role"], turn["text"], SPACE, now=at)
    else:
        memory.add(turn["text"], SPACE, role=turn["role"], source="eval", speaker=turn.get("speaker", ""), at=at)


def run_case(case: dict, tz_name: str, settings: dict, llm, model: str, embedder, samples: int, baseline: bool,
             workdir: Path, log=print) -> tuple[CaseScore, list[dict]]:
    tz = clock.zone(tz_name)
    config = SmrutiConfig(data_dir=workdir / case["id"], timezone=tz_name, **settings)
    memory = Memory(config=config, embedder=embedder, llm=llm, model=model)
    score = CaseScore(case["id"])
    answers = []
    try:
        last = 0.0
        for session in case["sessions"]:
            start = clock.parse_time(session["at"], None, tz)
            for i, turn in enumerate(session["turns"]):
                last = start + i * TURN_GAP
                add(memory, turn, last)
            began = time.time()
            report = memory.sleep(SPACE, now=last + SLEEP_AFTER)
            score.sleep_seconds += time.time() - began
            if report.failed_batches:
                score.checks.append(Check("sleep", "sleep batch failed", False, str(report.failed_batches)))
        score.checks += score_memory(memory.store, SPACE, case.get("expect", {}), tz)
        score.states_total = len(memory.store.current_states(SPACE))
        asked_at = last + ASK_AFTER
        for spec in case.get("finds", []):
            found, detail = finds(memory.recaller.search(SPACE, spec["q"], asked_at), spec["terms"])
            score.checks.append(Check("finds", f"{spec['q']} -> {' '.join(spec['terms'])}", found, detail[:80]))
        for question in case.get("questions", []):
            messages = memory.build_context([{"role": "user", "content": question["q"]}], SPACE, now=asked_at)
            runs = [(True, llm.chat(messages, model=model, temperature=0.0 if i == 0 else 0.7)) for i in range(samples)]
            if baseline:
                runs.append((False, llm.chat([{"role": "user", "content": question["q"]}], model=model, temperature=0.0)))
            for with_memory, answer in runs:
                ok = answer_passes(answer, question.get("must", []), question.get("must_not", []))
                answers.append({"case": case["id"], "q": question["q"], "memory": with_memory, "passed": ok,
                                "answer": answer})
            passed = sum(a["passed"] for a in answers if a["q"] == question["q"] and a["memory"])
            score.checks.append(Check("question", question["q"], passed == samples, f"{passed}/{samples} samples"))
    finally:
        memory.store.close()
    failed = [c for c in score.checks if not c.passed]
    log(f"  {case['id']:<16} {len(score.checks) - len(failed)}/{len(score.checks)} checks, "
        f"{score.states_total} states, sleep {score.sleep_seconds:.0f}s")
    for check in failed:
        log(f"      FAIL {check.kind:<8} {check.label}  [{check.detail}]")
    return score, answers


def summarize(scores: list[CaseScore], answers: list[dict]) -> dict:
    def rate(kind: str) -> tuple[int, int]:
        checks = [c for s in scores for c in s.of(kind)]
        return sum(c.passed for c in checks), len(checks)

    with_memory = [a for a in answers if a["memory"]]
    without = [a for a in answers if not a["memory"]]
    return {
        "states_found": rate("current"),
        "updates_closed": rate("closed"),
        "wrong_states_avoided": rate("absent"),
        "retrieval": rate("finds"),
        "event_dates": rate("event"),
        "user_aliases": rate("alias"),
        "answer_samples_with_memory": (sum(a["passed"] for a in with_memory), len(with_memory)),
        "answers_without_memory": (sum(a["passed"] for a in without), len(without)),
        "states_total": sum(s.states_total for s in scores),
        "sleep_seconds": round(sum(s.sleep_seconds for s in scores), 1),
    }


def table(runs: list[tuple[str, dict]]) -> str:
    keys = list(runs[0][1])
    lines = ["| metric | " + " | ".join(name for name, _ in runs) + " |", "|---" * (len(runs) + 1) + "|"]
    for key in keys:
        cells = []
        for _, summary in runs:
            value = summary.get(key)
            cells.append(f"{value[0]}/{value[1]}" if isinstance(value, (list, tuple)) else str(value))
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream", default="http://localhost:11434/v1", help="OpenAI-compatible base URL.")
    parser.add_argument("--model", help="Model for sleep and answers.")
    parser.add_argument("--cases", help="Comma-separated case ids (default: all).")
    parser.add_argument("--split", choices=["dev", "test"], default="dev",
                        help="dev cases (eval/cases.json) to tune on; test cases (eval/test_cases.json) only to report.")
    parser.add_argument("--cases-file", help="Overrides --split.")
    parser.add_argument("--samples", type=int, default=3, help="Answers per question with memory.")
    parser.add_argument("--no-baseline", action="store_true", help="Skip the answers without memory.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="SmrutiConfig override.")
    parser.add_argument("--label", default="run")
    parser.add_argument("--out", default=str(HERE / "runs"))
    parser.add_argument("--compare", nargs="+", metavar="RUN.json", help="Print saved runs side by side and exit.")
    args = parser.parse_args(argv)

    if args.compare:
        runs = [(Path(p).stem, json.loads(Path(p).read_text(encoding="utf-8"))["summary"]) for p in args.compare]
        print(table(runs))
        return
    if not args.model:
        parser.error("--model is required")

    cases_file = Path(args.cases_file) if args.cases_file else HERE / ("cases.json" if args.split == "dev" else
                                                                         "test_cases.json")
    tz_name, cases = load_cases(cases_file, set(args.cases.split(",")) if args.cases else None)
    settings = overrides(args.set)
    llm = LLMClient(args.upstream, model=args.model, timeout=600)
    defaults = SmrutiConfig(data_dir=Path("."))
    embedder = FastEmbedder(defaults.embedding_model, defaults.embedding_dim)  # loaded once, shared by every case
    print(f"eval '{args.label}' ({cases_file.name}): {len(cases)} cases on {args.model} via {args.upstream}"
          + (f", overrides {settings}" if settings else ""))
    started = time.time()
    scores, answers = [], []
    with tempfile.TemporaryDirectory(prefix="smruti-eval-") as workdir:
        for case in cases:
            score, case_answers = run_case(case, tz_name, settings, llm, args.model, embedder, args.samples,
                                           not args.no_baseline, Path(workdir))
            scores.append(score)
            answers += case_answers
    summary = summarize(scores, answers)
    summary["wall_seconds"] = round(time.time() - started, 1)
    print("\n" + table([(args.label, summary)]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{datetime.now():%Y%m%d-%H%M%S}-{args.label}.json"
    path.write_text(json.dumps({
        "label": args.label, "cases": cases_file.name, "model": args.model, "upstream": args.upstream,
        "overrides": settings,
        "summary": summary, "cases": [dataclasses.asdict(s) for s in scores], "answers": answers,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    sys.exit(main())
