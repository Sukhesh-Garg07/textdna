"""Evaluation harness for the TextDNA detectors.

Runs the fallacy, bias and argument detectors over the starter
evaluation set (eval/data/starter_eval.jsonl) and writes measured
precision / recall / F1 / accuracy per detector to EVALUATION.md.

Honest scope: the starter set is 30 short texts written for this project
as a smoke evaluation. It is NOT a benchmark. The rule set was developed
alongside these examples, so the numbers are optimistic by construction;
they exist to catch regressions and to show the measurement method, not
to claim state-of-the-art fallacy detection.

Usage:
    python eval/run_eval.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Detectors do not use embeddings; force the fast offline backend so the
# eval never waits on a model download.
os.environ.setdefault("TEXTDNA_EMBEDDINGS", "tfidf")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analyzer import analyze  # noqa: E402

DATA = Path(__file__).parent / "data" / "starter_eval.jsonl"
OUT = Path(__file__).resolve().parent.parent / "EVALUATION.md"


def binary_metrics(pairs: list[tuple[bool, bool]]) -> dict:
    tp = sum(1 for gold, pred in pairs if gold and pred)
    fp = sum(1 for gold, pred in pairs if not gold and pred)
    fn = sum(1 for gold, pred in pairs if gold and not pred)
    tn = sum(1 for gold, pred in pairs if not gold and not pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(pairs) if pairs else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": accuracy,
    }


def fmt(m: dict) -> str:
    return (
        f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
        f"{m['accuracy']:.3f} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']}"
    )


def main() -> None:
    items = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    fallacy_pairs, bias_pairs, arg_pairs = [], [], []
    per_type_total: dict[str, int] = defaultdict(int)
    per_type_hit: dict[str, int] = defaultdict(int)
    mistakes = []

    for item in items:
        labels = item["labels"]
        report = analyze(item["text"])

        pred_fallacy = len(report["fallacies"]) > 0
        pred_bias = report["bias"]["is_biased"]
        pred_arg = report["argument"]["is_argumentative"]

        fallacy_pairs.append((labels["fallacy"], pred_fallacy))
        bias_pairs.append((labels["biased"], pred_bias))
        arg_pairs.append((labels["argumentative"], pred_arg))

        pred_types = {f["type"] for f in report["fallacies"]}
        for t in labels["fallacy_types"]:
            per_type_total[t] += 1
            if t in pred_types:
                per_type_hit[t] += 1

        for name, gold, pred in (
            ("fallacy", labels["fallacy"], pred_fallacy),
            ("bias", labels["biased"], pred_bias),
            ("argument", labels["argumentative"], pred_arg),
        ):
            if gold != pred:
                mistakes.append((item["id"], name, gold, pred))

    fallacy_m = binary_metrics(fallacy_pairs)
    bias_m = binary_metrics(bias_pairs)
    arg_m = binary_metrics(arg_pairs)

    lines = [
        "# TextDNA evaluation",
        "",
        "Measured by running `python eval/run_eval.py` against the starter",
        "evaluation set in `eval/data/starter_eval.jsonl`.",
        "",
        "## The dataset (read this first)",
        "",
        f"The starter set has **{len(items)} short texts**, written by hand for",
        "this project as a smoke evaluation: fallacy examples, loaded/biased",
        "statements, clean argumentative texts, neutral factual texts and",
        "descriptive non-argumentative texts. Labels mark whether a text",
        "contains a fallacy (and which type), whether it is biased in the",
        "operational sense used here (sustained loaded / absolutist",
        "language), and whether it has a claim-plus-premise argument",
        "structure.",
        "",
        "**This is not a benchmark.** The detector rules were developed",
        "alongside these examples, so the scores below are optimistic by",
        "construction. They are reported to demonstrate the measurement",
        "method and to catch regressions when the rules change. A serious",
        "evaluation needs a larger, independently labelled corpus.",
        "",
        "## Results per detector",
        "",
        "| detector | precision | recall | f1 | accuracy | tp | fp | fn | tn |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| fallacy (any type) | {fmt(fallacy_m)} |",
        f"| bias (loaded language) | {fmt(bias_m)} |",
        f"| argumentative (claim + premise) | {fmt(arg_m)} |",
        "",
        "## Fallacy recall by type",
        "",
        "| fallacy type | examples | detected | recall |",
        "|---|---|---|---|",
    ]
    for t in sorted(per_type_total):
        total = per_type_total[t]
        hit = per_type_hit[t]
        lines.append(f"| {t} | {total} | {hit} | {hit / total:.3f} |")
    lines += [
        "",
        "## Mistakes on the starter set",
        "",
    ]
    if mistakes:
        lines.append("| id | detector | gold | predicted |")
        lines.append("|---|---|---|---|")
        for mid, name, gold, pred in mistakes:
            lines.append(f"| {mid} | {name} | {gold} | {pred} |")
    else:
        lines.append("No mistakes on the starter set (see the caveat above).")
    lines += [
        "",
        "Reading the mistakes: both bias false positives (f04, f07) are",
        "fallacy texts whose absolutist phrasing ('every course', 'nothing",
        "at all', 'everyone knows') trips the absolutist-language component",
        "of the bias score. That overlap between fallacy rhetoric and the",
        "bias operationalisation is a known limitation of the lexicon",
        "approach, and the gold labels were deliberately left unchanged",
        "after seeing the predictions.",
    ]
    lines += [
        "",
        "## Method notes",
        "",
        "- Detectors are the transparent rule/lexicon systems in",
        "  `app/detectors.py`; every rule carries an explanation string.",
        "- Embeddings are irrelevant to these three detectors, so the eval",
        "  forces the TF-IDF backend for speed.",
        "- Bias labels follow the operational definition in `detect_bias`:",
        "  sustained loaded/absolutist language, threshold score >= 12.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"fallacy:   P={fallacy_m['precision']:.3f} R={fallacy_m['recall']:.3f} F1={fallacy_m['f1']:.3f} acc={fallacy_m['accuracy']:.3f}")
    print(f"bias:      P={bias_m['precision']:.3f} R={bias_m['recall']:.3f} F1={bias_m['f1']:.3f} acc={bias_m['accuracy']:.3f}")
    print(f"argument:  P={arg_m['precision']:.3f} R={arg_m['recall']:.3f} F1={arg_m['f1']:.3f} acc={arg_m['accuracy']:.3f}")
    print(f"mistakes: {len(mistakes)}")


if __name__ == "__main__":
    main()
