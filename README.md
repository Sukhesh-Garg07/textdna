# TextDNA

A multi-layer NLP text-intelligence engine. Paste in a piece of text and
TextDNA takes it apart layer by layer -- linguistic structure, meaning,
argument shape, and cognitive red flags -- then folds the findings into a
single composite **TextDNA Score** with an auditable breakdown.

Everything runs **locally on CPU**. No text is sent anywhere.

![layers](https://img.shields.io/badge/layers-linguistic%20%C2%B7%20semantic%20%C2%B7%20argument%20%C2%B7%20cognitive%20%C2%B7%20generative-blue)

## What it does

| layer | what it extracts | how |
|---|---|---|
| linguistic | named entities, POS profile, dependency profile, lexical diversity | spaCy `en_core_web_sm` |
| semantic | coherence and drift between consecutive sentences | sentence embeddings (see below) |
| argument-mining | per-sentence claim / premise roles, evidence markers, is the text argumentative? | documented cue patterns |
| cognitive | fallacy signals (8 types) and a bias score from loaded / absolutist language | transparent rules + lexicons, every hit explained |
| generative | composite TextDNA Score (0-100) with five sub-scores, plus a plain-language summary | fixed published weights |

### Fallacy and bias detection you can audit

There is no black-box classifier here. Each fallacy type (ad hominem,
straw man, slippery slope, false dilemma, bandwagon, hasty
generalization, whataboutism, appeal to vague authority) is a set of
regex patterns in `app/detectors.py`, and every finding returns the
matched text plus a plain-language explanation of why the pattern is a
signal. The bias score is a lexicon-density measure with a published
formula and threshold. You can disagree with a rule -- and you can point
at exactly which rule fired.

The honest cost of this approach is recall: subtle or paraphrased
fallacies will be missed. The evaluation harness below measures the
rules instead of asserting they work.

### Semantic layer: two backends, reported honestly

1. **sentence-transformers** (`all-MiniLM-L6-v2`, dense embeddings) --
   used automatically when the package is installed and the model loads.
2. **TF-IDF fallback** (scikit-learn) -- always available, fully offline,
   but only captures lexical overlap, so coherence values run lower.

Every API response names the backend that actually ran
(`semantic.backend`). Force the fallback with
`TEXTDNA_EMBEDDINGS=tfidf`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# optional, for dense embeddings:
pip install sentence-transformers

uvicorn app.main:app --port 8000
```

Open http://127.0.0.1:8000, paste text, and read the report. Or use the
API directly:

```bash
curl -s -X POST http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text": "The city should invest in bus lanes because they move more people per hour than car lanes. According to the transit authority, the pilot corridor in Berlin cut commute times by 18%."}'
```

Response (abridged):

```json
{
  "linguistic": {"entities": [{"text": "Berlin", "label": "GPE"}]},
  "semantic": {"backend": "sentence-transformers:all-MiniLM-L6-v2",
               "coherence": 0.3774, "drift": 0.6226},
  "argument": {"is_argumentative": true, "claim_count": 1, "premise_count": 2},
  "fallacies": [],
  "bias": {"bias_score": 0.0, "is_biased": false},
  "textdna_score": {
    "overall": 82.7,
    "sub_scores": {"coherence": 50.32, "argumentation": 82.5,
                   "neutrality": 100.0, "soundness": 100.0, "structure": 95.0}
  }
}
```

(Real output from a local run with the sentence-transformers backend.
With the TF-IDF fallback the same text reports lower coherence, because
word overlap is all that backend can see -- the response always names
the backend that produced the numbers.)

## The TextDNA Score

A weighted blend of five sub-scores, weights fixed in `app/scoring.py`:

| sub-score | weight | meaning |
|---|---|---|
| coherence | 0.25 | mean similarity of consecutive sentences (ramp 0.0-0.75 -> 0-100) |
| argumentation | 0.25 | claims + premises + balance + evidence markers |
| neutrality | 0.20 | 100 minus twice the bias score |
| soundness | 0.20 | penalty per fallacy type and finding |
| structure | 0.10 | sentence-length regularity and lexical diversity |

The score is a summary heuristic, not a judgement of truth. A false text
can score well if it is coherent, structured and polite; a true text can
score badly if it is rambling. Read the sub-scores, not just the number.

## Evaluation

`eval/` contains a starter evaluation set -- **30 short hand-written
texts** (`eval/data/starter_eval.jsonl`) labelled for fallacy presence
and type, bias, and argumentative structure -- and a harness that runs
the detectors over it:

```bash
python eval/run_eval.py   # writes EVALUATION.md
```

Measured on the starter set (see `EVALUATION.md` for the full tables):

| detector | precision | recall | f1 | accuracy |
|---|---|---|---|---|
| fallacy (any type) | 1.000 | 1.000 | 1.000 | 1.000 |
| bias (loaded language) | 0.818 | 1.000 | 0.900 | 0.933 |
| argumentative | 1.000 | 1.000 | 1.000 | 1.000 |

**Caveat, stated plainly:** this is a small starter set written for
smoke evaluation, and the rules were developed alongside it, so these
numbers are optimistic by construction. They demonstrate the measurement
method and catch regressions; they are not a benchmark and not evidence
of performance on real-world text. The two bias false positives are
fallacy texts whose absolutist phrasing ("every course", "everyone
knows") trips the bias measure -- a real, documented overlap, kept in
the report rather than labelled away.

## Tests

```bash
python -m pytest tests/ -q
```

23 tests covering the detectors, the analysis pipeline and the API.
The suite forces the TF-IDF backend so it never depends on a model
download.

## Project layout

```
app/
  main.py        FastAPI service (GET /, POST /analyze, GET /health)
  analyzer.py    the five-layer pipeline
  detectors.py   fallacy / bias / argument-role rules (all explained)
  embeddings.py  sentence-transformers with TF-IDF fallback
  scoring.py     composite TextDNA Score, published weights
  templates/     the single-page web UI
eval/
  data/starter_eval.jsonl   30 labelled starter texts
  run_eval.py               harness -> EVALUATION.md
tests/           pytest suite
Dockerfile       container build (CPU)
```

## Limitations

- Fallacy and bias detection is pattern-based: it catches common
  surface forms and misses paraphrase, sarcasm and context-dependent
  cases. Findings are prompts for a closer read, not verdicts.
- `en_core_web_sm` is spaCy's small English model; entity accuracy on
  names and organisations is limited, and other languages are not
  supported.
- The TF-IDF fallback measures word overlap, not meaning; coherence
  and drift values are only comparable within the same backend.
- The starter evaluation set is small and self-authored (see above).

## Next steps

- Larger, independently labelled evaluation corpus (e.g. existing
  fallacy datasets) and per-type precision, not just recall.
- Coreference-aware argument mining (link premises to the claims they
  support, not just label them).
- A fine-tuned transformer classifier as an *additional* signal next to
  the rules, with disagreement cases surfaced in the UI.
- Batch analysis endpoint and CSV export for classroom / newsroom use.
