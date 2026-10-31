# TextDNA evaluation

Measured by running `python eval/run_eval.py` against the starter
evaluation set in `eval/data/starter_eval.jsonl`.

## The dataset (read this first)

The starter set has **30 short texts**, written by hand for
this project as a smoke evaluation: fallacy examples, loaded/biased
statements, clean argumentative texts, neutral factual texts and
descriptive non-argumentative texts. Labels mark whether a text
contains a fallacy (and which type), whether it is biased in the
operational sense used here (sustained loaded / absolutist
language), and whether it has a claim-plus-premise argument
structure.

**This is not a benchmark.** The detector rules were developed
alongside these examples, so the scores below are optimistic by
construction. They are reported to demonstrate the measurement
method and to catch regressions when the rules change. A serious
evaluation needs a larger, independently labelled corpus.

## Results per detector

| detector | precision | recall | f1 | accuracy | tp | fp | fn | tn |
|---|---|---|---|---|---|---|---|---|
| fallacy (any type) | 1.000 | 1.000 | 1.000 | 1.000 | 11 | 0 | 0 | 19 |
| bias (loaded language) | 0.818 | 1.000 | 0.900 | 0.933 | 9 | 2 | 0 | 19 |
| argumentative (claim + premise) | 1.000 | 1.000 | 1.000 | 1.000 | 5 | 0 | 0 | 25 |

## Fallacy recall by type

| fallacy type | examples | detected | recall |
|---|---|---|---|
| ad_hominem | 2 | 2 | 1.000 |
| appeal_to_vague_authority | 1 | 1 | 1.000 |
| bandwagon | 1 | 1 | 1.000 |
| false_dilemma | 1 | 1 | 1.000 |
| hasty_generalization | 2 | 2 | 1.000 |
| slippery_slope | 2 | 2 | 1.000 |
| straw_man | 1 | 1 | 1.000 |
| whataboutism | 1 | 1 | 1.000 |

## Mistakes on the starter set

| id | detector | gold | predicted |
|---|---|---|---|
| f04 | bias | False | True |
| f07 | bias | False | True |

Reading the mistakes: both bias false positives (f04, f07) are
fallacy texts whose absolutist phrasing ('every course', 'nothing
at all', 'everyone knows') trips the absolutist-language component
of the bias score. That overlap between fallacy rhetoric and the
bias operationalisation is a known limitation of the lexicon
approach, and the gold labels were deliberately left unchanged
after seeing the predictions.

## Method notes

- Detectors are the transparent rule/lexicon systems in
  `app/detectors.py`; every rule carries an explanation string.
- Embeddings are irrelevant to these three detectors, so the eval
  forces the TF-IDF backend for speed.
- Bias labels follow the operational definition in `detect_bias`:
  sustained loaded/absolutist language, threshold score >= 12.
