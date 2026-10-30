# IntentBench / Tiny ML Router

![Tests](https://github.com/Hemanth-TS/intentbench-ml/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)
![License](https://img.shields.io/badge/license-MIT-22c55e)

A from-scratch multinomial Naive Bayes intent classifier with held-out evaluation, rejection and model serialization.

> **New portfolio demo · created 30 October 2026.** This is an original,
> AI-assisted implementation created for this portfolio. It is not a past
> client delivery, production deployment, or evidence of commercial results.
> All supplied data is synthetic. No popular project was copied or forked.

## Run in under a minute

Only Python 3.11 or later is required. No API keys, model downloads,
paid services or third-party Python packages are needed.

```sh
git clone https://github.com/Hemanth-TS/intentbench-ml.git
cd intentbench-ml
python intentbench.py
python -m unittest discover -s tests -v
```

## Model and evaluation

`Synthetic training set → Unicode tokens → multinomial Naive Bayes → posterior threshold → intent / unknown`

The implementation uses log probabilities, Laplace smoothing and a stable
softmax. It trains on 15 fictional examples across `book`, `cancel` and
`handoff`, then evaluates six separate held-out examples. Reports include
accuracy, macro F1, a confusion matrix and coverage (the accepted fraction).
Rejected examples count as errors for their true class.

```sh
python intentbench.py --text "please schedule a visit"
python intentbench.py --text "galactic weather" --threshold 0.8
python intentbench.py --save model.json
```

`Classifier.load(path)` restores a versioned JSON model without pickle or
executable deserialization. Exact case-insensitive training/test text overlap
is rejected by the CLI. Add paraphrases and independent data before drawing
conclusions; the current split is deliberately tiny.

## Limits and responsible interpretation

This is a **learned statistical baseline**, not an LLM, speech recognizer or
production intent service. Vocabulary, labels and examples are synthetic.
Posterior scores are not calibrated confidence estimates. Unknown words are
ignored; all-unknown inputs reject immediately. Mixed known/unknown inputs can
still be overconfident. No out-of-domain accuracy, customer performance or
robustness claim is made. Treat metrics as fixture checks, not benchmarks.
Use a human confirmation step before any real booking or cancellation.


## Repository guide

- `intentbench.py`: implementation and command-line interface.
- `samples/`: fictional input fixtures; safe to edit and reuse.
- `tests/`: regression tests, including boundary cases.
- `.github/workflows/tests.yml`: automated Windows/Linux checks.
- [CHANGELOG.md](CHANGELOG.md): release notes.
- [CONTRIBUTING.md](CONTRIBUTING.md): development and bug-report guidance.

## Maintenance and provenance

Version **0.1.0** is an educational demo, with no production support commitment.
Changes should include a regression test and an entry in the changelog.
Algorithms are implemented directly in this repository; there are no vendored
models, third-party source files, proprietary assets or external datasets.
Badges show GitHub's current CI status; sample outputs are not independent benchmarks.

[Portfolio](https://hemanth-ts.github.io/) · [MIT license](LICENSE)
