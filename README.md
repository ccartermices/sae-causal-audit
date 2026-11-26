# A Calibrated Causal Audit of Sparse Autoencoder Features in EEG Decoders

Code and archived results for the study; the article itself is submitted
separately and is not part of this repository.

Repository: <https://github.com/ccartermices/sae-causal-audit>

The study asks a narrow question with a checkable answer: when a sparse
autoencoder (SAE) feature of an EEG decoder is given a physiological name, does
that name tell you anything about whether the decoder *uses* the feature? We
built an instrument that measures the causal effect of removing one atom,
validated it on a paradigm with a known answer, and then tested three things
the literature routinely assumes.

**Result, in one line:** semantic agreement does not predict the causal effect;
a first-order score from one backward pass does; and no *reporting unit* — atom
identity across seeds, a bootstrap-stable cell set, a cross-subject cell map —
survives the audit.

## Layout

```
scripts/    analysis pipeline (d00-d42) and the number registry builder/validator
lean/       Lean 4 proofs of the estimators' algebraic core
registry/   numbers.yaml: 193 result entries, each bound to a source artefact,
            the expression that evaluates it, and the claim it supports
probe/      archived probe artefacts (JSON) the registry evaluates against
data/       small committed facts (registry_facts.json, locked/, CSVs)
```

## Verify the numbers (no raw data required)

```bash
python scripts/build_registry.py    # numbers.yaml -> numbers.{json,md,tex}
python scripts/check_registry.py    # 11 checks, see below
```

`check_registry.py` runs eleven checks: structural validity, evaluation,
declared bounds, provenance, staleness (source hashes), determinism (two builds,
same manifest hash), two poisoning tests (perturb a source CSV / a generated
value; the registry must react), zero literals, raw-file completeness, and a
digit scan over the manuscript sources (skipped here — the article is not in
this repository). Every entry recomputes from `probe/` and `data/` only.

## Lean proofs

```bash
export LEAN_BIN=/path/to/lean-4.34.1/bin/lean
export MATHLIB_LAKE=/path/to/mathlib4/.lake
bash lean/build.sh
```

Requires Lean 4.34.1 + Mathlib 4.34.1 (already `lake build`). Four propositions
proved without `sorry` or custom axioms: exactness of the first-order score
under an affine head, the single-feature corollary, the algebraic zero of
matched-norm ± pairs, and additivity of set ablation under a linear head.

## Data

All recordings are public and are **not** redistributed here: EEGMMIDB
(PhysioNet), BCI Competition IV dataset 2a, and the SSVEP Benchmark (Wang et
al., 2017). Fetch scripts: `scripts/d00_fetch.py`, `d01_audit.py`,
`d02_annot_checks.py`; `data/AVAILABILITY.md` records per source what was
reachable and what that cost the design. The registry keeps only the small
facts extracted from the recordings, so every number re-derives without a 9 GB
download.

## License

Code (`scripts/`, `lean/`) and committed data are released under the MIT
License; see `LICENSE`.
