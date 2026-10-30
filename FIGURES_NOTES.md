# FIGURES_NOTES.md -- Figure 31 (QPCA loadings) and Figure 32 (entanglement map)

Unzip over the repo root AFTER fix_oct28_pipeline_stats.zip and delta_target_patch.zip.
`run_pipeline.py` and `src/baselines/qgan_llm.py` here are CUMULATIVE (they already contain the earlier
patches); the other four files are new or extended.

## Why the figures were "Not generated"
- Figure 31: scripts/run_qpca.py existed but was not a pipeline step, was never run (no
  results/qpca_results.json), and the repo had no plotting code at all.
- Figure 32: the only entanglement code gave ONE number per run (half-vs-half entropy/purity); no
  pairwise quantity was computed or saved.

## What now happens
- Pipeline step `run_qpca` (optional, skippable once done) -> results/qpca_results.json.
- QGAN-LLM evaluate() saves results/entanglement_maps/<model>_seed<N>.json: the 20x20 (or 12x12)
  pairwise quantum mutual information I(i:j) = S(i)+S(j)-S(ij), in bits, from the generator statevector
  on the fixed reference input. ~1-2 s at 20 qubits; a failure only prints a warning, never breaks a run.
- Pipeline step `make_figures` (always re-runs, never blocks) -> results/figures/figure_31_qpca_loadings.png
  and figure_32_entanglement_map.png (one panel per saved map, shared colour scale).
- Standalone: `python -m scripts.run_qpca` then `python -m scripts.make_figures`.

## To get the maps for runs you already have
Maps are written during evaluate(), so re-evaluate the QGAN configs (generators are NOT retrained):
  python run_pipeline.py --steps "train_eval:QGAN-LLM,train_eval:QGAN-LLM (12-qubit),train_eval:QGAN-LLM (linear topology),train_eval:QGAN-LLM (no noise)"
  python run_pipeline.py --steps run_qpca,make_figures

## Reading them honestly
Exact simulator quantities on a fixed input (not measurement-reconstructed), describing the circuits
themselves. They show HOW entangled the generator is, not that entanglement helps: H4 is still
degenerate and H5's entanglement-vs-accuracy correlation rests on n=5 points. The QPCA runs on
classical features (its JSON `note` says so) and is a simulation, so it carries no speedup claim.
