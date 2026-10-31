# FIX_NOTES_OCT30.md -- prepare_data MemoryError, make_figures crash, QPCA eigenvector mismatch

Unzip over the repo root (overwrites 6 files). Do NOT use --reset.

1. prepare_data MemoryError (889 MiB, df.copy() in clean_indicators): clean_prices, clean_indicators and
   add_technical_indicators take copy=False and edit the 5.5M-row frame in place; prepare_data uses it and calls
   gc.collect() between steps. Results are identical (tested: copy=True and copy=False give equal frames).
   A MemoryError now prints what to do. Run prepare_data on its own, with no parallel train_eval jobs open.
2. make_figures: ModuleNotFoundError matplotlib. matplotlib is now in requirements.txt (`pip install matplotlib`),
   and the script prints a skip message and exits normally if it is missing.
3. QPCA: with n_clock=6 the clock bin is 1/64 wide, but the 5 eigenvalues of rho sit near 0.2 each, so the peaks
   merged and component 2 had overlap 0.0 with classical PCA (the saved qpca_results.json shows ~1e-16 for both).
   Synthetic check with a similar spectrum: n_clock=6 -> overlaps [1, 0]; n_clock=8 -> [1, 1].
   Default is now n_clock=8, auto_clock raises it (cap 10) until the top eigenvalue gaps span 4 bins, and
   run_qpca.py writes eigenvectors_match_classical and warns when overlap < 0.9.
   Re-run: python -m scripts.run_qpca && python -m scripts.make_figures
