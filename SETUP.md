# SETUP.md — Preparing the Environment

Follow this once, before your first run. Estimated time: 30–90 minutes,
most of it waiting on data exports and package installs, not active work.

## 1. Hardware

- **GPU strongly recommended** for the classical LSTM/GAN components at
  5.5M-row scale. A single consumer/cloud GPU (e.g. one RTX 4090, or an
  A10/A100 instance from any cloud provider) is enough — this is not a
  distributed-training-scale workload.
- **The quantum simulator (`default.qubit`) is CPU-only regardless of
  GPU**, since PennyLane's default simulator doesn't use CUDA. At 20
  qubits, `default.qubit` keeps a 2^20 ≈ 1-million-entry state vector in
  memory per forward pass — manageable on CPU, but consider
  `pennylane-lightning` (already in `requirements.txt`) as a faster
  drop-in CPU backend by setting `qgan_llm.quantum_device:
  "lightning.qubit"` in the config. `lightning.gpu` exists too if you have
  an NVIDIA GPU and want to install `pennylane-lightning-gpu` (not in the
  default requirements — it needs a matching CUDA toolkit version, check
  PennyLane's install docs for your exact CUDA version before adding it).
- **Disk space**: budget at least 15–20 GB free for the raw minute-level
  CSVs (5.5M + 5.2M rows across two sources) plus their processed/derived
  arrays.

## 2. Python environment

```bash
python3 -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```'
Se7rF1kGGAQ38Jjy9zjDB9DGRlb22wVO8x7Nyehy
FRED_API_KEY
[System.Environment]::SetEnvironmentVariable('FRED_API_KEY', '687e6d45e4a553c4e3e69e24c55fa1de', 'User')
[System.Environment]::SetEnvironmentVariable('NVIDIA_API_KEY', 'Se7rF1kGGAQ38Jjy9zjDB9DGRlb22wVO8x7Nyehy', 'User')
If you have a CUDA GPU, install the CUDA-matched PyTorch build *before*
installing the rest of `requirements.txt` (PyTorch's default pip package
may install a CPU-only build depending on your platform) — see
https://pytorch.org/get-started/locally/ for the exact command for your
CUDA version, then run `pip install -r requirements.txt` afterward; pip
will skip reinstalling torch if the version constraint is already met.

Verify:
```bash
python -m scripts.verify_environment
```
python -m scripts.prepare_data

This checks Python version, every required package, GPU availability,
config loading, and — once you've done steps 3–4 below — whether the
expected data files and API key are actually in place. Run it again after
each of the following steps; it's meant to be your progress checklist.

## 3. Acquiring the data

None of this data is bundled with the code — it comes from licensed or
rate-limited providers this sandbox cannot reach, and you need to place
it yourself at the paths named in `config/default_config.yaml`:

| Config key | Expected path | Source |
|---|---|---|
| `data.dukascopy_file` | `data/raw/dukascopy_eurusd_1min_2012_2025.csv` | Dukascopy Bank SA — primary source |
| `data.forexsb_file` | `data/raw/forexsb_eurusd_1min_2012_2023.csv` | ForexSB — supplementary/cross-validation |
| `data.fred_file` | `data/raw/fred_macro_2011_2025.csv` | FRED (Federal Reserve Economic Data) |
| (VIX) | cached automatically at `data/raw/vix_cache.csv` on first run | yfinance (`^VIX`), pulled live |

**Dukascopy**: Dukascopy Bank SA's historical data feed
(https://www.dukascopy.com/swiss/english/marketwatch/historical/) or
their JForex API, if you have an academic/institutional agreement.
Export as CSV with columns `timestamp, open, high, low, close, volume`
(header names are case-insensitive; the loader lowercases them).

**ForexSB**: their MetaTrader-format CSV export tool
(https://forexsb.com/historical-forex-data), same column expectations.
Note their exporter caps each download at 100,000 rows per file — you
will likely need to export multiple date-range chunks and concatenate
them into one CSV before placing it at the path above.

**FRED**: https://fred.stlouisfed.org/ — either their API (free API key,
instant signup at https://fred.stlouisfed.org/docs/api/api_key.html) or
manual CSV export per series. Expected format: `date` column plus one
column per series (wide format) — see `src/data/data_loader.py:load_fred`
if your export is in FRED's default long format (`date, series_id,
value`) instead; you'll need to pivot it to wide format first (a two-line
`pandas.pivot_table` call, not provided here since the exact series IDs
you use aren't specified anywhere in the manuscript).

**VIX**: handled automatically via the `yfinance` package on first run —
no manual download needed unless you're running somewhere without
internet access, in which case pre-populate `data/raw/vix_cache.csv` with
columns `date, vix` yourself.

## 4. NVIDIA API access (for LLAMA 3.3 fine-tuning)

**Never paste a real API key into a chat with any AI assistant, a config
file, or a script — including this one.** A key typed into a conversation
is exposed the moment it's sent, whether or not the assistant on the
other end has any way to use it. If you ever do paste a real key
somewhere by accident, treat it as compromised: revoke/rotate it
immediately at the provider's dashboard rather than assuming it's fine
because "nothing happened yet."

1. Create an account at https://build.nvidia.com/
2. Generate an API key from your account's API Keys page.
3. Set it as an environment variable — never hardcode it into any config
   file or script:
   ```bash
   export NVIDIA_API_KEY="your-key-here"
   ```
   Add this to your shell profile (`~/.bashrc`, `~/.zshrc`) if you want it
   to persist across terminal sessions.
4. Confirm your account has fine-tuning access to the specific model
   named in `config/default_config.yaml`'s `llm.base_model`
   (`meta/llama-3.3-70b-instruct`) — NVIDIA's NIM catalog and fine-tuning
   entitlements change over time; check their current documentation at
   https://docs.nvidia.com/nim/ rather than assuming the endpoint shape
   in `src/llm/nvidia_finetune.py` is still exactly correct by the time
   you run this.

`python -m scripts.verify_environment` checks that this environment
variable is set, but cannot verify the key is *valid* or has fine-tuning
entitlement — that only gets checked the first time
`src/llm/nvidia_finetune.py` actually submits a job, and it will fail
loudly (not silently) if the key or entitlement is wrong.

## 5. Final check

```bash
python -m scripts.verify_environment
```
should now report every check passing except possibly the "Processed
file present" checks, which come from Step 1 of RUNNING.md, not this
file. Once this looks right, move to **RUNNING.md**.
