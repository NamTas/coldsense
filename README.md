# 🧊 ColdSense — cold-storage monitoring prototype

> Which storage units are at risk of a temperature excursion, which sensors look faulty, and which unit should operations check first?

ColdSense is a small, end-to-end project built on **real sensor measurements from 123 domestic refrigerators in Bangkok**. It does the work a cold-chain operations / quality team needs:

- **cleans and validates** raw logger data and reports its quality,
- keeps an **excursion ledger** (every period a unit was outside its safe range: start, end, duration, peak, severity),
- produces a **compliance report per unit** (% time in range, mean kinetic temperature, door openings, grade A–D),
- **warns early**: predicts whether a unit will leave its safe range in the next 30 minutes,
- **flags suspicious sensors**,
- ranks units into a **work list with a recommended action** and a **shift-handover note**,
- writes **fleet-level findings** (when excursions happen, which positions run warm, …) into a ready-to-send HTML report,
- and shows all of it in a custom control-room dashboard.

**Scope, honestly:** the data come from household refrigerators, **not** pharmaceutical shipments. ColdSense is a monitoring *prototype* that shows the method; the safety limits are configurable (food 5 °C, vaccine 2–8 °C) so the same engine can be pointed at pharmaceutical or food cold-chain data.

---

## What it does

| Step | What | File |
|---|---|---|
| 1 | Read the raw logger files (auto-detects columns) | `src/loader.py` |
| 2 | Validate and clean: impossible values, duplicates, gaps, frozen sensors → data-quality report | `src/validate.py` |
| 3 | Features from the **past**, label from the **future** ("excursion in next 30 min?") | `src/features.py` |
| 4 | Compare a rule baseline, logistic regression and XGBoost; split **by fridge** | `src/models.py` |
| 5 | Sensor anomaly detection with Isolation Forest, tested with **injected** fake faults | `src/anomaly.py` |
| 6 | Alert priority score, status, plain-English reasons and recommended action | `src/priority.py` |
| 7 | Excursion ledger, unit compliance table (MKT, grade), fleet findings, HTML report | `src/reports.py` |
| – | Control-room dashboard | `app/streamlit_app.py` |

Bonus (optional): MLflow experiment tracking (automatic) and a small LSTM forecast vs. simple baselines (`extras/lstm_baseline.py`).

---

## How to run it (step by step)

### Step 1 — Install Python
Install **Python 3.10, 3.11 or 3.12** from python.org. On Windows, tick **“Add Python to PATH”** during install.
Check in a terminal (Command Prompt / PowerShell / Terminal):
```
python --version
```
(On Mac/Linux you may need `python3` instead of `python`.)

### Step 2 — Open the project and install the libraries
Unzip `coldsense.zip`, then open a terminal **inside the `coldsense` folder** (Windows: open the folder, click the address bar, type `cmd`, press Enter).

```
python -m venv .venv
```
Activate it:
- **Windows (Command Prompt):** `.venv\Scripts\activate`
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`  (if blocked: `Set-ExecutionPolicy -Scope Process Bypass`)
- **Mac / Linux:** `source .venv/bin/activate`

Then:
```
pip install -r requirements.txt
```
(This takes a few minutes the first time.)

### Step 3 — Quick test with demo data (5 minutes, optional but recommended)
This proves everything works on your PC *before* using the real data. The demo data is **synthetic** — never quote its results.
```
python scripts/02_run_pipeline.py --demo
streamlit run app/streamlit_app.py
```
Your browser opens the dashboard. Stop it with `Ctrl + C` in the terminal.

### Step 4 — Get the real dataset
1. Open **https://doi.org/10.57745/TMWYBQ** (“Dataset of household cold-chain conditions … Bangkok, Thailand”, Recherche Data Gouv).
2. Click **Access Dataset → Download** (all files as ZIP), or download the `.txt` files.
3. Unzip and copy **all the `.txt` data files** into the folder `data/raw/` of this project. (Files inside sub-folders are fine. The dataset's own README file may stay there; it is skipped automatically.)

### Step 5 — Check that the columns were understood
```
python scripts/01_inspect_data.py
```
You will see the first rows of two files and a table like `'Temp Top' -> temp_top`.
- Everything mapped correctly → go to Step 6.
- Something is under **UNMAPPED** or mapped wrongly → open `column_map.json`, add a `"sensors"` section as the script suggests (e.g. `{"sensors": {"T1": "temp_top"}}`), and run Step 5 again. Names must start with `temp_` or `hum_`.
- Still stuck? Copy the output of this script and ask for help — fixing the mapping is a 2-minute job.

### Step 6 — Run the pipeline on the real data
```
python scripts/02_run_pipeline.py
```
It prints 6 steps and writes everything to `outputs/`. Useful options:
```
python scripts/02_run_pipeline.py --profile vaccine      # 2–8 °C limits
python scripts/02_run_pipeline.py --profile household    # upper limit 8 °C
```
If the default 5 °C limit flags almost every moment (many home fridges run warm), the script warns you. Then try `--profile household`, or edit the limits in `config.py`.

### Step 7 — Open the dashboard
```
streamlit run app/streamlit_app.py
```
Use the **replay slider** in the sidebar to move through the recording as if it were happening live.
- **Unit wall**: every unit as a tile (status colour, current temperature, 6-hour sparkline, why it is flagged), a fleet status bar and a "units out of range over time" chart.
- **Work list**: what to inspect first + recommended action. Download the work list (CSV), a shift-handover note (TXT) and the fleet report (HTML).
- **Unit report**: compliance grade, time in range, MKT, door openings, the temperature chart with shaded excursions, and the excursion ledger for that unit.
- **Fleet insights**: plain-English findings, when excursions happen, which sensor positions run warm, and a **“what if the limit were different?”** explorer.
- **Trust & method**: honest model comparison, sensor-fault test, data quality, limitations.

The pipeline also leaves ready-to-use files in `outputs/reports/`:
`excursion_events.csv`, `unit_compliance.csv`, `fleet_insights.json` and **`coldsense_fleet_report.html`** (double-click to open in a browser; attach it to an e-mail or portfolio).

### Step 8 (optional) — Bonus extras
```
mlflow ui --backend-store-uri sqlite:///mlflow.db        # experiment tracking page (http://127.0.0.1:5000)
pip install tensorflow                                    # large download
python extras/lstm_baseline.py                            # LSTM vs. simple forecasters
pytest -q                                                 # run the small test-suite
```

---

## Results (fill in from YOUR real run — do not copy the demo numbers)

After Step 6, open `outputs/run_summary.json` and `outputs/model_comparison.csv` and write the real numbers here:

| Model | PR-AUC (test fridges) | ROC-AUC | Recall | Precision |
|---|---|---|---|---|
| Rule baseline (closer to limit = riskier) | _fill_ | _fill_ | – | – |
| Logistic regression | _fill_ | _fill_ | _fill_ | _fill_ |
| XGBoost | _fill_ | _fill_ | _fill_ | _fill_ |

- Share of test moments with a real excursion (prevalence): _fill_
- Sensor-fault test: _X_ of _Y_ injected fake faults caught; false-alarm rate _Z_ % (from `anomaly_eval`)
- Fleet findings: copy the 4–5 sentences printed at the end of the pipeline (also in `outputs/reports/fleet_insights.json`), e.g. "X of 123 units had at least one excursion…"
- Compliance: how many units got grade A / B / C / D (`outputs/reports/unit_compliance.csv`)

Add 1–2 dashboard screenshots to a `docs/` folder and link them here.

---

## Design decisions (good to explain in an interview)

- **No labels exist** for real alerts or sensor faults. Excursions are defined by the safety limit; sensor faults are *injected* into a copy of held-out fridges to test the detector.
- **Prediction, not rule-copying:** the model is trained only on moments when the unit is *currently inside* its safe range and must predict a *new* excursion in the next 30 minutes.
- **No leakage:** features use only past readings (there is a unit test for it); train / validation / test are split **by fridge**; the final model is chosen on validation fridges, never on the test fridges.
- **Honest baselines:** a simple rule baseline is included so you can see whether machine learning actually adds value.
- **Rare events:** PR-AUC is the headline metric because excursions are the minority class.
- **Simple, explainable priority score:** 55 % risk + 25 % time outside the limit in the last hour + 20 % sensor alert (`config.py`).
- **Configurable limits:** one config, different profiles (food / household / vaccine).
- **Deliverables, not just models:** the outputs a quality team can act on — excursion ledger, per-unit compliance with MKT, work list with actions, handover note, HTML report.
- **Excursion definition:** at least 10 minutes outside the range in a row (`MIN_EXCURSION_MIN` in `config.py`); severity brief (< 30 min), moderate (< 2 h), prolonged (2 h or more).
- **MKT (mean kinetic temperature):** the standard pharma summary of a temperature history; it weights warm periods more than cold ones.

## Limitations

- Household refrigerators, not pharmaceutical storage or shipments.
- Roughly five days of recording per fridge.
- No ground truth for real faults or alerts; fault detection is validated on simulated faults.
- Domestic door-opening patterns differ from commercial or pharmaceutical units.

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found | Reinstall Python and tick “Add to PATH”, or try `python3` / `py` |
| `No data files found` | Files must be in `data/raw/` (Step 4) |
| Column unmapped / wrong | Step 5, then edit `column_map.json` |
| `Could not find a date/time column` | Add `"datetime": "ColumnName"` (or `"date"` + `"time"`) to `column_map.json` |
| `split has no excursions` | Try `--profile household`, or change limits in `config.py` |
| Warning about few usable rows | Most of the time the unit is already above the limit → use a higher limit profile |
| Dashboard says “No results found” | Run Step 6 first |
| XGBoost fails to install | Remove it from `requirements.txt`; the code falls back to scikit-learn boosting automatically |

## Put it on GitHub

```
git init
git add .
git commit -m "ColdSense: cold-storage monitoring prototype"
git branch -M main
git remote add origin https://github.com/<your-username>/coldsense.git
git push -u origin main
```
Raw data files are ignored by `.gitignore` (the dataset has its own download page). Add the dataset citation below to your README.

## Data citation

Paviet-Salomon Y., Chaomuang N., Potisuwan C., Derens-Bertheau E., Nukunudompanich M., Laguerre O. (2026). *Dataset of household cold-chain conditions: temperature and relative humidity monitoring in domestic refrigerators in Bangkok Thailand.* Recherche Data Gouv. https://doi.org/10.57745/TMWYBQ
Related paper: Chaomuang et al. (2025), *Applied Food Research* 5(2), 101512 — https://doi.org/10.1016/j.afres.2025.101512

---

## Project structure
```
coldsense/
├── README.md
├── config.py                  # limits, horizon, weights (everything you may want to change)
├── column_map.json            # only if column names need fixing
├── requirements.txt
├── data/raw/                  # ← put the real dataset .txt files here
├── scripts/
│   ├── 01_inspect_data.py     # check column detection
│   ├── 02_run_pipeline.py     # whole pipeline
│   └── make_demo_data.py      # synthetic data for testing only
├── src/                       # loader, validate, features, models, anomaly, priority, reports
├── app/streamlit_app.py       # dashboard
├── extras/lstm_baseline.py    # optional LSTM vs. baselines
├── tests/test_smoke.py        # small unit tests
├── .streamlit/config.toml     # dashboard colour theme
└── outputs/                   # created by the pipeline
    └── reports/               # excursion log, unit compliance, fleet report (HTML)
```
