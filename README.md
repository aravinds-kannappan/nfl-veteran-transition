# NFL Veterans Team Transition Predictor

**Analyzing how veteran NFL players perform after switching teams using causal inference and ML ensembles.**

---

## Problem Statement

When a veteran NFL offensive player (QB, RB, WR, TE) changes teams, does their performance systematically change — or are observed shifts explained by aging, scheme fit, and selection bias? This project isolates the **causal effect of team transitions** from confounding factors using longitudinal mixed-effects models and stacked ML ensembles, applied to 2015–2024 NFL play-by-play data.

This matters for front offices evaluating free-agent signings and for analysts trying to separate signal from noise in player valuation.

## Key Design Decisions

- **Hierarchical mixed-effects models** with player-level random intercepts and slopes — chosen over fixed-effects regression to account for unobserved heterogeneity across players while sharing strength across positions.
- **Position-specific efficiency metrics**: YPC (RB), YPRR (WR/TE), EPA/play (QB), ANY/A (QB) — raw counting stats would conflate opportunity with skill.
- **Longitudinal window**: 2 seasons pre-transition and 2 seasons post-transition, excluding 2020–2021 COVID seasons to avoid confounding from shortened/abnormal play.
- **Stacked ensemble prediction**: combines gradient boosting, ridge regression, and random forest for post-transition performance forecasting, with temporal feature engineering capturing trajectory trends.
- **Controls for confounders**: age curves, team offensive quality, opponent strength, and snap count to mitigate selection bias (better players get more opportunity on new teams).

## How the System Works End-to-End

```
nflfastR / nfl_data_py          Hierarchical Mixed-Effects     Stacked ML Ensemble
┌──────────────────┐            ┌──────────────────────┐       ┌─────────────────────┐
│  Play-by-play &  │──Clean──▶  │  Causal estimation   │       │  Post-transition     │
│  weekly stats    │  & merge   │  of transition effect │──▶    │  performance         │
│  (2015-2024)     │            │  (lme4 / statsmodels) │       │  prediction          │
└──────────────────┘            └──────────────────────┘       └─────────────────────┘
        │                                                               │
        ▼                                                               ▼
   Veteran transition              Position-specific               Interactive
   identification &                effect estimates &              visualizations
   feature engineering             confidence intervals            & reports
```

1. **Data Collection** — Pull play-by-play and roster data via `nflfastR` (R) or `nfl_data_py` (Python). No API keys needed.
2. **Preprocessing** — Identify veteran transitions (≥3 seasons experience), compute rolling efficiency metrics, build longitudinal panel.
3. **Causal Modeling** — Fit hierarchical mixed-effects models per position to estimate the transition effect while controlling for aging and team quality.
4. **Prediction** — Stacked ensemble forecasts post-transition performance using pre-transition trajectory features.
5. **Visualization** — Generate position-level effect plots, individual player trajectories, and model diagnostics.

## Tradeoffs & Future Improvements

- **Mixed-effects vs. difference-in-differences**: Mixed-effects models handle unbalanced panels better (not all players have equal pre/post data), but assume linear random effects. A DiD approach could better handle time-varying confounders.
- **Sample size**: ~110–150 veteran transitions is modest. Expanding to defensive positions or longer time windows would increase power but introduce more heterogeneity.
- **Selection bias**: Players who switch teams aren't random — they were either released (negative signal) or pursued in free agency (positive signal). Current controls help but an instrumental variable approach could strengthen causal claims.
- **No scheme-level features**: Offensive system differences (air raid vs. west coast) likely mediate transition effects but aren't currently encoded.

## Repository Structure

```
├── data/                    # Raw and processed datasets
├── results/                 # Model outputs, plots, tables
├── src/
│   ├── python/
│   │   ├── data_collection.py
│   │   ├── preprocessing.py
│   │   └── modeling.py
│   └── r/
│       ├── 01_data_collection.R
│       ├── 02_preprocessing.R
│       ├── 03_modeling.R
│       └── 04_visualization.R
├── NFL_Veterans_Transition_Paper.pdf   # Full writeup
├── requirements.txt
├── nfl-veteran-transitions.Rproj
└── README.md
```

## How to Run

### Python
```bash
git clone https://github.com/aravinds-kannappan/nfl-veteran-transition.git
cd nfl-veteran-transition
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline
python -c "
from src.python import data_collection, preprocessing, modeling
data_collection.download_nfl_data(seasons=range(2015, 2025))
df = preprocessing.create_veteran_transitions_dataset()
model = modeling.fit_hierarchical_model(df, position='RB')
modeling.plot_model_results(model)
"
```

### R
```r
install.packages(c("nflfastR", "dplyr", "tidyr", "lme4", "nlme",
                   "ggplot2", "broom.mixed", "performance"))

source("src/r/01_data_collection.R")
source("src/r/02_preprocessing.R")
source("src/r/03_modeling.R")
source("src/r/04_visualization.R")
```

## Data Sources

All data freely available via `nfl_data_py` (Python) and `nflfastR` (R). No API keys required.

- **Timespan**: 2015–2024 (excl. 2020–2021 COVID seasons)
- **Positions**: QB, RB, WR, TE
- **Sample**: ~110–150 veteran transitions

## Paper

Full methodology and results: [`NFL_Veterans_Transition_Paper.pdf`](./NFL_Veterans_Transition_Paper.pdf)

## Authorship

All code, modeling, and analysis in this repository is solely my own work. Data is sourced from the open-source `nflfastR` / `nfl_data_py` packages developed by Ben Baldwin, Sebastian Carl, and contributors.

## Contact

Aravind Kannappan — [ak12124@nyu.edu](mailto:ak12124@nyu.edu) — [LinkedIn](https://linkedin.com/in/aravindkannappan)

## License

MIT
