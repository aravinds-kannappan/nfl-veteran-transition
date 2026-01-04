# NFL Veterans Team Change Effects: A Multi-Position Longitudinal Mixed-Effects Analysis

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![R](https://img.shields.io/badge/R-4.0+-blue.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the complete implementation of a hierarchical mixed-effects modeling analysis examining whether veteran NFL offensive skill-position players (QB, RB, TE, WR) experience systematic performance changes when switching teams. The analysis uses play-by-play and weekly statistics from 2015-2024 via the free nflfastR package.

**Key Features:**
- Dual implementation in both Python and R
- Free data access (no API keys required)
- Hierarchical mixed-effects models with random intercepts and slopes
- Position-specific efficiency metrics (YPC, YPRR, EPA/play)
- Comprehensive control for age decline, team quality, and selection bias
- Interactive visualizations and reports


### Python Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/nfl-veteran-transitions.git
cd nfl-veteran-transitions

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### R Setup

```r
# Install required packages
install.packages(c("nflfastR", "dplyr", "tidyr", "lme4", "nlme", 
                   "ggplot2", "broom.mixed", "performance"))

# Or use renv for reproducibility
renv::restore()
```

## Quick Start

### Python

```python
from src.python import data_collection, preprocessing, modeling

# 1. Download data
data_collection.download_nfl_data(seasons=range(2015, 2025))

# 2. Preprocess and identify veteran transitions
df = preprocessing.create_veteran_transitions_dataset()

# 3. Fit mixed-effects model
model = modeling.fit_hierarchical_model(df, position='RB')

# 4. Visualize results
modeling.plot_model_results(model)
```

### R

```r
source("src/r/utils.R")

# 1. Download data
source("src/r/01_data_collection.R")

# 2. Preprocess
source("src/r/02_preprocessing.R")

# 3. Fit models
source("src/r/03_modeling.R")

# 4. Visualize
source("src/r/04_visualization.R")
```

## Data Sources

All data is freely available via:
- **Python**: `nfl_data_py` package
- **R**: `nflfastR` package

No API keys or authentication required.

### Data Coverage
- **Timespan**: 2015-2024 (excluding 2020-2021 COVID seasons)
- **Positions**: QB, RB, WR, TE
- **Metrics**: EPA, YPC, YPRR, ANY/A, Success Rate
- **Sample Size**: ~110-150 veteran transitions

## Methodology

### Key Features
1. **Hierarchical Mixed-Effects Models**: Random intercepts and slopes for individual players
2. **Position-Specific Metrics**: YPC (RB), YPRR (WR/TE), EPA/play (QB)
3. **Comprehensive Controls**: Age, experience, team quality, opponent strength
4. **Longitudinal Design**: 2 years pre-transition, 2 years post-transition

Full results available in `data/plots`

## Citation

If you use this code or methodology, please cite:

```bibtex
@misc{kannappan2025nfl,
  author = {Kannappan, Aravind},
  title = {NFL Veterans Team Change Effects: A Multi-Position Longitudinal Mixed-Effects Analysis},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/nfl-veteran-transitions}
}
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - see LICENSE file for details

## Contact

Aravind Kannappan - ak12124@nyu.edu

## Acknowledgments

- nflfastR team for comprehensive NFL data
- Ben Baldwin and Sebastian Carl for nflfastR development
- nfl_data_py developers for Python port
