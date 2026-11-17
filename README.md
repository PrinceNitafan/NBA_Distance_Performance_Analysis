# Evaluating the Impact of Inter-Game Travel Distance on NBA Offensive Rating

By: Prince Nitafan

Evaluates whether inter-game travel distance (`distance`, in kilometers) affects single-game NBA regular-season offensive efficiency (`OFF_RATING`, points scored per 100 possessions) across 12,142 team-game observations from the `2021-22` through `2025-26` seasons. Across both Ordinary Least Squares (OLS) regressions and Random Forest models, travel distance shows no statistically significant or generalizable effect once baseline team offense, opponent defense, rest days, and home-court advantage are controlled for.

For the full methodology, statistical tables, residual diagnostics, and limitations analysis, see [`NBA_Distance_Performance_Analysis.pdf`](NBA_Distance_Performance_Analysis.pdf).

## Repository Layout

```text
NBA_Distance_Performance_Analysis/
├── Analysis/
│   ├── analysis1/
│   │   ├── Original_distance_vs_off_rating.png                  # Raw scatter plot (distance vs. OFF_RATING)
│   │   ├── a1_ols_summary.txt                                   # Test 1 univariate OLS regression summary
│   │   └── regression_distance_vs_off_rating.png                # Test 1 scatter plot with OLS fit line
│   ├── analysis2/
│   │   ├── a2_ols_with_dist_summary.txt                         # Test 3 multivariate OLS summary (with distance)
│   │   ├── a2_ols_without_dist_summary.txt                      # Test 2 multivariate OLS summary (baseline controls)
│   │   ├── ols_multi_regress_with_dist_coefficients.png         # Test 3 standardized coefficient bar chart
│   │   └── ols_multi_regress_without_dist_coefficients.png      # Test 2 standardized coefficient bar chart
│   └── analysis3/
│       ├── a3_rf_regress_with_dist_scores.txt                   # Test 5 Random Forest train/validation R^2 scores
│       ├── a3_rf_regress_without_dist_scores.txt                # Test 4 Random Forest train/validation R^2 scores
│       ├── randfor_regress_with_dist_importance_scores.png      # Test 5 feature importance bar chart
│       └── randfor_regress_without_dist_importance_scores.png   # Test 4 feature importance bar chart
├── Datasets/
│   ├── cleaned_data/
│   │   └── nba_team_performance_cleaned.csv                     # Game logs merged with venue coordinates
│   ├── features_data/
│   │   └── nba_team_perf_full_feats.csv                         # Final modeling dataset (12,142 games, 5 features)
│   └── raw_data/
│       ├── nba_locations.csv                                    # Filtered NBA team arena coordinates
│       ├── nba_team_regular_performances.csv                    # Raw nba_api team game logs (2021-22 to 2025-26)
│       └── pro_venues_clean_final.csv                           # Kaggle pro sports venue coordinates input
├── src/
│   ├── analysis.py                                              # Main entrypoint running Tests 1 through 5
│   ├── extract.py                                               # Fetches regular-season team game logs via nba_api
│   ├── features.py                                              # Engineers travel distance and rolling control features
│   ├── transform.py                                             # Filters arena locations and cleans raw game logs
│   └── utils.py                                                 # Vectorized Haversine distance calculation (km)
├── .gitignore                                                   # Git ignore rules
├── NBA_Distance_Performance_Analysis.pdf                        # Full technical report and empirical analysis
├── README.md                                                    # Project documentation and runbook
└── environment.yml                                              # Conda environment specification (nba_distance_analysis)
```

## Prerequisites and Environment Setup

1. Install Anaconda or Miniconda.
2. From the repository root, create and activate the `nba_distance_analysis` Conda environment defined in [`environment.yml`](environment.yml):
   ```bash
   conda env create -f environment.yml
   conda activate nba_distance_analysis
   ```
3. Confirm that the required directory tree exists:
   - `Datasets/raw_data/`, `Datasets/cleaned_data/`, `Datasets/features_data/`
   - `Analysis/analysis1/`, `Analysis/analysis2/`, `Analysis/analysis3/`

## Running the Pipeline

Pre-extracted and feature-engineered CSV files are included under `Datasets/`, so the analysis script can run directly without calling the NBA API.

1. Run Tests 1 through 5 from the repository root:
   ```bash
   python src/analysis.py
   ```
   Expected output:
   ```text
   ANALYSIS: Completed analyis 1.
   ANALYSIS: Completed analyis 2.
   ANALYSIS: Completed analyis 3.
   ```
2. (Optional) To re-fetch raw game logs from `nba_api` and regenerate all intermediate datasets from scratch, ensure [`Datasets/raw_data/pro_venues_clean_final.csv`](Datasets/raw_data/pro_venues_clean_final.csv) ([Kaggle Pro Sports Venues dataset](https://www.kaggle.com/datasets/peteyy/pro-sports-venues-nba-nfl-mlb-nhl/data)) is present, then either uncomment `setup()` in `main()` of [`src/analysis.py`](src/analysis.py) or run the pipeline stages in order:
   ```bash
   python src/extract.py
   python src/transform.py
   python src/features.py
   python src/analysis.py
   ```
