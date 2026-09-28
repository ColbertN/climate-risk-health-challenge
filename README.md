# Climate Risk and Health Prediction Challenge

## Project purpose

This repository contains a complete machine-learning project for the Zindi **Climate Risk and Health Prediction Challenge**.

The challenge is based on mortality records from Uganda. Each record contains information such as age, gender, location, date of death, temperature, rainfall, latitude, and longitude. Some records are labelled as climate-sensitive and others are not.

The goal is to learn patterns from the labelled training data and predict, for each unseen test record:

- the probability that the death is climate-sensitive; and
- a binary prediction derived from that probability using the required default threshold of `0.5`.

This is a population-level machine-learning benchmark. It is not a clinical diagnostic system and should not be used to make decisions about individual patients.

The project aims to produce a model that is accurate, reproducible, explainable, and compliant with the competition rules. The practical objective is to build a trustworthy model that generalizes to unseen records while avoiding target leakage.

## Competition metric

The challenge combines two metrics:

```text
Final score = 0.60 x F1-Score + 0.40 x ROC-AUC
```

F1 is weighted more heavily because it measures how well the model identifies climate-sensitive cases while balancing precision and recall. ROC-AUC measures how well the model ranks climate-sensitive cases above non-sensitive cases across probability thresholds.

The required submission columns are:

| Column | Meaning |
|---|---|
| `ID` | Original test-record identifier |
| `TargetF1` | Binary prediction from `TargetRAUC >= 0.5` |
| `TargetRAUC` | Raw predicted probability |

The pipeline does not tune a custom submission threshold and does not round the submitted probabilities.

## Machine-learning lifecycle

### 1. Problem framing

This is a supervised binary-classification problem:

- input: demographic, geographic, temporal, and environmental variables;
- target: `is_climate_sensitive`;
- output: a probability and a default-threshold class prediction; and
- success criterion: the weighted F1/ROC-AUC competition score.

The target is imbalanced, so accuracy alone would be misleading. The workflow therefore tracks F1, ROC-AUC, and the official weighted score throughout development.

### 2. Data understanding

The supplied files are:

- `Train.csv`: 3,146 labelled mortality records;
- `Test.csv`: 1,030 records for prediction;
- `climate_features.csv`: 4,176 climate and environmental enrichments joined by `ID`;
- `data_dictionary.csv`: descriptions of the original variables; and
- `downloaded_climate_features_data_dictionary.csv`: descriptions of the supplied environmental features.

The records cover multiple years and locations in Uganda. The training target is imbalanced. The project explores target balance, missingness, temporal patterns, demographic patterns, climate distributions, correlations, and geographic structure before modeling.

### 3. Data governance and leakage control

Only climate and environmental external data are used. The project does not use external mortality, health, disease, demographic, socioeconomic, census, healthcare-access, or cause-of-death data.

The pipeline never joins an external mortality source to recover the target for a test row. Such a join would reveal the answer rather than provide a legitimate covariate.

Unsupervised support counts are calculated from train and test covariates without using the target. Validation predictions are generated from folds that do not contain the corresponding validation labels.

### 4. Exploratory data analysis

The EDA answers practical questions before modeling:

- How imbalanced is the target?
- Does the climate-sensitive rate vary by month or year?
- Are there differences by age group, gender, or zone?
- Which climate variables have different distributions across the two classes?
- Are there geographic clusters or sparse regions?
- Which variables contain missing values or strong correlations?

Generated outputs include target balance, temporal rates, demographic rates, climate distributions, spatial plots, a correlation heatmap, missingness, and feature importance. The figures are under [`reports/figures`](reports/figures), with numeric tables under [`reports/tables`](reports/tables).

![Target balance](reports/figures/01_target_balance.png)

![Temporal target rates](reports/figures/02_temporal_target_rates.png)

![Climate distributions](reports/figures/04_climate_distributions.png)

![Spatial distribution](reports/figures/05_spatial_distribution.png)

### 5. Feature engineering

The raw variables are transformed into features representing plausible climate-health mechanisms and stable geographic structure.

#### Time features

- year, month, quarter, week, and day of year;
- sine/cosine seasonal cycles; and
- date support counts.

#### Demographic features

- age squared;
- infant, child, working-age, and older-adult indicators;
- age bands; and
- missing-age indicators.

#### Geographic features

- location, district, and region text fields;
- latitude and longitude squares;
- latitude-longitude interaction;
- rounded spatial cells; and
- supplied elevation and slope.

#### Climate features

- temperature ranges and temperature-centre contrasts;
- rainfall log transforms;
- 7-day, 30-day, and 90-day rainfall ratios;
- rainfall intensity and extreme-rainfall share;
- temperature-window changes;
- hot-day share;
- NDVI change between windows; and
- heat-stress and wet-heat proxies.

The purpose is to expose seasonal stress, rainfall accumulation, heat extremes, vegetation conditions, and geographic vulnerability to nonlinear models.

### 6. External climate data enrichment

The challenge-supplied climate file contains features derived from external environmental products:

- [CHIRPS rainfall](https://chc.ucsb.edu/data/chirps/);
- [ERA5-Land reanalysis](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=documentation);
- [NASA MODIS vegetation indices](https://modis.gsfc.nasa.gov/data/dataprod/mod13.php); and
- [SRTM terrain](https://lpdaac.usgs.gov/products/srtmgl1v003/).

The project additionally downloads public NASA POWER daily weather data using eight 0.5-degree environmental grid cells covering the competition records. The downloader creates pre-death rolling summaries for temperature, precipitation, humidity, wind, solar radiation, evapotranspiration, and soil wetness.

NASA POWER is accessed through its [official Daily Point API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/point/). The cached result is stored in [`data/external/nasa_power_features.csv`](data/external/nasa_power_features.csv), with source metadata in the same folder.

### 7. Validation design

The main validation protocol is a shuffled, stratified five-fold split. Stratification preserves the target ratio in every fold. The same folds are used to compare models and calculate out-of-fold predictions.

This is more reliable than a single train/validation split and provides a consistent basis for comparing feature sets and model families.

### 8. Model development

#### CatBoost

CatBoost is used because it handles mixed numeric and categorical variables well, captures nonlinear interactions, and represents location-related categories without fragile manual target encodings.

#### LightGBM

LightGBM is trained on one-hot encoded features as a complementary model. Its different tree-building behaviour provides ensemble diversity.

#### XGBoost

XGBoost was tested as a third ensemble member. Its out-of-fold score was weaker and its selected blend weight was zero, so it is retained as an experiment rather than used in the primary submission.

### 9. Hyperparameter tuning and feature selection

The project uses a small explicit search rather than an AutoML system. The tested CatBoost configurations vary tree depth, learning rate, regularisation, and random strength.

Feature importance is averaged across validation folds. Importance-ranked subsets are evaluated to test whether removing weak or noisy variables improves generalisation. The top-20 subset was retained as a challenger, but the full engineered feature set performed better under the full five-fold comparison.

A smoothed target-encoding LightGBM experiment was evaluated using fold-specific mappings. Its score was weaker, so it was rejected.

### 10. Ensembling

The primary candidate blends external-data CatBoost and LightGBM probabilities. Blend weights are chosen from out-of-fold predictions.

The selected weights are approximately:

```text
35% CatBoost + 65% LightGBM
```

XGBoost receives zero weight because it did not improve the out-of-fold score.

### 11. Post-processing and submission validation

Before writing a submission, the pipeline checks exact column names, test-row count, test-ID order, ID uniqueness, binary labels, finite probabilities, probability bounds, and consistency between `TargetF1` and `TargetRAUC >= 0.5`.

The primary submission passed all checks for 1,030 test rows.

## Results

These are five-fold out-of-fold results from the project validation protocol:

| Candidate | F1 | ROC-AUC | Weighted score | Decision |
|---|---:|---:|---:|---|
| Original CatBoost all features | 0.8049 | 0.8136 | 0.8084 | baseline |
| NASA POWER CatBoost | 0.8077 | 0.8099 | 0.8086 | retained component |
| NASA POWER LightGBM | 0.8043 | 0.8154 | 0.8087 | retained component |
| NASA POWER CatBoost + LightGBM | **0.8101** | **0.8155** | **0.8123** | **primary candidate** |
| NASA POWER target-encoded LightGBM | 0.7879 | 0.7818 | 0.7854 | rejected |

The primary submission is [`submissions/nasa_power_catboost_lgbm_blend_submission.csv`](submissions/nasa_power_catboost_lgbm_blend_submission.csv).

Feature importance is available in [`reports/tables/feature_importance.csv`](reports/tables/feature_importance.csv) and [`reports/tables/lightgbm_feature_importance.csv`](reports/tables/lightgbm_feature_importance.csv).

![Feature importance](reports/figures/08_feature_importance.png)

## Reproduce the project

Place the challenge files in the project root:

```text
Train.csv
Test.csv
climate_features.csv
```

Install the open-source dependencies:

```bash
pip install -r requirements.txt
```

Run the lifecycle from beginning to end:

```bash
python scripts/make_eda.py
python scripts/download_nasa_power.py
python scripts/tune_hyperparameters.py
python scripts/train_model.py --all-features --run-name nasa_power_all_features
python scripts/blend_models.py
python scripts/validate_submission.py --submission submissions/nasa_power_catboost_lgbm_blend_submission.csv
python scripts/make_feature_report.py
```

The repository also contains individual scripts for feature selection, target-encoding experiments, feature-importance plots, and submission validation.

## Repository map

```text
src/climate_health/       Reusable feature engineering and submission logic
scripts/                  EDA, download, tuning, training, blending, and validation
data/external/            Cached NASA POWER climate enrichment and metadata
reports/                  EDA figures, validation tables, model summaries
submissions/              Competition-ready CSV candidates
models/                   Small model configuration and schema files
```

Raw challenge files are intentionally excluded from Git by `.gitignore`. The repository contains the code, derived public climate enrichment, methodology, analysis, validation results, and submission outputs needed to understand and reproduce the work.
