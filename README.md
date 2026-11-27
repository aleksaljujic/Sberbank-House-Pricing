# Sberbank Russian Housing Market — Apartment Price Prediction

This project predicts the sale price of apartments in Moscow (`price_doc`, in rubles) using data
from the Kaggle competition
[Sberbank Russian Housing Market](https://www.kaggle.com/c/sberbank-russian-housing-market).
It covers exploratory data analysis, data cleaning, feature engineering and a comparison of
several regression models, from linear models to a stacking ensemble.

## Project structure

```
.
├── data/
│   ├── raw/                    # original Kaggle data
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── macro.csv
│   │   ├── sample_submission.csv
│   │   └── data_dictionary.txt
│   └── processed/              # intermediate notebook outputs
│       ├── data_cleaned.csv    # output of 01_EDA
│       ├── data_ceaned.csv     # input to 02_Model
│       └── data_model.csv      # output of 02_Model, input to 03_Test
├── models/
│   ├── stack_pipeline.pkl      # final stacking pipeline
│   └── xgb_best_pipeline.pkl   # best XGBoost pipeline
├── notebooks/
│   ├── 01_EDA.ipynb            # analysis and data cleaning
│   ├── 02_Model.ipynb          # model training and evaluation
│   └── 03_Test.ipynb           # testing the saved model
├── requirements.txt
└── README.md
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in order: `01_EDA` → `02_Model` → `03_Test`. Paths are resolved relative
to the project root, so the notebooks work whether Jupyter is started from the root folder or
from `notebooks/`.

> The saved models (`.pkl`) were created with `scikit-learn==1.7.1`. Loading them with a
> different version may fail or produce warnings.

## Workflow

### 1. Exploratory analysis and cleaning (`01_EDA.ipynb`)

- only apartments within 30 km of the Kremlin are kept
- the target is strongly right-skewed, so the models predict `log1p(price_doc)`
- new features: `year`, `month`, `floor_ratio`, `build_age`, `sqm_per_room`, `price_per_m2`
- columns with more than 30% missing values and weak correlation with the target are dropped
- illogical values (e.g. `life_sq > full_sq`, 0 or more than 10 rooms, unrealistic distances)
  are set to `NaN`, and a binary flag is added for each case
- correlation analysis and removal of highly correlated feature pairs

### 2. Modeling (`02_Model.ipynb`)

- **Time-based split**: data is sorted by year and month. The first 80% is used for training
  and the last 20% for testing, which prevents leakage from the future.
- **Preprocessing** (`ColumnTransformer`):
  - nominal columns: most-frequent imputation + one-hot encoding
  - `ecology`: ordinal encoding
  - `sub_area`: frequency encoding (custom `FrequencyEncoder`)
  - numeric columns: median imputation + `QuantileTransformer` (normal output distribution)
- **Models**: LassoCV, Linear Regression + SelectKBest, Random Forest, XGBoost
  (`RandomizedSearchCV`) and Stacking (Ridge + XGBoost + LightGBM, with Linear Regression as
  the meta-model)

### 3. Testing (`03_Test.ipynb`)

Loads `models/stack_pipeline.pkl` and compares its predictions with actual prices on random samples.

## Results

Metrics are computed on the time-based test set (the last 20%), with prices converted back to rubles:

| Model                             | RMSE (RUB)    | MAE (RUB)     | R²        | RMSLE     |
|-----------------------------------|--------------:|--------------:|----------:|----------:|
| Linear Regression + SelectKBest   | 3,262,516     | 2,259,717     | 0.418     | 0.453     |
| Random Forest                     | 2,987,428     | 2,144,492     | 0.512     | 0.440     |
| LassoCV                           | 2,742,855     | 1,698,246     | 0.589     | 0.443     |
| XGBoost                           | 2,655,654     | 1,860,474     | 0.615     | 0.419     |
| **Stacking (Ridge + XGB + LGBM)** | **2,448,644** | **1,599,511** | **0.672** | **0.410** |

The stacking model gives the best results and is saved as `models/stack_pipeline.pkl`.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, matplotlib, seaborn, Jupyter.
