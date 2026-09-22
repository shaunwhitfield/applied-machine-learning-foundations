"""
09_model_deployment_02.py

Cell 191 of section "Using MLFlow" from ml-foundations.ipynb, runnable on its own
from the terminal:

    .venv/bin/python cells/09_model_deployment_02.py

Contains everything it needs (imports, data, helpers, models). The dataset is
read from raw_housing.csv in this directory.
"""

import warnings
from pathlib import Path

import polars as pl
import polars.selectors as cs

warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / 'raw_housing.csv'
PLOTS_DIR = HERE / 'plots'

# King County House Sales dataset from OpenML (includes Seattle).
# ARFF file: 31 lines of metadata/attributes, then one row per house sale.
# Downloaded once to raw_housing.csv so individual cells don't re-fetch it.
COLS = ['id', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
        'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
        'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'date_year', 'date_month', 'date_day']

raw = pl.read_csv(DATA_PATH, new_columns=COLS, skip_rows=31, has_header=False)


def print_df(obj, n=10):
    """Show a (possibly large) frame the way the notebook would display it."""
    if isinstance(obj, pl.DataFrame):
        print(obj.slice(0, n))
        print(f'(shape: {obj.shape})')
    else:
        print(obj)


def save_chart(chart, name):
    """Save a polars .plot (holoviews) chart as a standalone HTML file."""
    import holoviews as hv
    from bokeh.io import save
    from bokeh.resources import INLINE

    PLOTS_DIR.mkdir(exist_ok=True)
    path = PLOTS_DIR / f'{name}.html'
    save(hv.render(chart), str(path), title=name, resources=INLINE)
    print(f'Saved plot to {path}')

# ---- data processing helper (notebook: "Data Preprocessing") ----

def tweak_housing(df):
    return (df
            .with_columns(zipcode=pl.col('zipcode').cast(pl.String).cast(pl.Categorical),
                          date=pl.date(pl.col('date_year'), pl.col('date_month'), pl.col('date_day')),
                          yr_renovated=pl.col('yr_renovated').replace(0, None),
                          )
            .select(['id', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                     'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement',
                     'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15',
                     'sqft_lot15', 'date',  #'date_year', 'date_month', 'date_day',
                     ])
    )


# ---- sklearn setup (notebook: "Sklearn Pipelines") ----
# A pipeline is a sequence of transformers; a ColumnTransformer applies
# different transformers to different columns of the input.

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config

set_config(transform_output='polars')


# ---- full preprocessor (notebook: "make the pipeline") ----

numeric_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
                    'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
                    'lat', 'long', 'sqft_living15', 'sqft_lot15', 'zip_mean']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['zipcode']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore',
                              sparse_output=False), categorical_features)])

tweak_transformer = FunctionTransformer(tweak_housing)


# ---- custom transformer: adds the mean price per zip code ----

class ZipAvgPriceAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # assume X is a polars dataframe
        self.zip_avg_price = (X
                              .group_by('zipcode')
                              .agg(zip_mean=pl.col('price').mean())
        )
        return self

    def transform(self, X, y=None):
        return X.join(self.zip_avg_price, on='zipcode')


# ---- train/test split ----

X = raw
y = raw.select('price')  # Note sklearn wants a Polars dataframe for y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ---- linear regression ----

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr_pipe = Pipeline(steps=[('tweak', tweak_transformer),
                          ('zip_avg_price', ZipAvgPriceAdder()),
                          ('preprocessor', preprocessor),
                          ('lr', lr)])
lr_pipe.fit(X_train, y_train)

# ---- notebook cell ----

import os

import mlflow

# keep mlruns/ inside the cells/ directory no matter where we run from
os.chdir(HERE)

model_info = mlflow.sklearn.log_model(lr_pipe, artifact_path='lr_pipe')
print(model_info)
