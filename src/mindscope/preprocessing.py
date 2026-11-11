import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

TARGET = "Mental_Health_Risk"
ID_COL = "Student_ID"
RANDOM_STATE = 42


def load_data(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])
    df = df.drop(columns=[ID_COL])
    return df.reset_index(drop=True)


def split_data(df, test_size=0.2, random_state=RANDOM_STATE):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


class IQRCapper(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    #Cap the values outside [Q1 - k*IQR, Q3 + k*IQR]. Bounds are learned on the training data only

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = np.where(iqr > 0, q1 - self.factor * iqr, -np.inf)
        self.upper_ = np.where(iqr > 0, q3 + self.factor * iqr, np.inf)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)


def build_preprocessor(X):
    #Imputation + outlier capping + scaling for numbers, imputation + one-hot for categories
    num_cols = X.select_dtypes(include="number").columns.tolist()
    ordinal_categories = {
        "Year_of_Study": [1, 2, 3, 4, 5],
        "Physical_Activity": ["Low", "Moderate", "High"],
        "Diet_Quality": ["Poor", "Average", "Good"],
    }
    ordinal_columns = list(ordinal_categories)
    nominal_columns = ["Gender", "Part_Time_Job", "Previous_Mental_Health_History"]    

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("capper", IQRCapper()),
        ("scaler", StandardScaler()),
    ])
    nominal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ordinal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal_encoder", OrdinalEncoder(categories=list(ordinal_categories.values())))
    ])
    return ColumnTransformer(
        [("num", num_pipe, num_cols), ("hot", nominal_pipe, nominal_columns), ("ordinal", ordinal_pipe, ordinal_columns)],
        sparse_threshold=0,
    )


def build_pipeline(model, X, use_smote=False):
    #Full pipeline: preprocessing, optional SMOTE (only applied when fitting) and the model
    steps = [("preprocessor", build_preprocessor(X))]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", model))
    return ImbPipeline(steps)