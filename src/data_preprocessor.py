import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def preprocess_data(df, label_col=None):
    """
    Preprocesses a DataFrame by keeping only numerical columns,
    imputing missing values, and scaling.
    Converts label column to integer codes if categorical.
    Returns preprocessed features as a DataFrame, labels (as integers if categorical), and the fitted transformer.
    """
    # Find which columns are numbers
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Separate out the label column (target) if provided
    if label_col and label_col in df.columns:
        X = df.drop(label_col, axis=1)
        y = df[label_col]
        # Convert categorical target to integer codes
        if y.dtype == 'object' or str(y.dtype).startswith('category'):
            y = y.astype('category').cat.codes
        # Remove label column from feature list
        if label_col in numerical_cols:
            numerical_cols.remove(label_col)
    else:
        X = df.copy()
        y = None

    # Only keep numerical columns
    X = X[numerical_cols]

    # Set up how to handle missing values and scale numbers
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Fit the preprocessor and transform the data
    X_preprocessed = numerical_transformer.fit_transform(X)

    # Make the output a pandas DataFrame
    X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=numerical_cols, index=X.index)
    return X_preprocessed_df, y, numerical_transformer
