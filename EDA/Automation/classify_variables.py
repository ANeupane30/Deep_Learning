import pandas as pd
import numpy as np
import re

# Define the Adaptive Variable Classification Function
def classify_variables(df):
    """
    Adaptive classification of variables into analytical types.
    Returns a dictionary with column lists by category.
    """
    
    n_rows = len(df)
    variable_types = {
        'continuous': [],
        'discrete': [],
        'categorical': [],
        'high_cardinality_cat': [],
        'binary': [],
        'temporal': [],
        'id_or_text': [],
        'constant': [],
        'unknown': []
    }
    
    for col in df.columns:
        series = df[col]
        dtype = series.dtype
        n_unique = series.nunique(dropna=True)
        unique_ratio = n_unique / n_rows if n_rows > 0 else 0
        
        # Handle constants
        if n_unique <= 1:
            variable_types['constant'].append(col)
            continue
        
        # Numeric columns
        if pd.api.types.is_numeric_dtype(series):
            if n_unique == 2:
                variable_types['binary'].append(col)
            elif unique_ratio > 0.05 and dtype == 'float':
                variable_types['continuous'].append(col)
            elif unique_ratio <= 0.05 or dtype == 'int':
                variable_types['discrete'].append(col)
            else:
                variable_types['continuous'].append(col)
        
        # Datetime columns
        elif pd.api.types.is_datetime64_any_dtype(series):
            variable_types['temporal'].append(col)
        
        # Boolean columns
        elif pd.api.types.is_bool_dtype(series):
            variable_types['binary'].append(col)
        
        # Object / String columns
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            # Check for datetime-like strings
            if series.astype(str).str.match(r'\d{4}-\d{2}-\d{2}').any():
                variable_types['temporal'].append(col)
                continue
                
            if unique_ratio > 0.95:
                variable_types['id_or_text'].append(col)
            elif n_unique <= min(50, 0.05 * n_rows):
                variable_types['categorical'].append(col)
            else:
                variable_types['high_cardinality_cat'].append(col)
        
        else:
            variable_types['unknown'].append(col)
    
    return variable_types


# Turing the Result into a DataFrame for better visualization
def variable_summary(df, var_types):
    data = []
    for vtype, cols in var_types.items():
        for col in cols:
            data.append({
                'column': col,
                'type': vtype,
                'dtype': str(df[col].dtype),
                'n_unique': df[col].nunique(),
                'missing': df[col].isna().sum(),
                'missing_%': 100 * df[col].isna().mean()
            })
    return pd.DataFrame(data)

summary = variable_summary(df, variable_types)
summary.sort_values('type').head(15)
