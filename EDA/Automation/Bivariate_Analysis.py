import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import classify_variables

def bivariate_analysis(df, var1, var2, var_types):
    """
    Automatically selects the correct visualization and test for two variables.
    """

    def get_type(col):
        for k, v in var_types.items():
            if col in v:
                return k
        return "unknown"
    
    t1, t2 = get_type(var1), get_type(var2)

    print(f"🔍 Analyzing relationship between '{var1}' ({t1}) and '{var2}' ({t2})")

    # ===============================
    # Case 1: Numeric vs Numeric
    # ===============================
    if t1 in ['continuous', 'discrete'] and t2 in ['continuous', 'discrete']:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=var1, y=var2, data=df)
        plt.title(f"{var1} vs {var2}")
        plt.show()

        # Compute correlation
        pearson_corr = df[[var1, var2]].corr(method='pearson').iloc[0,1]
        spearman_corr = df[[var1, var2]].corr(method='spearman').iloc[0,1]
        print(f"📈 Pearson Corr: {pearson_corr:.3f} | Spearman Corr: {spearman_corr:.3f}")

    # ===============================
    # Case 2: Categorical vs Numeric
    # ===============================
    elif (t1 in ['categorical', 'binary'] and t2 in ['continuous', 'discrete']) or \
         (t2 in ['categorical', 'binary'] and t1 in ['continuous', 'discrete']):
        
        cat_col = var1 if t1 in ['categorical', 'binary'] else var2
        num_col = var2 if cat_col == var1 else var1

        plt.figure(figsize=(6,4))
        sns.boxplot(x=cat_col, y=num_col, data=df)
        plt.title(f"{num_col} distribution across {cat_col}")
        plt.show()

        # Run ANOVA/Kruskal test if multiple categories
        groups = [group.dropna().values for name, group in df.groupby(cat_col)[num_col]]
        if len(groups) > 1:
            stat, p = stats.f_oneway(*groups)
            print(f"📊 ANOVA F-statistic = {stat:.3f}, p-value = {p:.4f}")
            if p < 0.05:
                print("✅ Significant difference between group means.")
            else:
                print("⚪ No significant difference between group means.")
    
    # ===============================
    # Case 3: Categorical vs Categorical
    # ===============================
    elif t1 in ['categorical', 'binary'] and t2 in ['categorical', 'binary']:
        ctab = pd.crosstab(df[var1], df[var2])
        print("\n📋 Cross Tabulation:\n", ctab)

        plt.figure(figsize=(6,4))
        sns.heatmap(ctab, annot=True, fmt='d', cmap='YlGnBu')
        plt.title(f"{var1} vs {var2}")
        plt.show()

        # Chi-square test
        chi2, p, dof, exp = stats.chi2_contingency(ctab)
        print(f"📊 Chi-Square = {chi2:.3f}, p-value = {p:.4f}")
        if p < 0.05:
            print("✅ Significant relationship between variables.")
        else:
            print("⚪ No significant relationship detected.")

    # ===============================
    # Case 4: Anything with Temporal
    # ===============================
    elif 'temporal' in [t1, t2]:
        temp_col = var1 if t1 == 'temporal' else var2
        other_col = var2 if temp_col == var1 else var1

        df_sorted = df.sort_values(temp_col)
        plt.figure(figsize=(8,4))
        sns.lineplot(x=temp_col, y=other_col, data=df_sorted)
        plt.title(f"Trend of {other_col} over time ({temp_col})")
        plt.show()
    
    else:
        print("⚠️ Relationship type not supported or insufficient data.")


## Example usage
df = sns.load_dataset('titanic')

# Use your previous classify_variables() function
var_types = classify_variables(df)

# Run examples
bivariate_analysis(df, 'age', 'fare', var_types)          # numeric vs numeric
bivariate_analysis(df, 'class', 'fare', var_types)         # categorical vs numeric
bivariate_analysis(df, 'sex', 'survived', var_types)       # categorical vs categorical
