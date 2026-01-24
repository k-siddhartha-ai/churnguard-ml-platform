import pandas as pd
import numpy as np


def logistic_coefficients(model, feature_names):
    """
    Extracts and ranks Logistic Regression coefficients.

    Parameters:
    - model: trained sklearn Pipeline (Logistic Regression)
    - feature_names: list of transformed feature names

    Returns:
    - DataFrame sorted by absolute impact
    """
    coefs = model.named_steps["model"].coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "impact": np.abs(coefs)
    }).sort_values("impact", ascending=False)

    return coef_df


def random_forest_importance(model, feature_names):
    """
    Extracts and ranks Random Forest feature importances.

    Parameters:
    - model: trained sklearn Pipeline (Random Forest)
    - feature_names: list of transformed feature names

    Returns:
    - DataFrame sorted by importance
    """
    importances = model.named_steps["model"].feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    return importance_df
