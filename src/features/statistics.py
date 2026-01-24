def churn_statistics(df):
    """
    Compute basic churn statistics.
    """
    return {
        "churn_rate": df["Churn"].mean(),
        "avg_tenure_churn": df[df["Churn"] == 1]["tenure"].mean()
    }
