import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.linear_model import ARDRegression
from sklearn.preprocessing import SplineTransformer

import logging
logger = logging.getLogger(__name__)

def build_gam_model(df: pl.DataFrame):
    """
    Build a GAM-like model using periodic spline transformations for Hour and Month,
    with an interaction term for post-2020 data.

    Parameters:
      df (pl.DataFrame): DataFrame with columns 'Year', 'Month', 'Hour', and 'Volume'.

    Returns:
      model (ARDRegression): Fitted regression model.
      spline_transformer_hour (SplineTransformer): Fitted spline transformer for Hour.
      spline_transformer_month (SplineTransformer): Fitted spline transformer for Month.
      n_knots_hour (int): Number of knots used for Hour splines.
      n_knots_month (int): Number of knots used for Month splines.
    """
    n_knots_hour = 15
    n_knots_month = 4
    knots_interaction = np.array([0., 4., 6., 8., 10., 15., 17., 19., 24.])
    n_knots_interaction = len(knots_interaction)

    hour_values = df.select("Hour").to_numpy()
    month_values = df.select("Month").to_numpy()

    # periodic cubic spline basis for Hour
    spline_transformer_hour = SplineTransformer(
        n_knots=n_knots_hour,
        degree=3,
        knots=np.linspace(0, 24, n_knots_hour)[:, None],
        extrapolation="periodic"
    )
    X_hour = spline_transformer_hour.fit_transform(hour_values)

    # periodic cubic spline basis interaction
    spline_transformer_interaction = SplineTransformer(
        n_knots=len(knots_interaction),
        degree=3,
        knots=knots_interaction[:, None],
        extrapolation="periodic",
        include_bias=False
    )
    X_interaction = spline_transformer_interaction.fit_transform(hour_values)

    # periodic cubic spline basis for Month
    spline_transformer_month = SplineTransformer(
        n_knots=n_knots_month,
        degree=3,
        knots=np.linspace(0, 12, n_knots_month)[:, None],
        extrapolation="periodic",
        include_bias=False
    )
    X_month = spline_transformer_month.fit_transform(month_values)

    # binary indicator for post-2020 data
    df = df.with_columns((pl.col("Year") > 2020).cast(pl.Int64).alias("post2020"))

    post2020_values = df["post2020"].to_numpy()

    # Interaction: multiply hour splines by post2020 indicator
    X_interaction = X_interaction * post2020_values[:, None]

    # Assemble design matrix
    X_design = np.concatenate([X_hour, X_interaction, X_month], axis=1)
    y = df["Volume"].to_numpy()

    # Bayesian model comes with parameter covariance for easy confidence intervals
    model = ARDRegression(verbose=True, fit_intercept=False, compute_score=True, tol=1e-8)#, alpha_1=1e-3)
    model.fit(X_design, y)

    return model, spline_transformer_hour, spline_transformer_interaction, spline_transformer_month, X_hour, X_interaction, X_month 
