from matplotlib.gridspec import GridSpec
import numpy as np
import os
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # for 3d plotting
from matplotlib import colormaps
from sklearn.linear_model import ARDRegression, BayesianRidge, GammaRegressor, LinearRegression, PoissonRegressor, Ridge
from sklearn.preprocessing import MinMaxScaler, SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from scipy import stats
from scipy.interpolate import CubicSpline
import traffic_data_analysis as tda

import logging
logger = logging.getLogger(__name__)

#df_traffic, df_meta = tda.utils.load_directional_hourly()

def gam_bayes(location_id, direction, start='2018-01-01', end='2025-01-01', epsilon=1.0, alpha=1e-2, degree=2, n_knots=15, percentile=0.95):
    '''
    Fits a GAM using a spline basis and linear regression on the log-transformed volume,
    then uses Torch to compute the Hessian of the mean-squared error loss with respect
    to the regression parameters. Interpreting the (scaled) inverse Hessian as an
    approximate posterior covariance, the function draws samples from the posterior,
    computes predictions on a grid of hours, and then produces Bayesian prediction
    intervals (after exponentiating back to the original volume scale).
    
    Parameters
    ----------
    location_id : numeric or string
        Identifier for the location.
    direction : string
        Traffic direction (e.g., 'N', 'S', etc.).
    start : str, optional
        Start date (default '2018-01-01').
    end : str, optional
        End date (default '2025-01-01').
    n_samples : int, optional
        Number of parameter samples to draw (default 1000).
    
    Returns
    -------
    mean_pred : array, shape (n_new,)
        The mean predicted volume (after exponentiating) on a grid of hours (0 to 23).
    bands : dict
        A dictionary with keys 0.80, 0.90, and 0.95. Each value is a tuple (lower, upper)
        giving the corresponding prediction interval (on the original scale).
    filtered_df : pandas.DataFrame
        The subset of df_traffic used for training.
    pipeline : sklearn.pipeline.Pipeline
        The fitted pipeline (with the SplineTransformer and LinearRegression).
    H_np : ndarray
        The Hessian (of the loss function with respect to the regression parameters)
        computed using Torch.
    '''
    # --- 1. Filter the data ---
    # (Assumes your global DataFrame is named `df_traffic` with columns:
    #  'LocationID', 'Direction', 'DateTime', 'Weekend', 'Hour', 'Volume')
    filtered_df = df_traffic[
        (df_traffic['LocationID'] == location_id) &
        (df_traffic['Direction'] == direction) &
        (df_traffic['DateTime'] >= start) &
        (df_traffic['DateTime'] <= end) &
        (df_traffic['Weekend'] == False)
    ].copy()
    #filtered_df = remove_hourly_outliers(filtered_df)

    # --- 2. Prepare the data ---
    # Predictor: Hour (2D array); Response: Volume (we will log-transform)
    #X = filtered_df[['Hour']].values + 0.5 # shape (n_samples, 1)
    X = filtered_df[['Hour']].values
    y = filtered_df['Volume'].values
    to_save = np.hstack((X,y.reshape(-1,1)))
    year = 2023
    np.savetxt(f'{year}_y.txt', y, fmt='%d')
    print(to_save.shape)
    np.savetxt(f'{year}_rawdata.csv', to_save, fmt='%d, %d')
    y = y.astype(float)
    X = X.astype(float)# + 0.5
    #noise = np.random.uniform(0.0, 1.0, size=X.shape)
    #X += noise
    filtered_df['Hour'] = X
    #y = np.log1p(y)

    param = 1e-6
    alpha_1 = param 
    alpha_2 = param 
    lambda_1 = param 
    lambda_2 = param 

    spline = SplineTransformer(degree=degree, knots=np.linspace(0, 24, n_knots)[:, None], extrapolation='periodic',include_bias=True)
    #reg = BayesianRidge(alpha_init=epsilon, lambda_init=alpha, tol=1e-12, max_iter=5000, fit_intercept=False, compute_score=True, lambda_1=lambda_1, lambda_2=lambda_2, alpha_1=alpha_1, alpha_2=alpha_2)
    reg = ARDRegression(tol=1e-12, max_iter=10000, fit_intercept=False, compute_score=True, threshold_lambda=1)
    #reg = GammaRegressor(tol=1e-12, max_iter=5000, fit_intercept=False)

    pipeline = make_pipeline(spline, reg)
    pipeline.fit(X, y)  # training on log(y) so when we transform back we stay positive
    X_save = pipeline.named_steps['splinetransformer'].transform(X)  # shape: (n_samples, d)
    print(X_save.shape)
    np.savetxt(f'{year}_X.txt', X_save)
    return
    #params = len(reg.coef_)  # shape: (d,)
    
    #text = f'$\\alpha={reg.alpha_:.3e}$\n$\\lambda={reg.lambda_:.2e}$\n$L={reg.scores_[-1]:.2e}$'
    text = f'$\\alpha={reg.alpha_:.3e}$\n$L={reg.scores_[-1]:.2e}$'
    print(text)
    #print(reg.lambda_)
    print(reg.coef_)
    print(reg.lambda_)
    print(np.diag(reg.sigma_))
    print(reg.sigma_)

    hour_new = np.linspace(0, 24, 250).reshape(-1, 1)
    mean, std = pipeline.predict(hour_new, return_std=True)
    #mean = pipeline.predict(hour_new)
    #print(std)
    #print()

    alpha = (1 - percentile) / 2
    lower = stats.norm.ppf(alpha, loc=mean, scale=std)
    upper = stats.norm.ppf(1 - alpha, loc=mean, scale=std)

    return mean, lower, upper, filtered_df
    #return np.expm1(mean), np.expm1(lower), np.expm1(upper), filtered_df
    #return mean, filtered_df


def gam_bayes_sklearn_torch(location_id, direction, start='2018-01-01', end='2025-01-01', epsilon=1.0, alpha=1e-2, degree=2, n_knots=15, percentile=0.95):
    '''
    Fits a GAM using a spline basis and linear regression on the log-transformed volume,
    then uses Torch to compute the Hessian of the mean-squared error loss with respect
    to the regression parameters. Interpreting the (scaled) inverse Hessian as an
    approximate posterior covariance, the function draws samples from the posterior,
    computes predictions on a grid of hours, and then produces Bayesian prediction
    intervals (after exponentiating back to the original volume scale).
    
    Parameters
    ----------
    location_id : numeric or string
        Identifier for the location.
    direction : string
        Traffic direction (e.g., 'N', 'S', etc.).
    start : str, optional
        Start date (default '2018-01-01').
    end : str, optional
        End date (default '2025-01-01').
    n_samples : int, optional
        Number of parameter samples to draw (default 1000).
    
    Returns
    -------
    mean_pred : array, shape (n_new,)
        The mean predicted volume (after exponentiating) on a grid of hours (0 to 23).
    bands : dict
        A dictionary with keys 0.80, 0.90, and 0.95. Each value is a tuple (lower, upper)
        giving the corresponding prediction interval (on the original scale).
    filtered_df : pandas.DataFrame
        The subset of df_traffic used for training.
    pipeline : sklearn.pipeline.Pipeline
        The fitted pipeline (with the SplineTransformer and LinearRegression).
    H_np : ndarray
        The Hessian (of the loss function with respect to the regression parameters)
        computed using Torch.
    '''
    # --- 1. Filter the data ---
    # (Assumes your global DataFrame is named `df_traffic` with columns:
    #  'LocationID', 'Direction', 'DateTime', 'Weekend', 'Hour', 'Volume')
    filtered_df = df_traffic[
        (df_traffic['LocationID'] == location_id) &
        (df_traffic['Direction'] == direction) &
        (df_traffic['DateTime'] >= start) &
        (df_traffic['DateTime'] <= end) &
        (df_traffic['Weekend'] == False)
    ].copy()
    filtered_df = remove_hourly_outliers(filtered_df)

    # --- 2. Prepare the data ---
    # Predictor: Hour (2D array); Response: Volume (we will log-transform)
    X = filtered_df[['Hour']].values  # shape (n_samples, 1)
    y = filtered_df['Volume'].values   # counts, add 1.0 to make log happy

    #y_log = np.log(y)                   # take natural log
    
    # We'll use a SplineTransformer to create a spline basis.
    # To incorporate an intercept in the linear model, we set include_bias=True
    # and use LinearRegression with fit_intercept=False.
    spline = SplineTransformer(degree=degree, knots=np.linspace(0, 24, n_knots)[:, None], extrapolation='periodic',include_bias=True)
    #scalar = StandardScaler();
    model = Ridge(alpha=alpha, fit_intercept=False)
    #pipeline = make_pipeline(spline, scalar, model)
    pipeline = make_pipeline(spline, model)
    pipeline.fit(X, y)  # training on log(y) so when we transform back we stay positive
    
    X_design = pipeline.named_steps['splinetransformer'].transform(X)  # shape: (n_samples, d)
    w = pipeline.named_steps['ridge'].coef_  # shape: (d,)
    hessian = 2. * (np.matmul(X_design.transpose(), X_design) + alpha * np.eye(w.shape[0]))
    cov_params = (1.0 / epsilon) * np.linalg.inv(hessian)
    print(np.diag(cov_params))
    print(cov_params)
    eigs = np.linalg.eigvalsh(cov_params)
    plt.plot(eigs)
    
    hour_new = np.linspace(0, 24, 250).reshape(-1, 1)
    X_new_design = pipeline.named_steps['splinetransformer'].transform(hour_new)  # shape: (n_new, d)
    
    mean_pred_log = pipeline.predict(hour_new)
    #mean_pred = np.exp(mean_pred_log) - 1.0
    mean_pred = mean_pred_log

    # Model output std dev
    std_dev = np.sum(np.matmul(X_new_design, cov_params) * X_new_design, axis=1) ** 0.5
    for t in range(0, 23):
        print(f'{t}: {std_dev[t*10]}')
    alpha = (1 - percentile) / 2
    lower = stats.norm.ppf(alpha, loc=mean_pred_log, scale=std_dev)
    upper = stats.norm.ppf(1 - alpha, loc=mean_pred_log, scale=std_dev)

    return mean_pred, lower, upper, filtered_df

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
    # Define number of knots for each variable
    n_knots_hour = 15
    n_knots_month = 4
    knots_interaction = np.array([0., 4., 6., 8., 10., 15., 17., 19., 24.])
    n_knots_interaction = len(knots_interaction)

    # Convert Polars dataframe to NumPy for sklearn compatibility
    hour_values = df.select("Hour").to_numpy()
    month_values = df.select("Month").to_numpy()

    # Create periodic cubic spline basis for Hour
    spline_transformer_hour = SplineTransformer(
        n_knots=n_knots_hour,
        degree=3,
        knots=np.linspace(0, 24, n_knots_hour)[:, None],
        extrapolation="periodic"
    )
    X_hour = spline_transformer_hour.fit_transform(hour_values)

    # Create periodic cubic spline basis interaction
    spline_transformer_interaction = SplineTransformer(
        n_knots=n_knots_hour,
        degree=3,
        knots=knots_interaction[:, None],
        extrapolation="periodic",
        include_bias=False
    )
    X_interaction = spline_transformer_interaction.fit_transform(hour_values)

    # Create periodic cubic spline basis for Month
    spline_transformer_month = SplineTransformer(
        n_knots=n_knots_month,
        degree=3,
        knots=np.linspace(0, 12, n_knots_month)[:, None],
        extrapolation="periodic",
        include_bias=False
    )
    X_month = spline_transformer_month.fit_transform(month_values)

    # Create binary indicator for post-2020 data
    df = df.with_columns((pl.col("Year") > 2020).cast(pl.Int64).alias("post2020"))

    # Convert to NumPy
    post2020_values = df["post2020"].to_numpy()

    # Interaction: multiply hour splines by post2020 indicator
    X_interaction = X_interaction * post2020_values[:, None]

    # Assemble design matrix
    X_design = np.concatenate([X_hour, X_interaction, X_month], axis=1)
    y = df["Volume"].to_numpy()

    # Fit model
    model = ARDRegression(verbose=True, fit_intercept=False)
    #model = Ridge(10., fit_intercept=False)
    model.fit(X_design, y)

    return model, spline_transformer_hour, spline_transformer_interaction, spline_transformer_month, n_knots_hour, n_knots_interaction, n_knots_month

def plot_gam_model(model, spline_transformer_hour, spline_transformer_month, n_knots_hour, n_knots_month):
    """
    Generate plots based on the fitted GAM-like model.

    Parameters:
      model (ARDRegression): Fitted regression model.
      spline_transformer_hour (SplineTransformer): Spline transformer for Hour.
      spline_transformer_month (SplineTransformer): Spline transformer for Month.
      n_knots_hour (int): Number of knots used for Hour splines.
      n_knots_month (int): Number of knots used for Month splines.
    """
    # Create a grid for Hour and Month
    hours = np.linspace(0, 24, 100)
    months = np.linspace(0, 12, 100)
    HH, MM = np.meshgrid(hours, months)

    # Flatten grid for prediction
    HH_flat = HH.ravel().reshape(-1, 1)
    MM_flat = MM.ravel().reshape(-1, 1)

    # Transform the grid values using the fitted spline transformers
    X_hour_grid = spline_transformer_hour.transform(HH_flat)
    X_month_grid = spline_transformer_month.transform(MM_flat)

    # Retrieve model coefficients and intercept
    coef = model.coef_
    intercept = model.intercept_

    # --- Pre-2020 Predictions (post2020 = 0) ---
    y_pred_pre = (intercept +
                  np.dot(X_hour_grid, coef[:n_knots_hour]) +
                  np.dot(X_month_grid, coef[2*n_knots_hour:2*n_knots_hour+n_knots_month]))
    y_pred_pre = y_pred_pre.reshape(HH.shape)

    # --- Post-2020 Predictions (post2020 = 1) ---
    y_pred_post = (intercept +
                   np.dot(X_hour_grid, (coef[:n_knots_hour] + coef[n_knots_hour:2*n_knots_hour])) +
                   np.dot(X_month_grid, coef[2*n_knots_hour:2*n_knots_hour+n_knots_month]))
    y_pred_post = y_pred_post.reshape(HH.shape)

    # --- 3D Surface Plot for Pre-2020 ---
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    surf1 = ax1.plot_surface(HH, MM, y_pred_pre, cmap='viridis', edgecolor='none', alpha=0.8)
    ax1.set_title("Pre-2020 Volume Surface")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Month")
    ax1.set_zlabel("Volume")
    fig1.colorbar(surf1, shrink=0.5, aspect=10)

    # --- 3D Surface Plot for Post-2020 ---
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    surf2 = ax2.plot_surface(HH, MM, y_pred_post, cmap='plasma', edgecolor='none', alpha=0.8)
    ax2.set_title("Post-2020 Volume Surface")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Month")
    ax2.set_zlabel("Volume")
    fig2.colorbar(surf2, shrink=0.5, aspect=10)

    # --- Partial Dependence Plot for the Interaction Term ---
    # This plot shows the effect of the interaction term, i.e. the additional effect of Hour in post-2020 data.
    # We compute this as the dot product of the hour spline basis with the interaction coefficients.
    hours_pd = np.linspace(0, 24, 100).reshape(-1, 1)
    X_hour_pd = spline_transformer_hour.transform(hours_pd)
    interaction_effect = np.dot(X_hour_pd, coef[n_knots_hour:2*n_knots_hour])
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(hours_pd, interaction_effect, label='Post-2020 Hourly Interaction Effect', color='darkred')
    ax3.set_xlabel("Hour")
    ax3.set_ylabel("Effect on Volume")
    ax3.set_title("Partial Dependence: Interaction Term (Hourly Effect)")
    ax3.legend()
    
    plt.show()
#alpha_init = 2e-3   # precision of model noise
#lambda_init = 1e-2     # precision of parameter prior (regularization strength)
#degree = 2
#n_knots = 15

#location_id = 26024
#direction = 'SB'
#start ='2014-01-01'
#end ='2025-01-01'

#np.set_printoptions(precision=3)
#np.set_printoptions(linewidth=1000)

#all_plots(df_traffic)

#start ='2019-01-01'
#end ='2020-01-01'
#pre = gam_bayes(location_id, direction, start=start, end=end, epsilon=alpha_init, alpha=lambda_init, degree=degree, n_knots=n_knots)
#pre = gam_bayes_sklearn_torch(location_id, direction, start=start, end=end, epsilon=alpha_init, alpha=lambda_init, degree=degree, n_knots=n_knots)

'''
medians = model[1].groupby(['Hour'])['Volume'].agg(
        mean='median',
        Q1=lambda x: x.quantile(0.25),
        Q3=lambda x: x.quantile(0.75)
        ).reset_index()
y = medians['mean'].values
y_24     = y[0]
y = np.append(y, y_24)
hour_knots= np.arange(25)      # 0..24
cs = CubicSpline(hour_knots, y, bc_type='periodic')
hour_fine = np.linspace(0, 24, 250)
quartiles = cs(hour_fine)

plot_multi([model, [quartiles]], ['model', 'quartiles'])
'''



#start ='2023-01-01'
#end ='2024-01-01'
#post = gam_bayes(location_id, direction, start=start, end=end, epsilon=alpha_init, alpha=lambda_init, degree=degree, n_knots=n_knots)
#post = gam_bayes_sklearn_torch(location_id, direction, start=start, end=end, epsilon=alpha_init, alpha=lambda_init, degree=degree, n_knots=n_knots)
#plot_multi([pre, post], ['pre', 'post'])
