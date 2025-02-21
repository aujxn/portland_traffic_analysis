"""
Traffic Data Analysis Package

This package provides tools for traffic data preprocessing, 
linear models, neural networks, visualization, and utilities.
"""

from . import config
from . import preprocess
from . import utils
from . import linear_models
from . import plots

# Handle optional neural network module
try:
    from . import neural_network
    __all__ = ["preprocess", "utils", "config", "linear_models", "plots", "neural_network"]
except ImportError:
    __all__ = ["preprocess", "utils", "config", "linear_models", "plots"]

__version__ = "0.1.0"
