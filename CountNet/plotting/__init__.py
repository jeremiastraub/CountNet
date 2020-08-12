"""The analysis module takes care of the plotting"""

from .metrics import (plot_loss_single, plot_loss_multiple,
                      plot_metric_single, plot_metric_multiple)
from .plot_utils import (extract_loss_information, extract_metric_information,
                         load_trainer)
from .model_output import plot_example_output, plot_count_distribution
