"""
Configuration loader.

Either define the options in this file and return the dictionary or use it to define
another configuration option loading function.
"""
import imp
import logging
from os import path

LOGGER = logging.getLogger(__name__)


def get_config():
    """Get the experiment configuration."""
    conf_fp = path.join(path.dirname(__file__), "options.py")
    LOGGER.info("Loading experiment configuration from `%s`...", conf_fp)
    options = imp.load_source(
        "_options", path.abspath(path.join(path.dirname(__file__), "options.py"))
    )
    LOGGER.info("Done.")
    return options.config


def adjust_config(config, mode):  # pylint: disable=unused-argument
    """Hot-patch the experiment configuration for a specific mode."""
    # Don't misuse this!
    # Results are better if batchnorm is always used in 'training' mode and
    # normalizes the observed distribution. That's why it's important to leave
    # the batchsize unchanged.
    # if mode not in ['train', 'trainval']:
    #    config['batch_size'] = 1
    return config


def get_hyperparameters():
    """Load experiment hyperparameters."""
    conf_fp = path.join(path.dirname(__file__), "options.py")
    LOGGER.info("Loading hyperparameter configuration from `%s`...", conf_fp)
    options = imp.load_source(
        "_options", path.abspath(path.join(path.dirname(__file__), "options.py"))
    )
    # with open(CONF_FP, 'r') as inf:
    #     config = json.loads(inf.read())
    LOGGER.info("Done.")
    return options.hyperparameters
