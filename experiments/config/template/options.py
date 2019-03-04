"""The experiment options."""
from hyperopt import hp  # pylint: disable=wrong-import-position

# pylint: disable=invalid-name
config = {
    "seed": 1,  # Random seed.
    ######################### Data.
    # The following values specify the data hierarchy within the main folder as:
    # data/[data_prefix]/[data_version]/[stage].
    "data_prefix": "dummy",
    "data_version": "1",
    # Other data options.
    "width": 28,
    "height": 28,
    ######################### Model.
    # Supported run modes for this experiment.
    "supp_modes": ["train", "val", "trainval", "test"],
    # Other model parameters.
    "num_classes": 2,
    ######################### Optimizer.
    # The batch size to use.
    "batch_size": 2,
    # Optimizer learning rate.
    "opt_learning_rate": 0.05820132900208654,
    # Optimizer weight decay.
    "opt_weight_decay": 1e-2,
    # The number of epochs to train.
    "num_epochs": 30,
    # The minimum number of epochs to train during a parameter sweep before early
    # stopping according to the hyperband criterion.
    "min_num_epochs": 5,
    # Learning rate decay steps.
    "lr_decay_epochs": 14,
    # Learning rate decay factor.
    "lr_decay": 0.1,
    ######################### Summaries.
    # Write full summaries every x steps.
    "summary_full_every": 100,
    # How often to write a model checkpoint (every x epochs).
    "checkpoint_every": 3,
    # Only keep the last x checkpoints. If x is <= 0, keep all.
    "checkpoint_keep": 5,
}

# Specify experiment hyperparameters that you may want to do a sweep around.
hyperparameters = {
    "opt_learning_rate": hp.uniform("learning_rate", 1e-4, 1e-1),
    "lr_decay_epochs": hp.uniform("lr_decay_epochs", 10, 20),
    "lr_decay": hp.uniform("lr_decay", 0.001, 0.1),
}
