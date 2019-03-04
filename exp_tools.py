"""Experiment management tools."""
import logging
import os
from glob import glob
from os import path

LOGGER = logging.getLogger(__name__)
LOGFORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


def setup_paths(exp_fp):
    """
    Setup the paths for an experiment.

    Requires the exp_fp to start with `experiments/config`. It returns feature and log
    filepaths relative to the execution directory. During a parameter sweep, this is
    relative to the sweep experiment folder. If you're using `run.py`, it's in the
    standard directory tree.

    If the paths do not exist, they are created.
    """
    exp_fp = exp_fp.strip("/")  # Standardized format.
    assert exp_fp.startswith(path.join("experiments", "config"))
    exp_name = path.basename(exp_fp)
    LOGGER.info("Experiment name & path: %s, `%s`", exp_name, exp_fp)
    exp_feat_fp = path.join("experiments", "features", exp_name)
    exp_log_fp = path.join("experiments", "states", exp_name)
    if not path.exists(exp_feat_fp):
        os.makedirs(exp_feat_fp)
    if not path.exists(exp_log_fp):
        os.makedirs(exp_log_fp)
    return exp_fp, exp_name, exp_feat_fp, exp_log_fp


def setup_logging(log_fp):
    """
    Setup the logging facilities for the experiment.
    """
    import socket

    handler = logging.FileHandler(log_fp)
    formatter = logging.Formatter(LOGFORMAT)
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    LOGGER.info("Running on host: %s", socket.getfqdn())


def get_config(exp_fp, mode, custom_options, num_threads):
    """
    Get an experiment configuration.

    Assumes a valid `exp_fp`. The configuration is always resolved relative to this
    module.
    """
    import imp
    import ast

    exp_config_mod = imp.load_source(
        "_exp_config", path.join(path.dirname(__file__), exp_fp, "config.py")
    )
    exp_config = exp_config_mod.adjust_config(exp_config_mod.get_config(), mode)
    assert mode in exp_config["supp_modes"], (
        "Unsupported mode by this model: %s, available: %s."
        % (mode, str(exp_config["supp_modes"]))
    )
    if custom_options != "":
        custom_options = ast.literal_eval(custom_options)
        exp_config.update(custom_options)
    exp_config["num_threads"] = num_threads
    LOGGER.info("Configuration:")
    for key, val in exp_config.items():
        LOGGER.info("%s = %s", key, val)
    return exp_config


def get_hyperparameters(exp_fp):
    """
    Get an experiment hyperparameter set.

    Assumes a valid `exp_fp`. The configuration is always resolved relative to this
    module.
    """
    import imp
    import ast

    exp_config_mod = imp.load_source(
        "_exp_config", path.join(path.dirname(__file__), exp_fp, "config.py")
    )
    exp_hypers = exp_config_mod.adjust_config(
        exp_config_mod.get_hyperparameters(), "train"
    )
    LOGGER.info("Hyperparameters:")
    for key, val in exp_hypers.items():
        LOGGER.info("%s = %s", key, val)
    return exp_hypers


def get_desc(exp_fp, mode, exp_config):
    """Get a model."""
    import imp

    model_mod = imp.load_source(
        "_model", path.join(path.dirname(__file__), exp_fp, "model.py")
    )
    exp_desc = model_mod.ExpDesc(mode, exp_config)

    return exp_desc


def get_checkpoints(exp_name):
    """
    Retrieve a list of all checkpoint filepaths for this experiment (sorted ascending).
    """
    from natsort import natsorted

    check_fp = path.join("experiments", "states", exp_name)
    checkpoint_fps = natsorted(glob(path.join(check_fp, "checkpoint-*.pt")))
    return checkpoint_fps


def get_latest_checkpoint(exp_name):
    """
    Retrieve an automatic checkpoint.

    Constructs the file path relative to the current execution directory. Returns None
    if no checkpoint is available.
    """
    checkpoint_fps = get_checkpoints(exp_name)
    if checkpoint_fps:
        return checkpoint_fps[-1]
    else:
        return None


def get_checkpoint_fp(log_fp, model_steps):
    """Get the checkpoint filepath for a given log path and amount of model steps."""
    return path.join(log_fp, "checkpoint-%d.pt" % (model_steps))


def update_checkpoints(exp_log_fp, exp_desc, model_steps, model_epoch, exp_config):
    """Update the available model checkpoints, honoring the exp configuration."""
    exp_name = path.basename(exp_log_fp)
    if model_epoch > 0 and model_epoch % exp_config["checkpoint_every"] == 0:
        exp_desc.save(
            get_checkpoint_fp(exp_log_fp, model_steps), model_steps, model_epoch
        )
    if exp_config["checkpoint_keep"] > 0:
        checkpoint_fps = get_checkpoints(exp_name)
        if len(checkpoint_fps) > exp_config["checkpoint_keep"]:
            for checkpoint_fp in checkpoint_fps[: -exp_config["checkpoint_keep"]]:
                os.remove(checkpoint_fp)


def train_val(exp_fp, config, device, num_threads, reporter=None):
    """
    Self-contained training with validation set monitoring for a given experiment.

    This routine is used for the parameter sweep facilities, but you can use it
    independently.
    """
    import torch
    import numpy as np
    import imp
    import data
    import config
    import tqdm
    import sys

    exp_fp, exp_name, exp_feat_fp, exp_log_fp = setup_paths(  # pylint: disable=unused-variable
        exp_fp
    )
    setup_logging(path.join(exp_log_fp, "train_val.log"))
    LOGGER.info("Running joined training / validation.")
    LOGGER.info("Using device `%s`", str(device))
    device = torch.device(device)  # pylint: disable=no-member
    # Config.
    exp_config = get_config(exp_fp, "train", "", num_threads)
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(exp_config["seed"])
    np.random.seed(exp_config["seed"])
    # Data.
    exp_prep_mod = imp.load_source(
        "_exp_preprocessing",
        path.join(path.dirname(__file__), exp_fp, "preprocessing.py"),
    )
    loader_trn = data.get_dataset(
        config.EXP_DATA_FP,
        "train",
        exp_config,
        exp_prep_mod.get_transformations("train", exp_config),
    )
    loader_val = data.get_dataset(
        config.EXP_DATA_FP,
        "val",
        exp_config,
        exp_prep_mod.get_transformations("val", exp_config),
    )

    # Model.
    exp_desc = get_desc(exp_fp, "train", exp_config)
    model_steps, model_epoch = 0, 0
    # Execution.
    exp_desc.setup_for_device(device, exp_config)

    epoch_pbar = tqdm.tqdm(range(model_epoch, exp_config["num_epochs"]), desc="Epochs")
    min_loss = sys.float_info.max
    for _ in epoch_pbar:
        exp_desc.lr_updater.step()
        epoch_loss = 0.0
        batch_pbar = tqdm.tqdm(
            loader_trn, unit_scale=exp_config["batch_size"], desc="Iterations"
        )
        for batch_data in batch_pbar:
            # Get the inputs and labels
            inputs = [
                dta.to(device) if isinstance(dta, torch.Tensor) else dta
                for dta in batch_data
            ]
            labels = inputs[1]
            outputs = exp_desc.model(inputs)
            # Loss computation
            loss = exp_desc.criterion(outputs, labels)
            # Backpropagation
            exp_desc.optimizer.zero_grad()
            loss.backward()
            exp_desc.optimizer.step()
            model_steps += 1
            # Keep track of loss for current epoch
            epoch_loss += loss.item() / len(loader_trn) / exp_config["batch_size"]
            batch_pbar.set_description(
                "Iterations (loss: %03f)" % (loss.item() / exp_config["batch_size"])
            )
        epoch_pbar.set_description("Epochs (loss: %03f)" % (epoch_loss))
        model_epoch += 1
        # Run on validation.
        batch_pbar = tqdm.tqdm(
            loader_val, unit_scale=exp_config["batch_size"], desc="Iterations"
        )
        epoch_loss = 0.0
        for batch_data in batch_pbar:
            # Get the inputs and labels
            inputs = [
                dta.to(device) if isinstance(dta, torch.Tensor) else dta
                for dta in batch_data
            ]
            labels = inputs[1]

            outputs = exp_desc.model(inputs)
            # Loss computation
            loss = exp_desc.criterion(outputs, labels)

            # Keep track of loss for current epoch
            epoch_loss += loss.item() / len(loader_val) / exp_config["batch_size"]
            batch_pbar.set_description(
                "Iterations (loss: %03f)" % (loss.item() / exp_config["batch_size"])
            )
        if reporter:
            reporter(
                loss=epoch_loss,
                neg_loss=-epoch_loss,
                timesteps_total=model_epoch,
                # checkpoint=
            )

        if epoch_loss < min_loss:
            LOGGER.info(
                "New minimum loss detected: `%f` for epoch %d.", epoch_loss, model_epoch
            )
            min_loss = epoch_loss

    LOGGER.info("Done.")
    return min_loss
