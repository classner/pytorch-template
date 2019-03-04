#!/usr/bin/env python
"""Main control file."""
# pylint: disable=bad-continuation, wrong-import-position, too-many-locals
# pylint: disable=too-many-branches, too-many-statements
import imp
import json
import logging
import signal
import sys
from os import path

import click
import coloredlogs
import numpy as np
import torch
import tqdm
from tensorboardX import SummaryWriter  # pylint: disable=import-error

sys.path.insert(0, path.dirname(__file__))  # isort:skip

import config  # isort:skip
import data  # isort:skip
import exp_tools  # isort:skip


LOGGER = logging.getLogger(__name__)


@click.command()
@click.argument("stage", type=click.Choice(["train", "val", "trainval", "test"]))
@click.argument("exp_fp", type=click.Path(exists=True, writable=True, file_okay=False))
@click.option(
    "--device",
    type=click.STRING,
    default="cuda:0",
    help="Set the PyTorch device to use. Default: `cuda:0`.",
)
@click.option(
    "--num_threads",
    type=click.INT,
    default=8,
    help="Number of data preprocessing threads.",
)
@click.option(
    "--no_checkpoint", type=click.BOOL, is_flag=True, help="Ignore checkpoints."
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Checkpoint to use for restoring (+.meta).",
)
@click.option(
    "--out_fp",
    type=click.Path(writable=True),
    default=None,
    help="If specified, write test or sample results there.",
)
@click.option(  # pylint: disable=too-many-locals
    "--custom_options",
    type=click.STRING,
    default="",
    help="Provide model specific custom options.",
)
@click.option(
    "--summarize_graph",
    type=click.BOOL,
    default=False,
    is_flag=True,
    help="If set, summarize the graph for tensorboard.",
)
def cli(stage, exp_fp, **kwargs):
    """autoenc main control file."""
    exp_fp, exp_name, exp_feat_fp, exp_log_fp = exp_tools.setup_paths(  # pylint: disable=unused-variable
        exp_fp
    )
    exp_tools.setup_logging(path.join(exp_log_fp, "run.py.log"))
    LOGGER.info("Running stage `%s`.", stage)
    LOGGER.info("Using device `%s`", kwargs["device"])
    device = torch.device(kwargs["device"])  # pylint: disable=no-member
    # Config.
    exp_config = exp_tools.get_config(
        exp_fp, stage, kwargs["custom_options"], kwargs["num_threads"]
    )
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(exp_config["seed"])
    np.random.seed(exp_config["seed"])
    # Data.
    exp_prep_mod = imp.load_source(
        "_exp_preprocessing", path.join(exp_fp, "preprocessing.py")
    )
    loader = data.get_dataset(
        config.EXP_DATA_FP,
        stage,
        exp_config,
        exp_prep_mod.get_transformations(stage, exp_config),
    )
    # Model.
    exp_desc = exp_tools.get_desc(exp_fp, stage, exp_config)
    model_steps, model_epoch = 0, 0
    # Checkpoint.
    if not kwargs["no_checkpoint"]:
        if kwargs["checkpoint"]:
            model_steps, model_epoch = exp_desc.restore(kwargs["checkpoint"])
        else:
            model_steps, model_epoch = exp_desc.restore(
                exp_tools.get_latest_checkpoint(exp_name)
            )
    # Summaries.
    writer = SummaryWriter(log_dir=path.join(exp_log_fp, stage))
    exp_sum_mod = imp.load_source("_exp_summary", path.join(exp_fp, "summary.py"))
    summarizer = exp_sum_mod.ExpSummarizer(exp_desc, exp_config)
    graph_exported = False
    # Execution.
    exp_desc.setup_for_device(device, exp_config)
    shutdown_requested = [False]

    def SIGINT_handler(  # pylint: disable=unused-argument, redefined-outer-name, invalid-name
        signal, frame
    ):
        LOGGER.warning("Received SIGINT.")
        shutdown_requested[0] = True

    signal.signal(signal.SIGINT, SIGINT_handler)
    if stage in ["train", "trainval"]:
        epoch_pbar = tqdm.tqdm(
            range(model_epoch, exp_config["num_epochs"]), desc="Epochs"
        )
    else:
        epoch_pbar = tqdm.tqdm(range(1), desc="Epochs")
    for _ in epoch_pbar:
        exp_desc.lr_updater.step()
        epoch_loss = 0.0
        batch_pbar = tqdm.tqdm(
            loader, unit_scale=exp_config["batch_size"], desc="Iterations"
        )
        summarized_tst = False
        for batch_data in batch_pbar:
            # Get the inputs and labels
            inputs = [
                dta.to(device) if isinstance(dta, torch.Tensor) else dta
                for dta in batch_data
            ]
            labels = inputs[1]
            outputs = exp_desc.model(inputs)
            if not graph_exported and kwargs["summarize_graph"]:
                LOGGER.info("Summarizing graph...")
                try:
                    writer.add_graph(exp_desc.model, inputs, True)
                except Exception as ex:  # pylint: disable=broad-except
                    LOGGER.error("Could not export graph: %s.", str(ex))
                graph_exported = True
            # Loss computation
            loss = exp_desc.criterion(outputs, labels)
            if stage in ["train", "trainval"]:
                writer.add_scalar("loss/loss", loss.item(), model_steps)
                writer.add_scalar(
                    "parameters/lr_0", exp_desc.lr_updater.get_lr()[0], model_steps
                )
                if (
                    model_steps % exp_config["summary_full_every"] == 0
                    and model_steps > 0
                ):
                    summarizer.add_summary(writer, model_steps)
                # Backpropagation
                exp_desc.optimizer.zero_grad()
                loss.backward()
                exp_desc.optimizer.step()
                model_steps += 1
            else:
                if not summarized_tst:
                    summarizer.add_summary(writer, model_steps)
                    summarized_tst = True

            if shutdown_requested[0]:
                break
            # Keep track of loss for current epoch
            epoch_loss += loss.item() / len(loader) / exp_config["batch_size"]
            batch_pbar.set_description(
                "Iterations (loss: %03f)" % (loss.item() / exp_config["batch_size"])
            )
        epoch_pbar.set_description("Epochs (loss: %03f)" % (epoch_loss))
        if stage not in ["train", "trainval"]:
            writer.add_scalar("loss/loss", epoch_loss, model_steps)
            result_json_fp = path.join(
                exp_log_fp, "checkpoint-%d-%s-score.json" % (model_steps, stage)
            )
            with open(result_json_fp, "w") as outf:
                json.dump({"loss": epoch_loss}, outf)
        if not shutdown_requested[0]:
            model_epoch += 1
        exp_tools.update_checkpoints(
            exp_log_fp, exp_desc, model_steps, model_epoch, exp_config
        )
        if shutdown_requested[0]:
            break
    # Optionally export scalar data to JSON for external processing.
    # writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    LOGGER.info("Done.")


if __name__ == "__main__":
    coloredlogs.install(level=logging.INFO)
    cli()  # pylint: disable=no-value-for-parameter
