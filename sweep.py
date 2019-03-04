#!/usr/bin/env python
"""Run a parameter sweep for an experiment."""
# pylint: disable=wrong-import-order, import-error, ungrouped-imports, wrong-import-position
# pylint: disable=too-many-locals, too-many-arguments, bad-continuation
import logging
import sys
from os import path

import click
import coloredlogs
import ray
import torch
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import HyperOptSearch

sys.path.insert(0, path.dirname(__file__))  # isort:skip

import exp_tools  # isort:skip


LOGGER = logging.getLogger(__name__)


@click.command()
@click.argument("exp_fp", type=click.Path(exists=True, file_okay=False, readable=True))
@click.option(
    "--n_samples", type=click.INT, default=10, help="How many configurations to test."
)
@click.option(
    "--n_jobs", type=click.INT, default=1, help="How many jobs to run in parallel."
)
@click.option(
    "--ray_redis",
    type=click.STRING,
    default="",
    help="Ray redis server address or nothing (run local).",
)
@click.option(
    "--use_cpu",
    type=click.BOOL,
    is_flag=True,
    default=False,
    help="Use CPUs only to run.",
)
@click.option(
    "--data_threads",
    type=click.INT,
    default=4,
    help="Data loading threads to use on each worker.",
)
@click.option(
    "--no_resume",
    type=click.BOOL,
    is_flag=True,
    default=False,
    help="If specified, don't resume a sweep but start from scratch.",
)
def cli(
    exp_fp,
    n_samples=10,
    n_jobs=1,
    ray_redis="",
    use_cpu=False,
    data_threads=4,
    no_resume=False,
):
    """Run a hyperparameter sweep for an experiment."""
    LOGGER.info("Running hyperparameter sweep for experiment `%s`.", exp_fp)
    LOGGER.info("Using %d samples with %d parallel jobs.", n_samples, n_jobs)
    LOGGER.info("Initializing ray connection...")
    if ray_redis == "":
        LOGGER.info("Using local server.")
        ray.init()
    else:
        LOGGER.info("Connecting to server `%s`...", ray_redis)
        ray.init(redis_address=ray_redis)
    LOGGER.info("Creating configurations...")
    hyper_config = exp_tools.get_hyperparameters(exp_fp)
    basic_config = exp_tools.get_config(exp_fp, "train", "", 0)
    # exp_runner = ExpRunner(
    #    exp_fp, torch.device("cuda:0"), 8
    # )  # pylint: disable=no-member

    def run_experiment(config, reporter):
        """Experiment closure."""
        if not use_cpu:
            device = torch.device("cpu:0")  # pylint: disable=no-member
        else:
            device = torch.device("cuda:0")  # pylint: disable=no-member
        exp_tools.train_val(exp_fp, config, device, data_threads, reporter)

    hyperband = AsyncHyperBandScheduler(
        time_attr="timesteps_total",
        reward_attr="neg_loss",
        max_t=basic_config["num_epochs"] + 1,
        grace_period=basic_config["min_num_epochs"],
    )
    hyperopt_search = HyperOptSearch(
        hyper_config, reward_attr="neg_loss", max_concurrent=n_jobs
    )
    n_gpus = 1
    if use_cpu:
        n_gpus = 0
    exp_config = tune.Experiment(
        "sweep_%s" % (path.basename(exp_fp)),
        run=run_experiment,
        local_dir=path.join(path.dirname(__file__), "sweep"),
        num_samples=n_samples,
        resources_per_trial={"cpu": data_threads, "gpu": n_gpus},
        config=hyper_config,
    )
    LOGGER.info("Running experiments...")
    trials = tune.run_experiments(
        exp_config,
        scheduler=hyperband,
        search_alg=hyperopt_search,
        resume=not no_resume,
    )
    LOGGER.info("Analyzing results...")
    min_loss = sys.float_info.max
    best_trial = None
    for trial in trials:
        if trial.last_result["loss"] < min_loss:
            min_loss = trial.last_result["loss"]
            best_trial = trial
    LOGGER.info("Best parameters: %s.", str(best_trial.config))
    LOGGER.info("Done.")


if __name__ == "__main__":
    coloredlogs.install(level=logging.INFO)
    cli()  # pylint: disable=no-value-for-parameter
