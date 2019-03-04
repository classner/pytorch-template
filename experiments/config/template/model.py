"""Model description."""
import logging

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

LOGGER = logging.getLogger(__name__)


class Dummy(nn.Module):
    """
    Create a parameterized dummy network
    """

    def __init__(self, exp_config):  # pylint: disable=unused-argument
        super().__init__()

        # Module parts.
        self.fc1 = nn.Linear(28 * 28, 28 * 28)

        # For logging.
        self.log = {"in": [None, "img"], "id": [None, "txt"], "out": [None, "img"]}

    def forward(self, x):  # pylint: disable=arguments-differ
        self.log["in"][0] = x[1]
        self.log["id"][0] = x[0]
        x = x[1]

        x = x.reshape((-1, 28 * 28))
        x = self.fc1(x)
        x = x.reshape((-1, 28, 28))

        self.log["out"][0] = x

        return x


class ExpDesc:

    """
    Encapsulates an experiment description.
    """

    def __init__(self, mode, exp_config):  # pylint: disable=unused-argument
        self.model = Dummy(exp_config)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=exp_config["opt_learning_rate"],
            weight_decay=exp_config["opt_weight_decay"],
        )
        self.lr_updater = lr_scheduler.StepLR(
            self.optimizer, exp_config["lr_decay_epochs"], exp_config["lr_decay"]
        )

    def restore(self, checkpoint_fp):
        """Load the model and optimizer parameters from a checkpoint."""
        if not checkpoint_fp:
            return 0, 0
        LOGGER.info("Restoring model from `%s`", checkpoint_fp)
        checkpoint = torch.load(checkpoint_fp)
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        except Exception as ex:  # pylint: disable=broad-except
            LOGGER.error("Could not restore model state! Exception: %s.", str(ex))
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as ex:  # pylint: disable=broad-except
            LOGGER.error("Could not restore optimizer state! Exception: %s.", str(ex))
        try:
            self.lr_updater.load_state_dict(checkpoint["lr_updater_state_dict"])
        except Exception as ex:  # pylint: disable=broad-except
            LOGGER.error("Could not restore lr_updater state! Exception: %s.", str(ex))
        return checkpoint["steps"], checkpoint["epochs"]

    def save(self, checkpoint_fp, steps, epochs):
        """Save the model and optimizer to a checkpoint."""
        LOGGER.info("Saving model in `%s`", checkpoint_fp)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_updater_state_dict": self.lr_updater.state_dict(),
                "steps": steps,
                "epochs": epochs,
            },
            checkpoint_fp,
        )

    def setup_for_device(self, device, exp_config):
        """Move the model and optimizer to a device."""
        self.model.to(device)
        self.criterion.to(device)
        # Now the optimizer and lr scheduler must be recreated!
        # (For more info, see https://github.com/pytorch/pytorch/issues/2830 .)
        old_state_dict = self.optimizer.state_dict()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=exp_config["opt_learning_rate"],
            weight_decay=exp_config["opt_weight_decay"],
        )
        self.optimizer.load_state_dict(old_state_dict)
        old_state_dict = self.lr_updater.state_dict()
        self.lr_updater = lr_scheduler.StepLR(
            self.optimizer, exp_config["lr_decay_epochs"], exp_config["lr_decay"]
        )
        self.lr_updater.load_state_dict(old_state_dict)
