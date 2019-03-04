"""Set up the preprocessing pipeline."""
import logging

from torchvision import transforms

LOGGER = logging.getLogger(__name__)


def get_transformations(stage, exp_config):  # pylint: disable=unused-argument
    """Get the transformations for every processed tensor."""
    input_trans = transforms.Compose(
        [
            transforms.Resize((exp_config["height"], exp_config["width"])),
            transforms.ToTensor(),
        ]
    )
    return {"input": input_trans, "id": None}
