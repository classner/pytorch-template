"""Summarization of this model."""
import torchvision.utils as vutils


class ExpSummarizer:  # pylint: disable=too-few-public-methods

    """Create summaries for this experiment."""

    def __init__(self, exp_desc, exp_config):  # pylint: disable=unused-argument
        self.exp_desc = exp_desc

    def add_summary(self, writer, step):
        """Add a summary for the provided tensorboardx summary writer."""
        log = self.exp_desc.model.log
        for name, (val, tpe) in log.items():
            if val is not None:
                if tpe == "img":
                    sum_img = vutils.make_grid(val, normalize=True, scale_each=True)
                    writer.add_image(name, sum_img, step)
                elif tpe == "txt" and name == "id":
                    pass
                else:
                    raise Exception("Summary type `%s` not implemented." % (tpe))
