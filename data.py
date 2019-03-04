"""Data handling tools."""
import logging
from glob import glob
from os import path

import tqdm
from PIL import Image
from torch.utils import data

LOGGER = logging.getLogger(__name__)


class FSBacked(data.Dataset):  # pylint: disable=too-many-instance-attributes
    """
    Filesystem backed dataset.

    Represents a folder with continuously named files according to the pattern
    `00000_name.suffix`, where the number of zeros determines a fixed length number
    for the entire directory, `name` specifies a certain input name and suffix the
    input type. Currently implemented are images (.jpg, .png) and text (.txt).

    The returned dataset tuple is ordered by the `name`s of the elements.
    """

    def __init__(self, exp_data_fp, transformations):
        LOGGER.info("Loading data from `%s`...", exp_data_fp)
        self.exp_data_fp = exp_data_fp
        assert path.exists(self.exp_data_fp), "Data folder `%s` doesn't exist!" % (
            self.exp_data_fp
        )
        self.fillwidth = 0
        self.colnames = []
        self.coltypes = []
        self.colendings = []
        self.colfullnames = []
        # Populate the column info.
        self._scan_columns()
        self.nsamples = 0
        # Locate all samples.
        self._scan_samples()
        # Finalize the transformation configuration.
        self.transformations = []
        for colname in self.colnames:
            if colname not in transformations:
                LOGGER.warning("`%s` not found in data transformations.", colname)
                self.transformations.append(None)
            else:
                self.transformations.append(transformations[colname])

    def _scan_columns(self):
        for fillwidth in range(1, 9):
            zero_entries = sorted(
                glob(
                    path.join(
                        self.exp_data_fp, "{0:0{width}}_*.*".format(0, width=fillwidth)
                    )
                )
            )
            if zero_entries:
                break
        colnames, colfullnames, colendings, coltypes = [], [], [], []
        self.fillwidth = fillwidth  # pylint: disable=undefined-loop-variable
        LOGGER.info("Columns:")
        for fp in zero_entries:  # pylint: disable=invalid-name
            fn = path.basename(fp)  # pylint: disable=invalid-name
            if len(fn.split(".")) > 2:
                raise Exception("No dots apart from file-ending allowed!")
            colt = None
            colfulln = fn[fn.find("_") + 1 : fn.find(".")]
            if ":" in colfulln:
                coln = colfulln[: colfulln.find(":")]
                colt = colfulln[colfulln.find(":") + 1 :]
                assert colt in ["jpg", "jpeg", "png", "webp"], (
                    "':' only allowed in colname as image storage spec in "
                    "[jpg, jpeg, png, webp] (is `%s`)!" % (colt)
                )
            else:
                coln = colfulln
            fending = fn[fn.find(".") + 1 :]
            if colt is None:
                if fending in ["jpg", "jpeg", "png", "webp"]:
                    colt = fending
                elif fending == "txt":
                    colt = "txt"
                elif fending == "npy":
                    colt = "plain"
            LOGGER.info("  %s: %s", colfulln, colt)
            colnames.append(coln)
            colfullnames.append(colfulln)
            coltypes.append(colt)
            colendings.append(fending)
        assert colfullnames, "No columns found!"
        self.colnames = colnames
        self.coltypes = coltypes
        self.colendings = colendings
        self.colfullnames = colfullnames

    def _scan_samples(self):
        LOGGER.info("Scanning...")
        nsamples = 0
        scan_complete = False
        pbar = tqdm.tqdm()
        while not scan_complete:
            for coln, cole in zip(self.colfullnames, self.colendings):
                if not path.exists(
                    path.join(  # pylint: disable=bad-continuation
                        self.exp_data_fp,
                        "{0:0{width}}_{1}.{2}".format(
                            nsamples, coln, cole, width=self.fillwidth
                        ),
                    )
                ):
                    scan_complete = True
                    break
            if scan_complete:
                break
            nsamples += 1
            pbar.update(1)
        self.nsamples = nsamples
        pbar.close()
        LOGGER.info("%d complete examples located.", nsamples)

    def __getitem__(self, index):
        ret = []
        for colname, coltype, transform in zip(
            self.colnames,  # pylint: disable=bad-continuation
            self.coltypes,  # pylint: disable=bad-continuation
            self.transformations,  # pylint: disable=bad-continuation
        ):
            data_path = path.join(
                self.exp_data_fp,
                "{0:0{fillwidth}}_{colname}.{coltype}".format(
                    index, fillwidth=self.fillwidth, colname=colname, coltype=coltype
                ),
            )
            if coltype in ["jpg", "jpeg", "png"]:
                sample = Image.open(data_path)
                if transform:
                    sample = transform(sample)
                ret.append(sample)
            elif coltype == "txt":
                with open(data_path, "r") as inf:
                    text = "\n".join(inf.readlines()).strip()
                    ret.append(text)
            else:
                raise Exception("Column type `%s` not implemented." % (coltype))
        return tuple(ret)

    def __len__(self):
        return self.nsamples


def get_dataset(exp_data_fp, mode, exp_config, transformations):
    """Create a `DataLoader` for a requested dataset."""
    dset = FSBacked(
        path.join(
            exp_data_fp, exp_config["data_prefix"], exp_config["data_version"], mode
        ),
        transformations,
    )
    use_shuffle = mode in ["train", "trainval"]
    LOGGER.info("Shuffle: `%s`.", str(use_shuffle))
    loader = data.DataLoader(
        dset,
        batch_size=exp_config["batch_size"],
        shuffle=use_shuffle,
        num_workers=exp_config["num_threads"],
    )
    LOGGER.info("%d examples prepared, %d steps per epoch", dset.nsamples, len(loader))
    return loader
