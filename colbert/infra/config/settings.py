import os
from dataclasses import dataclass
from typing import Optional

import __main__
import torch
from colbert.utils.utils import timestamp

from .core_config import DefaultVal


@dataclass
class RunSettings:
    """
    The defaults here have a special status in Run(), which initially calls assign_defaults(),
    so these aren't soft defaults in that specific context.
    """

    overwrite: bool = False

    root: str = os.path.join(os.getcwd(), "experiments")
    experiment: str = "default"

    index_root: str = None
    name: str = timestamp(daydir=True)

    rank: int = 0
    nranks: int = 1
    amp: bool = True

    total_visible_gpus = torch.cuda.device_count()
    gpus: int = total_visible_gpus

    avoid_fork_if_possible: bool = False

    @property
    def gpus_(self):
        value = self.gpus

        if isinstance(value, int):
            value = list(range(value))

        if isinstance(value, str):
            value = value.split(",")

        value = list(map(int, value))
        value = sorted(list(set(value)))

        assert all(device_idx in range(0, self.total_visible_gpus) for device_idx in value), value

        return value

    @property
    def index_root_(self):
        return self.index_root or os.path.join(self.root, self.experiment, "indexes/")

    @property
    def script_name_(self):
        if "__file__" in dir(__main__):
            cwd = os.path.abspath(os.getcwd())
            script_path = os.path.abspath(__main__.__file__)
            root_path = os.path.abspath(self.root)

            if script_path.startswith(cwd):
                script_path = script_path[len(cwd) :]

            else:
                try:
                    commonpath = os.path.commonpath([script_path, root_path])
                    script_path = script_path[len(commonpath) :]
                except:
                    pass

            assert script_path.endswith(".py")
            script_name = script_path.replace("/", ".").strip(".")[:-3]

            assert len(script_name) > 0, (script_name, script_path, cwd)

            return script_name

        return "none"

    @property
    def path_(self):
        return os.path.join(self.root, self.experiment, self.script_name_, self.name)

    @property
    def device_(self):
        return self.gpus_[self.rank % self.nranks]


@dataclass
class TokenizerSettings:
    query_token_id: str = "[unused0]"
    doc_token_id: str = "[unused1]"
    query_token: str = "[Q]"
    doc_token: str = "[D]"


@dataclass
class ResourceSettings:
    checkpoint: str = None
    triples: str = None
    collection: str = None
    queries: str = None
    index_name: str = None


@dataclass
class DocSettings:
    dim: int = 128
    doc_maxlen: int = 220
    mask_punctuation: bool = True


@dataclass
class QuerySettings:
    query_maxlen: int = 32
    attend_to_mask_tokens: bool = False
    interaction: str = "colbert"


@dataclass
class TrainingSettings:
    similarity: str = "cosine"

    bsize: int = 32

    accumsteps: int = 1

    lr: float = 3e-06

    maxsteps: int = 500_000

    save_every: int = None

    log_every: Optional[int] = None

    resume: bool = False

    ## NEW:
    warmup: int = None

    warmup_bert: int = None

    relu: bool = False

    nway: int = 2

    use_ib_negatives: bool = False

    drop_duplciate_passages: bool = False

    reranker: bool = False

    distillation_alpha: float = 1.0

    ignore_scores: bool = False

    model_name: str = None  # DefaultVal('bert-base-uncased'


@dataclass
class IndexingSettings:
    index_path: str = None

    index_bsize: int = 64

    nbits: int = 1

    kmeans_niters: int = 4

    resume: bool = False

    @property
    def index_path_(self):
        return self.index_path or os.path.join(self.index_root_, self.index_name)


@dataclass
class SearchSettings:
    ncells: int = None
    centroid_score_threshold: float = None
    ndocs: int = None
    load_index_with_mmap: bool = False
