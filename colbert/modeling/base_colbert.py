import torch
from colbert.infra.config import ColBERTConfig
from colbert.utils.utils import torch_load_dnn
from transformers import AutoConfig, AutoModel, AutoTokenizer, T5EncoderModel


class BaseColBERT(torch.nn.Module):
    """
    Shallow module that wraps the ColBERT parameters, custom configuration, and underlying tokenizer.
    This class provides direct instantiation and saving of the model/colbert_config/tokenizer package.

    Like HF, evaluation mode is the default.
    """

    def __init__(self, name_or_path, colbert_config=None):
        super().__init__()

        self.colbert_config = ColBERTConfig.from_existing(
            ColBERTConfig.load_from_checkpoint(name_or_path), colbert_config
        )
        self.name = self.colbert_config.model_name or name_or_path

        # Load model as HF model
        self.hf_config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)

        if self.hf_config.model_type == "t5":
            self.bert = T5EncoderModel.from_pretrained(name_or_path)
        else:
            self.bert = AutoModel.from_pretrained(name_or_path, config=self.colbert_config)

        # Add Linear projection layer and set some attributes to behave like native HF model
        self.linear = torch.nn.Linear(self.hf_config.hidden_size, self.colbert_config.dim, bias=False)

        self.raw_tokenizer = AutoTokenizer.from_pretrained(name_or_path)

        self.eval()

    @property
    def device(self):
        return self.bert.device

    @property
    def score_scaler(self):
        return self.model.score_scaler

    def save(self, path):
        assert not path.endswith(".dnn"), f"{path}: We reserve *.dnn names for the deprecated checkpoint format."

        self.model.save_pretrained(path)
        self.raw_tokenizer.save_pretrained(path)

        self.colbert_config.save_for_checkpoint(path)
