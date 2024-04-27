from transformers.tokenization_utils_base import BatchEncoding


class TokenizerCallMixin:
    def __call__(
        self,
        batch_text,
        context=None,
        full_length_search=None,
        padding=None,
        max_length=None,
        truncation=None,
        return_tensors="pt",
    ):
        batch = self.tensorize(
            batch_text,
            bsize=len(batch_text),
            context=context,
            full_length_search=full_length_search,
        )
        return BatchEncoding({"input_ids": batch[0][0], "attention_mask": batch[0][1]})


def add_special_tokens(tokenizer, config):
    if config.query_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [config.query_token]}, replace_additional_special_tokens=False
        )
    if config.doc_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [config.doc_token]}, replace_additional_special_tokens=False
        )

    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
