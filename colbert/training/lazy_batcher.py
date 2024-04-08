from functools import partial

import numpy as np
from colbert.data.collection import Collection
from colbert.data.examples import Examples
from colbert.data.queries import Queries
from colbert.infra.config.config import ColBERTConfig
from colbert.modeling.tokenization import (DocTokenizer, QueryTokenizer,
                                           tensorize_triples)
from colbert.utils.utils import zipstar

# from colbert.utils.runs import Run


class LazyBatcher:
    def __init__(
        self, config: ColBERTConfig, triples, queries, collection, rank=0, nranks=1
    ):
        self.bsize, self.accumsteps = config.bsize, config.accumsteps
        self.nway = config.nway

        self.query_tokenizer = QueryTokenizer(config)
        self.doc_tokenizer = DocTokenizer(config)
        self.tensorize_triples = partial(
            tensorize_triples, self.query_tokenizer, self.doc_tokenizer
        )
        self.position = 0

        self.triples = Examples.cast(triples, nway=self.nway).tolist(rank, nranks)
        self.queries = Queries.cast(queries)
        self.collection = Collection.cast(collection)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.triples)

    def __next__(self):
        offset, endpos = self.position, min(
            self.position + self.bsize, len(self.triples)
        )
        self.position = endpos

        if offset + self.bsize > len(self.triples):
            raise StopIteration

        all_queries, all_passages, all_scores = [], [], []

        for position in range(offset, endpos):
            query, *pids = self.triples[position]
            pids = pids[: self.nway]

            query = self.queries[query]

            try:
                pids, scores = zipstar(pids)
            except:
                scores = []

            passages = [self.collection[pid] for pid in pids]

            all_queries.append(query)
            all_passages.extend(passages)
            all_scores.extend(scores)

        assert len(all_scores) in [0, len(all_passages)], len(all_scores)

        print(all_queries[0])
        return self.collate(all_queries, all_passages, all_scores)

    def collate(self, queries, passages, scores):
        assert len(queries) == self.bsize
        assert len(passages) == self.nway * self.bsize

        return self.tensorize_triples(
            queries, passages, scores, self.bsize // self.accumsteps, self.nway
        )

    # def skip_to_batch(self, batch_idx, intended_batch_size):
    #     Run.warn(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')
    #     self.position = intended_batch_size * batch_idx


class LazyBatcherDistinctPassages(LazyBatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_state = np.random.RandomState(1234)

    def __next__(self):
        offset, endpos = self.position, min(
            self.position + self.bsize, len(self.triples)
        )
        self.position = endpos

        if offset + self.bsize > len(self.triples):
            raise StopIteration

        all_queries, all_passages, all_scores = [], [], []

        positive_pids = []
        negative_pids = []
        for position in range(offset, endpos):
            query, *pids = self.triples[position]
            pids = pids[: self.nway]

            query = self.queries[query]

            try:
                pids, scores = zipstar(pids)
            except:
                scores = []

            # passages = [self.collection[pid] for pid in pids]

            all_queries.append(query)
            # all_passages.extend(passages)
            all_scores.extend(scores)
            positive_pids.append(pids[0])
            negative_pids.append(pids[1:])

        # find negatives which are equal to positives and replace them with random passage
        negative_pids = np.array(negative_pids)
        replace_mask = np.ones_like(negative_pids, dtype=bool)
        max_repeats = 3
        i = 0
        while i < max_repeats and np.any(replace_mask):
            for pid in positive_pids:
                replace_mask[negative_pids != pid] = False

            n_duplicates = replace_mask.sum()
            if n_duplicates:
                random_pids = self.random_state.randint(len(self.collection), size=n_duplicates)
                negative_pids[replace_mask] = random_pids
            i += 1

        for positive, negatives in zip(positive_pids, negative_pids):
            all_passages.extend([self.collection[pid] for pid in [positive] + negatives.tolist()])


        assert len(all_scores) in [0, len(all_passages)], len(all_scores)

        return self.collate(all_queries, all_passages, all_scores)

