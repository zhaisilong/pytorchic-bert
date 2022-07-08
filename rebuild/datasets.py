# Input file format :
# 1. One sentence per line. These should ideally be actual sentences,
#    not entire paragraphs or arbitrary spans of text. (Because we use
#    the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed
#    so that the "next sentence prediction" task doesn't span between documents.
import torch

from utils import truncate_tokens_pair, get_random_word
from random import randint, shuffle, random


def seek_random_offset(f, back_margin=2000):
    """ seek random offset of file pointer """
    f.seek(0, 2)
    # we remain some amount of text to read
    max_offset = f.tell() - back_margin
    f.seek(randint(0, max_offset), 0)
    f.readline()  # throw away an incomplete sentence


class SentPairDataLoader():
    """ Load sentence pair (sequential or random order) from corpus """

    def __init__(self, file, batch_size, tokenize, max_len, short_sampling_prob=0.1, pipeline=[]):
        super().__init__()
        self.f_pos = open(file, "r", encoding='utf-8', errors='ignore')  # 正采样
        self.f_neg = open(file, "r", encoding='utf-8', errors='ignore')  # 负（随机）采样
        self.tokenize = tokenize  # tokenize function
        self.max_len = max_len  # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob  # 采样短行的概率
        self.pipeline = pipeline  # 函数列表，逐个执行
        self.batch_size = batch_size

    def read_tokens(self, f, length, discard_last_and_restart=True):
        """Read tokens from file pointer with limited length
        读取固定长度的 tokens，一行不够两行，三行...
        如果中间遇到空行，丢弃之前读的所有 token
        """
        tokens = []
        while len(tokens) < length:
            line = f.readline()  # 读取一行
            if not line:  # end of file
                return None
            if not line.strip():  # blank line (delimiter of documents)
                if discard_last_and_restart:
                    tokens = []  # throw all and restart
                    continue
                else:
                    return tokens  # return last tokens in the document
            tokens.extend(self.tokenize(line.strip()))
        return tokens

    def __iter__(self):  # iterator to load data
        while True:
            batch = []
            for i in range(self.batch_size):
                # sampling length of each tokens_a and tokens_b
                # sometimes sample a short sentence to match between train and test sequences
                len_tokens = randint(1, int(self.max_len / 2)) \
                    if random() < self.short_sampling_prob \
                    else int(self.max_len / 2)

                is_next = random() < 0.5  # whether token_b is next to token_a or not

                tokens_a = self.read_tokens(self.f_pos, len_tokens, True)  # 如果抽到空行则丢弃
                seek_random_offset(self.f_neg)  # 随机置负采样指针
                f_next = self.f_pos if is_next else self.f_neg  # 正负采样概率均为 50%
                tokens_b = self.read_tokens(f_next, len_tokens, False)  # 保留空行

                if tokens_a is None or tokens_b is None:  # end of file
                    self.f_pos.seek(0, 0)  # reset file pointer
                    return

                instance = (is_next, tokens_a, tokens_b)
                for proc in self.pipeline:
                    instance = proc(instance)

                batch.append(instance)

            # To Tensor
            batch_tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*batch)]
            yield batch_tensors


class Pipeline():
    """ Pre-process Pipeline Class : callable """

    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Preprocess4Pretrain(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512):
        super().__init__()
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len

    def __call__(self, instance):
        is_next, tokens_a, tokens_b = instance

        # -3  for special tokens [CLS], [SEP], [SEP]
        truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3)

        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        input_mask = [1] * len(tokens)

        # For masked Language Models
        masked_tokens, masked_pos = [], []
        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = min(self.max_pred, max(1, int(round(len(tokens) * self.mask_prob))))
        # candidate positions of masked tokens
        cand_pos = [i for i, token in enumerate(tokens)
                    if token != '[CLS]' and token != '[SEP]']  # cand_pos is idx
        shuffle(cand_pos)
        for pos in cand_pos[:n_pred]:
            masked_tokens.append(tokens[pos])
            masked_pos.append(pos)
            if random() < 0.8:  # 80% mask n_pred of tokens
                tokens[pos] = '[MASK]'
            elif random() < 0.5:  # 10% sample from vocab
                tokens[pos] = get_random_word(self.vocab_words)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1] * len(masked_tokens)  # same len as masked_tokens

        # Token Indexing
        input_ids = self.indexer(tokens)
        masked_ids = self.indexer(masked_tokens)

        # Zero Padding To Max Len
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)
        input_mask.extend([0] * n_pad)

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
            masked_weights.extend([0] * n_pad)

        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next)
