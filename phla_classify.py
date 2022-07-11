# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Fine-tuning on A Classification Task with pretrained Transformer """

import itertools
import csv
import time
from tqdm.auto import tqdm
import fire

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional import auroc
import tokenization
import models
import optim
import train
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from utils import set_seeds, get_device, truncate_tokens_pair, iter_count, get_logger, pbar
import logging
from torch.nn.functional import softmax

log = get_logger(__name__)
log.setLevel(logging.INFO)


class CsvDataset(Dataset):
    """ Dataset Class for CSV file """
    labels = None

    def __init__(self, file, pipeline=[]):  # cvs file and pipeline object
        Dataset.__init__(self)
        data = []
        with open(file, "r") as f:
            # list of splitted lines : line is also list
            lines = csv.reader(f, delimiter=',', quotechar=None)

            @pbar(self.get_instances(lines), totol=iter_count(file), description='Loading data')
            def do(i, pipeline, data):
                for proc in pipeline:  # a bunch of pre-processing
                    i = proc(i)
                data.append(i)

            do(pipeline=pipeline, data=data)

        # To Tensors
        self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def get_instances(self, lines):
        """ get instance array from (csv-separated) line list """
        raise NotImplementedError


class PHLA(CsvDataset):
    """ Dataset class for MRPC """
    labels = ("0", "1")  # label names

    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None):  # skip header
            # label, text_a, text_b
            yield ' '.join(list(line[4])), ' '.join(list(line[1])), ' '.join(list(line[5]))  # 手动分词


def dataset_class(task):
    """ Mapping from task string to Dataset Class """
    table = {'phla': PHLA}
    return table[task]


class Pipeline(object):
    """ Preprocess Pipeline Class : callable """

    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Tokenizing(Pipeline):
    """ Tokenizing sentence pair """

    def __init__(self, preprocessor, tokenize):
        super().__init__()
        self.preprocessor = preprocessor  # e.g. text normalization
        self.tokenize = tokenize  # tokenize function

    def __call__(self, instance):
        label, text_a, text_b = instance

        label = self.preprocessor(label)

        tokens_a = self.tokenize(self.preprocessor(text_a))

        tokens_b = self.tokenize(self.preprocessor(text_b)) \
            if text_b else []

        return (label, tokens_a, tokens_b)


class AddSpecialTokensWithTruncation(Pipeline):
    """ Add special tokens [CLS], [SEP] with truncation """

    def __init__(self, max_len=512):
        super().__init__()
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        # -3 special tokens for [CLS] text_a [SEP] text_b [SEP]
        # -2 special tokens for [CLS] text_a [SEP]
        _max_len = self.max_len - 3 if tokens_b else self.max_len - 2
        truncate_tokens_pair(tokens_a, tokens_b, _max_len)

        # Add Special Tokens
        tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        tokens_b = tokens_b + ['[SEP]'] if tokens_b else []

        return (label, tokens_a, tokens_b)


class TokenIndexing(Pipeline):
    """ Convert tokens into token indexes and do zero-padding """

    def __init__(self, indexer, labels, max_len=512):
        super().__init__()
        self.indexer = indexer  # function : tokens to indexes
        # map from a label name to a label index
        self.label_map = {name: i for i, name in enumerate(labels)}
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        input_ids = self.indexer(tokens_a + tokens_b)
        segment_ids = [0] * len(tokens_a) + [1] * len(tokens_b)  # token type ids
        input_mask = [1] * (len(tokens_a) + len(tokens_b))

        label_id = self.label_map[label]

        # zero padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)
        input_mask.extend([0] * n_pad)

        return (input_ids, segment_ids, input_mask, label_id)


class Classifier(nn.Module):
    """ Classifier with Transformer """

    def __init__(self, cfg, n_labels):
        super().__init__()
        self.transformer = models.Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.dim, n_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        h = self.transformer(input_ids, segment_ids, input_mask)
        # only use the first h in the sequence
        pooled_h = self.activ(self.fc(h[:, 0]))
        logits = self.classifier(self.drop(pooled_h))
        return logits


# pretrain_file='../uncased_L-12_H-768_A-12/bert_model.ckpt',
# pretrain_file='../exp/bert/pretrain_100k/model_epoch_3_steps_9732.pt',

DEBUG = True

def main(task='phla',
         train_cfg='config/phla_train.json',
         model_cfg='config/phla_bert.json',
         # data_file='data/finetune.csv',
         # model_file=None,
         # pretrain_file='models/model_steps_20000.pt',
         # mode='train',
         data_file='data/independent_set.csv',
         model_file='exp/model_steps_44400.pt',
         pretrain_file=None,
         mode='eval',
         data_parallel=True,
         vocab='data/vocab.txt',
         save_dir='exp',
         max_len=52,):
    log.info('loading config...')
    cfg = train.Config.from_json(train_cfg)
    model_cfg = models.Config.from_json(model_cfg)

    set_seeds(cfg.seed)

    log.info(f'Loading Data from {data_file}...')
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=False)
    TaskDataset = dataset_class(task)  # task dataset class according to the task
    pipeline = [Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
                AddSpecialTokensWithTruncation(max_len),
                TokenIndexing(tokenizer.convert_tokens_to_ids,
                              TaskDataset.labels, max_len)]
    dataset = TaskDataset(data_file, pipeline)
    data_iter = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)


    log.info(f'loading model from {model_file}...')
    model = Classifier(model_cfg, len(TaskDataset.labels))
    criterion = nn.CrossEntropyLoss()
    log.info('start training...')
    trainer = train.Trainer(cfg,
                            model,
                            data_iter,
                            optim.optim4GPU(cfg, model),
                            save_dir, get_device())

    if mode == 'train':
        def get_loss(model, batch, global_step):  # make sure loss is a scalar tensor
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            loss = criterion(logits, label_id)
            return loss

        trainer.train(get_loss, model_file, pretrain_file, data_parallel)

    elif mode == 'eval':
        def evaluate(model, batch):

            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            _, label_pred = logits.max(1)
            result = (label_pred == label_id).float()  # .cpu().numpy()
            accuracy = result.mean()

            return accuracy, result, softmax(logits, dim=1), label_id


        results, logits, labels = trainer.eval(evaluate, model_file, data_parallel)
        total_accuracy = torch.cat(results).mean().item()
        log.debug(torch.cat(logits))
        log.debug(torch.cat(labels))
        auc_score = auroc(torch.cat(logits), torch.cat(labels),  num_classes=2)
        log.info(f'Accuracy: {total_accuracy}')
        log.info(f'AUC_Score: {auc_score}')


if __name__ == '__main__':
    fire.Fire(main)
