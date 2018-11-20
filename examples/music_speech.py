"""
This is an example of audiomate using for the whole process of classifying audio as either music or speech.
"""

import os
import random
from timeit import default_timer

import numpy as np

import torch
from torch import nn
from torch.utils import data

from ignite import engine
from ignite import metrics

import audiomate
from audiomate.corpus import assets
from audiomate.corpus import io
from audiomate.corpus import subset
from audiomate.processing import pipeline
from audiomate.utils import units
from audiomate import encoding
from audiomate import feeding

FRAME_SIZE = 2048
HOP_SIZE = 1024
SAMPLING_RATE = 16000

BATCH_SIZE = 10

OUTPUT_LABELS = ['music', 'speech']

#
#   Download data
#
corpus_path = 'output/endtoend/gtzan'

if not os.path.isdir(corpus_path):
    io.GtzanDownloader().download(corpus_path)

#
#   Load corpus
#
corpus = audiomate.Corpus.load(corpus_path, reader='gtzan')

#
#   Create splits
#
train_path = 'output/endtoend/train'
dev_path = 'output/endtoend/dev'
test_path = 'output/endtoend/test'

if not os.path.isdir(train_path) or not os.path.isdir(dev_path) or not os.path.isdir(test_path):
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(dev_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    splitter = subset.Splitter(corpus, random_seed=111)
    splits = splitter.split_by_proportionally_distribute_labels({
        'train': 0.7,
        'dev': 0.15,
        'test': 0.15
    }, use_lengths=True)

    train_corpus = audiomate.Corpus.from_corpus(splits['train'])
    train_corpus.save_at(train_path)

    dev_corpus = audiomate.Corpus.from_corpus(splits['dev'])
    dev_corpus.save_at(dev_path)

    test_corpus = audiomate.Corpus.from_corpus(splits['test'])
    test_corpus.save_at(test_path)

    # Print length of every subset, to ensure it was able to create reasonable subsets
    for split_name, split in splits.items():
        print('Subset "{}" duration: {}'.format(split_name, split.total_duration))
else:
    train_corpus = audiomate.Corpus.load(train_path)
    dev_corpus = audiomate.Corpus.load(dev_path)
    test_corpus = audiomate.Corpus.load(test_path)

#
#   Extract Features
#
train_feat_path = 'output/endtoend/train_feats.hdf5'
dev_feat_path = 'output/endtoend/dev_feats.hdf5'
test_feat_path = 'output/endtoend/test_feats.hdf5'

if not os.path.isfile(train_feat_path) or not os.path.isfile(dev_feat_path) or not os.path.isfile(test_feat_path):
    mfcc = pipeline.MFCC(n_mfcc=13, n_mels=128)
    power_to_db = pipeline.PowerToDb(ref=np.max, parent=mfcc)
    deltas = pipeline.Delta(parent=power_to_db)
    stack = pipeline.Stack(parents=[power_to_db, deltas])

    var_pool = pipeline.VarPool(10, parent=stack)
    avg_pool = pipeline.AvgPool(10, parent=stack)

    pool_stack = pipeline.Stack(parents=[avg_pool, var_pool])

    train_feats = pool_stack.process_corpus(train_corpus,
                                            train_feat_path,
                                            frame_size=FRAME_SIZE,
                                            hop_size=HOP_SIZE,
                                            sr=SAMPLING_RATE)

    dev_feats = pool_stack.process_corpus(dev_corpus,
                                          dev_feat_path,
                                          frame_size=FRAME_SIZE,
                                          hop_size=HOP_SIZE,
                                          sr=SAMPLING_RATE)

    test_feats = pool_stack.process_corpus(test_corpus,
                                           test_feat_path,
                                           frame_size=FRAME_SIZE,
                                           hop_size=HOP_SIZE,
                                           sr=SAMPLING_RATE)
else:
    train_feats = assets.FeatureContainer(train_feat_path)
    dev_feats = assets.FeatureContainer(dev_feat_path)
    test_feats = assets.FeatureContainer(test_feat_path)

train_feats.open()
dev_feats.open()
test_feats.open()

#
#   Extract targets
#
train_target_path = 'output/endtoend/train_targets.hdf5'
dev_target_path = 'output/endtoend/dev_targets.hdf5'
test_target_path = 'output/endtoend/test_targets.hdf5'

if not os.path.isfile(train_target_path) or not os.path.isfile(dev_target_path) or not os.path.isfile(test_target_path):
    fs = units.FrameSettings(train_feats.frame_size, train_feats.hop_size)
    encoder = encoding.FrameHotEncoder(OUTPUT_LABELS, audiomate.corpus.LL_DOMAIN, fs, SAMPLING_RATE)

    train_targets = encoder.encode_corpus(train_corpus, train_target_path)
    dev_targets = encoder.encode_corpus(dev_corpus, dev_target_path)
    test_targets = encoder.encode_corpus(test_corpus, test_target_path)
else:
    train_targets = assets.Container(train_target_path)
    dev_targets = assets.Container(dev_target_path)
    test_targets = assets.Container(test_target_path)

train_targets.open()
dev_targets.open()
test_targets.open()


#
#   Train
#
class Wrap:

    def __init__(self, ds):
        self.ds = ds
        self.iterable = None
        self.count_to = len(self)
        self.count = self.count_to

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        if self.count >= self.count_to:
            self.iterable = self.ds.partitioned_iterator('1G', shuffle=True, seed=1)
            self.count = 0

        x = next(self.iterable)
        self.count += 1

        return x


train_set = Wrap(feeding.FrameDataset(train_corpus, [train_feats, train_targets]))
train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=0, shuffle=False, pin_memory=False)

dev_set = Wrap(feeding.FrameDataset(dev_corpus, [dev_feats, dev_targets]))
dev_loader = data.DataLoader(dev_set, batch_size=BATCH_SIZE, num_workers=0, shuffle=False, pin_memory=False)

test_set = Wrap(feeding.FrameDataset(test_corpus, [test_feats, test_targets]))
test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=0, shuffle=False, pin_memory=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on device {}'.format(device.type))

model = nn.Sequential(
    nn.Linear(52, 20),
    nn.ReLU(),
    nn.BatchNorm1d(20),
    nn.Linear(20, 2),
    nn.Softmax(dim=1)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

trainer = engine.create_supervised_trainer(model, optimizer, loss_fn)

probs_to_class = lambda x: (x[0], torch.max(x[1], 1)[1])
eval_metrics = {
    'bce': metrics.Loss(loss_fn),
    'precision': metrics.Precision(output_transform=probs_to_class),
    'recall': metrics.Recall(output_transform=probs_to_class)
}
evaluator = engine.create_supervised_evaluator(model, metrics=eval_metrics)


# @trainer.on(engine.Events.ITERATION_COMPLETED)
# def log_training_loss(trainer):
#     print('Epoch[{}] Loss: {:.2f}'.format(trainer.state.epoch, trainer.state.output))


@trainer.on(engine.Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print('Training Results - Epoch: {} Avg loss: {:.2f}, Avg precision: {:.2f}, Avg recall: {:.2f}'
          .format(trainer.state.epoch, metrics['bce'], metrics['precision'].mean(), metrics['recall'].mean()))


@trainer.on(engine.Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(dev_loader)
    metrics = evaluator.state.metrics
    print('Validation Results - Epoch: {} Avg loss: {:.2f}, Avg precision: {:.2f}, Avg recall: {:.2f}'
          .format(trainer.state.epoch, metrics['bce'], metrics['precision'].mean(), metrics['recall'].mean()))


start = default_timer()
trainer.run(train_loader, max_epochs=10)
end = default_timer()

evaluator.run(test_loader)
metrics = evaluator.state.metrics
print('Test Results - Epoch: {} Avg loss: {:.2f}, Avg precision: {:.2f}, Avg recall: {:.2f}'
      .format(trainer.state.epoch, metrics['bce'], metrics['precision'].mean(), metrics['recall'].mean()))
print('Training time: {}'.format(end - start))
