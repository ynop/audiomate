import os
import tempfile

import audiomate
from audiomate.annotations import Label, LabelList
from audiomate.issuers import Issuer, Speaker, Gender
from audiomate.corpus.subset import subview


def get_resource_path(sub_path_components):
    """ Get the absolute path of a file in the resources folder with its relative path components. """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), *sub_path_components))


def sample_wav_file(name):
    """ Return the path to a wav file of the `wav_files` folder with its name. """
    return get_resource_path(['wav_files', name])


def sample_corpus_path(name):
    """ Return the path to a sample corpus path with its name. """
    return get_resource_path(['sample_corpora', name])


def create_dataset():
    temp_path = tempfile.mkdtemp()

    ds = audiomate.Corpus(temp_path)

    wav_1_path = sample_wav_file('wav_1.wav')
    wav_2_path = sample_wav_file('wav_2.wav')
    wav_3_path = sample_wav_file('wav_3.wav')
    wav_4_path = sample_wav_file('wav_4.wav')

    file_1 = ds.new_file(wav_1_path, track_idx='wav-1')
    file_2 = ds.new_file(wav_2_path, track_idx='wav_2')
    file_3 = ds.new_file(wav_3_path, track_idx='wav_3')
    file_4 = ds.new_file(wav_4_path, track_idx='wav_4')

    issuer_1 = Speaker('spk-1', gender=Gender.MALE)
    issuer_2 = Speaker('spk-2', gender=Gender.FEMALE)
    issuer_3 = Issuer('spk-3')

    ds.import_issuers([issuer_1, issuer_2, issuer_3])

    utt_1 = ds.new_utterance('utt-1', file_1.idx, issuer_idx=issuer_1.idx)
    utt_2 = ds.new_utterance('utt-2', file_2.idx, issuer_idx=issuer_1.idx)
    utt_3 = ds.new_utterance('utt-3', file_3.idx, issuer_idx=issuer_2.idx, start=0, end=1.5)
    utt_4 = ds.new_utterance('utt-4', file_3.idx, issuer_idx=issuer_2.idx, start=1.5, end=2.5)
    utt_5 = ds.new_utterance('utt-5', file_4.idx, issuer_idx=issuer_3.idx)

    utt_1.set_label_list(LabelList(audiomate.corpus.LL_WORD_TRANSCRIPT,
                                   labels=[Label('who am i')]))
    utt_2.set_label_list(LabelList(audiomate.corpus.LL_WORD_TRANSCRIPT,
                                   labels=[Label('who are you', meta={'a': 'hey', 'b': 2})]))
    utt_3.set_label_list(LabelList(audiomate.corpus.LL_WORD_TRANSCRIPT,
                                   labels=[Label('who is he')]))
    utt_4.set_label_list(LabelList(audiomate.corpus.LL_WORD_TRANSCRIPT,
                                   labels=[Label('who are they')]))
    utt_5.set_label_list(LabelList(audiomate.corpus.LL_WORD_TRANSCRIPT,
                                   labels=[Label('who is she')]))

    train_filter = subview.MatchingUtteranceIdxFilter(utterance_idxs={'utt-1', 'utt-2', 'utt-3'})
    sv_train = subview.Subview(ds, filter_criteria=[train_filter])

    dev_filter = subview.MatchingUtteranceIdxFilter(utterance_idxs={'utt-4', 'utt-5'})
    sv_dev = subview.Subview(ds, filter_criteria=[dev_filter])

    ds.import_subview('train', sv_train)
    ds.import_subview('dev', sv_dev)

    ds.new_feature_container('mfcc', '/some/dummy/path')
    ds.new_feature_container('mel', '/some/dummy/path_mel')

    return ds


def create_multi_label_corpus():
    ds = audiomate.Corpus()

    wav_1_path = sample_wav_file('wav_1.wav')
    wav_2_path = sample_wav_file('wav_2.wav')
    wav_3_path = sample_wav_file('wav_3.wav')
    wav_4_path = sample_wav_file('wav_4.wav')

    file_1 = ds.new_file(wav_1_path, track_idx='wav-1')
    file_2 = ds.new_file(wav_2_path, track_idx='wav_2')
    file_3 = ds.new_file(wav_3_path, track_idx='wav_3')
    file_4 = ds.new_file(wav_4_path, track_idx='wav_4')

    issuer_1 = ds.new_issuer('spk-1')
    issuer_2 = ds.new_issuer('spk-2')
    issuer_3 = ds.new_issuer('spk-3')

    utt_1 = ds.new_utterance('utt-1', file_1.idx, issuer_idx=issuer_1.idx)
    utt_2 = ds.new_utterance('utt-2', file_2.idx, issuer_idx=issuer_1.idx)
    utt_3 = ds.new_utterance('utt-3', file_3.idx, issuer_idx=issuer_2.idx, start=0, end=15)
    utt_4 = ds.new_utterance('utt-4', file_3.idx, issuer_idx=issuer_2.idx, start=15, end=25)
    utt_5 = ds.new_utterance('utt-5', file_3.idx, issuer_idx=issuer_2.idx, start=25, end=40)
    utt_6 = ds.new_utterance('utt-6', file_4.idx, issuer_idx=issuer_3.idx, start=0, end=15)
    utt_7 = ds.new_utterance('utt-7', file_4.idx, issuer_idx=issuer_3.idx, start=15, end=25)
    utt_8 = ds.new_utterance('utt-8', file_4.idx, issuer_idx=issuer_3.idx, start=25, end=40)

    utt_1.set_label_list(LabelList(labels=[
        Label('music', 0, 5),
        Label('speech', 5, 12),
        Label('music', 13, 15)
    ]))

    utt_2.set_label_list(LabelList(labels=[
        Label('music', 0, 5),
        Label('speech', 5, 12),
        Label('music', 13, 15)
    ]))

    utt_3.set_label_list(LabelList(labels=[
        Label('music', 0, 1),
        Label('speech', 2, 6)
    ]))

    utt_4.set_label_list(LabelList(labels=[
        Label('music', 0, 5),
        Label('speech', 5, 12),
        Label('music', 13, 15)
    ]))

    utt_5.set_label_list(LabelList(labels=[
        Label('speech', 0, 7)
    ]))

    utt_6.set_label_list(LabelList(labels=[
        Label('music', 0, 5),
        Label('speech', 5, 12),
        Label('music', 13, 15)
    ]))

    utt_7.set_label_list(LabelList(labels=[
        Label('music', 0, 5),
        Label('speech', 5, 11)
    ]))

    utt_8.set_label_list(LabelList(labels=[
        Label('music', 0, 10)
    ]))

    train_filter = subview.MatchingUtteranceIdxFilter(utterance_idxs={'utt-4', 'utt-5', 'utt-6'})
    sv_train = subview.Subview(ds, filter_criteria=[train_filter])

    dev_filter = subview.MatchingUtteranceIdxFilter(utterance_idxs={'utt-7', 'utt-8'})
    sv_dev = subview.Subview(ds, filter_criteria=[dev_filter])

    ds.import_subview('train', sv_train)
    ds.import_subview('dev', sv_dev)

    ds.new_feature_container('mfcc', '/some/dummy/path/secondmfcc')
    ds.new_feature_container('energy', '/some/dummy/path/energy')

    return ds


def create_single_label_corpus():
    ds = audiomate.Corpus()

    wav_1_path = sample_wav_file('wav_1.wav')
    wav_2_path = sample_wav_file('wav_2.wav')
    wav_3_path = sample_wav_file('wav_3.wav')
    wav_4_path = sample_wav_file('wav_4.wav')

    file_1 = ds.new_file(wav_1_path, track_idx='wav-1')
    file_2 = ds.new_file(wav_2_path, track_idx='wav_2')
    file_3 = ds.new_file(wav_3_path, track_idx='wav_3')
    file_4 = ds.new_file(wav_4_path, track_idx='wav_4')

    issuer_1 = ds.new_issuer('spk-1')
    issuer_2 = ds.new_issuer('spk-2')
    issuer_3 = ds.new_issuer('spk-3')

    utt_1 = ds.new_utterance('utt-1', file_1.idx, issuer_idx=issuer_1.idx)
    utt_2 = ds.new_utterance('utt-2', file_2.idx, issuer_idx=issuer_1.idx)
    utt_3 = ds.new_utterance('utt-3', file_3.idx, issuer_idx=issuer_2.idx, start=0, end=15)
    utt_4 = ds.new_utterance('utt-4', file_3.idx, issuer_idx=issuer_2.idx, start=15, end=25)
    utt_5 = ds.new_utterance('utt-5', file_3.idx, issuer_idx=issuer_2.idx, start=25, end=40)
    utt_6 = ds.new_utterance('utt-6', file_4.idx, issuer_idx=issuer_3.idx, start=0, end=15)
    utt_7 = ds.new_utterance('utt-7', file_4.idx, issuer_idx=issuer_3.idx, start=15, end=25)
    utt_8 = ds.new_utterance('utt-8', file_4.idx, issuer_idx=issuer_3.idx, start=25, end=40)

    utt_1.set_label_list(LabelList(labels=[
        Label('music')
    ]))

    utt_2.set_label_list(LabelList(labels=[
        Label('music')
    ]))

    utt_3.set_label_list(LabelList(labels=[
        Label('speech')
    ]))

    utt_4.set_label_list(LabelList(labels=[
        Label('music')
    ]))

    utt_5.set_label_list(LabelList(labels=[
        Label('speech')
    ]))

    utt_6.set_label_list(LabelList(labels=[
        Label('music')
    ]))

    utt_7.set_label_list(LabelList(labels=[
        Label('speech')
    ]))

    utt_8.set_label_list(LabelList(labels=[
        Label('music')
    ]))

    return ds
