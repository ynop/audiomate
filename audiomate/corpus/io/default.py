import collections
import glob
import os
import re
import json

import audiomate
from audiomate import containers
from audiomate import tracks
from audiomate import annotations
from audiomate import issuers
from audiomate.corpus.subset import subview
from audiomate.utils import textfile
from audiomate.utils import jsonfile
from . import base

FILES_FILE_NAME = 'files.txt'
ISSUER_FILE_NAME = 'issuers.json'
UTTERANCE_FILE_NAME = 'utterances.txt'
UTT_ISSUER_FILE_NAME = 'utt_issuers.txt'
LABEL_FILE_PREFIX = 'labels'
FEAT_CONTAINER_FILE_NAME = 'features.txt'
AUDIO_CONTAINER_FILE_NAME = 'audio.txt'
SUBVIEW_FILE_PREFIX = 'subview'

LABEL_META_REGEX = r'(.*) \[(\{.*\})\]'
META_PATTERN = re.compile(LABEL_META_REGEX)


class DefaultReader(base.CorpusReader):
    """
    Reads corpora in the Default format.
    """

    @classmethod
    def type(cls):
        return 'default'

    def _check_for_missing_files(self, path):
        necessary_files = [FILES_FILE_NAME, UTTERANCE_FILE_NAME]
        missing_files = []

        for file_name in necessary_files:
            file_path = os.path.join(path, file_name)

            if not os.path.isfile(file_path):
                missing_files.append(file_name)

        return missing_files

    def _load(self, path):
        file_path = os.path.join(path, FILES_FILE_NAME)
        audio_path = os.path.join(path, AUDIO_CONTAINER_FILE_NAME)
        issuer_path = os.path.join(path, ISSUER_FILE_NAME)
        utt_issuer_path = os.path.join(path, UTT_ISSUER_FILE_NAME)
        utterance_path = os.path.join(path, UTTERANCE_FILE_NAME)
        feat_path = os.path.join(path, FEAT_CONTAINER_FILE_NAME)

        corpus = audiomate.Corpus(path=path)

        DefaultReader.read_files(file_path, corpus)
        DefaultReader.read_tracks_from_audio_containers(audio_path, corpus)
        DefaultReader.read_issuers(issuer_path, corpus)
        utt_id_to_issuer = DefaultReader.read_utt_to_issuer_mapping(utt_issuer_path, corpus)
        DefaultReader.read_utterances(utterance_path, corpus, utt_id_to_issuer)
        DefaultReader.read_labels(path, corpus)
        DefaultReader.read_feature_containers(feat_path, corpus)
        DefaultReader.read_subviews(path, corpus)

        return corpus

    @staticmethod
    def read_files(file_path, corpus):
        path = os.path.dirname(file_path)
        for file_idx, file_path in textfile.read_key_value_lines(file_path, separator=' ').items():
            corpus.new_file(os.path.join(path, file_path), track_idx=file_idx, copy_file=False)

    @staticmethod
    def read_issuers(file_path, corpus):
        if not os.path.isfile(file_path):
            return

        data = jsonfile.read_json_file(file_path)

        for issuer_idx, issuer_data in data.items():
            issuer_type = issuer_data.get('type', None)
            issuer_info = issuer_data.get('info', {})

            if issuer_type == 'speaker':
                gender = issuers.Gender(issuer_data.get('gender', 'unknown').lower())
                age_group = issuers.AgeGroup(issuer_data.get('age_group', 'unknown').lower())
                native_language = issuer_data.get('native_language', None)

                issuer = issuers.Speaker(issuer_idx,
                                         gender=gender,
                                         age_group=age_group,
                                         native_language=native_language,
                                         info=issuer_info)
            elif issuer_type == 'artist':
                name = issuer_data.get('name', None)

                issuer = issuers.Artist(issuer_idx,
                                        name=name,
                                        info=issuer_info)
            else:
                issuer = issuers.Issuer(issuer_idx, info=issuer_info)

            corpus.import_issuers(issuer)

    @staticmethod
    def read_utt_to_issuer_mapping(utt_issuer_path, corpus):
        utt_issuers = {}

        if os.path.isfile(utt_issuer_path):
            for utt_id, issuer_idx in textfile.read_key_value_lines(utt_issuer_path, separator=' ').items():
                if issuer_idx in corpus.issuers.keys():
                    utt_issuers[utt_id] = corpus.issuers[issuer_idx]
                else:
                    utt_issuers[utt_id] = corpus.new_issuer(issuer_idx=issuer_idx)

        return utt_issuers

    @staticmethod
    def read_utterances(utterance_path, corpus, utt_idx_to_issuer):
        utterances = textfile.read_separated_lines_with_first_key(utterance_path, separator=' ', max_columns=4)

        for utterance_idx, utt_info in utterances.items():
            issuer_idx = None
            start = 0
            end = float('inf')

            if len(utt_info) > 1:
                start = float(utt_info[1])

            if len(utt_info) > 2:
                end = float(utt_info[2])

                if end == -1:
                    end = float('inf')

            if utterance_idx in utt_idx_to_issuer.keys():
                issuer_idx = utt_idx_to_issuer[utterance_idx].idx

            corpus.new_utterance(utterance_idx, utt_info[0], issuer_idx=issuer_idx, start=start, end=end)

    @staticmethod
    def read_labels(path, corpus):

        for label_file in glob.glob(os.path.join(path, '{}_*.txt'.format(LABEL_FILE_PREFIX))):
            file_name = os.path.basename(label_file)
            key = file_name[len('{}_'.format(LABEL_FILE_PREFIX)):len(file_name) - len('.txt')]

            utterance_labels = collections.defaultdict(list)

            labels = textfile.read_separated_lines_generator(label_file, separator=' ', max_columns=4)

            for record in labels:
                label = record[3]
                start = float(record[1])
                end = float(record[2])
                meta = None
                meta_match = META_PATTERN.match(label)

                if end == -1:
                    end = float('inf')

                if meta_match is not None:
                    meta_json = meta_match.group(2)
                    meta = json.loads(meta_json)
                    label = meta_match.group(1)

                utterance_labels[record[0]].append(annotations.Label(label, start, end, meta=meta))

            for utterance_idx, labels in utterance_labels.items():
                ll = annotations.LabelList(idx=key, labels=labels)
                corpus.utterances[utterance_idx].set_label_list(ll)

    @staticmethod
    def read_feature_containers(feat_path, corpus):
        if os.path.isfile(feat_path):
            base_path = os.path.dirname(feat_path)
            containers = textfile.read_key_value_lines(feat_path, separator=' ')
            for container_name, container_path in containers.items():
                corpus.new_feature_container(container_name, path=os.path.join(base_path, container_path))

    @staticmethod
    def read_tracks_from_audio_containers(audio_path, corpus):
        if os.path.isfile(audio_path):
            base_path = os.path.dirname(audio_path)
            audio_tracks = textfile.read_separated_lines(audio_path,
                                                         separator=' ',
                                                         max_columns=3)

            audio_containers = {}

            for entry in audio_tracks:
                track_idx = entry[0]
                container_path = entry[1]
                key = entry[2]

                if container_path in audio_containers.keys():
                    container = audio_containers[key]
                else:
                    abs_path = os.path.abspath(os.path.join(base_path, container_path))
                    container = containers.AudioContainer(abs_path)

                track = tracks.ContainerTrack(track_idx, container, key)
                corpus.import_tracks(track)

    @staticmethod
    def read_subviews(path, corpus):
        for sv_file in glob.glob(os.path.join(path, '{}_*.txt'.format(SUBVIEW_FILE_PREFIX))):
            file_name = os.path.basename(sv_file)
            key = file_name[len('{}_'.format(SUBVIEW_FILE_PREFIX)):len(file_name) - len('.txt')]

            with open(sv_file, 'r') as f:
                content = f.read().strip()

            sv = subview.Subview.parse(content)
            corpus.import_subview(key, sv)


class DefaultWriter(base.CorpusWriter):
    """
    Writes corpora in the Default format.
    """

    @classmethod
    def type(cls):
        return 'default'

    def _save(self, corpus, path):
        file_path = os.path.join(path, FILES_FILE_NAME)
        audio_path = os.path.join(path, AUDIO_CONTAINER_FILE_NAME)
        issuer_path = os.path.join(path, ISSUER_FILE_NAME)
        utterance_path = os.path.join(path, UTTERANCE_FILE_NAME)
        utt_issuer_path = os.path.join(path, UTT_ISSUER_FILE_NAME)
        container_path = os.path.join(path, FEAT_CONTAINER_FILE_NAME)

        DefaultWriter.write_file_tracks(file_path, corpus, path)
        DefaultWriter.write_container_tracks(audio_path, corpus, path)
        DefaultWriter.write_issuers(issuer_path, corpus)
        DefaultWriter.write_utterances(utterance_path, corpus)
        DefaultWriter.write_utt_to_issuer_mapping(utt_issuer_path, corpus)
        DefaultWriter.write_labels(path, corpus)
        DefaultWriter.write_feature_containers(container_path, corpus)
        DefaultWriter.write_subviews(path, corpus)

    @staticmethod
    def write_file_tracks(file_path, corpus, path):
        file_records = []

        for file in corpus.tracks.values():
            if isinstance(file, tracks.FileTrack):
                file_records.append([
                    file.idx,
                    os.path.relpath(file.path, path)
                ])

        textfile.write_separated_lines(file_path, file_records, separator=' ', sort_by_column=0)

    @staticmethod
    def write_container_tracks(audio_path, corpus, path):
        container_records = set({})

        for track in corpus.tracks.values():
            if isinstance(track, tracks.ContainerTrack):
                rel_path = os.path.relpath(track.container.path, path)
                container_records.add((
                    track.idx,
                    rel_path,
                    track.key
                ))

        textfile.write_separated_lines(
            audio_path,
            container_records,
            separator=' ',
            sort_by_column=0
        )

    @staticmethod
    def write_issuers(file_path, corpus):
        data = {}

        for issuer in corpus.issuers.values():
            issuer_data = {}

            if issuer.info is not None and len(issuer.info) > 0:
                issuer_data['info'] = issuer.info

            if type(issuer) == issuers.Speaker:
                issuer_data['type'] = 'speaker'

                if issuer.gender != issuers.Gender.UNKNOWN:
                    issuer_data['gender'] = issuer.gender.value

                if issuer.age_group != issuers.AgeGroup.UNKNOWN:
                    issuer_data['age_group'] = issuer.age_group.value

                if issuer.native_language not in ['', None]:
                    issuer_data['native_language'] = issuer.native_language

            elif type(issuer) == issuers.Artist:
                if issuer.name not in ['', None]:
                    issuer_data['name'] = issuer.name

            data[issuer.idx] = issuer_data

        jsonfile.write_json_to_file(file_path, data)

    @staticmethod
    def write_utterances(utterance_path, corpus):
        utterance_records = {}

        for utterance in corpus.utterances.values():
            track_idx = utterance.track.idx
            start = utterance.start
            end = utterance.end

            if end == float('inf'):
                end = -1

            utterance_records[utterance.idx] = [track_idx, start, end]

        textfile.write_separated_lines(utterance_path, utterance_records, separator=' ', sort_by_column=0)

    @staticmethod
    def write_utt_to_issuer_mapping(utt_issuer_path, corpus):
        utt_issuer_records = {}

        for utterance in corpus.utterances.values():
            if utterance.issuer is not None:
                utt_issuer_records[utterance.idx] = utterance.issuer.idx

        textfile.write_separated_lines(utt_issuer_path, utt_issuer_records, separator=' ', sort_by_column=0)

    @staticmethod
    def write_labels(path, corpus):
        records = collections.defaultdict(list)

        for utterance in corpus.utterances.values():
            for label_list_idx, label_list in utterance.label_lists.items():
                utt_records = []
                for l in label_list:
                    start = l.start
                    end = l.end

                    if end == float('inf'):
                        end = -1

                    if len(l.meta) > 0:
                        value = '{} [{}]'.format(l.value, json.dumps(l.meta, sort_keys=True))
                        utt_records.append((utterance.idx, start, end, value))
                    else:
                        utt_records.append((utterance.idx, start, end, l.value))

                records[label_list_idx].extend(utt_records)

        for label_list_idx, label_list_records in records.items():
            file_path = os.path.join(path, '{}_{}.txt'.format(LABEL_FILE_PREFIX, label_list_idx))
            textfile.write_separated_lines(file_path, label_list_records, separator=' ')

    @staticmethod
    def write_feature_containers(container_path, corpus):
        feat_records = [(idx, container.path) for idx, container in corpus.feature_containers.items()]
        textfile.write_separated_lines(container_path, feat_records, separator=' ')

    @staticmethod
    def write_subviews(path, corpus):
        for name, sv in corpus.subviews.items():
            sv_path = os.path.join(path, '{}_{}.txt'.format(SUBVIEW_FILE_PREFIX, name))
            with open(sv_path, 'w') as f:
                f.write(sv.serialize())
