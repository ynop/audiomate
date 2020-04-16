import os
import re
import xml.etree.ElementTree as ET

from . import base

import audiomate
from audiomate.utils import jsonfile
from audiomate import issuers
from audiomate import annotations
from audiomate import tracks
from . import downloader


URLS = {
    'de': 'https://www2.informatik.uni-hamburg.de/nats/pub/SWC/SWC_German.tar',
    'en': 'https://www2.informatik.uni-hamburg.de/nats/pub/SWC/SWC_English.tar',
    'nl': 'https://www2.informatik.uni-hamburg.de/nats/pub/SWC/SWC_Dutch.tar'
}

READER_NAME_PATTERN = re.compile(r'user_name\s+=\s+(.*?)\n')
READER_GENDER_PATTERN = re.compile(r'(gender|geschlecht)\s+=\s+(.*?)\n')

MIN_SEGMENT_DURATION = 1.0


class SWCDownloader(downloader.ArchiveDownloader):
    """
    Downloader for the MUSAN Corpus.

    Args:
        lang (str): The language to download.
    """

    def __init__(self, lang='de'):
        url = URLS[lang]

        super(SWCDownloader, self).__init__(
            url,
            move_files_up=True
        )

    @classmethod
    def type(cls):
        return 'swc'


class SWCReader(base.CorpusReader):
    """ Reader for the Spoken Wikipedia Corpus. """

    @classmethod
    def type(cls):
        return 'swc'

    def _check_for_missing_files(self, path):
        return []

    def _load(self, path):
        corpus = audiomate.Corpus()

        article_paths = sorted(self.get_articles(path))
        reader_map = {}
        file_map = {}

        for article_path in article_paths:
            audio_files = self.get_audio_file_info(article_path)
            reader_name, reader_gender = self.get_reader_info(article_path)
            segments = self.get_segments(article_path)

            if reader_name not in reader_map.keys():
                speaker = issuers.Speaker(
                    '{:0>8}'.format(len(reader_map)),
                    gender=reader_gender
                )
                reader_map[reader_name] = speaker
                corpus.import_issuers(speaker)
            else:
                speaker = reader_map[reader_name]

            for start, end, text in segments:
                file_path = self.find_audio_file_for_segment(start, end, audio_files)

                if file_path is not None:
                    if file_path not in file_map.keys():
                        track = tracks.FileTrack(
                            '{:0>10}'.format(len(file_map)),
                            file_path
                        )
                        file_map[file_path] = track
                        corpus.import_tracks(track)
                    else:
                        track = file_map[file_path]

                    track_offset = audio_files[file_path]
                    utt_start = start - track_offset
                    utt_end = end - track_offset

                    utt_idx = '{}_{}_{}_{}'.format(
                        speaker.idx,
                        track.idx,
                        int(start * 1000),
                        int(end * 1000)
                    )

                    if utt_idx not in self.invalid_utterance_ids:
                        utt = corpus.new_utterance(
                            utt_idx,
                            track.idx,
                            issuer_idx=speaker.idx,
                            start=utt_start,
                            end=utt_end
                        )

                        ll = annotations.LabelList.create_single(
                            text,
                            audiomate.corpus.LL_WORD_TRANSCRIPT
                        )

                        utt.set_label_list(ll)

        return audiomate.Corpus.from_corpus(corpus)

    def get_articles(self, path):
        """ Return the list of article-paths """
        article_paths = []

        for dirname in os.listdir(path):
            dirpath = os.path.join(path, dirname)

            audio_meta_file = os.path.join(dirpath, 'audiometa.txt')
            info_file = os.path.join(dirpath, 'info.json')
            align_file = os.path.join(dirpath, 'aligned.swc')

            if os.path.isfile(audio_meta_file) and \
                    os.path.isfile(info_file) and \
                    os.path.isfile(align_file):
                article_paths.append(dirpath)

        return article_paths

    def get_audio_file_info(self, article_path):
        """
        Return info about the audio files.
        List of tuples with (path, offset).
        """

        info_path = os.path.join(article_path, 'info.json')
        info = jsonfile.read_json_file(info_path)
        audio_files = {}

        if len(info['audio_files']) == 1:
            path = os.path.join(article_path, 'audio.ogg')

            if 'offset' not in info['audio_files'][0].keys():
                return {}

            offset = info['audio_files'][0]['offset']
            audio_files[path] = offset

        else:
            for i, af in enumerate(info['audio_files']):
                path = os.path.join(article_path, 'audio{}.ogg'.format(i+1))
                offset = af['offset']
                audio_files[path] = offset

        return audio_files

    def get_reader_info(self, article_path):
        """ Return info about the reader of the article. """

        meta_path = os.path.join(article_path, 'audiometa.txt')
        with open(meta_path, 'r', encoding='utf-8') as f:
            content = f.read()

        name_match = READER_NAME_PATTERN.search(content)

        if name_match is not None:
            name = name_match.group(1).strip()
        else:
            name = os.path.basename(article_path)

        gender_match = READER_GENDER_PATTERN.search(content)
        gender_str = ''

        if gender_match is not None:
            gender_str = gender_match.group(2).strip()

        if gender_str.lower() in ['male', 'männlich', 'mänlich', 'mann', 'm', 'malee', 'männ', 'maennlich']:
            gender = issuers.Gender.MALE
        elif gender_str.lower() in ['female', 'weiblich']:
            gender = issuers.Gender.FEMALE
        else:
            gender = issuers.Gender.UNKNOWN

        return name, gender

    def find_audio_file_for_segment(self, start, end, audio_files):
        """
        Find the correct audio file for the segment.
        Return index of matching audio file.
        """

        items = sorted(audio_files.items(), key=lambda x: x[1])

        for i in range(len(items) - 1):
            if end <= items[i+1][1]:
                # Segment belongs to audiofile i
                if start >= items[i][1]:
                    return items[i][0]

                # Segment crosses audiofile boundaries, ignore
                else:
                    return None

        if start >= items[-1][1]:
            return items[-1][0]
        else:
            return None

    def get_segments(self, article_path):
        """
        Parse segments from alignment file.
        Return list with tuples (start, end, text).
        """
        aligned_path = os.path.join(article_path, 'aligned.swc')
        tree = ET.parse(aligned_path)
        root = tree.getroot()

        segments = []

        for p in root:
            if p.tag == 'd':
                q = self.parse_element(p)
                segments.extend(q)

        # Filter segments on minimal duration
        segments = [s for s in segments if s[1] - s[0] > MIN_SEGMENT_DURATION]

        return segments

    def parse_element(self, parent):
        segments = []

        for c in parent:
            if c.tag in ['extra', 'section', 'sectioncontent', 'p']:
                q = self.parse_element(c)
                segments.extend(q)
            elif c.tag in ['sectiontitle', 's']:
                q = self.parse_sentence(c)
                segments.extend(q)

        return segments

    def parse_sentence(self, element):
        tokens = []

        for index, token in enumerate(element):
            if token.tag == 't':
                tokens.append((index, token))

        return self.parse_tokens(tokens)

    def parse_tokens(self, tokens):
        """ Get segments from sequence of tokens. """
        valid_tokens = self.get_valid_tokens(tokens)

        if len(valid_tokens) <= 0:
            return []

        segmented_tokens = []
        current_tokens = [valid_tokens[0]]

        for t in valid_tokens[1:]:
            # Tokens have consecutive indices
            # so same segment
            if current_tokens[-1][0] + 1 == t[0]:
                current_tokens.append(t)

            # start new segment
            else:
                segmented_tokens.append(current_tokens)
                current_tokens = [t]

        segmented_tokens.append(current_tokens)

        segments = []

        for token_group in segmented_tokens:
            text = ' '.join(t[3].strip() for t in token_group)
            start = token_group[0][1]
            end = token_group[-1][2]
            segments.append((start, end, text))

        return segments

    def get_valid_tokens(self, tokens):
        """
        Return only valid tokens
        (have a normalization with start and end time).
        """
        valid_tokens = []

        for index, token in tokens:
            normalizations = self.parse_normalizations(token)

            if len(normalizations) > 0:
                normalized_text = ' '.join(n[0] for n in normalizations)
                start = normalizations[0][1]
                end = normalizations[-1][2]

                if start != -1 and end != -1:
                    valid_tokens.append((
                        index,
                        start,
                        end,
                        normalized_text
                    ))

        return valid_tokens

    def parse_normalizations(self, token):
        """
        Parse normalizations of a token.
        Return list of tuples (text, start, end).
        """
        norms = []

        for c in token:
            if c.tag == 'n':
                pronunciation = c.attrib['pronunciation']
                start = -1
                end = -1

                if 'start' in c.attrib.keys():
                    start = int(c.attrib['start']) / 1000.0

                if 'end' in c.attrib.keys():
                    end = int(c.attrib['end']) / 1000.0

                norms.append((
                    pronunciation,
                    start,
                    end
                ))

        return norms
