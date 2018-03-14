import os

import pytest

from pingu.corpus import io
from tests import resources


@pytest.fixture
def reader():
    return io.TudaReader()


@pytest.fixture
def data_path():
    return resources.sample_tuda_ds_path()


class TestTudaReader:

    def test_get_ids_from_folder(self, data_path):
        assert io.TudaReader.get_ids_from_folder(os.path.join(data_path, 'train'), 'train') == {
            '2014-03-17-09-45-16',
            '2014-03-27-13-39-34',
            '2014-03-27-13-39-53'
        }
        assert io.TudaReader.get_ids_from_folder(os.path.join(data_path, 'dev'), 'dev') == {
            '2015-01-27-11-31-32',
            '2015-01-27-11-31-41'
        }
        assert io.TudaReader.get_ids_from_folder(os.path.join(data_path, 'test'), 'test') == {
            '2015-01-27-12-32-58',
            '2015-01-27-12-34-46',
            '2015-01-27-12-34-36'
        }

    def test_get_ids_from_folder_ignore_bad_files(self, data_path):
        ids = io.TudaReader.get_ids_from_folder(os.path.join(data_path, 'train'), 'train')
        assert '2014-08-05-11-08-34-Parliament' not in ids
        assert '2014-03-24-13-39-24' not in ids

    def test_load_files(self, reader, data_path):
        ds = reader.load(data_path)

        dev = os.path.join(data_path, 'dev')
        test = os.path.join(data_path, 'test')
        train = os.path.join(data_path, 'train')

        assert ds.num_files == 8

        assert ds.files['2015-01-27-11-31-32'].idx == '2015-01-27-11-31-32'
        assert ds.files['2015-01-27-11-31-32'].path == os.path.join(dev, '2015-01-27-11-31-32-beamformedSignal.wav')

        assert ds.files['2015-01-27-11-31-41'].idx == '2015-01-27-11-31-41'
        assert ds.files['2015-01-27-11-31-41'].path == os.path.join(dev, '2015-01-27-11-31-41-beamformedSignal.wav')

        assert ds.files['2015-01-27-12-32-58'].idx == '2015-01-27-12-32-58'
        assert ds.files['2015-01-27-12-32-58'].path == os.path.join(test, '2015-01-27-12-32-58-beamformedSignal.wav')

        assert ds.files['2015-01-27-12-34-36'].idx == '2015-01-27-12-34-36'
        assert ds.files['2015-01-27-12-34-36'].path == os.path.join(test, '2015-01-27-12-34-36-beamformedSignal.wav')

        assert ds.files['2015-01-27-12-34-46'].idx == '2015-01-27-12-34-46'
        assert ds.files['2015-01-27-12-34-46'].path == os.path.join(test, '2015-01-27-12-34-46-beamformedSignal.wav')

        assert ds.files['2014-03-17-09-45-16'].idx == '2014-03-17-09-45-16'
        assert ds.files['2014-03-17-09-45-16'].path == os.path.join(train, '2014-03-17-09-45-16-beamformedSignal.wav')

        assert ds.files['2014-03-27-13-39-34'].idx == '2014-03-27-13-39-34'
        assert ds.files['2014-03-27-13-39-34'].path == os.path.join(train, '2014-03-27-13-39-34-beamformedSignal.wav')

        assert ds.files['2014-03-27-13-39-53'].idx == '2014-03-27-13-39-53'
        assert ds.files['2014-03-27-13-39-53'].path == os.path.join(train, '2014-03-27-13-39-53-beamformedSignal.wav')

    def test_load_speakers(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_issuers == 4

        assert ds.issuers['755d9b71-f36e-45a6-a437-edebcfaee08d'].idx == '755d9b71-f36e-45a6-a437-edebcfaee08d'
        assert ds.issuers['755d9b71-f36e-45a6-a437-edebcfaee08d'].info['gender'] == 'male'

        assert ds.issuers['9e6a00c9-80f0-479d-8b36-4139a9571217'].idx == '9e6a00c9-80f0-479d-8b36-4139a9571217'
        assert ds.issuers['9e6a00c9-80f0-479d-8b36-4139a9571217'].info['gender'] == 'male'

        assert ds.issuers['58b8b441-684f-4753-aa16-589f1e149fa0'].idx == '58b8b441-684f-4753-aa16-589f1e149fa0'
        assert ds.issuers['58b8b441-684f-4753-aa16-589f1e149fa0'].info['gender'] == 'male'

        assert ds.issuers['2a0995a7-47d8-453f-9864-5940efd3c71a'].idx == '2a0995a7-47d8-453f-9864-5940efd3c71a'
        assert ds.issuers['2a0995a7-47d8-453f-9864-5940efd3c71a'].info['gender'] == 'male'

    def test_load_utterances(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_utterances == 8

        assert ds.utterances['2015-01-27-11-31-32'].idx == '2015-01-27-11-31-32'
        assert ds.utterances['2015-01-27-11-31-32'].file.idx == '2015-01-27-11-31-32'

        assert ds.utterances['2015-01-27-11-31-41'].idx == '2015-01-27-11-31-41'
        assert ds.utterances['2015-01-27-11-31-41'].file.idx == '2015-01-27-11-31-41'

        assert ds.utterances['2015-01-27-12-32-58'].idx == '2015-01-27-12-32-58'
        assert ds.utterances['2015-01-27-12-32-58'].file.idx == '2015-01-27-12-32-58'

        assert ds.utterances['2015-01-27-12-34-36'].idx == '2015-01-27-12-34-36'
        assert ds.utterances['2015-01-27-12-34-36'].file.idx == '2015-01-27-12-34-36'

        assert ds.utterances['2015-01-27-12-34-46'].idx == '2015-01-27-12-34-46'
        assert ds.utterances['2015-01-27-12-34-46'].file.idx == '2015-01-27-12-34-46'

        assert ds.utterances['2014-03-17-09-45-16'].idx == '2014-03-17-09-45-16'
        assert ds.utterances['2014-03-17-09-45-16'].file.idx == '2014-03-17-09-45-16'

        assert ds.utterances['2014-03-27-13-39-34'].idx == '2014-03-27-13-39-34'
        assert ds.utterances['2014-03-27-13-39-34'].file.idx == '2014-03-27-13-39-34'

        assert ds.utterances['2014-03-27-13-39-53'].idx == '2014-03-27-13-39-53'
        assert ds.utterances['2014-03-27-13-39-53'].file.idx == '2014-03-27-13-39-53'

    def test_load_transcriptions(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.utterances['2015-01-27-11-31-32'].label_lists['transcription'][0].value == \
            'Manche Wagen haben die Besonderheit dass der Fahrer rechts sitzt um das Greifen der ' \
            'Mülltonnen besser steuern zu können'
        assert ds.utterances['2015-01-27-11-31-32'].label_lists['transcription_raw'][0].value == \
            'Manche Wagen haben die Besonderheit , dass der Fahrer rechts sitzt , um das Greifen der ' \
            'Mülltonnen besser steuern zu können .'

        assert ds.utterances['2015-01-27-11-31-41'].label_lists['transcription'][0].value == \
            'Überlegen Sie sich was Sie tun gute Frau'
        assert ds.utterances['2015-01-27-11-31-41'].label_lists['transcription_raw'][0].value == \
            'Überlegen Sie sich, was Sie tun, gute Frau.'

        assert ds.utterances['2015-01-27-12-32-58'].label_lists['transcription'][0].value == \
            'Das ist auch der Grund warum das Parlament der Einigung so viel Bedeutung beimisst die ' \
            'Seite an Seite mit der Kommission und den zukünftigen Ratsvorsitzen getroffen wurde um einen ' \
            'Weg zu finden mit dem die Finanzierung zukünftiger politischer Maßnahmen über den Haushalt der ' \
            'Gemeinschaft aus neuen Ressourcen die nicht mehr eine Belastung für die nationalen Haushalte ' \
            'darstellen und über die nationalen Haushalte selbst die in der Summe zwanzig Mal größer sind als ' \
            'der winzige europäische Gesamthaushalt gesichert wird'
        assert ds.utterances['2015-01-27-12-32-58'].label_lists['transcription_raw'][0].value == \
            'Das ist auch der Grund, warum das Parlament der Einigung so viel Bedeutung beimisst, die ' \
            'Seite an Seite mit der Kommission und den zukünftigen Ratsvorsitzen getroffen wurde, um einen ' \
            'Weg zu finden, mit dem die Finanzierung zukünftiger politischer Maßnahmen über den Haushalt der ' \
            'Gemeinschaft aus neuen Ressourcen, die nicht mehr eine Belastung für die nationalen Haushalte ' \
            'darstellen, und über die nationalen Haushalte selbst, die in der Summe 20 Mal größer sind als der ' \
            'winzige europäische Gesamthaushalt, gesichert wird.'

        assert ds.utterances['2015-01-27-12-34-36'].label_lists['transcription'][0].value == \
            'In seiner eigenen schriftstellerischen Existenz bemühte er sich ebenfalls stets um Unabhängigkeit'
        assert ds.utterances['2015-01-27-12-34-36'].label_lists['transcription_raw'][0].value == \
            'In seiner eigenen schriftstellerischen Existenz bemühte er sich ebenfalls stets um Unabhängigkeit .'

        assert ds.utterances['2015-01-27-12-34-46'].label_lists['transcription'][0].value == \
            'Eigentlich hat die Assistenzzeit meine Erwartungen übertroffen da ich wirklich auf sehr nette und ' \
            'offene Kollegen getroffen bin mit denen die Arbeit sehr viel Freude bereitet hat'
        assert ds.utterances['2015-01-27-12-34-46'].label_lists['transcription_raw'][0].value == \
            'Eigentlich hat die Assistenzzeit meine Erwartungen übertroffen, da ich wirklich auf sehr nette und ' \
            'offene Kollegen getroffen bin, mit denen die Arbeit sehr viel Freude bereitet hat.'

        assert ds.utterances['2014-03-17-09-45-16'].label_lists['transcription'][0].value == \
            'Hannibal nahm den Landweg durch das südliche Gallien überquerte die Alpen und fiel mit einem Heer in ' \
            'Italien ein wobei er mehrere römische Armeen nacheinander vernichtete'
        assert ds.utterances['2014-03-17-09-45-16'].label_lists['transcription_raw'][0].value == \
            'Hannibal nahm den Landweg durch das südliche Gallien, überquerte die Alpen und fiel mit einem Heer in ' \
            'Italien ein, wobei er mehrere römische Armeen nacheinander vernichtete.'

        assert ds.utterances['2014-03-27-13-39-34'].label_lists['transcription'][0].value == \
            'Abschließend möchte ich die Kommission ersuchen diese fünf Punkte bei der Formulierung der ' \
            'Schlußfolgerungen der vier Pfeiler zu berücksichtigen denn nach meiner Auffassung muß die Bindung der ' \
            'Bevölkerung an den ländlichen Raum eines der vorrangigen Ziele der Europäischen Union bilden'
        assert ds.utterances['2014-03-27-13-39-34'].label_lists['transcription_raw'][0].value == \
            'Abschließend möchte ich die Kommission ersuchen, diese fünf Punkte bei der Formulierung der ' \
            'Schlußfolgerungen der vier Pfeiler zu berücksichtigen, denn nach meiner Auffassung muß die Bindung der ' \
            'Bevölkerung an den ländlichen Raum eines der vorrangigen Ziele der Europäischen Union bilden.'

        assert ds.utterances['2014-03-27-13-39-53'].label_lists['transcription'][0].value == \
            'Die diesbezüglichen Vorgaben der Kommission stellen aber entgegen ihrem Anspruch Orientierung zu ' \
            'geben vielmehr einen Angebotskatalog möglicher Maßnahmen im Rahmen der Politikfelder dar'
        assert ds.utterances['2014-03-27-13-39-53'].label_lists['transcription_raw'][0].value == \
            'Die diesbezüglichen Vorgaben der Kommission stellen aber entgegen ihrem Anspruch, Orientierung zu ' \
            'geben, vielmehr einen Angebotskatalog möglicher Maßnahmen im Rahmen der Politikfelder dar.'

    def test_load_subviews(self, reader, data_path):
        ds = reader.load(data_path)

        assert ds.num_subviews == 3

        assert ds.subviews['train'].num_utterances == 3
        assert ds.subviews['dev'].num_utterances == 2
        assert ds.subviews['test'].num_utterances == 3
