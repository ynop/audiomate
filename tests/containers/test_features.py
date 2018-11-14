import pytest

from audiomate import containers
from tests import resources


@pytest.fixture()
def sample_feature_container():
    container_path = resources.get_resource_path(
        ['sample_files', 'feat_container']
    )
    sample_container = containers.FeatureContainer(container_path)
    sample_container.open()
    yield sample_container
    sample_container.close()


class TestFeatureContainer:

    def test_frame_size(self, sample_feature_container):
        assert sample_feature_container.frame_size == 400

    def test_hop_size(self, sample_feature_container):
        assert sample_feature_container.hop_size == 160

    def test_sampling_rate(self, sample_feature_container):
        assert sample_feature_container.sampling_rate == 16000

    def test_stats_per_key(self, sample_feature_container):
        utt_stats = sample_feature_container.stats_per_key()

        assert utt_stats['utt-1'].min == pytest.approx(0.0071605651933048797)
        assert utt_stats['utt-1'].max == pytest.approx(0.9967182746569494)
        assert utt_stats['utt-1'].mean == pytest.approx(0.51029100520776705)
        assert utt_stats['utt-1'].var == pytest.approx(0.079222738766221268)
        assert utt_stats['utt-1'].num == 100

        assert utt_stats['utt-2'].min == pytest.approx(0.01672865642756316)
        assert utt_stats['utt-2'].max == pytest.approx(0.99394433783429104)
        assert utt_stats['utt-2'].mean == pytest.approx(0.46471979908661543)
        assert utt_stats['utt-2'].var == pytest.approx(0.066697466410977804)
        assert utt_stats['utt-2'].num == 65

        assert utt_stats['utt-3'].min == pytest.approx(0.014999482706963607)
        assert utt_stats['utt-3'].max == pytest.approx(0.99834417857609881)
        assert utt_stats['utt-3'].mean == pytest.approx(0.51042690965262705)
        assert utt_stats['utt-3'].var == pytest.approx(0.071833200069641057)
        assert utt_stats['utt-3'].num == 220

    def test_stats_per_key_not_open(self, sample_feature_container):
        sample_feature_container.close()

        with pytest.raises(ValueError):
            sample_feature_container.stats_per_key()

    def test_stats(self, sample_feature_container):
        stats = sample_feature_container.stats()

        assert stats.min == pytest.approx(0.0071605651933048797)
        assert stats.max == pytest.approx(0.99834417857609881)
        assert stats.mean == pytest.approx(0.50267482489606408)
        assert stats.var == pytest.approx(0.07317811077366114)

    def test_stats_not_open(self, sample_feature_container):
        sample_feature_container.close()

        with pytest.raises(ValueError):
            sample_feature_container.stats()
