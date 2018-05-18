import pytest

from audiomate.corpus import subset
from tests import resources


@pytest.fixture()
def corpus():
    return resources.create_single_label_corpus()


class TestSubsetGenerator:

    def test_random_subset(self, corpus):
        g = subset.SubsetGenerator(corpus, random_seed=20)
        sv = g.random_subset(0.5)

        assert sv.num_utterances == 4

    def test_random_subset_balanced(self, corpus):
        g = subset.SubsetGenerator(corpus, random_seed=20)
        sv = g.random_subset(0.5, balance_labels=True)

        label_count = sv.label_count()

        assert sv.num_utterances == 4
        assert label_count['music'] / label_count['speech'] == pytest.approx(1.0, abs=0.5)

    def test_random_subset_by_duration(self, corpus):
        g = subset.SubsetGenerator(corpus, random_seed=20)
        sv = g.random_subset_by_duration(0.5)

        assert sv.total_duration == pytest.approx(corpus.total_duration * 0.5, abs=5)

    def test_random_subset_by_duration_balanced(self, corpus):
        g = subset.SubsetGenerator(corpus, random_seed=20)
        sv = g.random_subset_by_duration(0.5, balance_labels=True)

        label_durations = sv.label_durations()

        assert sv.total_duration == pytest.approx(corpus.total_duration * 0.5, abs=15)
        assert label_durations['music'] / label_durations['speech'] == pytest.approx(1.0, abs=0.5)

    def test_random_subsets(self, corpus):
        g = subset.SubsetGenerator(corpus, random_seed=20)
        svs = g.random_subsets([0.6, 0.4, 0.2])

        assert svs.keys() == {0.6, 0.4, 0.2}

        assert svs[0.6].num_utterances == 5
        assert svs[0.4].num_utterances == 3
        assert svs[0.2].num_utterances == 2

        assert set(svs[0.4].utterances.keys()).issubset(set(svs[0.6].utterances.keys()))
        assert set(svs[0.2].utterances.keys()).issubset(set(svs[0.4].utterances.keys()))
