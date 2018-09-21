from audiomate.corpus import assets
from audiomate.corpus.utils import label_cleaning


def test_merge_consecutive_labels_with_same_values():
    ll = assets.LabelList(labels=[
        assets.Label('a', 0, 0.993),
        assets.Label('a', 1.001, 2.8),
        assets.Label('b', 2.8, 3.94)
    ])

    label_cleaning.merge_consecutive_labels_with_same_values(ll, threshold=0.01)

    assert len(ll) == 2

    assert ll[0] == assets.Label('a', 0, 2.8)
    assert ll[1] == assets.Label('b', 2.8, 3.94)


def test_merge_consecutive_labels_with_same_values_3_in_a_row():
    ll = assets.LabelList(labels=[
        assets.Label('a', 0, 0.993),
        assets.Label('a', 1.001, 2.8),
        assets.Label('a', 2.8, 3.94),
        assets.Label('b', 3.94, 4.5)
    ])

    label_cleaning.merge_consecutive_labels_with_same_values(ll, threshold=0.01)

    assert len(ll) == 2

    print(ll.labels)

    assert ll[0] == assets.Label('a', 0, 3.94)
    assert ll[1] == assets.Label('b', 3.94, 4.5)
