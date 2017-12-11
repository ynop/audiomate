from pingu.corpus import assets
from pingu.utils import textfile


class UnmappedLabelsException(Exception):
    def __init__(self, message):
        super(Exception, self).__init__(message)
        self.message = message


def relabel(label_list, projections):
    """
    Relabel an entire :py:class:`pingu.corpus.assets.LabelList` using user-defined
    projections. Labels can be renamed, removed or overlapping labels can be flattened to a single
    label per segment.

    This method raises a :py:class:`pingu.corpus.assets.UnmappedLabelsException` if a projection
    for one or more combinations of labels is not defined.

    Args:
        label_list (pingu.corpus.assets.LabelList): The label list to relabel
        projections (dict): A dictionary that maps tuples of label combinations to string
                            labels.
    Returns:
        pingu.corpus.assets.LabelList: New label list with remapped labels

    Example:
        >>> projections = {
        ...     ('a',): 'a',
        ...     ('b',): 'b',
        ...     ('c',): 'c',
        ...     ('a', 'b',): 'a_b',
        ...     ('a', 'b', 'c',): 'a_b_c',
        ...     ('b', 'c',): 'b_c',
        ... }
        >>> label_list = assets.LabelList(labels=[
        ...     assets.Label('a', 3.2, 4.5),
        ...     assets.Label('b', 4.0, 4.9),
        ...     assets.Label('c', 4.2, 5.1)
        ... ])
        >>> ll = relabel(label_list, projections)
        >>> [l.value for l in ll]
        ['a', 'a_b', 'a_b_c', 'b_c', 'c']
    """
    unmapped_combinations = find_missing_projections(label_list, projections)
    if len(unmapped_combinations) > 0:
        raise UnmappedLabelsException('Unmapped combinations: {}'.format(unmapped_combinations))

    new_labels = []
    for labeled_segment in label_list.ranges():
        combination = tuple(sorted([label.value for label in labeled_segment[2]]))
        label_mapping = projections[combination]

        if label_mapping == '':
            continue

        new_labels.append(assets.Label(label_mapping, labeled_segment[0], labeled_segment[1]))

    return assets.LabelList(idx=label_list.idx, labels=new_labels)


def find_missing_projections(label_list, projections):
    """
    Finds all combinations of labels in `label_list` that are not covered by an entry in the
    `projections`. Returns a list containing tuples of uncovered label combinations or en empty
    list if there are none. All uncovered label combinations are naturally sorted.

    Args:
        label_list (pingu.corpus.assets.LabelList): The label list to relabel
        projections (dict): A dictionary that maps tuples of label combinations to string
                            labels.

    Returns:
        dict: Dictionary where the keys are naturally sorted tuples of unmapped label combinations
              that should be relabeled to the key's value

    Example:
        >>> ll = assets.LabelList(labels=[
        ...     assets.Label('b', 3.2, 4.5),
        ...     assets.Label('a', 4.0, 4.9),
        ...     assets.Label('c', 4.2, 5.1)
        ... ])
        >>> find_missing_projections(ll, {('b',): 'new_label'})
        [('a', 'b'), ('a', 'b', 'c'), ('a', 'c'), ('c',)]
    """
    unmapped_combinations = []
    for labeled_segment in label_list.ranges():
        combination = tuple(sorted([label.value for label in labeled_segment[2]]))

        if combination not in projections:
            unmapped_combinations.append(combination)

    return unmapped_combinations


def load_projections(projections_file):
    """
    Loads projections defined in the given `projections_file`.

    The `projections_file` is expected to be in the following format::

        old_label_1 | new_label_1
        old_label_1 old_label_2 | new_label_2
        old_label_3 |

    You can define one projection per line. Each projection starts with a list of one or multiple
    old labels (separated by a single whitespace) that are separated from the new label by a pipe
    (`|`). In the code above, the segment labeled with `old_label_1` will be labeled with
    `new_label_1` after applying the projection. Segments that are labeled with `old_label_1`
    **and** `old_label_2` concurrently are relabeled to `new_label_2`. All segments labeled with
    `old_label_3` are dropped. Combinations of multiple labels are automatically sorted in natural
    order.

    Args:
        projections_file (str): Path to the file with projections

    Returns:
        dict: Dictionary where the keys are tuples of labels to project to the key's value

    Example:
        >>> load_projections('/path/to/projections.txt')
        {('b',): 'foo', ('a', 'b'): 'a_b', ('a',): 'bar'}
    """
    projections = {}
    for parts in textfile.read_separated_lines_generator(projections_file, '|'):
        combination = tuple(sorted([label.strip() for label in parts[0].split(' ')]))
        new_label = parts[1].strip()

        projections[combination] = new_label

    return projections
