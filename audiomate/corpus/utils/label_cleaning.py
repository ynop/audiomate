"""
Module containing functions for cleaning label-lists.
"""


def merge_consecutive_labels_with_same_values(label_list, threshold=0.01):
    """
    If there are consecutive (end equals start) labels, those two labels are merged into one.

    Args:
         label_list (LabelList): The label-list to clean.
         threshold (float): Labels are considered consecutive, if the duration between the end of the first and
                            the start of the second label is smaller than this threshold (default 0.01 seconds).
    """

    sorted_labels = sorted(label_list.labels, key=lambda x: x.start)

    index = 0

    while index < len(sorted_labels) - 1:
        current_label = sorted_labels[index]
        next_label = sorted_labels[index + 1]

        if current_label.value == next_label.value and (next_label.start - current_label.end) < threshold:
            label_list.labels.remove(next_label)
            sorted_labels.remove(next_label)
            current_label.end = next_label.end
        else:
            index += 1
