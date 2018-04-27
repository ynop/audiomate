def length_of_overlap(first_start, first_end, second_start, second_end):
    """
    Find the length of the overlapping part of two segments.

    Args:
        first_start (float): Start of the first segment.
        first_end (float): End of the first segment.
        second_start (float): Start of the second segment.
        second_end (float): End of the second segment.

    Return:
        float: The amount of overlap or 0 if they don't overlap at all.
    """
    if first_end <= second_start or first_start >= second_end:
        return 0.0

    if first_start < second_start:
        if first_end < second_end:
            return abs(first_end - second_start)
        else:
            return abs(second_end - second_start)

    if first_start > second_start:
        if first_end > second_end:
            return abs(second_end - first_start)
        else:
            return abs(first_end - first_start)
