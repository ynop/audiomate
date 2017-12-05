from pingu.corpus import assets

class LabelMapper(object):
    """
    The LabelMapper class has two functions.

    1. After instantiation it needs to load a filter from a txt file with the load_filter() function.
    The load_filter() function takes a path to the txt file and returns a list of filters back

    2. The map() function takes the returned filter list  and  the LabelList which has to be mapped
       and returns the mapped LabelList
    """
    def load_filter(self,filter_path):

        """
        :param filter_path: String
        :return: filter: List
        """
        filter = []
        with open(filter_path) as file:
            for line in file:
                line = line.strip()
                filter.append(line)
        return filter

    def map(self,label_list,filter):
        """
        :param label_list: LabelList: The LabelList which has to be reduced
        :param filter: List: The filter with with input labels and mapped output label
        :return: endLabelList: LabelList

        Example:
        >>>label_list = assets.LabelList(labels=[
        >>>    assets.Label('a', 3.2, 4.5),
        >>>    assets.Label('speech', 5.1, 8.9),
        >>>    assets.Label('music', 7.2, 10.5),
        >>>    assets.Label('d', 10.5, 14),
        >>>    assets.Label('e', 11.5, 16),  # todo: throws error with this data: 'e', 11.5 , 13
        >>>])

        Example:
        >>>filter = ["a b| a_b",
        >>>          "b c| b_c",
        >>>          "x c| x_c",
        >>>          "b c| b_c",
        >>>          "d e| MusikSprach",
        >>>          "speech music| muspeech"]
        """
        filterListMap = []  # [['a b', ' a_b'], ['b c', ' b_c'], ['x c', ' x_c'], ['b c', ' b_c'], ['c d', ' c_d']]
        for x in filter:
            filterListMap.append(x.split("|"))

        print("filterListMap",filterListMap)
        print("X")
        filterSet = []  # [['a', 'b'], ['b', 'c'], ['x', 'c'], ['b', 'c'], ['c', 'd']]
        for x in filterListMap:
            filterSet.append(x[0].split(" "))
        print("filterSet:",filterSet)
        print("X")
        ranges = label_list.ranges()
        reduced_label = ""
        endLabelLabels = []
        for range in ranges:
            currlabelSet = [str(val.value) for val in range[2]]
            for idx, fs in enumerate(filterSet):
                if (sorted(fs) == sorted(currlabelSet)):
                    reduced_label = filterListMap[idx][1]
                    break
                else:
                    reduced_label = currlabelSet[0]
            reduced_label = reduced_label.strip()
            endLabel = assets.Label(reduced_label, range[0], range[1])
            endLabelLabels.append(endLabel)
            print(range[0], range[1], reduced_label)

        endLabelList = assets.LabelList(labels=endLabelLabels)

        return endLabelList