import unittest

from pingu.corpus import assets


class CorpusTest(unittest.TestCase):
    def test_ranges(self):
        ll = assets.LabelList(labels=[
            assets.Label('a', 3.2, 4.5),
            assets.Label('speech', 5.1, 8.9),
            assets.Label('music', 7.2, 10.5),
            assets.Label('d', 10.5, 14),
            assets.Label('e', 11.5, 16),    #=>>> gibt fehler wenn 'e', 11.5 , 13   ===> warum?
        ])

        filter = ["a b| a_b",
                  "b c| b_c",
                  "x c| x_c",
                  "b c| b_c",
                  "d e| MusikSprach",
                  "speech music| muspeech"]

        filterListMap = [] # [['a b', ' a_b'], ['b c', ' b_c'], ['x c', ' x_c'], ['b c', ' b_c'], ['c d', ' c_d']]
        for x in filter:
            filterListMap.append(x.split("|"))

        print("filterListMap:",filterListMap)

        print("X")

        filterSet = [] # [['a', 'b'], ['b', 'c'], ['x', 'c'], ['b', 'c'], ['c', 'd']]
        for x in filterListMap:
            filterSet.append(x[0].split(" "))

        print("filterSet:",filterSet)

        ranges = ll.ranges()
        reduced_label =""
        endLabelLabels = []
        for range in ranges:
            #reduced_label = '_'.join([str(val.value) for val in range[2]])
            currlabelSet = [str(val.value) for val in range[2]]
            for idx, fs in enumerate(filterSet):
                #print("filterSet:",fs)
                #print("currentLab:",currlabelSet)
                #print("CURR:",currlabelSet)
                if(sorted(fs) == sorted(currlabelSet)):
                    #print("YESSS")
                    reduced_label = filterListMap[idx][1]
                else:
                    reduced_label = currlabelSet
            #print(range[0], range[1], range[2][0].value)
            endLabel = assets.Label(reduced_label,range[0],range[1])
            endLabelLabels.append(endLabel)
            print(range[0], range[1], reduced_label)

        endLabelList = assets.LabelList(endLabelLabels)
