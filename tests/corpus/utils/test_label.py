from pingu.corpus.utils import label
from pingu.corpus import assets
import unittest
import os

class LabelMapperTest(unittest.TestCase):

    def test_map(self):
        labelMapper = label.LabelMapper()

        label_list = assets.LabelList(labels=[
            assets.Label('a', 3.2, 4.5),
            assets.Label('speech', 5.1, 8.9),
            assets.Label('music', 7.2, 10.5),
            assets.Label('d', 10.5, 14),
            assets.Label('e', 11.5, 16),
        ])

        filter_path = os.path.join(os.path.dirname(__file__), 'filter.txt')

        filter_list = labelMapper.load_filter(filter_path)


        result_mapped_label_list = labelMapper.map(label_list,filter_list)

        should_mapped_label_list = assets.LabelList(labels=[
            assets.Label('a', 3.2, 4.5),
            assets.Label('speech', 5.1, 7.2),
            assets.Label('muspeech', 7.2, 8.9),
            assets.Label('music', 8.9, 10.5),
            assets.Label('d', 10.5, 11.5),
            assets.Label('MusikSprach', 11.5, 14),
            assets.Label('e', 14, 16)
        ])

        result_list=[]
        for lbl in result_mapped_label_list.ranges():
            result_list.append([lbl[0],lbl[1],lbl[2][0].value])

        should_list=[]
        for lbl in should_mapped_label_list.ranges():
            should_list.append([lbl[0],lbl[1],lbl[2][0].value])


        print("oo----------------------------------------oo")
        print("res_list",result_list)
        print("should_l",should_list)
        print("oo----------------------------------------oo")
        print("filter:",filter_list)
        print("is:",result_mapped_label_list)
        print("should:",should_mapped_label_list)


        self.assertEqual(str(result_list),str(should_list))