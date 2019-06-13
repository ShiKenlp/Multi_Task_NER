# 
# @author: Allan
#
from common.sentence import Sentence
class Instance:

    def __init__(self, input: Sentence, output, prefix_label, dataset_num):
        self.input = input
        self.output = output
        self.prefix_label = prefix_label
        self.dataset_num = dataset_num
        self.elmo_vec = None
        self.word_ids = None
        self.char_ids = None

        self.prefix_label_ids = None
        self.conll_label_ids = None
        self.notes_label_ids = None

    def __len__(self):
        return len(self.input)
