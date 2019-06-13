# 
# @author: Allan
#

import numpy as np
from tqdm import tqdm
from typing import List
from common.instance import Instance
from config.utils import PAD, START, STOP
import torch
from enum import Enum
from termcolor import colored



class ContextEmb(Enum):
    none = 0
    elmo = 1
    bert = 2
    flair = 3

class Config:
    def __init__(self, args):

        self.PAD = PAD
        self.B = "B-"
        self.I = "I-"
        self.S = "S-"
        self.E = "E-"
        self.O = "O"
        self.START_TAG = START
        self.STOP_TAG = STOP
        self.UNK = "<UNK>"
        self.unk_id = -1

        '''
        task specific parameter
        '''
        self.is_base = True
        # self.device = torch.device("cuda" if args.gpu else "cpu")
        self.embedding_file = args.embedding_file
        self.embedding_dim = args.embedding_dim
        self.context_emb = ContextEmb[args.context_emb]
        self.context_emb_size = 0
        self.embedding, self.embedding_dim = self.read_pretrain_embedding()
        self.word_embedding = None
        self.seed = args.seed
        self.digit2zero = args.digit2zero

        self.dataset1 = args.dataset1
        self.dataset2 = args.dataset2
        # self.train_file = "data/" + self.dataset + "/train.txt"
        # self.dev_file = "data/" + self.dataset + "/dev.txt"
        # ## following datasets do not have development set
        # if self.dataset in ("abc", "cnn", "mnb", "nbc", "p25", "pri", "voa"):
        #     self.dev_file = "data/" + self.dataset + "/test.conllx"
        # self.test_file = "data/" + self.dataset + "/test.txt"

        self.train_file_1 = "data/" + self.dataset1 + "/train.conllx"
        self.dev_file_1 = "data/" + self.dataset1 + "/dev.conllx"
        self.test_file_1 = "data/" + self.dataset1 + "/test.conllx"

        self.train_file_2 = "data/" + self.dataset2 + "/train.conllx"
        self.dev_file_2 = "data/" + self.dataset2 + "/dev.conllx"
        self.test_file_2 = "data/" + self.dataset2 + "/test.conllx"

        self.label2idx_prefix = {}
        self.idx2labels_prefix = []

        self.label2idx_0 = {}
        self.idx2labels_0 = []

        self.label2idx_1 = {}
        self.idx2labels_1 = []

        self.char2idx = {}
        self.idx2char = []
        self.num_char = 0


        self.optimizer = args.optimizer.lower()
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.l2 = args.l2
        self.num_epochs = args.num_epochs
        self.use_dev = True
        self.train_num = args.train_num
        self.dev_num = args.dev_num
        self.test_num = args.test_num
        self.batch_size = args.batch_size
        self.clip = 5
        self.lr_decay = args.lr_decay
        self.device = torch.device(args.device)
        self.hidden_dim = args.hidden_dim_base
        # self.tanh_hidden_dim = args.tanh_hidden_dim
        self.use_brnn = True
        self.num_layers = 1
        self.dropout = args.dropout
        self.char_emb_size = 25
        self.charlstm_hidden_dim = 50
        self.use_char_rnn = args.use_char_rnn






    # def print(self):
    #     print("")
    #     print("\tuse gpu: " + )

    '''
      read all the  pretrain embeddings
    '''
    def read_pretrain_embedding(self):
        print("reading the pretraing embedding: %s" % (self.embedding_file))
        if self.embedding_file is None:
            print("pretrain embedding in None, using random embedding")
            return None, self.embedding_dim
        embedding_dim = -1
        embedding = dict()
        with open(self.embedding_file, 'r', encoding='utf-8') as file:
            for line in tqdm(file.readlines()):
                line = line.strip()
                if len(line) == 0:
                    continue
                tokens = line.split()
                if embedding_dim < 0:
                    embedding_dim = len(tokens) - 1
                else:
                    # print(tokens)
                    # print(embedding_dim)
                    assert (embedding_dim + 1 == len(tokens))
                embedd = np.empty([1, embedding_dim])
                embedd[:] = tokens[1:]
                first_col = tokens[0]
                embedding[first_col] = embedd
        return embedding, embedding_dim


    def build_word_idx(self, train_insts, dev_insts, test_insts):
        self.word2idx = dict()
        self.idx2word = []
        self.word2idx[self.PAD] = 0
        self.idx2word.append(self.PAD)
        self.word2idx[self.UNK] = 1
        self.unk_id = 1
        self.idx2word.append(self.UNK)

        self.char2idx[self.PAD] = 0
        self.idx2char.append(self.PAD)
        self.char2idx[self.UNK] = 1
        self.idx2char.append(self.UNK)

        ##extract char on train, dev, test
        for inst in train_insts + dev_insts + test_insts:
            for word in inst.input.words:
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word.append(word)
        ##extract char only on train
        for inst in train_insts:
            for word in inst.input.words:
                for c in word:
                    if c not in self.char2idx:
                        self.char2idx[c] = len(self.idx2char)
                        self.idx2char.append(c)
        self.num_char = len(self.idx2char)
    '''
        build the embedding table
        obtain the word2idx and idx2word as well.
    '''
    def build_emb_table(self):
        print("Building the embedding table for vocabulary...")
        scale = np.sqrt(3.0 / self.embedding_dim)
        if self.embedding is not None:
            print("[Info] Use the pretrained word embedding to initialize: %d x %d" % (len(self.word2idx), self.embedding_dim))
            self.word_embedding = np.empty([len(self.word2idx), self.embedding_dim])
            for word in self.word2idx:
                if word in self.embedding:
                    self.word_embedding[self.word2idx[word], :] = self.embedding[word]
                elif word.lower() in self.embedding:
                    self.word_embedding[self.word2idx[word], :] = self.embedding[word.lower()]
                else:
                    # self.word_embedding[self.word2idx[word], :] = self.embedding[self.UNK]
                    self.word_embedding[self.word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])
            self.embedding = None
        else:
            self.word_embedding = np.empty([len(self.word2idx), self.embedding_dim])
            for word in self.word2idx:
                self.word_embedding[self.word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])


    def build_label_idx(self, insts):
        self.label2idx_prefix[self.PAD] = len(self.label2idx_prefix)
        self.idx2labels_prefix.append(self.PAD)
        for inst in insts:
            for label in inst.output:
                prefix = label.split('-')[0]
                if prefix not in self.label2idx_prefix:
                    self.idx2labels_prefix.append(prefix)
                    self.label2idx_prefix[prefix] = len(self.label2idx_prefix)

        self.label2idx_prefix[self.START_TAG] = len(self.label2idx_prefix)
        self.idx2labels_prefix.append(self.START_TAG)
        self.label2idx_prefix[self.STOP_TAG] = len(self.label2idx_prefix)
        self.idx2labels_prefix.append(self.STOP_TAG)
        self.label_size_prefix = len(self.label2idx_prefix)
        print("#labels_prefix: " + str(self.label_size_prefix))
        print("label 2idx: " + str(self.label2idx_prefix))

        self.label2idx_0[self.PAD] = len(self.label2idx_0)
        self.idx2labels_0.append(self.PAD)
        for inst in insts:
            if inst.dataset_num == 1:
                continue
            for label in inst.output:
                if label not in self.label2idx_0:
                    self.idx2labels_0.append(label)
                    self.label2idx_0[label] = len(self.label2idx_0)

        self.label2idx_0[self.START_TAG] = len(self.label2idx_0)
        self.idx2labels_0.append(self.START_TAG)
        self.label2idx_0[self.STOP_TAG] = len(self.label2idx_0)
        self.idx2labels_0.append(self.STOP_TAG)
        self.label_size_0 = len(self.label2idx_0)
        print("#labels_0: " + str(self.label_size_0))
        print("label 2idx 0: " + str(self.label2idx_0))

        self.label2idx_1[self.PAD] = len(self.label2idx_1)
        self.idx2labels_1.append(self.PAD)
        for inst in insts:
            if inst.dataset_num == 0:
                continue
            for label in inst.output:
                if label not in self.label2idx_1:
                    self.idx2labels_1.append(label)
                    self.label2idx_1[label] = len(self.label2idx_1)

        self.label2idx_1[self.START_TAG] = len(self.label2idx_1)
        self.idx2labels_1.append(self.START_TAG)
        self.label2idx_1[self.STOP_TAG] = len(self.label2idx_1)
        self.idx2labels_1.append(self.STOP_TAG)
        self.label_size_1 = len(self.label2idx_1)
        print("#labels_1: " + str(self.label_size_1))
        print("label 2idx 1: " + str(self.label2idx_1))

        self.label_size = self.label_size_prefix
        self.label2idx = self.label2idx_prefix
        self.idx2labels = self.idx2labels_prefix

    def use_iobes(self, insts):
        for inst in insts:
            output = inst.output
            for pos in range(len(inst)):
                curr_entity = output[pos]
                if pos == len(inst) - 1:
                    if curr_entity.startswith(self.B):
                        output[pos] = curr_entity.replace(self.B, self.S)
                    elif curr_entity.startswith(self.I):
                        output[pos] = curr_entity.replace(self.I, self.E)
                else:
                    next_entity = output[pos + 1]
                    if curr_entity.startswith(self.B):
                        if next_entity.startswith(self.O) or next_entity.startswith(self.B):
                            output[pos] = curr_entity.replace(self.B, self.S)
                    elif curr_entity.startswith(self.I):
                        if next_entity.startswith(self.O) or next_entity.startswith(self.B):
                            output[pos] = curr_entity.replace(self.I, self.E)

    def map_insts_ids(self, insts: List[Instance]):
        insts_ids = []
        for inst in insts:
            words = inst.input.words
            inst.word_ids = []
            inst.char_ids = []
            inst.dep_label_ids = []
            inst.dep_head_ids = []
            inst.prefix_label_ids = []
            inst.conll_label_ids = []
            inst.notes_label_ids = []
            for word in words:
                if word in self.word2idx:
                    inst.word_ids.append(self.word2idx[word])
                else:
                    inst.word_ids.append(self.word2idx[self.UNK])
                char_id = []
                for c in word:
                    if c in self.char2idx:
                        char_id.append(self.char2idx[c])
                    else:
                        char_id.append(self.char2idx[self.UNK])
                inst.char_ids.append(char_id)
            for label in inst.output:
                prefix = label.split('-')[0]
                inst.prefix_label_ids.append(self.label2idx_prefix[prefix])
                if inst.dataset_num == 0:
                    inst.conll_label_ids.append(self.label2idx_0[label])
                    inst.notes_label_ids.append(0)
                if inst.dataset_num == 1:
                    inst.notes_label_ids.append(self.label2idx_1[label])
                    inst.conll_label_ids.append(0)
            insts_ids.append([inst.word_ids, inst.char_ids, inst.prefix_label_ids, inst.conll_label_ids, inst.notes_label_ids])
        return insts_ids


class Config_conll:
    def __init__(self, args):

        # self.label2idx = config.label2idx
        # self.labels = config.idx2labels
        # self.start_idx = self.label2idx[START]
        # self.end_idx = self.label2idx[STOP]
        # self.pad_idx = self.label2idx[PAD]

        self.PAD = PAD
        self.B = "B-"
        self.I = "I-"
        self.S = "S-"
        self.E = "E-"
        self.O = "O"
        self.START_TAG = START
        self.STOP_TAG = STOP
        self.UNK = "<UNK>"
        self.unk_id = -1

        '''
        task specific parameter
        '''
        self.is_base = False
        self.label_size = 0
        # self.device = torch.device("cuda" if args.gpu else "cpu")
        self.embedding_file = args.embedding_file
        self.embedding_dim = args.hidden_dim_base
        self.context_emb = None
        self.context_emb_size = 0
        # self.embedding, self.embedding_dim = self.read_pretrain_embedding()
        self.word_embedding = None
        self.seed = args.seed
        self.digit2zero = args.digit2zero

        self.dataset1 = args.dataset1
        self.dataset2 = args.dataset2
        # self.train_file = "data/" + self.dataset + "/train.txt"
        # self.dev_file = "data/" + self.dataset + "/dev.txt"
        # ## following datasets do not have development set
        # if self.dataset in ("abc", "cnn", "mnb", "nbc", "p25", "pri", "voa"):
        #     self.dev_file = "data/" + self.dataset + "/test.conllx"
        # self.test_file = "data/" + self.dataset + "/test.txt"

        self.train_file_1 = "data/" + self.dataset1 + "/train.conllx"
        self.dev_file_1 = "data/" + self.dataset1 + "/dev.conllx"
        self.test_file_1 = "data/" + self.dataset1 + "/test.conllx"

        self.train_file_2 = "data/" + self.dataset2 + "/train.conllx"
        self.dev_file_2 = "data/" + self.dataset2 + "/dev.conllx"
        self.test_file_2 = "data/" + self.dataset2 + "/test.conllx"

        self.label2idx = {}
        self.idx2labels = []

        self.char2idx = {}
        self.idx2char = []
        self.num_char = 0


        self.optimizer = args.optimizer.lower()
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.l2 = args.l2
        self.num_epochs = args.num_epochs
        self.use_dev = True
        self.train_num = args.train_num
        self.dev_num = args.dev_num
        self.test_num = args.test_num
        self.batch_size = args.batch_size
        self.clip = 5
        self.lr_decay = args.lr_decay
        self.device = torch.device(args.device)

        self.hidden_dim = args.hidden_dim_conll
        # self.tanh_hidden_dim = args.tanh_hidden_dim
        self.use_brnn = True
        self.num_layers = 1
        self.dropout = args.dropout
        self.char_emb_size = 25
        self.charlstm_hidden_dim = 50
        self.use_char_rnn = False


class Config_ontonotes:
    def __init__(self, args):

        self.PAD = PAD
        self.B = "B-"
        self.I = "I-"
        self.S = "S-"
        self.E = "E-"
        self.O = "O"
        self.START_TAG = START
        self.STOP_TAG = STOP
        self.UNK = "<UNK>"
        self.unk_id = -1

        '''
        task specific parameter
        '''
        self.is_base = False
        self.label_size = 0
        # self.device = torch.device("cuda" if args.gpu else "cpu")
        self.embedding_file = args.embedding_file
        self.embedding_dim = args.hidden_dim_base
        self.context_emb = None
        self.context_emb_size = 0
        # self.embedding, self.embedding_dim = self.read_pretrain_embedding()
        self.word_embedding = None
        self.seed = args.seed
        self.digit2zero = args.digit2zero

        self.dataset1 = args.dataset1
        self.dataset2 = args.dataset2
        # self.train_file = "data/" + self.dataset + "/train.txt"
        # self.dev_file = "data/" + self.dataset + "/dev.txt"
        # ## following datasets do not have development set
        # if self.dataset in ("abc", "cnn", "mnb", "nbc", "p25", "pri", "voa"):
        #     self.dev_file = "data/" + self.dataset + "/test.conllx"
        # self.test_file = "data/" + self.dataset + "/test.txt"

        self.train_file_1 = "data/" + self.dataset1 + "/train.conllx"
        self.dev_file_1 = "data/" + self.dataset1 + "/dev.conllx"
        self.test_file_1 = "data/" + self.dataset1 + "/test.conllx"

        self.train_file_2 = "data/" + self.dataset2 + "/train.conllx"
        self.dev_file_2 = "data/" + self.dataset2 + "/dev.conllx"
        self.test_file_2 = "data/" + self.dataset2 + "/test.conllx"

        self.label2idx = {}
        self.idx2labels = []

        self.char2idx = {}
        self.idx2char = []
        self.num_char = 0


        self.optimizer = args.optimizer.lower()
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.l2 = args.l2
        self.num_epochs = args.num_epochs
        self.use_dev = True
        self.train_num = args.train_num
        self.dev_num = args.dev_num
        self.test_num = args.test_num
        self.batch_size = args.batch_size
        self.clip = 5
        self.lr_decay = args.lr_decay
        self.device = torch.device(args.device)

        self.hidden_dim = args.hidden_dim_ontonotes
        # self.tanh_hidden_dim = args.tanh_hidden_dim
        self.use_brnn = True
        self.num_layers = 1
        self.dropout = args.dropout
        self.char_emb_size = 25
        self.charlstm_hidden_dim = 50
        self.use_char_rnn = False
