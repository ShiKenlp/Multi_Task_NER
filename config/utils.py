import numpy as np
import torch
from typing import List
from common.instance import Instance


START = "<START>"
STOP = "<STOP>"
PAD = "<PAD>"


def log_sum_exp_pytorch(vec):
    """

    :param vec: [batchSize * from_label * to_label]
    :return: [batchSize * to_label]
    """
    maxScores, idx = torch.max(vec, 1)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0] ,1 , vec.shape[2]).expand(vec.shape[0], vec.shape[1], vec.shape[2])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 1))



def simple_batching(config, insts: List[Instance]):
    from config.config import ContextEmb
    """

    :param config:
    :param insts:
    :return:
        word_seq_tensor,
        word_seq_len,
        char_seq_tensor,
        char_seq_len,
        label_seq_tensor,
        # task specific 
        prefix_label_seq_tensor,
        conll_label_seq_tensor,
        notes_label_seq_tensor,
        mask_base,
        mask_conll,
        mask_ontonotes
    """
    batch_size = len(insts)
    batch_data = sorted(insts, key=lambda inst: len(inst.input.words), reverse=True) ##object-based not direct copy
    word_seq_len = torch.LongTensor(list(map(lambda inst: len(inst.input.words), batch_data)))
    max_seq_len = word_seq_len.max()
    ### TODO: the 1 here might be used later?? We will make this as padding, because later we have to do a deduction.
    #### Use 1 here because the CharBiLSTM accepts
    char_seq_len = torch.LongTensor([list(map(len, inst.input.words)) + [1] * (int(max_seq_len) - len(inst.input.words)) for inst in batch_data])
    max_char_seq_len = char_seq_len.max()

    word_emb_tensor = None
    if config.context_emb != ContextEmb.none:
        emb_size = insts[0].elmo_vec.shape[1]
        word_emb_tensor = torch.zeros((batch_size, max_seq_len, emb_size))

    word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    # label_seq_tensor =  torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    '''
    task specific tensors.
    '''
    prefix_label_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    conll_label_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    notes_label_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)

    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_char_seq_len), dtype=torch.long)
    mask_base = torch.ones(batch_size, dtype=torch.float)
    mask_conll = torch.zeros(batch_size, dtype=torch.float)
    mask_ontonotes = torch.zeros(batch_size, dtype=torch.float)

    for idx in range(batch_size):
        if batch_data[idx].dataset_num == 0:
            mask_conll[idx] = 1.
            mask_ontonotes[idx] = 0.
        else:
            mask_conll[idx] = 0.
            mask_ontonotes[idx] = 1.
        # mask_conll[idx] = batch_data[idx].dataset_num
        # mask_ontonotes[idx] = abs(batch_data[idx].dataset_num - 1)
        word_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].word_ids)
        # label_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].output_ids)
        prefix_label_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].prefix_label_ids)
        conll_label_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].conll_label_ids)
        notes_label_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].notes_label_ids)
        if config.context_emb != ContextEmb.none:
            word_emb_tensor[idx, :word_seq_len[idx], :] = torch.from_numpy(batch_data[idx].elmo_vec)

        for word_idx in range(word_seq_len[idx]):
            char_seq_tensor[idx, word_idx, :char_seq_len[idx, word_idx]] = torch.LongTensor(batch_data[idx].char_ids[word_idx])
        for wordIdx in range(word_seq_len[idx], max_seq_len):
            char_seq_tensor[idx, wordIdx, 0: 1] = torch.LongTensor([config.char2idx[PAD]])   ###because line 119 makes it 1, every single character should have a id. but actually 0 is enough

    # word_seq_tensor = word_seq_tensor.to(config.device)
    # # label_seq_tensor = label_seq_tensor.to(config.device)
    # char_seq_tensor = char_seq_tensor.to(config.device)
    # word_seq_len = word_seq_len.to(config.device)
    # char_seq_len = char_seq_len.to(config.device)
    #
    # '''
    # task specific mask parameters.
    # '''
    # prefix_label_seq_tensor = prefix_label_seq_tensor.to(config.device)
    # conll_label_seq_tensor = conll_label_seq_tensor.to(config.device)
    # notes_label_seq_tensor = notes_label_seq_tensor.to(config.device)
    # mask_base = mask_base.to(config.device)
    # mask_conll = mask_conll.to(config.device)
    # mask_ontonotes = mask_ontonotes.to(config.device)
    # if config.use_elmo:
    #     word_emb_tensor = word_emb_tensor.to(config.device)

    return word_seq_tensor, word_seq_len, word_emb_tensor, char_seq_tensor, char_seq_len, prefix_label_seq_tensor, conll_label_seq_tensor, notes_label_seq_tensor, mask_base, mask_conll, mask_ontonotes


def lr_decay(config, optimizer, epoch):
    lr = config.learning_rate / (1 + config.lr_decay * (epoch - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('learning rate is set to: ', lr)
    return optimizer



