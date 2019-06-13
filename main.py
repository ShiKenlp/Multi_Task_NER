import argparse
import random
import numpy as np
from config.reader import Reader
from config import eval
from config.config import Config, ContextEmb, Config_conll, Config_ontonotes
import time
# from model.lstmcrf import NNCRF
from model.mt_lstmcrf import MT_LSTMCRF
import torch
import torch.optim as optim
import torch.nn as nn
from config.utils import lr_decay, simple_batching
from typing import List
from common.instance import Instance
from termcolor import colored


def setSeed(opt, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if opt.device.startswith("cuda"):
        print("using GPU...", torch.cuda.current_device())
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--mode', type=str, default='train', choices=["train","test"], help="training mode or testing mode")
    parser.add_argument('--device', type=str, default="cuda:1", choices=['cpu','cuda:0','cuda:1','cuda:2'],help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--digit2zero', action="store_true", default=True, help="convert the number to 0, make it true is better")
    parser.add_argument('--dataset1', type=str, default="conll2003")
    parser.add_argument('--dataset2', type=str, default='ontonotes')
    parser.add_argument('--embedding_file', type=str, default="data/glove/glove.6B.100d.txt")
    # parser.add_argument('--embedding_file', type=str, default=None)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--learning_rate', type=float, default=0.004) #0.01 ##only for sgd now
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_num', type=int, default=-1)
    parser.add_argument('--dev_num', type=int, default=-1)
    parser.add_argument('--test_num', type=int, default=-1)

    ##model hyperparameter
    parser.add_argument('--hidden_dim_base', type=int, default=200, help="hidden size of the LSTM of base layer")
    parser.add_argument('--hidden_dim_conll', type=int, default=300, help='hidden size of the LSTM of conll layer')
    parser.add_argument('--hidden_dim_ontonotes', type=int, default=300, help='hidden size of the LSTM of ontonotes layer')

    ##NOTE: this dropout applies to many places
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding") #0.5
    parser.add_argument('--use_char_rnn', type=int, default=1, choices=[0, 1], help="use character-level lstm, 0 or 1")
    parser.add_argument('--context_emb', type=str, default="none", choices=["none", "bert", "elmo", "flair"], help="contextual word embedding")




    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def get_optimizer(config: Config, model: nn.Module):
    params = model.parameters()
    if config.optimizer.lower() == "sgd":
        print(colored("Using SGD: lr is: {}, L2 regularization is: {}".format(config.learning_rate, config.l2), 'yellow'))
        return optim.SGD(params, lr=config.learning_rate, weight_decay=float(config.l2))
    elif config.optimizer.lower() == "adam":
        print(colored("Using Adam", 'yellow'))
        return optim.Adam(params)
    else:
        print("Illegal optimizer: {}".format(config.optimizer))
        exit(1)


def batching_list_instances(config: Config, insts:List[Instance]):
    train_num = len(insts)
    batch_size = config.batch_size
    total_batch = train_num // batch_size + 1 if train_num % batch_size != 0 else train_num // batch_size
    batched_data = []
    for batch_id in range(total_batch):
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        batched_data.append(simple_batching(config, one_batch_insts))

    return batched_data


def learn_from_insts(config:Config, conf_conll, conf_ontonotes, epoch: int, train_insts, dev_insts, test_insts):
    # train_insts: List[Instance], dev_insts: List[Instance], test_insts: List[Instance], batch_size: int = 1
    model = MT_LSTMCRF(config, conf_conll, conf_ontonotes)
    optimizer = get_optimizer(config, model)
    train_num = len(train_insts)
    print("number of instances: %d" % (train_num))
    print(colored("[Shuffled] Shuffle the training instance ids", "red"))
    random.shuffle(train_insts)



    batched_data = batching_list_instances(config, train_insts)
    dev_batches = batching_list_instances(config, dev_insts)
    test_batches = batching_list_instances(config, test_insts)

    best_dev = [-1, 0]
    best_test = [-1, 0]

    model_name = "model_files/lstm_{}_crf_{}_{}_dep_{}_elmo_{}_lr_{}.m".format(config.hidden_dim, config.dataset1, config.train_num, config.context_emb.name, config.optimizer.lower(), config.learning_rate)
    res_name = "results/lstm_{}_crf_{}_{}_dep_{}_elmo_{}_lr_{}.results".format(config.hidden_dim, config.dataset1, config.train_num, config.context_emb.name, config.optimizer.lower(), config.learning_rate)
    print("[Info] The model will be saved to: %s, please ensure models folder exist" % (model_name))

    for i in range(1, epoch + 1):
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad()
        if config.optimizer.lower() == "sgd":
            optimizer = lr_decay(config, optimizer, i)
        for index in np.random.permutation(len(batched_data)):
        # for index in range(len(batched_data)):
            model.train()
            batch_word, batch_wordlen, batch_context_emb, batch_char, batch_charlen, batch_prefix_label, batch_conll_label, batch_notes_label, mask_base, mask_conll, mask_ontonotes = batched_data[index]
            loss = model.neg_log_obj_total(batch_word, batch_wordlen, batch_context_emb, batch_char, batch_charlen, batch_prefix_label, batch_conll_label, batch_notes_label, mask_base, mask_conll, mask_ontonotes)
            epoch_loss += loss.item()
            loss.backward()
            # # torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip) ##clipping the gradient
            optimizer.step()
            model.zero_grad()

        end_time = time.time()
        print("Epoch %d: %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time), flush=True)

        model.eval()
        dev_metrics = evaluate_model(config, conf_conll, conf_ontonotes, model, dev_batches, "dev", dev_insts)
        test_metrics = evaluate_model(config, conf_conll, conf_ontonotes, model, test_batches, "test", test_insts)
        if dev_metrics[2] > best_dev[0]:
            print("saving the best model...")
            best_dev[0] = dev_metrics[2]
            best_dev[1] = i
            best_test[0] = test_metrics[2]
            best_test[1] = i
            torch.save(model.state_dict(), model_name)
            write_results(res_name, test_insts)
        model.zero_grad()

    print("The best dev: %.2f" % (best_dev[0]))
    print("The corresponding test: %.2f" % (best_test[0]))
    print("Final testing.")
    model.load_state_dict(torch.load(model_name))
    model.eval()
    evaluate_model(config, model, test_batches, "test", test_insts)
    write_results(res_name, test_insts)


def evaluate_model(config:Config, conf_conll, conf_ontonotes, model: MT_LSTMCRF, batch_insts_ids, name:str, insts: List[Instance]):
    ## evaluation
    metrics_conll = np.asarray([0, 0, 0], dtype=int)
    metrics_notes = np.asarray([0, 0, 0], dtype=int)
    batch_id = 0
    batch_size = config.batch_size
    for batch in batch_insts_ids:
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        sorted_batch_insts = sorted(one_batch_insts, key=lambda inst: len(inst.input.words), reverse=True)
        # batch_max_scores, batch_max_ids = model.decode(batch)
        bestScores_conll, decodeIdx_conll, bestScores_notes, decodeIdx_notes, mask_conll, mask_ontonotes = model.decode(batch)
        metrics_conll += eval.evaluate_num(sorted_batch_insts, decodeIdx_conll, batch[6], batch[1], conf_conll.idx2labels, mask_conll)
        metrics_notes += eval.evaluate_num(sorted_batch_insts, decodeIdx_notes, batch[7], batch[1], conf_ontonotes.idx2labels, mask_ontonotes)
        batch_id += 1
    p_conll, total_predict_conll, total_entity_conll = metrics_conll[0], metrics_conll[1], metrics_conll[2]
    precision_conll = p_conll * 1.0 / total_predict_conll * 100 if total_predict_conll != 0 else 0
    recall_conll = p_conll * 1.0 / total_entity_conll * 100 if total_entity_conll != 0 else 0
    fscore_conll = 2.0 * precision_conll * recall_conll / (precision_conll + recall_conll) if precision_conll != 0 or recall_conll != 0 else 0
    print("[%s conll set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision_conll, recall_conll,fscore_conll), flush=True)

    p_notes, total_predict_notes, total_entity_notes = metrics_notes[0], metrics_notes[1], metrics_notes[2]
    precision_notes = p_notes * 1.0 / total_predict_notes * 100 if total_predict_notes != 0 else 0
    recall_notes = p_notes * 1.0 / total_entity_notes * 100 if total_entity_notes != 0 else 0
    fscore_notes = 2.0 * precision_notes * recall_notes / (precision_notes + recall_notes) if precision_notes != 0 or recall_notes != 0 else 0
    print("[%s notes set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision_notes, recall_notes, fscore_notes), flush=True)

    # return [precision_notes, recall_notes, fscore_notes]
    return [precision_conll, recall_conll, fscore_conll, precision_notes, recall_notes, fscore_notes]


def test_model(config: Config, test_insts):
    model_name = "./model_files/lstm_{}_crf_{}_{}_dep_{}_elmo_{}_lr_{}.m".format(config.hidden_dim, config.dataset,
                                                                               config.train_num,
                                                                               config.context_emb.name,
                                                                               config.optimizer.lower(),
                                                                               config.learning_rate)
    res_name = "./results/lstm_{}_crf_{}_{}_dep_{}_elmo_{}_lr_{}.results".format(config.hidden_dim, config.dataset,
                                                                               config.train_num,
                                                                               config.context_emb.name,
                                                                               config.optimizer.lower(),
                                                                               config.learning_rate)


    model = MT_LSTMCRF(config)
    model.load_state_dict(torch.load(model_name))
    model.eval()
    test_batches = batching_list_instances(config, test_insts)
    evaluate_model(config, model, test_batches, "test", test_insts)
    write_results(res_name, test_insts)

def write_results(filename:str, insts):
    f = open(filename, 'w', encoding='utf-8')
    for inst in insts:
        # debug
        # if inst.dataset_num == 0:
        #     continue
        for i in range(len(inst.input)):
            words = inst.input.words
            tags = inst.input.pos_tags
            output = inst.output
            prediction = inst.prediction
            assert  len(output) == len(prediction)
            f.write("{}\t{}\t{}\t{}\t{}\n".format(i, words[i], tags[i], output[i], prediction[i]))
        f.write("\n")
    f.close()

def main():
    print('Reading arguments')
    parser = argparse.ArgumentParser(description="LSTM CRF implementation")
    opt = parse_arguments(parser)
    conf = Config(opt)
    conf_conll = Config_conll(opt)
    conf_ontonotes = Config_ontonotes(opt)

    reader = Reader(conf.digit2zero)
    setSeed(opt, conf.seed)

    trains_0 = reader.read_conll(conf.train_file_1, 0, conf.train_num, True)
    devs_0 = reader.read_conll(conf.dev_file_1, 0, conf.dev_num, False)
    tests_0 = reader.read_conll(conf.test_file_1, 0, conf.test_num, False)

    trains_1 = reader.read_conll(conf.train_file_2, 1, conf.train_num, True)
    devs_1 = reader.read_conll(conf.dev_file_2, 1, conf.dev_num, False)
    tests_1 = reader.read_conll(conf.test_file_2, 1, conf.test_num, False)

    trains_all = trains_0 + trains_1
    devs_all = devs_0 + devs_1
    tests_all = tests_0 + tests_1



    if conf.context_emb != ContextEmb.none:
        print('Loading the elmo vectors for all datasets.')
        conf.context_emb_size = reader.load_elmo_vec(conf.train_file_1+ "."+conf.context_emb.name+".vec", trains_1)
        reader.load_elmo_vec(conf.dev_file_1+ "."+conf.context_emb.name+".vec", devs_1)
        reader.load_elmo_vec(conf.test_file_1+ "."+conf.context_emb.name+".vec", tests_1)

    conf.use_iobes(trains_all)
    conf.use_iobes(devs_all)
    conf.use_iobes(tests_all)
    conf.build_label_idx(trains_all)

    conf.build_word_idx(trains_all, devs_all, tests_all)
    conf.build_emb_table()

    ids_train = conf.map_insts_ids(trains_all)
    ids_dev = conf.map_insts_ids(devs_all)
    ids_test= conf.map_insts_ids(tests_all)

    conf_conll.label_size = conf.label_size_0
    conf_conll.label2idx = conf.label2idx_0
    conf_conll.idx2labels = conf.idx2labels_0
    conf_ontonotes.label_size = conf.label_size_1
    conf_ontonotes.label2idx = conf.label2idx_1
    conf_ontonotes.idx2labels = conf.idx2labels_1

    print("num chars: " + str(conf.num_char))
    # print(str(config.char2idx))

    print("num words: " + str(len(conf.word2idx)))
    # print(config.word2idx)
    if opt.mode == "train":
        learn_from_insts(conf, conf_conll, conf_ontonotes, conf.num_epochs, trains_all, devs_all, tests_all)
    else:
        ## Load the trained model.
        test_model(conf, tests_all)
        # pass

    print(opt.mode)

if __name__ == "__main__":
    main()