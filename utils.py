# -*- coding: utf-8 -*-
# @Author  : Anonymous Period

import numpy as np
import math
import random
import os
import json
import pickle
from scipy.sparse import csr_matrix

import torch
import torch.nn.functional as F

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')

def neg_sample(item_set, item_size):  # 前闭后闭
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            if score[i] > self.best_score[i]+self.delta:
                return False
        return True

    def __call__(self, score, model):
        # score HIT@10 NDCG@10

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0]*len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            # ({self.score_min:.6f} --> {score:.6f})
            print(f'Validation score increased.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)

def avg_pooling(x, dim):
    return x.sum(dim=dim)/x.size(dim)


def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def get_user_seqs(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    num_users = len(lines)
    num_items = max_item + 2

    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    return user_seq, max_item, valid_rating_matrix, test_rating_matrix

def get_user_seqs_long(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    long_sequence = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        long_sequence.extend(items) 
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    return user_seq, max_item, long_sequence

def get_user_seqs_and_sample(data_file, sample_file):
    """
        user_seq: list of user positive interacted items
        max_item: item vocab
        sample_seq: list, len user cnt, negatively sample items
    """
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    lineno = 0
    for line in lines:
        lineno += 1
        if lineno % 1000 == 0:
            print ("DEBUG: get_user_seqs_and_sample Reading User Sequence File Lineno %d" % lineno)
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    lines = open(sample_file).readlines()
    sample_seq = []
    lineno = 0
    for line in lines:
        lineno += 1
        if lineno % 1000 == 0:
            print ("DEBUG: get_user_seqs_and_sample Reading Sample Sequence File Lineno %d" % lineno)
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        sample_seq.append(items)

    assert len(user_seq) == len(sample_seq)

    return user_seq, max_item, sample_seq


def split_and_pad_query(query, sep, max_length):
    """
        query: sep separated keywords, e.g.  "111,34,56,33,222"
        sep: "," sepearator
        length: max_length of elements, default to 0
    """
    keywords_list = query.split(sep)
    keywords_id_list =[int(kw) for kw in keywords_list]
    final_keywords_list = []
    if len(keywords_id_list) <= max_length:
        final_keywords_list = keywords_id_list + [0] * (max_length - len(keywords_id_list))
    else:
        final_keywords_list = keywords_id_list[0: max_length]
    return final_keywords_list

def get_user_id_query_pair_seqs_and_sample(data_file, query_seq_file, sample_file, sample_neg_queries_file, query_max_token_num=5):
    """
        data_file: user interacted id sequence
        query_seq_file:  user interacted queries sequence
        query_max_token_num: pad or cut each query to query_max_token_num, e.g. default to  0,0,...,0

        output:
            sample_neg_queries_seq, optional
    """
    default_query_id = 0

    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    lineno = 0
    for line in lines:
        lineno += 1
        if lineno % 10000 == 0:
            print ("DEBUG: get_user_seqs_and_sample Reading User Item Sequence File Lineno %d" % lineno)
        # if lineno >= 1000:
        #     break
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    query_lines = open(query_seq_file).readlines()
    user_query_seq = []
    lineno = 0
    for line in query_lines:
        lineno += 1
        if lineno % 10000 == 0:
            print ("DEBUG: get_user_seqs_and_sample Reading User Query Sequence File Lineno %d" % lineno)
        # if lineno >= 1000:
        #     break
        user, queries_list = line.strip().split(' ', 1)
        queries = queries_list.split(' ')
        # [L, N_keywords]
        query_keyword_list = [split_and_pad_query(query, ",", query_max_token_num) for query in queries]
        user_query_seq.append(query_keyword_list)

    lines = open(sample_file).readlines()
    sample_seq = []
    lineno = 0
    for line in lines:
        lineno += 1
        if lineno % 10000 == 0:
            print ("DEBUG: get_user_seqs_and_sample Reading Negatively Sample Sequence File Lineno %d" % lineno)
        # if lineno >= 1000:
        #     break
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        sample_seq.append(items)

    sample_neg_queries_file_exist = os.path.exists(sample_neg_queries_file)
    sample_neg_queries_seq = []
    lineno = 0
    if sample_neg_queries_file_exist:
        lines = open(sample_neg_queries_file).readlines()
        for line in lines:
            lineno += 1
            if lineno % 10000 == 0:
                print ("DEBUG: get_user_seqs_and_sample Reading Negatively Sample Query Sequence File Lineno %d" % lineno)
                # print (sample_neg_queries_seq[len(sample_neg_queries_seq) - 1])
            # if lineno >= 1000:
            #     break
            # queries  w1,w2,w3 w4,w5,w6 w7,w8,w9
            user, queries = line.strip().split(' ', 1)
            cur_query_list = []
            queries_list = queries.split(" ")
            for query in queries_list:
                query_ids_str = query.split(",")
                query_ids = []
                if query_ids_str == "":
                    query_ids = [default_query_id]
                else:
                    query_ids = [int(id) if id.strip() != "" else default_query_id for id in query_ids_str]
                ## check needs default padding
                if len(query_ids) < query_max_token_num:
                    query_ids = query_ids + [0] * (query_max_token_num - len(query_ids))
                assert len(query_ids) == query_max_token_num
                cur_query_list.append(query_ids)
            # cur_query_list: [num_neg, num_token_query]
            sample_neg_queries_seq.append(cur_query_list)
    else:
        # padding 0 as default negatively sampled query
        num = len(sample_seq)
        for i in range(num):
            num_neg_sample = len(sample_seq[i])
            # each user sample num_neg_sample sample (e.g. num_neg_sample=99), with num_neg_sample queries
            neg_queries_list = [[0] * query_max_token_num] * num_neg_sample
            sample_neg_queries_seq.append(neg_queries_list)

    assert len(user_seq) == len(user_query_seq)
    assert len(user_seq) == len(sample_seq)
    assert len(user_seq) == len(sample_neg_queries_seq)

    # print ("DEBUG: sample_seq type list, size %d, sample_seq %s" % (len(sample_seq), str(sample_seq[0])))
    # sample_neg_queries_seq shape [num_user, num_neg, num_token_query]
    # print ("DEBUG: sample_neg_queries_seq type list, size [%d, %d], sample_seq %s" % (len(sample_neg_queries_seq), len(sample_neg_queries_seq[0]), str(sample_neg_queries_seq[0])))

    return user_seq, user_query_seq, max_item, sample_seq, sample_neg_queries_seq

def get_item2attribute_json(data_file):
    item2attribute = json.loads(open(data_file).readline())
    attribute_set = set()
    for item, attributes in item2attribute.items():
        attribute_set = attribute_set | set(attributes)
    attribute_size = max(attribute_set) # 331
    return item2attribute, attribute_size

def get_item2attribute_json_simplified(data_file, default_attribute_size):
    item2attribute = json.loads(open(data_file).readline())
    # attribute_set = set()
    # for item, attributes in item2attribute.items():
    #     attribute_set = attribute_set | set(attributes)
    # attribute_size = max(attribute_set) # 331
    return item2attribute, default_attribute_size

def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT /len(pred_list), NDCG /len(pred_list), MRR /len(pred_list)

def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)

def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res
