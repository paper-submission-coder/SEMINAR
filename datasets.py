# -*- coding: utf-8 -*-
# @Author  : Anonymous Period

import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.functional import normalize

from utils import neg_sample

class PretrainDataset(Dataset):

    def __init__(self, args, user_seq, long_sequence):
        self.args = args
        self.user_seq = user_seq
        self.long_sequence = long_sequence
        self.max_len = args.max_seq_length
        self.part_sequence = []
        self.split_sequence()

    def split_sequence(self):
        for seq in self.user_seq:
            input_ids = seq[-(self.max_len+2):-2] # keeping same as train set
            for i in range(len(input_ids)):
                self.part_sequence.append(input_ids[:i+1])

    def __len__(self):
        return len(self.part_sequence)

    def __getitem__(self, index):

        sequence = self.part_sequence[index] # pos_items
        # sample neg item for every masked item
        masked_item_sequence = []
        neg_items = []
        # Masked Item Prediction
        item_set = set(sequence)
        for item in sequence[:-1]:
            prob = random.random()
            if prob < self.args.mask_p:
                masked_item_sequence.append(self.args.mask_id)
                neg_items.append(neg_sample(item_set, self.args.item_size))
            else:
                masked_item_sequence.append(item)
                neg_items.append(item)

        # add mask at the last position
        masked_item_sequence.append(self.args.mask_id)
        neg_items.append(neg_sample(item_set, self.args.item_size))

        # Segment Prediction
        if len(sequence) < 2:
            masked_segment_sequence = sequence
            pos_segment = sequence
            neg_segment = sequence
        else:
            sample_length = random.randint(1, len(sequence) // 2)
            start_id = random.randint(0, len(sequence) - sample_length)
            neg_start_id = random.randint(0, len(self.long_sequence) - sample_length)
            pos_segment = sequence[start_id: start_id + sample_length]
            neg_segment = self.long_sequence[neg_start_id:neg_start_id + sample_length]
            masked_segment_sequence = sequence[:start_id] + [self.args.mask_id] * sample_length + sequence[
                                                                                      start_id + sample_length:]
            pos_segment = [self.args.mask_id] * start_id + pos_segment + [self.args.mask_id] * (
                        len(sequence) - (start_id + sample_length))
            neg_segment = [self.args.mask_id] * start_id + neg_segment + [self.args.mask_id] * (
                        len(sequence) - (start_id + sample_length))

        assert len(masked_segment_sequence) == len(sequence)
        assert len(pos_segment) == len(sequence)
        assert len(neg_segment) == len(sequence)

        # padding sequence
        pad_len = self.max_len - len(sequence)
        masked_item_sequence = [0] * pad_len + masked_item_sequence
        pos_items = [0] * pad_len + sequence
        neg_items = [0] * pad_len + neg_items
        masked_segment_sequence = [0]*pad_len + masked_segment_sequence
        pos_segment = [0]*pad_len + pos_segment
        neg_segment = [0]*pad_len + neg_segment

        masked_item_sequence = masked_item_sequence[-self.max_len:]
        pos_items = pos_items[-self.max_len:]
        neg_items = neg_items[-self.max_len:]

        masked_segment_sequence = masked_segment_sequence[-self.max_len:]
        pos_segment = pos_segment[-self.max_len:]
        neg_segment = neg_segment[-self.max_len:]

        # Associated Attribute Prediction
        # Masked Attribute Prediction
        attributes = []
        for item in pos_items:
            attribute = [0] * self.args.attribute_size
            try:
                now_attribute = self.args.item2attribute[str(item)]
                for a in now_attribute:
                    attribute[a] = 1
            except:
                pass
            attributes.append(attribute)


        assert len(attributes) == self.max_len
        assert len(masked_item_sequence) == self.max_len
        assert len(pos_items) == self.max_len
        assert len(neg_items) == self.max_len
        assert len(masked_segment_sequence) == self.max_len
        assert len(pos_segment) == self.max_len
        assert len(neg_segment) == self.max_len


        cur_tensors = (torch.tensor(attributes, dtype=torch.long),
                       torch.tensor(masked_item_sequence, dtype=torch.long),
                       torch.tensor(pos_items, dtype=torch.long),
                       torch.tensor(neg_items, dtype=torch.long),
                       torch.tensor(masked_segment_sequence, dtype=torch.long),
                       torch.tensor(pos_segment, dtype=torch.long),
                       torch.tensor(neg_segment, dtype=torch.long),)
        return cur_tensors

class SASRecDataset(Dataset):

    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0] # no use

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]] if len(items) >= 2 else [0]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]


        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long), # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)


TEST_SET_MODE_RECOMMENDATION  = "recommendation"
TEST_SET_MODE_SEARCH = "search"
TEST_SET_MODE_PAIR = "pair"

class MultiModalSARDataset(Dataset):
    """
        Multi-Modal Search and Recommendation Dataset
        A Unified Sequence of Search and Recommendation behavior of multi-modal features
        [A, Q, T, I]
        A: Attributes, Q: Query, T: Text, I: Image

        test_neg_items: list of list, e.g. [[1,2,3], [5, 7], [3, 8, 10], [10, 18, 20]]
        test_neg_queries: the sequence of test negative queries aligned with test_neg_items, list of list, e.g. [[1,2,3], [5, 7], [3, 8, 10], [10, 18, 20]]
    """
    def __init__(self, args, user_seq, query_seq, test_neg_items=None, test_neg_queries=None, data_type='train'):
        self.args = args
        self.user_seq = user_seq     ## dict, key: user_id, value: sequence of interacted item ids, length: L
        self.query_seq = query_seq   ## dict, key: user_id, value: sequence of interacted query keywords, length L * N_keywords
        # test_neg_items [num_user, num_neg_sample] e.g. [30000,99]
        self.test_neg_items = test_neg_items
        # test_neg_queries [num_user, num_neg_sample, num_query_token] e.g. [30000, 99, 5]
        self.test_neg_queries = test_neg_queries
        self.data_type = data_type
        self.max_len = args.max_seq_length
        ## item_id [0,1,2,...,max_item, <EOS>], 0 for padding, (item_size+1) for <EOS>, item_size = max_item + 2
        self.token_EOS = args.item_size - 1
        self.query_keywords_num = 5
        self.token_default_empty_query = 0
        self.test_set_mode = args.test_set_mode

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]        # [L]
        query_list = self.query_seq[index]  # [L, N_keywords]

        # aligned itemid sequence and query sequence with same length
        assert len(items) == len(query_list)

        assert self.data_type in {"pretrain", "train", "valid", "test"}

        # input_ids: [0, 1, 2, 3, 4, 5, 6]

        # pretrain [0, 1, 2, <EOS>]
        # target [1, 2, <EOS>, 3]

        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "pretrain":
            # input: [0:L-4] + ["<EOS>"], length (L-2)  output [L-3]
            input_ids = items[:-4] + [self.token_EOS]
            input_queries = query_list[:-4] + [[self.token_EOS] * self.query_keywords_num]  #[L-3, N_keywords]
            target_pos = items[1:-4] + [self.token_EOS] + [items[-4]]
            # print ("DEBUG: query_list shape %d, %d" % (len(query_list), len(query_list[0])))
            target_queries_pos = query_list[1:-4] + [[self.token_EOS] * self.query_keywords_num] + [query_list[-4]]
            # print ("DEBUG: target_queries_pos shape %d, %d" % (len(target_queries_pos), len(target_queries_pos[0])))
            answer = [0] # no use
            answer_query = [0]

        elif self.data_type == "train":
            # input: [0:L-3], length (L-2) output [L-2]
            input_ids = items[:-3]
            input_queries = query_list[:-3]  #[L-3, N_keywords]
            target_pos = items[1:-2]
            target_queries_pos = query_list[1:-2]  #[L-3, N_keywords]
            answer = [0] # no use
            answer_query = [0]

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            input_queries = query_list[:-2]
            target_pos = items[1:-1]
            target_queries_pos = query_list[1:-1]
            answer = [items[-2]] if len(items) >= 2 else [0]
            answer_query = [query_list[-2]] if len(items) >= 2 else [0]

        else:
            ## test
            input_ids = items[:-1]
            input_queries = query_list[:-1]
            target_pos = items[1:]
            target_queries_pos = query_list[1:]

            # test positive
            answer = [items[-1]]
            answer_query = [query_list[-1]]

        target_neg = []
        target_queries_neg = []
        # target_neg_queries now we use randomly shuffled target_queries_pos as neg_queries,
        # better way is to align the pair of negatively sampled (item, query), need futher improvement
        seq_set = set(items)
        for _ in input_ids:
            # neg_sample only sample 1 items
            sampled_item_id = neg_sample(seq_set, self.args.item_size)
            target_neg.append(sampled_item_id)
            sampled_item_query = target_queries_pos[random.randint(1, len(target_queries_pos)) - 1]
            target_queries_neg.append(sampled_item_query)

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        # print ("DEBUG: Padding input_queries shape max_len %d|input_ids len %d|pad_len %d" % (self.max_len, len(input_ids), pad_len))
        # print ("DEBUG: Padding input_queries shape before %s" % str(np.array(input_queries).shape))
        # print ("DEBUG: Padding target_queries_pos shape before %s" % str(np.array(target_queries_pos).shape))

        input_queries = [[0] * self.query_keywords_num] * pad_len + input_queries
        target_queries_pos = [[0] * self.query_keywords_num] * pad_len + target_queries_pos
        target_queries_neg = [[0] * self.query_keywords_num] * pad_len + target_queries_neg
        # print ("DEBUG: Padding input_queries shape after %s" % str(np.array(input_queries).shape))
        # print ("DEBUG: Padding target_queries_pos shape after %s" % str(np.array(target_queries_pos).shape))

        input_ids = input_ids[-self.max_len:]
        input_queries = input_queries[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_queries_pos = target_queries_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]
        target_queries_neg = target_queries_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(input_queries) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_queries_pos) == self.max_len
        assert len(target_neg) == self.max_len
        assert len(target_queries_neg) == self.max_len

        if self.test_neg_items is not None:
            # test_samples: list of int id, shape: [num_negative_sample]
            test_samples = self.test_neg_items[index]
            num_negative_sample = len(test_samples)
            # test_samples_queries: list of list of int id, shape: [num_negative_sample, num_token_query]
            test_samples_queries = self.test_neg_queries[index]
            if self.test_set_mode == TEST_SET_MODE_RECOMMENDATION:
                # positive: answer, negative: test_samples
                # set query of answer,test_samples to 0 for evaluation
                # print ("DEBUG: Test recommendation before answer_query is %s" % str(answer_query))
                # print ("DEBUG: Test recommendation before test_samples_queries is %s" % str(test_samples_queries))
                answer_query = [[0] * self.query_keywords_num] * len(answer_query)
                test_samples_queries = [[0] * self.query_keywords_num] * len(test_samples)
                # print ("DEBUG: Test recommendation before answer_query is %s" % str(answer_query))
                # print ("DEBUG: Test recommendation before test_samples_queries is %s" % str(test_samples_queries))

            elif self.test_set_mode == TEST_SET_MODE_SEARCH:
                # search mode, target pos and negatively sampled under the sample query
                # set the query of all test_samples_queries positive and negative example to the query[0] of positive pair, imitating
                # candidating under the same query: query_0
                test_samples_queries = [answer_query[0] for _ in test_samples_queries]
                # print ("DEBUG: Test recommendation after answer_query is %s" % str(answer_query))
                # print ("DEBUG: Test recommendation after test_samples_queries is %s" % str(test_samples_queries))
            else:
                ## doing nothing
                assert 1==1

            ## Return: Eval, Test
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long), # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_queries, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_queries_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(target_queries_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(answer_query, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
                torch.tensor(test_samples_queries, dtype=torch.long),
            )
        else:
            ## Return: Train
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_queries, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_queries_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(target_queries_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(answer_query, dtype=torch.long),
            )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)
