# -*- coding: utf-8 -*-
# @Author  : Anonymous Period

import numpy as np
from collections import defaultdict
import json
np.random.seed(12345)

def sample_test_data(data_name, test_num=99, sample_type='random'):
    """
    sample_type:
        random:  sample `test_num` negative items randomly.
        pop: sample `test_num` negative items according to item popularity.

        test_queries_file: query of test sampled data in file {data_name}_sample.txt
    """

    data_file = f'{data_name}.txt'
    item2dummy_queries_file = f'{data_name}_id2dummy_query.json'
    test_file = f'{data_name}_sample.txt'
    test_queries_file = f'{data_name}_sample_queries.txt'

    item_count = defaultdict(int)
    user_items = defaultdict()

    lines = open(data_file).readlines()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_items[user] = items
        for item in items:
            item_count[item] += 1

    all_item = list(item_count.keys())
    count = list(item_count.values())
    sum_value = np.sum([x for x in count])
    probability = [value / sum_value for value in count]

    ## load dict
    lines = open(item2dummy_queries_file).readlines()
    dummy_json_list = []
    for line in lines:
        dummy_json_list.append(json.loads(line))
    item2dummy_queries_json = dummy_json_list[0]

    user_neg_items = defaultdict()
    user_neg_queries = defaultdict()
    lineno = 0
    for user, user_seq in user_items.items():
        lineno+=1
        if lineno % 100 == 0:
            print ("DEBUG: Generating Random Sample for User %d" % lineno)
        test_samples = []
        test_samples_queries = []
        while len(test_samples) < test_num:
            if sample_type == 'random':
                sample_ids = np.random.choice(all_item, test_num, replace=False)
            else: # sample_type == 'pop':
                sample_ids = np.random.choice(all_item, test_num, replace=False, p=probability)
            sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in test_samples]
            test_samples.extend(sample_ids)
            # [] list of string, each string is separated by , as "3,5,7,10"
            sample_queries = [item2dummy_queries_json[id] for id in sample_ids]
            test_samples_queries.extend(sample_queries)
        test_samples = test_samples[:test_num]
        test_samples_queries = test_samples_queries[:test_num]
        user_neg_items[user] = test_samples
        user_neg_queries[user] = test_samples_queries
    print ("DEBUG: Total Line of Sample Generated %d" % len(user_neg_items))
    cnt = 0
    with open(test_file, 'w') as out:
        for user, samples in user_neg_items.items():
            cnt += 1
            if cnt % 1000 == 0:
                print ("DEBUG: Writing Line to test_file %d" % cnt)
            out.write(user+' '+' '.join(samples)+'\n')
    cnt = 0
    with open(test_queries_file, 'w') as out:
        for user, queries in user_neg_queries.items():
            cnt += 1
            if cnt % 1000 == 0:
                print ("DEBUG: Writing Line to test_queries_file %d" % cnt)
            out.write(user+' '+' '.join(queries)+'\n')

# data_names = ['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games', 'Yelp', 'LastFM']
# data_names = ['kuaisar']
# for data_name in data_names:
#     sample_test_data(data_name)

## Generate KuaiSAR Dataset
def generate_kuaisar_dataset():
    data_names = ['kuaisar']
    for data_name in data_names:
        sample_test_data(data_name)

def generate_amazon_dataset():
    data_names = ['Movies_and_TV']
    for data_name in data_names:
        sample_test_data(data_name)

def main():
    generate_amazon_dataset()

if __name__ == '__main__':
    main()
