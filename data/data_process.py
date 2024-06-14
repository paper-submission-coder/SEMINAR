# -*- coding: utf-8 -*-
# @Author  : Anonymous Period

from collections import defaultdict
import random
import numpy as np
import pandas as pd
import json
import pickle
import gzip
import tqdm
import os
import torch

def parse(path): # for Amazon
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)

# def parse_lines(path): # for Amazon
#     g = gzip.open(path, 'r')
#     for l in g:
#         yield l

# return (user item timestamp) sort in get_interaction
def Amazon(data_path, dataset_name, rating_score):
    '''
    reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    asin - ID of the product, e.g. 0000013714
    reviewerName - name of the reviewer
    helpful - helpfulness rating of the review, e.g. 2/3
    --"helpful": [2, 3],
    reviewText - text of the review
    --"reviewText": "I bought this for my husband who plays the piano. ..."
    overall - rating of the product
    --"overall": 5.0,
    summary - summary of the review
    --"summary": "Heavenly Highway Hymns",
    unixReviewTime - time of the review (unix time)
    --"unixReviewTime": 1252800000,
    reviewTime - time of the review (raw)
    --"reviewTime": "09 13, 2009"
    '''
    datas = []
    # older Amazon
    # data_path='/path/reviews_'
    data_file = os.path.join(data_path, dataset_name + '.json.gz')
    print ("DEBUG: data_file path is %s" % data_file)
    # latest Amazon
    # data_flie = '/home/hui_wang/data/new_Amazon/' + dataset_name + '.json.gz'
    for inter in parse(data_file):
        if float(inter['overall']) <= rating_score: # 小于一定分数去掉
            continue
        user = inter['reviewerID']
        item = inter['asin']
        time = inter['unixReviewTime']
        datas.append((user, item, int(time)))
    return datas

def Amazon_meta(data_path, dataset_name, data_maps):
    '''
    asin - ID of the product, e.g. 0000031852
    --"asin": "0000031852",
    title - name of the product
    --"title": "Girls Ballet Tutu Zebra Hot Pink",
    description
    price - price in US dollars (at time of crawl)
    --"price": 3.17,
    imUrl - url of the product image (str)
    --"imUrl": "http://ecx.images-amazon.com/images/I/51fAmVkTbyL._SY300_.jpg",
    related - related products (also bought, also viewed, bought together, buy after viewing)
    --"related":{
        "also_bought": ["B00JHONN1S"],
        "also_viewed": ["B002BZX8Z6"],
        "bought_together": ["B002BZX8Z6"]
    },
    salesRank - sales rank information
    --"salesRank": {"Toys & Games": 211836}
    brand - brand name
    --"brand": "Coxlures",
    categories - list of categories the product belongs to
    --"categories": [["Sports & Outdoors", "Other Sports", "Dance"]]
    '''
    datas = {}
    # meta_flie = '/path/meta_' + dataset_name + '.json.gz'
    meta_file = os.path.join(data_path, dataset_name + '.json.gz')
    item_asins = list(data_maps['item2id'].keys())
    cnt = 0
    for info in parse(meta_file):
        cnt += 1
        if cnt % 1000 == 0:
            print ("DEBUG: Amazon_meta processing line cnt %d" % cnt)
        if info['asin'] not in item_asins:
            continue
        datas[info['asin']] = info
    return datas

def KuaiSAR_meta(data_maps):
    '''
     {2632676: {'item_id': 2632676,
      'caption': '[2077636]',
      'author_id': 44552,
      'item_type': 'NORMAL',
      'upload_time': '2015-09-24',
      'upload_type': 'UNKNOWN',
      'music_id': 0,
      'first_level_category_id': 0,
      'first_level_category_name': 'empty',
      'second_level_category_id': 0,
      'second_level_category_name': 'empty',
      'third_level_category_id': 0,
      'third_level_category_name': 'empty',
      'fourth_level_category_id': 0,
      'fourth_level_category_name': 'empty',
      'first_level_category_name_en': 'empty',
      'second_level_category_name_en': 'empty',
      'third_level_category_name_en': 'empty',
      'fourth_level_category_name_en': 'empty'},
     ...}
    '''
    datas = {}
    content_meta_file = './KuaiSAR_final/item_features.csv'
    item_ids_set = set(data_maps['item2id'].keys())

    df_content = pd.read_csv(content_meta_file)
    print ("DEBUG: Total Content Size %d" % len(df_content))

    for index, row in df_content.iterrows():
        # print ("DEBUG: index %d and row is %s" % (index, str(row)))
        if int(index) % 100000 == 0:
            print ("DEBUG: Finish Readding Item Index %d" % index)
        if row["item_id"] not in item_ids_set:
           continue
        datas[row["item_id"]] = row.to_dict()
    return datas

def add_comma(num):
    # 1000000 -> 1,000,000
    str_num = str(num)
    res_num = ''
    for i in range(len(str_num)):
        res_num += str_num[i]
        if (len(str_num)-i-1) % 3 == 0:
            res_num += ','
    return res_num[:-1]

# categories brand is all attribute
def get_attribute_Amazon(meta_infos, datamaps, attribute_core):

    attributes = defaultdict(int)
    for iid, info in tqdm.tqdm(meta_infos.items()):
        for cates in info['category']:
            for cate in cates[1:]:
                attributes[cate] +=1
        try:
            attributes[info['brand']] += 1
        except:
            pass

    print(f'before delete, attribute num:{len(attributes)}')
    new_meta = {}
    for iid, info in tqdm.tqdm(meta_infos.items()):
        new_meta[iid] = []

        try:
            if attributes[info['brand']] >= attribute_core:
                new_meta[iid].append(info['brand'])
        except:
            pass
        for cates in info['category']:
            for cate in cates[1:]:
                if attributes[cate] >= attribute_core:
                    new_meta[iid].append(cate)
    attribute2id = {}
    id2attribute = {}
    attributeid2num = defaultdict(int)
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []

    for iid, attributes in new_meta.items():
        item_id = datamaps['item2id'][iid]
        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            attributeid2num[attribute2id[attribute]] += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    print(f'before delete, attribute num:{len(attribute2id)}')
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')
    datamaps['attribute2id'] = attribute2id
    datamaps['id2attribute'] = id2attribute
    datamaps['attributeid2num'] = attributeid2num
    return len(attribute2id), np.mean(attribute_lens), datamaps, items2attributes



# attributes:
# captain list of id,
# author_id: int
# music_id: int
# first_level_category_id: int
# second_level_category_id: int
# third_level_category_id: int

def get_attribute_Kuai(meta_infos, datamaps, attribute_core):
    """
    """
    ## get Attributes Original Cnt Map
    attributes = defaultdict(int)
    for iid, info in tqdm.tqdm(meta_infos.items()):
        item_attributes_list = []
        item_attributes_list.append("author_id_%s" % str(info['author_id']))
        item_attributes_list.append("music_id_%s" % str(info['music_id']))
        item_attributes_list.append("first_level_category_id_%s" % str(info['first_level_category_id']))
        item_attributes_list.append("second_level_category_id_%s" % str(info['second_level_category_id']))
        item_attributes_list.append("third_level_category_id_%s" % str(info['third_level_category_id']))
        ## add to attributes cnt map
        for attribute in item_attributes_list:
            attributes[attribute] += 1

    print(f'before delete, attribute num:{len(attributes)}')
    new_meta = {}
    # key: item_id (index), value: list of attributes (str)
    for iid, info in tqdm.tqdm(meta_infos.items()):
        # author_id
        item_attributes_list = []
        item_attributes_list.append("author_id_%s" % str(info['author_id']))
        item_attributes_list.append("music_id_%s" % str(info['music_id']))
        item_attributes_list.append("first_level_category_id_%s" % str(info['first_level_category_id']))
        item_attributes_list.append("second_level_category_id_%s" % str(info['second_level_category_id']))
        item_attributes_list.append("third_level_category_id_%s" % str(info['third_level_category_id']))
        # captain
        for captain_id in info['caption']:
            item_attributes_list.append("caption_id_%s" % str(captain_id))
        ## add attributes frequency cutoff
        new_meta[iid] = []
        for attribute in item_attributes_list:
            if attributes[attribute] >= attribute_core:
                new_meta[iid].append(attribute)

    attribute2id = {}
    id2attribute = {}
    attributeid2num = defaultdict(int)
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []

    for iid, attributes in new_meta.items():
        item_id = datamaps['item2id'][iid]
        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            attributeid2num[attribute2id[attribute]] += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    print(f'before delete, attribute num:{len(attribute2id)}')
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')
    datamaps['attribute2id'] = attribute2id
    datamaps['id2attribute'] = id2attribute
    datamaps['attributeid2num'] = attributeid2num
    return len(attribute2id), np.mean(attribute_lens), datamaps, items2attributes


# categories brand is all attribute
def get_attribute_KuaiSAR(meta_infos, datamaps, attribute_core):

    attributes = defaultdict(int)
    for iid, info in tqdm.tqdm(meta_infos.items()):
        for cates in info['categories']:
            for cate in cates[1:]:
                attributes[cate] +=1
        try:
            attributes[info['brand']] += 1
        except:
            pass

    print(f'before delete, attribute num:{len(attributes)}')
    new_meta = {}
    for iid, info in tqdm.tqdm(meta_infos.items()):
        new_meta[iid] = []

        try:
            if attributes[info['brand']] >= attribute_core:
                new_meta[iid].append(info['brand'])
        except:
            pass
        for cates in info['categories']:
            for cate in cates[1:]:
                if attributes[cate] >= attribute_core:
                    new_meta[iid].append(cate)

    attribute2id = {}
    id2attribute = {}
    attributeid2num = defaultdict(int)
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []

    for iid, attributes in new_meta.items():
        item_id = datamaps['item2id'][iid]
        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            attributeid2num[attribute2id[attribute]] += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    print(f'before delete, attribute num:{len(attribute2id)}')
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')
    datamaps['attribute2id'] = attribute2id
    datamaps['id2attribute'] = id2attribute
    datamaps['attributeid2num'] = attributeid2num
    return len(attribute2id), np.mean(attribute_lens), datamaps, items2attributes


def get_interaction(datas):
    """
        output: user, item1, item2, item3
        item1, item2, item3 positive interaction
    """
    user_seq = {}
    for data in datas:
        user, item, time = data
        if user in user_seq:
            user_seq[user].append((item, time))
        else:
            user_seq[user] = []
            user_seq[user].append((item, time))

    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])
        items = []
        for t in item_time:
            items.append(t[0])
        user_seq[user] = items
    return user_seq

def get_interaction_pair(datas):
    """
        output: user, item1, item2, item3
        item1, item2, item3 positive interaction
        key: user_id, value: list of string
    """
    pair_seq = {}
    for data in datas:
        user, item, query, time = data
        if user in pair_seq:
            pair_seq[user].append((item, query, time))
        else:
            pair_seq[user] = []
            pair_seq[user].append((item, query, time))

    item_id_seq = {}
    queries_seq = {}
    for user, pair_time in pair_seq.items():
        pair_time.sort(key=lambda x: x[2])
        items = []
        queries = []
        for t in pair_time:
            items.append(t[0])
            queries.append(t[1])
        item_id_seq[user] = items
        queries_seq[user] = queries
    return item_id_seq, queries_seq

# K-core user_core item_core
def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True

def filter_Kcore(user_items, user_core, item_core):
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core:
                user_items.pop(user)
            else:
                for item in user_items[user]:
                    if item_count[item] < item_core:
                        user_items[user].remove(item)
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    return user_items


def id_map(user_items): # user_items dict

    user2id = {} # raw 2 uid
    item2id = {} # raw 2 iid
    id2user = {} # uid 2 raw
    id2item = {} # iid 2 raw
    user_id = 1
    item_id = 1
    final_data = {}
    for user, items in user_items.items():
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        iids = [] # item id lists
        for item in items:
            if item not in item2id:
                item2id[item] = str(item_id)
                id2item[str(item_id)] = item
                item_id += 1
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    return final_data, user_id-1, item_id-1, data_maps


def main(data_name, data_type='Amazon'):
    assert data_type in {'Amazon', 'Yelp'}
    np.random.seed(12345)
    rating_score = 0.0  # rating score smaller than this score would be deleted
    # user 5-core item 5-core
    user_core = 5
    item_core = 5
    attribute_core = 0

    if data_type == 'Yelp':
        date_max = '2019-12-31 00:00:00'
        date_min = '2019-01-01 00:00:00'
        datas = Yelp(date_min, date_max, rating_score)
    else:
        datas = Amazon(data_name+'_5', rating_score=rating_score)

    user_items = get_interaction(datas)
    print(f'{data_name} Raw data has been processed! Lower than {rating_score} are deleted!')
    # raw_id user: [item1, item2, item3...]
    user_items = filter_Kcore(user_items, user_core=user_core, item_core=item_core)
    print(f'User {user_core}-core complete! Item {item_core}-core complete!')

    user_items, user_num, item_num, data_maps = id_map(user_items)  # new_num_id
    user_count, item_count, _ = check_Kcore(user_items, user_core=user_core, item_core=item_core)
    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)


    print('Begin extracting meta infos...')

    if data_type == 'Amazon':
        meta_infos = Amazon_meta(data_name, data_maps)
        attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_Amazon(meta_infos, data_maps, attribute_core)
    else:
        meta_infos = Yelp_meta(data_maps)
        attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_Yelp(meta_infos, data_maps, attribute_core)

    print(f'{data_name} & {add_comma(user_num)}& {add_comma(item_num)} & {user_avg:.1f}'
          f'& {item_avg:.1f}& {add_comma(interact_num)}& {sparsity:.2f}\%&{add_comma(attribute_num)}&'
          f'{avg_attribute:.1f} \\')

    # -------------- Save Data ---------------
    data_file = 'data/'+ data_name + '_neg.txt'
    item2attributes_file = 'data/'+ data_name + '_item2attributes.json'

    with open(data_file, 'w') as out:
        for user, items in user_items.items():
            out.write(user + ' ' + ' '.join(items) + '\n')
    json_str = json.dumps(item2attributes)
    with open(item2attributes_file, 'w') as out:
        out.write(json_str)

def run_kuaisar_dataset():
    data_name = "kuaisar"

    ## read kuai search and recommendation dataset
    print("Dataset Class SARDataset...")

    ## A Unified Search and Recommendation Pair Sequence
    user_action_datas = []

    ## load search dataset, keywords format: [1979292, 2548186, 2311293]
    dataset_path_se = "./KuaiSAR_final/src_inter.csv"
    cols_search = ["user_id",  "item_id", "keyword", "click_cnt", "search_session_timestamp"]
    df_search = pd.read_csv(dataset_path_se, usecols=cols_search)
    for index, row in df_search.iterrows():
        if index % 100000 == 0:
            print ("DEBUG: Finish Reading df_search index %d, Total Row %d" % (index, df_search.shape[0]))
        if int(row["click_cnt"]) > 0:
            keywords_value = row["keyword"]
            keywords_list = []
            try:
                keywords_list = json.loads(keywords_value)
            except Exception as e:
                keywords_list = []
            search_query = "0"
            if type(keywords_list) is list and len(keywords_list) > 0:
                search_query = ",".join([str(kw) for kw in keywords_list])
            else:
                search_query = "0"
            user_action_datas.append((row["user_id"], row["item_id"], search_query, int(row["search_session_timestamp"])))
    print ("DEBUG: user_action_datas df_search length %d" % len(user_action_datas))
    ## load recommendation dataset
    dataset_path_rec = "./KuaiSAR_final/rec_inter.csv"
    cols_rec = ["user_id", "item_id", "click", "timestamp"]
    df_rec = pd.read_csv(dataset_path_rec, usecols=cols_rec)
    for index, row in df_rec.iterrows():
        if index % 100000 == 0:
            print ("DEBUG: Finish Reading df_rec index %d, Total Row %d" % (index, df_rec.shape[0]))
        if int(row["click"]) > 0:
            default_query = "0"
            user_action_datas.append((row["user_id"], row["item_id"], default_query, int(row["timestamp"])))
    print ("DEBUG: user_action_datas length %d" % len(user_action_datas))

    ### Process User Search and Recommendation Sequence, datas [user, item, query, int(timestamp)]
    user_itemid_seq, user_queries_seq = get_interaction_pair(user_action_datas)
    print ("DEBUG: user_action_datas length %d" % len(user_action_datas))

    ## minimum user and item interaction filter
    ## input: user_itemid_seq, key original user id, output: user_items, key new indexed user id
    ## Each User Should have At Least 4 interactions
    user_core = 4
    item_core = 0
    user_itemid_seq = filter_Kcore(user_itemid_seq, user_core=user_core, item_core=item_core)
    print(f'User {user_core}-core complete! Item {item_core}-core complete!')
    print("After filter_Kcore filter, user_itemid_seq size %d" % len(user_itemid_seq))
    # user_items, key: str, value list of str
    user_items, user_num, item_num, data_maps = id_map(user_itemid_seq)  # new_num_id
    user_count, item_count, _ = check_Kcore(user_items, user_core=user_core, item_core=item_core)
    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)

    ## user_queries_seq key: original user_id; user_queries map key: indexed uid, output user_queries
    user2id_map = data_maps["user2id"]
    user_queries = {}
    for userid, queries in user_queries_seq.items():
        uid = user2id_map[userid] if userid in user2id_map else None
        if uid is not None:
            user_queries[uid] = queries
    # data inspection
    print ("DEBUG: aligned sequence user_items for user 1 len is %d|%s" % (len(user_items["1"]), str(user_items["1"])))
    print ("DEBUG: aligned sequence user_queries for user 1 len is %d|%s" % (len(user_queries["1"]), str(user_queries["1"])))
    print ("DEBUG: user_items map size %d, user_queries size %d" % (len(user_items), len(user_queries)))

    ## KuaiSAR 的 Meta信息
    meta_infos = KuaiSAR_meta(data_maps)
    print ("DEBUG: finish loading data_maps meta_infos...")
    attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_Kuai(meta_infos, data_maps, attribute_core=3)
    print ("DEBUG: finish loading get_attribute_Kuai attributes dict...")

    print(f'{data_name} & {add_comma(user_num)}& {add_comma(item_num)} & {user_avg:.1f}'
          f'& {item_avg:.1f}& {add_comma(interact_num)}& {sparsity:.2f}\%&{add_comma(attribute_num)}&'
          f'{avg_attribute:.1f} \\')

    # -------------- Save Data ---------------
    data_file = './'+ data_name + '.txt'
    data_query_file = './'+ data_name + '_queries.txt'
    item2attributes_file = './KuaiSAR_final/'+ data_name + '_item2attributes.json'

    with open(data_file, 'w') as out:
        for user, items in user_items.items():
            out.write(user + ' ' + ' '.join(items) + '\n')
    with open(data_query_file, 'w') as out:
        for user, queries in user_queries.items():
            out.write(str(user) + ' ' + ' '.join(queries) + '\n')

    json_str = json.dumps(item2attributes)
    with open(item2attributes_file, 'w') as out:
        out.write(json_str)


def generate_query_from_description(description):
    """
    """
    max_keep_keywords_cnt = 5
    keywords = description.split(" ")
    keywords_subset = []
    if len(keywords) >= max_keep_keywords_cnt:
        keywords_subset = keywords[0:max_keep_keywords_cnt]
    else:
        keywords_subset = keywords
    query = " ".join(keywords_subset)
    return query

def load_multi_modal_embedding(path_folder):
    """
        item_image_embedding, dict ,key: original asin id, value: tensor
        item_title_embedding, dict ,key: original asin id, value: tensor
    """
    item_image_embedding_file = os.path.join(path_folder, "item2image_emb.pth")
    item_title_embedding_file = os.path.join(path_folder, "item2title_emb.pth")

    item_image_embedding_dic = torch.load(item_image_embedding_file)
    item_title_embedding_dic = torch.load(item_title_embedding_file)
    return item_image_embedding_dic, item_title_embedding_dic

def run_main_Amazon(data_path, data_name, data_type='Amazon'):

    default_search_query = "<empty>"

    assert data_type in {'Amazon', 'Yelp'}
    np.random.seed(12345)
    rating_score = 0.0  # rating score smaller than this score would be deleted
    # user 5-core item 5-core
    user_core = 5
    item_core = 5
    attribute_core = 0

    if data_type == 'Yelp':
        date_max = '2019-12-31 00:00:00'
        date_min = '2019-01-01 00:00:00'
        datas = Yelp(date_min, date_max, rating_score)
    else:
        datas = Amazon(data_path, data_name+'_5', rating_score=rating_score)

    user_items = get_interaction(datas)
    print(f'{data_name} Raw data has been processed! Lower than {rating_score} are deleted!')
    # raw_id user: [item1, item2, item3...]
    user_items = filter_Kcore(user_items, user_core=user_core, item_core=item_core)
    print(f'User {user_core}-core complete! Item {item_core}-core complete!')

    user_items, user_num, item_num, data_maps = id_map(user_items)  # new_num_id
    user_count, item_count, _ = check_Kcore(user_items, user_core=user_core, item_core=item_core)
    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)


    print('Begin extracting meta infos...')

    if data_type == 'Amazon':
        # meta_infos = Amazon_meta(data_path, data_name, data_maps)
        meta_infos = Amazon_meta(data_path, "meta_" + data_name, data_maps)
        attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_Amazon(meta_infos, data_maps, attribute_core)
    else:
        meta_infos = Yelp_meta(data_maps)
        attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_Yelp(meta_infos, data_maps, attribute_core)

    print(f'{data_name} & {add_comma(user_num)}& {add_comma(item_num)} & {user_avg:.1f}'
          f'& {item_avg:.1f}& {add_comma(interact_num)}& {sparsity:.2f}\%&{add_comma(attribute_num)}&'
          f'{avg_attribute:.1f} \\')

    # process user to query json files
    user_queries_original = {}
    id2items_map = datamaps["id2item"]
    ## id2dummy_queries_map, query is generated from item's description
    for uid, item_id_list in user_items.items():
        query_list = []
        for id in item_id_list:
            original_asin_id = id2items_map[id]
            # item_title = meta_infos[item_id]["title"]
            # generate query from description
            item_description = default_search_query
            if original_asin_id in meta_infos and len(meta_infos[original_asin_id]["description"]) > 0:
                item_description = meta_infos[original_asin_id]["description"][0]
            query_list.append(generate_query_from_description(item_description))
        user_queries_original[uid] = query_list
    # id map
    keywords_set = set()
    for uid, query_list in user_queries_original.items():
        for query in query_list:
            keywords = query.split(" ")
            keywords_set.update(keywords)
    print ("DEBUG: keywords_set size %d" % len(keywords_set))
    keywords_list = list(keywords_set)
    keywords_to_id_map = {}
    id_to_keywords_map = {}
    keyword_id = 1
    for keyword in keywords_list:
        keywords_to_id_map[keyword] = keyword_id
        id_to_keywords_map[keyword_id] = keyword
        keyword_id += 1
    # user_queries_clean, k: uid, v: query_keywords_index
    user_queries = {}
    for uid, query_list in user_queries_original.items():
        query_to_id_list = []
        for query in query_list:
            keywords = query.split(" ")
            keywords_id_list = [keywords_to_id_map[kw] for kw in keywords]
            keywords_id_str = ",".join([str(id) for id in keywords_id_list])
            query_to_id_list.append(keywords_id_str)
        user_queries[uid] = query_to_id_list

    # Generate Id to Dummy Query Map
    id2dummy_query_map = {}
    for iid, original_asin_id in id2items_map.items():
        item_description = default_search_query
        if original_asin_id in meta_infos and len(meta_infos[original_asin_id]["description"]) > 0:
            item_description = meta_infos[original_asin_id]["description"][0]
        dummy_query = generate_query_from_description(item_description)
        dummy_keywords = dummy_query.split(" ")
        dummy_keywords_id_list = [keywords_to_id_map[kw] for kw in dummy_keywords]
        dummy_keywords_id_str = ",".join([str(id) for id in dummy_keywords_id_list])
        id2dummy_query_map[iid] = dummy_keywords_id_str

    ## Generate Id to Multi-Modal Embedding checkpoint
    item_image_embedding_dic, item_title_embedding_dic = load_multi_modal_embedding(data_path)
    iid_image_embedding_dic, iid_title_embedding_dic = {}, {}
    default_emb_size = 512
    match_image_cnt = 0
    match_title_cnt = 0
    total_cnt = len(id2items_map)
    for iid, original_asin_id in id2items_map.items():
        # image_emb = item_image_embedding_dic[original_asin_id] if original_asin_id in item_image_embedding_dic else torch.zeros(1, default_emb_size)
        # title_emb = item_title_embedding_dic[original_asin_id] if original_asin_id in item_title_embedding_dic else torch.zeros(1, default_emb_size)
        if original_asin_id in item_image_embedding_dic:
            match_image_cnt += 1
            image_emb = item_image_embedding_dic[original_asin_id]
        else:
            image_emb = torch.zeros(1, default_emb_size)

        if original_asin_id in item_title_embedding_dic:
            match_title_cnt += 1
            title_emb = item_title_embedding_dic[original_asin_id]
        else:
            title_emb = torch.zeros(1, default_emb_size)
        iid_image_embedding_dic[iid] = image_emb
        iid_title_embedding_dic[iid] = title_emb
    print ("DEBUG: Total Item Cnt %d, Match Image Ratio %f, Match Title Ratio %f" % (total_cnt, (match_image_cnt/total_cnt), (match_title_cnt/total_cnt)))

    # -------------- Save Data ---------------
    data_file = os.path.join(data_path, data_name + '.txt')
    data_query_file = os.path.join(data_path, data_name + '_queries.txt')
    item2attributes_file = os.path.join(data_path, data_name + '_item2attributes.json')
    id2dummy_query_file = os.path.join(data_path, data_name + '_id2dummy_query.json')
    image_emb_file = os.path.join(data_path, data_name + '_id2image_emb.pth')
    title_emb_file = os.path.join(data_path, data_name + '_id2title_emb.pth')

    with open(data_file, 'w') as out:
        for user, items in user_items.items():
            out.write(user + ' ' + ' '.join(items) + '\n')
    with open(data_query_file, 'w') as out:
        for user, queries in user_queries.items():
            out.write(str(user) + ' ' + ' '.join(queries) + '\n')
    json_str = json.dumps(item2attributes)
    with open(item2attributes_file, 'w') as out:
        out.write(json_str)
    id2dummy_query_json_str = json.dumps(id2dummy_query_map)
    with open(id2dummy_query_file, 'w') as out:
        out.write(id2dummy_query_json_str)
    # save embedding checkpoint
    torch.save(iid_image_embedding_dic, image_emb_file)
    torch.save(iid_title_embedding_dic, title_emb_file)


def run_data_sample():
    amazon_datas = ['Movies_and_TV']

    data_path = "${your_data_path}/SEMINAR/data/Amazon_movies_tv_5"
    for data_name in amazon_datas:
        run_main_Amazon(data_path, data_name, data_type='Amazon')
    # main('Yelp', data_type='Yelp')
    # LastFM()

def main():
    run_data_sample()

if __name__ == '__main__':
    main()
