# -*- coding: utf-8 -*-
# @Author  : Anonymous Period

import numpy as np
from numpy.linalg import norm

from collections import defaultdict

import torch
import clip
from PIL import Image
import json
import gzip
import requests
from io import BytesIO
import os
import faiss
import func_timeout
from func_timeout import func_set_timeout
import time

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.random.seed(12345)


def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l)

# set timeout 10s
@func_set_timeout(10)
def get_web_image_io(image_url):
    image_io = None
    try:
        response = requests.get(image_url)
        image_io = Image.open(BytesIO(response.content))
    except Exception as e:
        print(e)
    return image_io

def test_run_get_web_image_io():
    image_url = "https://images-na.ssl-images-amazon.com/images/I/515fDgR5EeL.jpg"
    image_io = get_web_image_io(image_url)

def download_item_multimodal_ckpt(path_to_meta_file, path_to_meta_folder, path_to_model_ckpt):
    """
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = os.path.join(path_to_model_ckpt, "ViT-B-32.pt")
    model, preprocess = clip.load(checkpoint_path, device=device)

    item2title_emb_json = {} # key: item_id, value: 64 dimensional emb
    item2image_emb_json = {} # key: item_id, value: 64 dimensional emb
    image_url_list = []
    total_cnt = 0
    max_title_token_cnt = 20

    image_success_cnt = 0
    text_success_cnt = 0
    for review in parse(path_to_meta_file):
        total_cnt += 1
        if total_cnt % 1000 == 0:
            print("DEBUG: read_item_meta_info reading line cnt %d, text_success_cnt %d, image_sucess_cnt %d"
                  % (total_cnt, text_success_cnt, image_success_cnt))
            # let crawler sleep 5s for every batch
            time.sleep(5)
        # list
        category = review['category'] if 'category' in review else ''
        title = review['title'] if 'title' in review else ''
        item_id = review['asin'] if 'asin' in review else ''
        image_url = review['image_url'][0] if 'image_url' in review and len(review['image_url']) >= 1 else None
        image_url_high_res = review['imageURLHighRes'][0] if 'imageURLHighRes' in review and len(review['imageURLHighRes']) >= 1 else None
        final_image_url = image_url_high_res if image_url_high_res is not None else image_url

        # proprocess title
        title_words = title.split(" ")
        title_clean = " ".join(title_words[0:max_title_token_cnt])

        image_io = None
        try:
            # response = requests.get(image_url_high_res)
            # image_io = Image.open(BytesIO(response.content))
            if final_image_url is not None:
                image_io = get_web_image_io(final_image_url)
        except func_timeout.exceptions.FunctionTimedOut as e:
            print (e)

        try:
            ## Image Modal
            if image_io is not None:
                image = preprocess(image_io).unsqueeze(0).to(device)
                with torch.no_grad():
                    # [torch.Size([1, 512])
                    image_features = model.encode_image(image)
                    item2image_emb_json[item_id] = image_features
                    image_success_cnt += 1

            ## Text Modal
            text = clip.tokenize([title_clean]).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text)
                item2title_emb_json[item_id] = text_features
                text_success_cnt += 1
        except Exception as e:
            print (e)

    print ("DEBUG: Total Item Cnt %d, image_success_cnt Cnt %d, text_success_cnt %d" % (total_cnt, image_success_cnt, text_success_cnt))

    ## torch export tensor
    torch.save(item2title_emb_json, os.path.join(path_to_meta_folder, "item2title_emb.pth"))
    torch.save(item2image_emb_json, os.path.join(path_to_meta_folder, "item2image_emb.pth"))

def load_item_embedding(path_folder):
    """
        item_image_embedding, dict ,key: original asin id, value: tensor
        item_title_embedding, dict ,key: original asin id, value: tensor
    """
    item_image_embedding_file = os.path.join(path_folder, "item2image_emb.pth")
    item_title_embedding_file = os.path.join(path_folder, "item2title_emb.pth")

    item_image_embedding_dic = torch.load(item_image_embedding_file)
    item_title_embedding_dic = torch.load(item_title_embedding_file)

    for item_id, img_emb in item_image_embedding_dic.items():
        print ("DEBUG: Item Image Embedding Item ID %s, Shape %s" % (item_id, str(img_emb.shape)))
        break
    for item_id, title_emb in item_title_embedding_dic.items():
        print ("DEBUG: Item Title Embedding Item ID %s, Shape %s" % (item_id, str(title_emb.shape)))
        break

def run_download_item_multimodal_ckpt():
    """
        # Change to ${your_data_path} to your local data_path
    """
    path_to_meta_folder = "${your_data_path}/SEMINAR/data/Amazon_movies_tv_5"
    path_to_meta_file = "${your_data_path}/SEMINAR/data/Amazon_movies_tv_5/meta_Movies_and_TV.json.gz"
    path_to_model_ckpt = "${your_data_path}/SEMINAR/download/"
    download_item_multimodal_ckpt(path_to_meta_file, path_to_meta_folder, path_to_model_ckpt)

def test_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "${your_data_path}/SEMINAR/download/ViT-B-32.pt"
    model, preprocess = clip.load(checkpoint_path, device=device)

    image_path = "${your_data_path}/SEMINAR/data/CLIP.png"
    ## shape: [1, 3, 224, 224]
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        print ("DEBUG: image_features shape: %s" % str(image_features.shape))
        print ("DEBUG: text_features shape: %s" % str(text_features.shape))

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

def init_item_multi_modal_embedding_tensor(item_size, emb_dim, embedding_dic):
    """
        init a tensor shape  [item_size, emb_dim]
        embedding_dic: key: int, iid. e.g. '1', '2'
    """
    item_embedding = np.zeros((item_size, emb_dim))
    for idx in range(item_size):
        key = idx
        pretrain_emb = embedding_dic[key] if key in embedding_dic else None
        if pretrain_emb is not None:
            item_embedding[idx] = pretrain_emb
    return item_embedding

def build_hnsw_index(multi_modal_emdbedding):
    """
        input shape: [L, M, D]
    """
    index_list = []
    shape_list = multi_modal_emdbedding.shape
    L, M, D = shape_list[0], shape_list[1], shape_list[2]
    full_dim = M * D
    for m in range(M):
        num_connection = 64  # number of connections each vertex will have
        ef_search = 32  # depth of layers explored during search
        ef_construction = 64  # depth of layers explored during index construction

        # initialize index (d == 128)
        index_hnsw = faiss.IndexHNSWFlat(D, num_connection)
        # set efConstruction and efSearch parameters
        index_hnsw.hnsw.efConstruction = ef_construction
        index_hnsw.hnsw.efSearch = ef_search
        # add data to index
        index_hnsw.add(multi_modal_emdbedding[:, m,:])
        index_list.append(index_hnsw)
    ## debug
    print ("DEBUG: build hnsw index index_list length is: %d" % len(index_list))
    return index_list

def normalize_l2(input_emb):
    """
        args: input_emb  [B, ..., D]
        normalize to last dimension
    """
    emb_dim = input_emb.shape[-1]
    shape_list_size = len(input_emb.shape)
    shape_list = [1] * (shape_list_size - 1) + [emb_dim]
    shape_list_tuple = tuple(shape_list)
    input_emb_norm_value = norm(input_emb, axis=-1, keepdims=-1)
    input_emb_norm = np.divide(input_emb, np.tile(input_emb_norm_value, reps=shape_list_tuple))
    return input_emb_norm

def build_cosine_similarity_index(user_emb_mat):
    """
        Normalize the input embedding before indexing
    """
    index_list = []
    shape_list = user_emb_mat.shape
    L, M, D = shape_list[0], shape_list[1], shape_list[2]
    full_dim = M * D
    # normalize the vector to unit vector
    user_emb_mat_norm = normalize_l2(user_emb_mat)
    for m in range(M):
        # initialize index (d == 128)
        # build inner product index
        index = faiss.IndexFlatIP(D)
        index.add(user_emb_mat_norm[:, m, :])  # add vectors to the index
        index_list.append(index)
    print ("DEBUG: build hnsw index index_list length is: %d" % len(index_list))
    return index_list

def run_text_image_retrieval():
    """
        Text-Image Retrieval on CLIP image embedding
    """
    path_to_meta_folder = "${your_data_path}/SEMINAR/data/Amazon_movies_tv_5"
    path_to_meta_file = "${your_data_path}/SEMINAR/data/Amazon_movies_tv_5/meta_Movies_and_TV.json.gz"

    # key: original item id
    item2title_emb_json = torch.load(os.path.join(path_to_meta_folder, "item2title_emb.pth"))
    item2image_emb_json = torch.load(os.path.join(path_to_meta_folder, "item2image_emb.pth"))
    id2title_emb_json, id2image_emb_json = {}, {}

    # Item Info Dict, key: item_id, value: dict {"title": , "url": ""}
    item_info_dict = {}
    total_cnt = 0
    id2itemid_map = {}
    itemid2id_map = {}
    matched_iid_list = []
    for review in parse(path_to_meta_file):
        total_cnt += 1
        if total_cnt % 1000 == 0:
            print("DEBUG: read_item_meta_info reading line cnt %d" % (total_cnt))
        # list
        category = review['category'] if 'category' in review else ''
        title = review['title'] if 'title' in review else ''
        item_id = review['asin'] if 'asin' in review else ''
        image_url_high_res = review['imageURLHighRes'][0] if 'imageURLHighRes' in review and len(
                review['imageURLHighRes']) >= 1 else ''
        item_info_dict[item_id] = review

        id = total_cnt-1
        id2itemid_map[total_cnt-1] = item_id
        itemid2id_map[item_id] = total_cnt-1
        if item_id in item2title_emb_json:
            matched_iid_list.append(id)
            id2title_emb_json[id] = item2title_emb_json[item_id]
        else:
            id2title_emb_json[id] = np.zeros((1, 512))
        id2image_emb_json[id] = item2image_emb_json[item_id] if item_id in item2image_emb_json else np.zeros((1, 512))

    # construct embedding
    item_size = total_cnt
    # Jackie Chan Case
    emb_dim = 512
    item_title_emb = init_item_multi_modal_embedding_tensor(item_size, emb_dim, id2title_emb_json)
    item_image_emb = init_item_multi_modal_embedding_tensor(item_size, emb_dim, id2image_emb_json)
    print ("DEBUG:item_title_emb shape %s" % str(item_title_emb))
    print ("DEBUG:item_image_emb shape %s" % str(item_image_emb))

    ## build index
    # max_item = 5000
    multi_modal_emdbedding = np.stack([item_title_emb, item_image_emb], axis=1)

    ## 截断Item [0:max_item]
    print ("DEBUG: multi_modal_emdbedding shape: %s" % str(multi_modal_emdbedding.shape))
    index_list = build_cosine_similarity_index(multi_modal_emdbedding)

    ## conduct image-text retrieval
    topK = 20
    query_id = 164931
    query_title = item_info_dict[id2itemid_map[query_id]]["title"]
    query_url = item_info_dict[id2itemid_map[query_id]]['imageURLHighRes'][0]
    print ("DEBUG: Target Item %d, Original ID %s, Title %s, URL %s------" % (query_id, id2itemid_map[query_id], query_title, query_url))

    # [M, D]
    query_vector_norm = normalize_l2(multi_modal_emdbedding[query_id])

    # query vectors expand to [1, D]
    query_vector_text = np.expand_dims(query_vector_norm[0], axis=0)
    query_vector_img = np.expand_dims(query_vector_norm[1], axis=0)

    t2t_topk_values, t2t_topk_idx = index_list[0].search(query_vector_text, topK)
    print ("--------------Text to Text Retrieval--------------")
    print ("Index|Original ID|Score|Title|URL")
    for value, idx in zip(t2t_topk_values.tolist()[0], t2t_topk_idx.tolist()[0]):
        origianl_id = id2itemid_map[idx]
        print ("%d|%s|%f|%s|%s" % (idx, origianl_id, value, item_info_dict[origianl_id]["title"] , item_info_dict[origianl_id]['imageURLHighRes'][0]))

    t2i_topk_values, t2i_topk_idx = index_list[1].search(query_vector_text, topK)
    print ("--------------Text to Image Retrieval--------------")
    print ("Index|Original ID|Score|Title|URL")
    for value, idx in zip(t2i_topk_values.tolist()[0], t2i_topk_idx.tolist()[0]):
        origianl_id = id2itemid_map[idx]
        print ("%d|%s|%f|%s|%s" % (idx, origianl_id, value, item_info_dict[origianl_id]["title"] , item_info_dict[origianl_id]['imageURLHighRes'][0]))

    i2t_topk_values, i2t_topk_idx = index_list[0].search(query_vector_img, topK)
    print ("--------------Image to Text Retrieval--------------")
    print ("Index|Original ID|Score|Title|URL")
    for value, idx in zip(i2t_topk_values.tolist()[0], i2t_topk_idx.tolist()[0]):
        origianl_id = id2itemid_map[idx]
        print ("%d|%s|%f|%s|%s" % (idx, origianl_id, value, item_info_dict[origianl_id]["title"] , item_info_dict[origianl_id]['imageURLHighRes'][0]))

    i2i_topk_values, i2i_topk_idx = index_list[1].search(query_vector_img, topK)
    print ("--------------Image to Image Retrieval--------------")
    print ("Index|Original ID|Score|Title|URL")
    for value, idx in zip(i2i_topk_values.tolist()[0], i2i_topk_idx.tolist()[0]):
        origianl_id = id2itemid_map[idx]
        print ("%d|%s|%f|%s|%s" % (idx, origianl_id, value, item_info_dict[origianl_id]["title"] , item_info_dict[origianl_id]['imageURLHighRes'][0]))

def sample_test_data(data_name, test_num=99, sample_type='random'):
    """
    sample_type:
        random:  sample `test_num` negative items randomly.
        pop: sample `test_num` negative items according to item popularity.
    """

    data_file = f'{data_name}.txt'
    test_file = f'{data_name}_sample.txt'

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

    user_neg_items = defaultdict()

    lineno = 0
    for user, user_seq in user_items.items():
        lineno+=1
        if lineno % 100 == 0:
            print ("DEBUG: Generating Random Sample for User %d" % lineno)
        test_samples = []
        while len(test_samples) < test_num:
            if sample_type == 'random':
                sample_ids = np.random.choice(all_item, test_num, replace=False)
            else: # sample_type == 'pop':
                sample_ids = np.random.choice(all_item, test_num, replace=False, p=probability)
            sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in test_samples]
            test_samples.extend(sample_ids)
        test_samples = test_samples[:test_num]
        user_neg_items[user] = test_samples

    print ("DEBUG: Total Line of Sample Generated %d" % len(user_neg_items))
    cnt = 0
    with open(test_file, 'w') as out:
        for user, samples in user_neg_items.items():
            cnt += 1
            if cnt % 1000 == 0:
                print ("DEBUG: Writing Line %d" % cnt)
            out.write(user+' '+' '.join(samples)+'\n')

# # data_names = ['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games', 'Yelp', 'LastFM']
# data_names = ['kuaisar']
# for data_name in data_names:
#     sample_test_data(data_name)

def main():
    run_download_item_multimodal_ckpt()
    # run_text_image_retrieval()

if __name__ == '__main__':
    main()
