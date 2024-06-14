# -*- coding: utf-8 -*-
# @Author  : Anonymous Period

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from numpy.linalg import norm
import faiss                   # make faiss available
import codecs

import pandas

import kmeans
import torch
from vector_quantize_pytorch import ResidualVQ, VectorQuantize

def build_hnsw(xb, dim, measure = faiss.METRIC_INNER_PRODUCT):
    """
        :param xb:  [B, D]
            dim: 64
            measure: faiss.METRIC_INNER_PRODUCT
        :return:
    """
    # measure = faiss.METRIC_L2
    param = 'HNSW64'
    index = faiss.index_factory(dim, param, measure)
    print(index.is_trained)
    index.add(xb)
    return index

def read_file(file_name):
    lines = []
    with codecs.open(file_name, encoding='utf-8') as f:
        for line in f:
            lines.append(line)
    return lines

def load_data_from_lines(lines):
    """
        lines : query_empty,query1,query_empty,query2,query2;item1,item2,item3,item4,item4 query2;item3
        :param lines:
        :return:
    """
    data = []
    for line in lines:
        items = line.split(" ")
        user_seq_id_str = items[0]
        target_item_pair_str = items[1]

        user_seq_id_items = user_seq_id_str.split(";")
        user_query_list = user_seq_id_items[0].split(",")
        user_item_id_list = user_seq_id_items[1].split(",")

        target_item_pairs = target_item_pair_str.split(";")
        target_query = target_item_pairs[0]
        target_item = target_item_pairs[1]
        #  (user_query_list, user_item_id_list, target_query, target_item) (list, list, id, id)
        data_group = (user_query_list, user_item_id_list, target_query, target_item)
        data.append(data_group)
    return data

def load_data(eval_file):
    # eval_file = "user_seq_eval.txt"
    lines = read_file(eval_file)
    data = []
    for line in lines:
        items = line.split(" ")
        user_seq_id_str = items[0]
        target_item_pair_str = items[1]

        user_seq_id_items = user_seq_id_str.split(";")
        user_query_list = user_seq_id_items[0].split(",")
        user_item_id_list = user_seq_id_items[1].split(",")

        target_item_pairs = target_item_pair_str.split(";")
        target_query = target_item_pairs[0]
        target_item = target_item_pairs[1]
        #  (user_query_list, user_item_id_list, target_query, target_item) (list, list, id, id)
        data_group = (user_query_list, user_item_id_list, target_query, target_item)
        data.append(data_group)
    return data

def transform_emb_to_array(emb_line):
    mat = []
    emb_line_new = emb_line.replace(":", ",")
    dim = 16
    emb_mat = np.array([float(d) for d in emb_line_new.split(",")]).reshape(-1, dim)
    emb_mat = np.squeeze(emb_mat)
    return emb_mat

def transform_emb_to_array_v2(emb_line):
    mat = []
    for sub in emb_line.split(":"):
        mat.append([float(d) for d in sub.split(",")])
    emb_mat = np.array(mat)
    # emb_vectors = np.array([float(w) for d in sub.split(";") for sub in emb_line.split(":")])
    # dim = 16
    # emb_mat = emb_vectors.reshape(-1, dim)
    return emb_mat

def load_embedding_data(eval_file):
    # eval_file = "user_seq_eval.txt"
    lines = read_file(eval_file)
    data = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        items = line.split("\t")

        target_query_emb = transform_emb_to_array(items[0])
        target_id_emb = transform_emb_to_array(items[1])
        target_text_emb = transform_emb_to_array(items[2])
        target_image_emb = transform_emb_to_array(items[3])

        user_query_emb = transform_emb_to_array(items[4])
        user_id_emb = transform_emb_to_array(items[5])
        user_text_emb = transform_emb_to_array(items[6])
        user_image_emb = transform_emb_to_array(items[7])

        #  (user_query_list, user_item_id_list, target_query, target_item) (list, list, id, id)
        data_group = (
            user_query_emb,user_id_emb,user_text_emb,user_image_emb, target_query_emb, target_id_emb,target_text_emb,target_image_emb
        )
        data.append(data_group)
    return data

def test_load_embedding_data():
    eval_file = "user_seq_eval_data_demo.txt"
    data = load_embedding_data_txt(eval_file)
    print (data)

def load_embedding_data_txt(eval_file):
    # eval_file = "user_seq_eval.txt"
    lines = read_file(eval_file)
    # 0, uuid
    # 1, label
    # 2,xlt_query_seq
    # 3,xlt_vst_code_seq
    # 4,seq_len
    # 5,query
    # 6,contentid
    # 7,attr_emb
    # 8,img_emb
    # 9,title_emb
    # 10,query_emb
    # 11,attr_emb_seq
    # 12,img_emb_seq
    # 13,title_emb_seq
    # 14,query_emb_seq
    data=[]
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        items = line.split("\t")
        target_query_emb = transform_emb_to_array(items[10])
        target_id_emb = transform_emb_to_array(items[7])
        target_text_emb = transform_emb_to_array(items[9])
        target_image_emb = transform_emb_to_array(items[8])

        user_query_emb = transform_emb_to_array(items[14])
        user_id_emb = transform_emb_to_array(items[11])
        user_text_emb = transform_emb_to_array(items[13])
        user_image_emb = transform_emb_to_array(items[12])

        #  (user_query_list, user_item_id_list, target_query, target_item) (list, list, id, id)
        data_group = (
            user_query_emb,user_id_emb,user_text_emb,user_image_emb, target_query_emb, target_id_emb,target_text_emb,target_image_emb
        )
        data.append(data_group)
    return data

# def read_data_from_odps(project_table_name):
#     """
#         :param table_name:
#         :return:
#     """
#     o = env_utils.get_odps_instance()
#     #reader = TableReader.from_ODPS_type(o, project_table_name, partition='1=1')
#     reader = TableReader.from_ODPS_type(o, project_table_name)
#     # 返回全量数据
#     data_df = reader.to_pandas()
#     return data_df

def load_embedding_from_lines(lines):
    """
        embedding: item_id \t f1,f2,...,fn
    """
    id_list = []
    emb_list = []
    embedding_dict = {}
    for line in lines:
        items = line.split(" ")
        k1 = items[0]
        k2 = items[1]
        digits = [float(d) for d in k2.split(",")]
        embedding_dict[k1] = np.array(digits)
        id_list.append(k1)
        emb_list.append(digits)
    emb_2d_array = np.array(emb_list)
    return embedding_dict, id_list, emb_2d_array


def load_embedding_table(input_file):
    """
        embedding: item_id \t f1,f2,...,fn
    """
    embedding_dict = {}
    lines = read_file(input_file)
    id_list = []
    emb_list = []
    for line in lines:
        items = line.split(" ")
        k1 = items[0]
        k2 = items[1]
        digits = [float(d) for d in k2.split(",")]
        embedding_dict[k1] = np.array(digits)
        id_list.append(k1)
        emb_list.append(digits)
    emb_2d_array = np.array(emb_list)
    return embedding_dict, id_list, emb_2d_array

def lookup_emb(id_list, emb_dict, ndim):
    emb_list = []
    for id in id_list:
        cur_emb = emb_dict[id] if id in emb_dict else np.zeros(ndim)
        emb_list.append(cur_emb)
    emb_array = np.array(emb_list)
    return emb_array

def top_K_idx(data, k):
    """
        data: 1-d array
    """
    data = np.array(np.squeeze(data))
    idx = data.argsort()[-k:][::-1]
    value = data[idx]
    return value, idx

def test_top_k():
    values = np.array([0.8, 0.1, 0.2, 0.5, 0.004])
    k = 3
    topk_values, topk_idx = top_K_idx(values, k)
    print (topk_values)
    print (topk_idx)

def full_attention(user_sequence_matrix, target_item_matrix, weight, topk):
    """
        target_item_matrix: [L, M, D]

        target_item_matrix: [1, M, D]
        weight [1, M]
    """
    # [L, D, M]
    user_sequence_trans = np.transpose(user_sequence_matrix, (0, 2, 1))
    # [M, 1]
    weight_trans = np.transpose(weight)
    # [L, D]
    #print ("DEBUG: user_sequence_trans shape %s" % str(user_sequence_trans.shape))
    #print ("DEBUG: weight_trans shape %s" % str(weight_trans.shape))
    user_sequence = np.matmul(user_sequence_trans, weight_trans)
    #print ("DEBUG: user_sequence shape %s" % str(user_sequence.shape))

    # [1, D]
    #print ("DEBUG: target_item_matrix shape %s" % str(target_item_matrix.shape))
    target_sequence = np.matmul(np.transpose(target_item_matrix, (1, 0)), weight_trans)
    #print ("DEBUG: target_sequence shape %s" % str(target_sequence.shape))

    attn_score = np.matmul(np.squeeze(user_sequence), target_sequence)
    #print ("DEBUG: attn_score shape is %s" % str(attn_score.shape))
    #print ("DEBUG: attn_score is %s" % str(attn_score))
    topk_values, topk_idx = top_K_idx(attn_score, topk)
    return topk_values, topk_idx

def build_hnsw_index(user_emb_mat):
    ## 保存M个模态index
    index_list = []
    ## 对齐
    shape_list = user_emb_mat.shape
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
        index_hnsw.add(user_emb_mat[:, m,:])
        index_list.append(index_hnsw)
    ## debug
    print ("DEBUG: build hnsw index index_list length is: %d" % len(index_list))
    return index_list

def build_cosine_index(user_emb_mat):
    ## 保存M个模态index
    index_list = []
    ## 对齐
    shape_list = user_emb_mat.shape
    L, M, D = shape_list[0], shape_list[1], shape_list[2]
    full_dim = M * D
    # normalize the vector to unit vector
    user_emb_mat_norm = np.divide(user_emb_mat, np.tile(norm(user_emb_mat, axis=-1, keepdims=-1), reps=(1, 1, D)))
    for m in range(M):
        # initialize index (d == 128)
        # build inner product index
        index = faiss.IndexFlatIP(D)
        print(index.is_trained)
        index.add(user_emb_mat_norm[:, m, :])  # add vectors to the index
        print(index.ntotal)
        index_list.append(index)
    ## debug
    print ("DEBUG: build hnsw index index_list length is: %d" % len(index_list))
    return index_list

def build_lsh_index(user_emb_mat):
    ## 保存M个模态index
    index_list = []
    ## 对齐
    shape_list = user_emb_mat.shape
    L, M, D = shape_list[0], shape_list[1], shape_list[2]
    full_dim = M * D
    for m in range(M):
        nbits = D * 4  # resolution of bucketed vectors
        # initialize index and add vectors
        index_lsh = faiss.IndexLSH(D, nbits)
        index_lsh.add(user_emb_mat[:, m,:])
        index_list.append(index_lsh)
    ## debug
    print ("DEBUG: build lsh index index_list length is: %d" % len(index_list))
    return index_list

def two_stage_ann_faiss_hnsw(user_emb_mat, target_emb, weight, topK, index_list):
    """
        :param user_emb_mat: [L, M, D]
        :param target_emb: [1, M, D]
        :param weight:  [M]
        :param topk:
        :param index_list:  list of hnsw index
            index, k1: user_id, k2: mode_id, value: L array
        :return:
    """
    num_mode = user_emb_mat.shape[1]
    print ("DEBUG: two_stage_ann_faiss_hnsw input num_mode is %d" % num_mode)
    topk_list = []
    ## Stage1, 
    for i in range(num_mode):
        # query的模态
        q = np.expand_dims(target_emb[i, :], axis=0)
        for j in range(num_mode):
            # q: # [1, D], K: [L, D]
            # FAISS ANN search
            #print ("DEBUG: two_stage_ann_faiss_hnsw q shape %s" % str(q.shape))
            topk_values, topk_idx = index_list[j].search(q, topK)
            #print ("DEBUG: topk_idx.tolist()" )
            #print (topk_idx.tolist())
            #topk_values, topk_idx = top_K_idx(relevance_score, topK)
            topk_list.extend(topk_idx.tolist()[0])
    ## Stage2
    topk_list_unique = list(set(list(topk_list)))
    # print ("DEBUG: Generated topk_list size %d, topk_list_unique size %d" % (len(topk_list), len(topk_list_unique)))

    q = target_emb
    # len: len(topk_list_unique)
    sim_score_list = []
    for id in topk_list_unique:
        ## [M, D]
        k_l = user_emb_mat[id, :, :]
        q_merge = np.matmul(weight, q)       # [1, D]
        k_l_merge = np.matmul(weight, k_l)   # [1, D]
        sim_score = np.sum(np.multiply(q_merge, k_l_merge))
        # mode_sim = np.sum(np.multiply(q, k_l), axis=1)
        # sim_score = np.sum(np.multiply(mode_sim, weight))
        # sim_score = np.sum(np.multiply(mode_sim, weight))
        sim_score_list.append(sim_score)
    final_topk_values, final_topk_idx = top_K_idx(np.array(sim_score_list), topK)
    final_topk_idx_orig = [topk_list_unique[idx] for idx in final_topk_idx]
    return final_topk_values, final_topk_idx_orig

def two_stage_ann_faiss_lsh(user_emb_mat, target_emb, weight, topK, index_list):
    """
        :param user_emb_mat: [L, M, D]
        :param target_emb: [1, M, D]
        :param weight:  [M]
        :param topk:
        :param index_list:  list of hnsw index
             k1: user_id, k2: mode_id, value: L array
        :return:
    """
    num_mode = user_emb_mat.shape[1]
    print ("DEBUG: two_stage_ann_faiss_hnsw input num_mode is %d" % num_mode)
    topk_list = []
    ## Stage1, 
    for i in range(num_mode):
        # 
        q = np.expand_dims(target_emb[i, :], axis=0)
        for j in range(num_mode):
            # q: # [1, D], K: [L, D]
            # FAISS ANN search
            #print ("DEBUG: two_stage_ann_faiss_hnsw q shape %s" % str(q.shape))
            topk_values, topk_idx = index_list[j].search(q, topK)
            #print ("DEBUG: topk_idx.tolist()" )
            #print (topk_idx.tolist())
            #topk_values, topk_idx = top_K_idx(relevance_score, topK)
            topk_list.extend(topk_idx.tolist()[0])
    ## Stage2
    topk_list_unique = list(set(list(topk_list)))
    # print ("DEBUG: Generated topk_list size %d, topk_list_unique size %d" % (len(topk_list), len(topk_list_unique)))

    q = target_emb
    # len: len(topk_list_unique)
    sim_score_list = []
    for id in topk_list_unique:
        ## [M, D]
        k_l = user_emb_mat[id, :, :]
        q_merge = np.matmul(weight, q)       # [1, D]
        k_l_merge = np.matmul(weight, k_l)   # [1, D]
        sim_score = np.sum(np.multiply(q_merge, k_l_merge))
        # mode_sim = np.sum(np.multiply(q, k_l), axis=1)
        # sim_score = np.sum(np.multiply(mode_sim, weight))
        # sim_score = np.sum(np.multiply(mode_sim, weight))
        sim_score_list.append(sim_score)
    final_topk_values, final_topk_idx = top_K_idx(np.array(sim_score_list), topK)
    final_topk_idx_orig = [topk_list_unique[idx] for idx in final_topk_idx]
    return final_topk_values, final_topk_idx_orig



def two_stage_ann(user_emb_mat, target_emb, weight, topK,
                  index_list):
    """
        :param user_emb_mat: [L, M, D]
        :param target_emb: [1, M, D]
        :param weight:  [M]
        :param topk:
        :param index_list:  list of hnsw index
            k1: user_id, k2: mode_id, value: L array
        :return:
    """
    num_mode = user_emb_mat.shape[1]
    topk_list = []
    ## Stage1,
    for i in range(num_mode):
        for j in range(num_mode):
            # q: # [1, D], K: [L, D]
            q = target_emb[i, :]
            k_l = user_emb_mat[:, j, :]
            relevance_score = np.matmul(k_l, np.transpose(q))
            topk_values, topk_idx = top_K_idx(relevance_score, topK)
            topk_list.extend(list(topk_idx))

    ## Stage2
    topk_list_unique = list(set(list(topk_list)))
    # print ("DEBUG: Generated topk_list size %d, topk_list_unique size %d" % (len(topk_list), len(topk_list_unique)))


    q = target_emb
    # len: len(topk_list_unique)
    sim_score_list = []
    for id in topk_list_unique:
        ## [M, D]
        k_l = user_emb_mat[id, :, :]
        q_merge = np.matmul(weight, q)       # [1, D]
        k_l_merge = np.matmul(weight, k_l)   # [1, D]
        sim_score = np.sum(np.multiply(q_merge, k_l_merge))
        # mode_sim = np.sum(np.multiply(q, k_l), axis=1)
        # sim_score = np.sum(np.multiply(mode_sim, weight))
        # sim_score = np.sum(np.multiply(mode_sim, weight))
        sim_score_list.append(sim_score)
    final_topk_values, final_topk_idx = top_K_idx(np.array(sim_score_list), topK)
    final_topk_idx_orig = [topk_list_unique[idx] for idx in final_topk_idx]
    return final_topk_values, final_topk_idx_orig

def two_stage_maxsim(user_emb_mat, target_emb, weight, topK,
                  index_list):
    """
        :param user_emb_mat: [L, M, D]
        :param target_emb: [1, M, D]
        :param weight:  [M]
        :param topk:
        :param index_list:  list of hnsw index
            index, k1: user_id, k2: mode_id, value: L array
        :return:
    """
    num_mode = user_emb_mat.shape[1]
    topk_list = []
    ## Stage1, ANN
    for i in range(num_mode):
        for j in range(num_mode):
            # q: # [1, D], K: [L, D]
            q = target_emb[i, :]
            k_l = user_emb_mat[:, j, :]
            relevance_score = np.matmul(k_l, np.transpose(q))
            topk_values, topk_idx = top_K_idx(relevance_score, topK)
            topk_list.extend(list(topk_idx))

    ## Stage2
    topk_list_unique = list(set(list(topk_list)))
    # print ("DEBUG: Generated topk_list size %d, topk_list_unique size %d" % (len(topk_list), len(topk_list_unique)))

    q = target_emb
    # len: len(topk_list_unique)
    sim_score_list = []
    for id in topk_list_unique:
        ## [M, D]
        k_l = user_emb_mat[id, :, :]
        ## MaxSim
        weight_sim = np.multiply(np.matmul(q, np.transpose(k_l)), weight)
        max_weight_sim = np.max(weight_sim, axis = 1)
        sim_score = np.sum(np.multiply(max_weight_sim, weight))
        # sim_score = np.sum(np.multiply(mode_sim, weight))
        sim_score_list.append(sim_score)
    final_topk_values, final_topk_idx = top_K_idx(np.array(sim_score_list), topK)
    final_topk_idx_orig = [topk_list_unique[idx] for idx in final_topk_idx]
    return final_topk_values, final_topk_idx_orig

def two_stage_maxmax_v1(user_emb_mat, target_emb, weight, topK,
                  index_list):
    """
        :param user_emb_mat: [L, M, D]
        :param target_emb: [1, M, D]
        :param weight:  [M]
        :param topk:
        :param index_list:  list of hnsw index
             k1: user_id, k2: mode_id, value: L array
        :return:

        q belongs to M^2   topK Item
    """
    num_mode = user_emb_mat.shape[1]
    topk_list = []
    ## Stage1
    for i in range(num_mode):
        for j in range(num_mode):
            # q: # [1, D], K: [L, D]
            q = target_emb[i, :]
            k_l = user_emb_mat[:, j, :]
            relevance_score = np.matmul(k_l, np.transpose(q))
            topk_values, topk_idx = top_K_idx(relevance_score, topK)
            topk_list.extend(list(topk_idx))

    ## Stage2
    topk_list_unique = list(set(list(topk_list)))
    # print ("DEBUG: Generated topk_list size %d, topk_list_unique size %d" % (len(topk_list), len(topk_list_unique)))

    q = target_emb
    # len: len(topk_list_unique)
    sim_score_list = []
    for id in topk_list_unique:
        ## [M, D]
        k_l = user_emb_mat[id, :, :]
        ## MAX Sim
        weight_sim = np.multiply(np.matmul(q, np.transpose(k_l)), weight)
        max_weight_sim = np.max(weight_sim, axis = 1)
        sim_score = np.max(np.multiply(max_weight_sim, weight))
        # sim_score = np.sum(np.multiply(mode_sim, weight))
        sim_score_list.append(sim_score)
    final_topk_values, final_topk_idx = top_K_idx(np.array(sim_score_list), topK)
    final_topk_idx_orig = [topk_list_unique[idx] for idx in final_topk_idx]
    return final_topk_values, final_topk_idx_orig


def one_stage_pq_distance(user_emb_mat_pq, target_emb_pq, weight, topK, centroids_dist):
    """

        :param user_emb_mat_pq:  int matrix, shape [L, M]
        :param target_emb_pq:    int matrix, shape [M]
        :param weight:        double matrix [1, M]
        :param topK:          int
        :param centroids_dist:   [K, K]
        :return:
    """
    # print ("DEBUG: one_stage_pq_distance user_emb_mat_pq is %s" % str(user_emb_mat_pq[0:2]))
    # print ("DEBUG: one_stage_pq_distance target_emb_pq is %s" % str(target_emb_pq))
    ##
    L, M = user_emb_mat_pq.shape[0], user_emb_mat_pq.shape[1]
    attn_score = []
    for l in range(L):
        q_codes = target_emb_pq[0, :].tolist() # [M]
        k_l_codes = user_emb_mat_pq[l,:].tolist() #[M]
        # print ("DEBUG: q_codes shape is %s value is %s" % (len(q_codes), str(q_codes)))
        # print ("DEBUG: k_l_codes shape is %s and value is %s" % (len(k_l_codes) ,str(k_l_codes)))
        total_dist = 0.0
        for i in range(M):
            for j in range(M):
                pair_dist = centroids_dist[q_codes[i], k_l_codes[j]]
                total_dist += (weight[0, i] * weight[0, j] * pair_dist)
                # print("DEBUG: one_stage_pq_distance pq distance at index (%d, %d) is %f" % (q_codes[i], k_l_codes[j], pair_dist))
        attn_score.append(total_dist)
    print ("DEBUG: one_stage_pq_distance attn_score is %s" % str(attn_score))
    ## 排序
    topk_values, topk_idx = top_K_idx(attn_score, topK)
    return topk_values, topk_idx



def one_stage_pq_supervised_distance_faiss_multi(user_emb_mat_pq, target_emb_pq, weight, topK, codebook_emb):
    """
        # output
            codebook_emb                         [M, Code_Size, K, subD]
            dataSetAssignment/user_emb_mat_pq    [M, L, Code_Size]
            testSetAssignment                    [M, 1, Code_Size]

            DEBUG: multi_modal_dataset shape: (4, 1028, 16)
            DEBUG: multi_modal_testset shape: (4, 1, 16)
            DEBUG: multi_modal_codebook_emb shape: (4, 16, 256, 2)

        :return:
    """
    # print ("DEBUG: one_stage_pq_supervised_distance user_emb_mat_pq shape is %s, value is %s" % (str(user_emb_mat_pq.shape), str(user_emb_mat_pq[0:2])))
    # print ("DEBUG: one_stage_pq_supervised_distance target_emb_pq shape is %s|value is %s" % (str(target_emb_pq.shape), str(target_emb_pq)))
    ## 构建长度为L的List
    # M = 4
    M, L, code_size = user_emb_mat_pq.shape[0], user_emb_mat_pq.shape[1], user_emb_mat_pq.shape[2]
    attn_score = []
    for l in range(L):
        q_codes = target_emb_pq[:, 0, :]
        k_l_codes = user_emb_mat_pq[:, l, :]
        # print ("DEBUG: q_codes shape is %s value is %s" % (len(q_codes), str(q_codes)))
        # print ("DEBUG: k_l_codes shape is %s and value is %s" % (len(k_l_codes) ,str(k_l_codes)))
        total_dist = 0.0
        for i in range(M):
            for j in range(M):
                # [code_size]
                cur_q_codes = q_codes[i]
                # [code_size]
                cur_k_l_codes = k_l_codes[j]
                pair_dist = 0.0
                code_size = len(cur_q_codes)
                for c in range(code_size):
                    e_1 = codebook_emb[i, c, cur_q_codes[c]]
                    e_2 = codebook_emb[j, c, cur_k_l_codes[c]]
                    pair_dist += kmeans.innerProductDistance(e_1, e_2)
                # pair_dist = kmeans.innerProductDistance(codebook_emb[i, int(q_codes[i])], codebook_emb[j, int(k_l_codes[j])])
                # total_dist += pair_dist
                total_dist += (weight[0, i] * weight[0, j] * pair_dist)
                # print("DEBUG: one_stage_pq_distance pq distance at index (%d, %d) is %f" % (q_codes[i], k_l_codes[j], pair_dist))
        attn_score.append(total_dist)
    # print ("DEBUG: one_stage_pq_supervised_distance_faiss_multi attn_score is %s" % str(attn_score))
    ## 排序
    topk_values, topk_idx = top_K_idx(attn_score, topK)
    return topk_values, topk_idx


def one_stage_pq_supervised_distance_decoder_quantized_multi(user_emb_mat_pq, target_emb_pq, weight, topK, codebook_emb,
                                                            user_emb_quantized, target_emb_quantized):
    """
        ## input has the original data and quantized dataset, encoder
        # output
            codebook_emb                         [M, Code_Size, K, subD]
            dataSetAssignment/user_emb_mat_pq    [M, L, Code_Size]
            testSetAssignment                    [M, 1, Code_Size]

            user_emb_mat_pq  [M, L, D]
            target_emb_pq    [M, L, D]
            user_emb_quantized   [M, L, D]
            target_emb_quantized  [M, 1, D]

            DEBUG: multi_modal_dataset shape: (4, 1028, 16)
            DEBUG: multi_modal_testset shape: (4, 1, 16)
            DEBUG: multi_modal_codebook_emb shape: (4, 16, 256, 2)

        :return:
    """
    # print ("DEBUG: one_stage_pq_supervised_distance user_emb_mat_pq shape is %s, value is %s" % (str(user_emb_mat_pq.shape), str(user_emb_mat_pq[0:2])))
    # print ("DEBUG: one_stage_pq_supervised_distance target_emb_pq shape is %s|value is %s" % (str(target_emb_pq.shape), str(target_emb_pq)))
    ## 构建长度为L的List
    # M = 4
    M, L, code_size = user_emb_mat_pq.shape[0], user_emb_mat_pq.shape[1], user_emb_mat_pq.shape[2]
    attn_score = []
    for l in range(L):
        total_dist = 0.0
        for i in range(M):
            query_quantized = target_emb_quantized[i]
            for j in range(M):
                # [code_size]
                key_quantized = user_emb_quantized[j, l]
                pair_dist = kmeans.innerProductDistance(query_quantized, key_quantized)
                total_dist += (weight[0, i] * weight[0, j] * pair_dist)
                # print("DEBUG: one_stage_pq_distance pq distance at index (%d, %d) is %f" % (q_codes[i], k_l_codes[j], pair_dist))
        attn_score.append(total_dist)
    # print ("DEBUG: one_stage_pq_supervised_distance_faiss_multi attn_score is %s" % str(attn_score))
    ## 排序
    topk_values, topk_idx = top_K_idx(attn_score, topK)
    return topk_values, topk_idx

def one_stage_pq_supervised_distance_faiss(user_emb_mat_pq, target_emb_pq, weight, topK, codebook_emb):
    """
        # output
            codebook_emb                         [M, K, D]
            dataSetAssignment/user_emb_mat_pq    [L, M]
            testSetAssignment                    [M]
        :return:
    """
    print ("DEBUG: one_stage_pq_supervised_distance user_emb_mat_pq shape is %s, value is %s" % (str(user_emb_mat_pq.shape), str(user_emb_mat_pq[0:2])))
    print ("DEBUG: one_stage_pq_supervised_distance target_emb_pq shape is %s|value is %s" % (str(target_emb_pq.shape), str(target_emb_pq)))
    M = 4
    L, code_size = user_emb_mat_pq.shape[0], user_emb_mat_pq.shape[1]
    attn_score = []
    for l in range(L):
        q_codes = target_emb_pq[0, :].tolist() # [M]
        k_l_codes = user_emb_mat_pq[l,:].tolist() #[M]
        # print ("DEBUG: q_codes shape is %s value is %s" % (len(q_codes), str(q_codes)))
        # print ("DEBUG: k_l_codes shape is %s and value is %s" % (len(k_l_codes) ,str(k_l_codes)))
        total_dist = 0.0
        for i in range(M):
            for j in range(M):
                pair_dist = kmeans.innerProductDistance(codebook_emb[i, int(q_codes[i])], codebook_emb[j, int(k_l_codes[j])])
                # total_dist += pair_dist
                total_dist += (weight[0, i] * weight[0, j] * pair_dist)
                # print("DEBUG: one_stage_pq_distance pq distance at index (%d, %d) is %f" % (q_codes[i], k_l_codes[j], pair_dist))
        attn_score.append(total_dist)
    # print ("DEBUG: one_stage_pq_supervised_distance attn_score is %s" % str(attn_score))
    ## 排序
    topk_values, topk_idx = top_K_idx(attn_score, topK)
    return topk_values, topk_idx


def one_stage_pq_supervised_distance(user_emb_mat_pq, target_emb_pq, weight, topK, codebook_emb):
    """
        # output
            codebook_emb                         [M, K, D]
            dataSetAssignment/user_emb_mat_pq    [L, M]
            testSetAssignment                    [M]
        :return:
    """
    print ("DEBUG: one_stage_pq_supervised_distance user_emb_mat_pq shape is %s, value is %s" % (str(user_emb_mat_pq.shape), str(user_emb_mat_pq[0:2])))
    print ("DEBUG: one_stage_pq_supervised_distance target_emb_pq shape is %s|value is %s" % (str(target_emb_pq.shape), str(target_emb_pq)))
    ## 构建长度为L的List
    L, M = user_emb_mat_pq.shape[0], user_emb_mat_pq.shape[1]
    attn_score = []
    for l in range(L):
        q_codes = target_emb_pq[0, :].tolist() # [M]
        k_l_codes = user_emb_mat_pq[l,:].tolist() #[M]
        # print ("DEBUG: q_codes shape is %s value is %s" % (len(q_codes), str(q_codes)))
        # print ("DEBUG: k_l_codes shape is %s and value is %s" % (len(k_l_codes) ,str(k_l_codes)))
        total_dist = 0.0
        for i in range(M):
            for j in range(M):
                pair_dist = kmeans.innerProductDistance(codebook_emb[i, int(q_codes[i])], codebook_emb[j, int(k_l_codes[j])])
                # total_dist += pair_dist
                total_dist += (weight[0, i] * weight[0, j] * pair_dist)
                # print("DEBUG: one_stage_pq_distance pq distance at index (%d, %d) is %f" % (q_codes[i], k_l_codes[j], pair_dist))
        attn_score.append(total_dist)
    print ("DEBUG: one_stage_pq_supervised_distance attn_score is %s" % str(attn_score))
    ## 排序
    topk_values, topk_idx = top_K_idx(attn_score, topK)
    return topk_values, topk_idx

def calc_recall_k_rate(truth, prediction):
    """
        :param truth:
        :param prediction:
        :return:
    """
    match_list = [1 if p in truth else 0 for p in prediction]
    hit_rate = sum(match_list)/len(match_list)
    return hit_rate


def normalize_tensor(v):
    """
        Tensor
    """
    if len(v.shape) == 1:
        normalized_v = np.linalg.norm(v)
        v_norm = v/normalized_v
        return v_norm
    else:
        shape_len = len(v.shape)
        normalized_v = np.linalg.norm(v, axis=shape_len-1)
        normalized_v = np.expand_dims(normalized_v, axis=1)
        v_norm = np.divide(v, normalized_v)
        return v_norm

def get_centroid(embedding_list, center_cnt, metric):
    """
        embedding_list list of numpy array [B, D], Batch Size varies
    """
    samples = np.concatenate(embedding_list, axis=0)
    # print ("DEBUG: kmeans get_centroid shape %s" % str(samples.shape))
    if metric == "eucledian":
        centroids, clusterAssment = kmeans.kmeans(samples, center_cnt)
        cluster_id_cnt_map = kmeans.count_cluster_result(clusterAssment)
    elif metric == "inner_product":
        centroids, clusterAssment = kmeans.kmeans_max_inner_product(samples, center_cnt)
        cluster_id_cnt_map = kmeans.count_cluster_result(clusterAssment)
    else:
        centroids, clusterAssment = kmeans.kmeans(samples, center_cnt)
        cluster_id_cnt_map = kmeans.count_cluster_result(clusterAssment)
    # print ("DEBUG: ------centroids------")
    # print (centroids.shape)
    # print ("DEBUG: ------cluster_id_cnt_map------")
    # print (cluster_id_cnt_map)
    return centroids


def cluster_supervised_multi_modal_faiss(embedding_list, query_embedding_list, center_cnt, inner_product_loss_enable = False, dist_func = "euclDistance"):
    """
        embedding_list: list size M, [L, D]
        query_embedding_list: list size M, [1, D]
        # output
            codebook_emb         [M, K, D]
            dataSetAssignment    [L, M]
            testSetAssignment    [M]
    """

    # hyperparameters
    code_size = 16
    nbits = 8

    ## [L, M, D]
    data_samples = np.stack(embedding_list, axis=1)
    print ("DEBUG: data_samples shape %s" % str(data_samples.shape))

    ## [M, D]
    test_query_samples = np.squeeze(np.stack(query_embedding_list, axis=1))
    print ("DEBUG: test_query_samples shape %s" % str(test_query_samples.shape))

    ## 对齐
    shape_list = data_samples.shape
    L, M, D = shape_list[0], shape_list[1], shape_list[2]
    # full_dim = M * D
    # data_samples_reshape = np.reshape(data_samples, [L, full_dim])
    # test_query_samples_reshape = np.reshape(test_query_samples, [1, full_dim])
    # data_merge = np.concatenate([data_samples_reshape, test_query_samples_reshape], axis=0)

    multi_modal_dataset_list = []
    multi_modal_testset_list = []
    multi_modal_codebook_list = []
    for i in range(M):
        # [L+1, D]
        cur_mode_data = np.concatenate([data_samples[:, i, :], np.expand_dims(test_query_samples[i, :], axis=0)], axis=0)
        index_pq = faiss.ProductQuantizer(D, code_size, nbits)
        index_pq.train(cur_mode_data)
        # index_pq = faiss.IndexPQ(full_dim, M, nbits)
        # index_pq.train(data_merge)
        # pq bits string of ceil(nbit * M / 8)
        dataSetAssignment = index_pq.compute_codes(data_samples[:, i, :])
        testSetAssignment = index_pq.compute_codes( np.expand_dims(test_query_samples[i, :], axis=0))
        codebook_emb = faiss.vector_to_array(index_pq.centroids).reshape(index_pq.M, index_pq.ksub, index_pq.dsub)

        # [L, D/m, 1]
        # print ("DEBUG: cluster_supervised_multi_modal_faiss dataSetAssignment shape: %s" % str(dataSetAssignment.shape))
        # # [1, D/m, 1]
        # print ("DEBUG: cluster_supervised_multi_modal_faiss testSetAssignment shape: %s" % str(testSetAssignment.shape))
        # print ("DEBUG: cluster_supervised_multi_modal_faiss codebook_emb shape: %s" % str(codebook_emb.shape))

        multi_modal_dataset_list.append(dataSetAssignment)
        multi_modal_testset_list.append(testSetAssignment)
        multi_modal_codebook_list.append(codebook_emb)
    ## 返回的Shape
    multi_modal_dataset = np.stack(multi_modal_dataset_list, axis=0)
    multi_modal_testset = np.stack(multi_modal_testset_list, axis=0)
    multi_modal_codebook_emb = np.stack(multi_modal_codebook_list, axis=0)
    print("DEBUG: multi_modal_dataset shape: %s" % str(multi_modal_dataset.shape))
    print("DEBUG: multi_modal_testset shape: %s" % str(multi_modal_testset.shape))
    print("DEBUG: multi_modal_codebook_emb shape: %s" % str(multi_modal_codebook_emb.shape))
    return multi_modal_codebook_emb, multi_modal_dataset, multi_modal_testset


def cluster_rq_vae(embedding_list, query_embedding_list, center_cnt, inner_product_loss_enable = False, dist_func = "euclDistance"):
    """
        embedding_list: list size M, [L, D]
        query_embedding_list: list size M, [1, D]
        # output
            codebook_emb         [M, K, D]
            dataSetAssignment    [L, M]
            testSetAssignment    [M]
    """
    ## [L, M, D]
    data_samples = np.stack(embedding_list, axis=1)
    print ("DEBUG: data_samples shape %s" % str(data_samples.shape))

    ## [M, D]
    test_query_samples = np.squeeze(np.stack(query_embedding_list, axis=1))
    print ("DEBUG: test_query_samples shape %s" % str(test_query_samples.shape))

    ## 对齐
    shape_list = data_samples.shape
    L, M, D = shape_list[0], shape_list[1], shape_list[2]
    # full_dim = M * D
    # data_samples_reshape = np.reshape(data_samples, [L, full_dim])
    # test_query_samples_reshape = np.reshape(test_query_samples, [1, full_dim])
    # data_merge = np.concatenate([data_samples_reshape, test_query_samples_reshape], axis=0)

    multi_modal_dataset_list = []
    multi_modal_testset_list = []
    multi_modal_codebook_list = []

    ## 保存原始的 quantize的向量
    multi_modal_dataset_quantized_list=[]
    multi_modal_testset_quantized_list=[]

    code_size = 32
    nbits = 8
    codebook_size = 1024

    for i in range(M):
        # [L+1, D]
        x = torch.from_numpy(np.concatenate([data_samples[:, i, :], np.expand_dims(test_query_samples[i, :], axis=0)], axis=0))
        residual_vq = ResidualVQ(
            dim=D,
            num_quantizers=code_size,  # specify number of quantizers
            codebook_size=codebook_size,  # codebook size
        )
        quantized, indices, commit_loss, all_codes = residual_vq(x, return_all_codes = True)
        x_reconstruct = torch.sum(all_codes, dim=0)

        # pq bits string of ceil(nbit * M / 8)
        # dataSetAssignment = x_reconstruct[0:x.shape[0]-1].numpy()
        # testSetAssignment = np.expand_dims(x_reconstruct[x.shape[0]-1].numpy(), axis=0)

        dataSetAssignment = indices[0:x.shape[0]-1].numpy()
        testSetAssignment = np.expand_dims(indices[x.shape[0]-1].numpy(), axis=0)
        codebook_emb = residual_vq.codebooks.numpy()
        dataset_quantized = quantized[0:quantized.shape[0]-1].numpy()
        testset_quantized = np.expand_dims(quantized[quantized.shape[0]-1].numpy(), axis=0)

        # [L, D/m, 1]
        # print ("DEBUG: cluster_supervised_multi_modal_faiss dataSetAssignment shape: %s" % str(dataSetAssignment.shape))
        # # [1, D/m, 1]
        # print ("DEBUG: cluster_supervised_multi_modal_faiss testSetAssignment shape: %s" % str(testSetAssignment.shape))
        # print ("DEBUG: cluster_supervised_multi_modal_faiss codebook_emb shape: %s" % str(codebook_emb.shape))

        multi_modal_dataset_list.append(dataSetAssignment)
        multi_modal_testset_list.append(testSetAssignment)
        multi_modal_codebook_list.append(codebook_emb)
        multi_modal_dataset_quantized_list.append(dataset_quantized)
        multi_modal_testset_quantized_list.append(testset_quantized)

    ## 返回的Shape
    multi_modal_dataset = np.stack(multi_modal_dataset_list, axis=0)
    multi_modal_testset = np.stack(multi_modal_testset_list, axis=0)
    multi_modal_codebook_emb = np.stack(multi_modal_codebook_list, axis=0)
    multi_modal_dataset_quantized = np.stack(multi_modal_dataset_quantized_list, axis=0)
    multi_modal_testset_quantized = np.stack(multi_modal_testset_quantized_list, axis=0)

    print("DEBUG: cluster_rq_vae multi_modal_dataset shape: %s" % str(multi_modal_dataset.shape))
    print("DEBUG: cluster_rq_vae multi_modal_testset shape: %s" % str(multi_modal_testset.shape))
    print("DEBUG: cluster_rq_vae multi_modal_codebook_emb shape: %s" % str(multi_modal_codebook_emb.shape))
    print("DEBUG: cluster_vq_vae multi_modal_dataset_quantized shape: %s" % str(multi_modal_dataset_quantized.shape))
    print("DEBUG: cluster_vq_vae multi_modal_testset_quantized shape: %s" % str(multi_modal_testset_quantized.shape))

    return multi_modal_codebook_emb, multi_modal_dataset, multi_modal_testset, multi_modal_dataset_quantized, multi_modal_testset_quantized


def cluster_vq_vae(embedding_list, query_embedding_list, center_cnt, inner_product_loss_enable = False, dist_func = "euclDistance"):
    """
        VQ-VAE
        embedding_list: list size M, [L, D]
        query_embedding_list: list size M, [1, D]
        # output
            codebook_emb         [M, K, D]
            dataSetAssignment    [L, M]
            testSetAssignment    [M]
    """
    ## [L, M, D]
    data_samples = np.stack(embedding_list, axis=1)
    print ("DEBUG: data_samples shape %s" % str(data_samples.shape))

    ## [M, D]
    test_query_samples = np.squeeze(np.stack(query_embedding_list, axis=1))
    print ("DEBUG: test_query_samples shape %s" % str(test_query_samples.shape))

    ## 对齐
    shape_list = data_samples.shape
    L, M, D = shape_list[0], shape_list[1], shape_list[2]
    # full_dim = M * D
    # data_samples_reshape = np.reshape(data_samples, [L, full_dim])
    # test_query_samples_reshape = np.reshape(test_query_samples, [1, full_dim])
    # data_merge = np.concatenate([data_samples_reshape, test_query_samples_reshape], axis=0)

    multi_modal_dataset_list = []
    multi_modal_testset_list = []
    multi_modal_codebook_list = []

    ## 保存原始的 quantize的向量
    multi_modal_dataset_quantized_list=[]
    multi_modal_testset_quantized_list=[]

    code_size = 16
    nbits = 8
    codebook_size = 1024

    for i in range(M):
        # [L+1, D]
        x = torch.from_numpy(np.concatenate([data_samples[:, i, :], np.expand_dims(test_query_samples[i, :], axis=0)], axis=0))
        # residual_vq = ResidualVQ(
        #     dim=D,
        #     num_quantizers=code_size,  # specify number of quantizers
        #     codebook_size=codebook_size,  # codebook size
        # )
        vq = VectorQuantize(
            dim=D,
            codebook_size=codebook_size,  # codebook size
            decay=0.8,  # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=1.  # the weight on the commitment loss
        )

        quantized, indices, commit_loss = vq(x)

        # pq bits string of ceil(nbit * M / 8)
        # dataSetAssignment = x_reconstruct[0:x.shape[0]-1].numpy()
        # testSetAssignment = np.expand_dims(x_reconstruct[x.shape[0]-1].numpy(), axis=0)

        dataSetAssignment = np.expand_dims(indices[0:x.shape[0]-1].numpy(), axis=-1)
        testSetAssignment = np.expand_dims(np.expand_dims(indices[x.shape[0]-1].numpy(), axis=0), axis=0)
        codebook_emb = np.expand_dims(vq.codebook.numpy(), axis=0)

        dataset_quantized = quantized[0:quantized.shape[0]-1].numpy()
        testset_quantized = np.expand_dims(quantized[quantized.shape[0]-1].numpy(), axis=0)

        # [L, D/m, 1]
        # print ("DEBUG: cluster_supervised_multi_modal_faiss dataSetAssignment shape: %s" % str(dataSetAssignment.shape))
        # [1, D/m, 1]
        # print ("DEBUG: cluster_supervised_multi_modal_faiss testSetAssignment shape: %s" % str(testSetAssignment.shape))
        # print ("DEBUG: cluster_supervised_multi_modal_faiss codebook_emb shape: %s" % str(codebook_emb.shape))
        multi_modal_dataset_list.append(dataSetAssignment)
        multi_modal_testset_list.append(testSetAssignment)
        multi_modal_codebook_list.append(codebook_emb)

        multi_modal_dataset_quantized_list.append(dataset_quantized)
        multi_modal_testset_quantized_list.append(testset_quantized)

    ## 返回的Shape
    multi_modal_dataset = np.stack(multi_modal_dataset_list, axis=0)
    multi_modal_testset = np.stack(multi_modal_testset_list, axis=0)
    multi_modal_codebook_emb = np.stack(multi_modal_codebook_list, axis=0)

    multi_modal_dataset_quantized = np.stack(multi_modal_dataset_quantized_list, axis=0)
    multi_modal_testset_quantized = np.stack(multi_modal_testset_quantized_list, axis=0)

    print("DEBUG: cluster_vq_vae multi_modal_dataset shape: %s" % str(multi_modal_dataset.shape))
    print("DEBUG: cluster_vq_vae multi_modal_testset shape: %s" % str(multi_modal_testset.shape))
    print("DEBUG: cluster_vq_vae multi_modal_codebook_emb shape: %s" % str(multi_modal_codebook_emb.shape))
    print("DEBUG: cluster_vq_vae multi_modal_dataset_quantized shape: %s" % str(multi_modal_dataset_quantized.shape))
    print("DEBUG: cluster_vq_vae multi_modal_testset_quantized shape: %s" % str(multi_modal_testset_quantized.shape))

    return multi_modal_codebook_emb, multi_modal_dataset, multi_modal_testset, multi_modal_dataset_quantized, multi_modal_testset_quantized

def cluster_supervised_faiss(embedding_list, query_embedding_list, center_cnt, inner_product_loss_enable = False, dist_func = "euclDistance"):
    """
        embedding_list: list size M, [L, D]
        query_embedding_list: list size M, [1, D]
        # output
            codebook_emb         [M, K, D]
            dataSetAssignment    [L, M]
            testSetAssignment    [M]
    """
    ## [L, M, D]
    data_samples = np.stack(embedding_list, axis=1)
    print ("DEBUG: data_samples shape %s" % str(data_samples.shape))

    ## [M, D]
    test_query_samples = np.squeeze(np.stack(query_embedding_list, axis=1))
    print ("DEBUG: test_query_samples shape %s" % str(test_query_samples.shape))

    ## 对齐
    shape_list = data_samples.shape
    L, M, D = shape_list[0], shape_list[1], shape_list[2]
    full_dim = M * D
    data_samples_reshape = np.reshape(data_samples, [L, full_dim])
    test_query_samples_reshape = np.reshape(test_query_samples, [1, full_dim])
    data_merge = np.concatenate([data_samples_reshape, test_query_samples_reshape], axis=0)

    nbits = 8
    index_pq = faiss.ProductQuantizer(full_dim, M, nbits)
    index_pq.train(data_merge)

    # index_pq = faiss.IndexPQ(full_dim, M, nbits)
    # index_pq.train(data_merge)
    # pq bits string of ceil(nbit * M / 8)
    dataSetAssignment = index_pq.compute_codes(data_samples_reshape)
    testSetAssignment = index_pq.compute_codes(test_query_samples_reshape)
    codebook_emb = faiss.vector_to_array(index_pq.centroids).reshape(index_pq.M, index_pq.ksub, index_pq.dsub)

    print ("DEBUG: dataSetAssignment shape: %s" % str(dataSetAssignment.shape))
    print ("DEBUG: testSetAssignment shape: %s" % str(testSetAssignment.shape))
    print ("DEBUG: codebook_emb shape: %s" % str(codebook_emb.shape))
    return codebook_emb, dataSetAssignment, testSetAssignment


def product_quantizer(centroids, list_2d_array, metric):
    """
        centroids: [K, D]
        list_2d_array: list of ndarray shape [B, D], embedding_list list of numpy array [B, D], Batch Size varies
    """
    list_assignment_index = []
    for array_2d in list_2d_array:
        # print ("DEBUG: array_2d %s" % str(array_2d))
        # print ("DEBUG: array_2d type %s" % str(array_2d))
        # print ("DEBUG: array_2d shape %s" % str(array_2d.shape))
        # assignment_index [B, 1]
        assignment_index = kmeans.assign_cluster(centroids, array_2d, metric)
        list_assignment_index.append(assignment_index)
    # [B, list_size]
    codes = np.concatenate(list_assignment_index, axis=-1)
    return codes

def get_norm_of_matrix_lastdim(tensor):
    """
        tensor = np.random.randn(20, 8)
        :param tensor:
        :return:
    """
    tensor_norm = np.linalg.norm(tensor, axis=-1, keepdims=True)
    average_norm = np.mean(tensor_norm)
    return average_norm

def eval_hitrate():
    ## Whether To Use Synthetic Dataset
    if_dummy = True
    if_random_norm_same = True
    if_equal_weight = False
    ndim = 16
    L = 2048
    topK = 32
    K_centroid = 512

    # weight_attr = np.array([[0.25, 0.25, 0.25, 0.25]])
    weight_attr = np.array([[0.25, 0.25, 0.25, 0.25]]) if if_equal_weight else np.array([[0.1, 0.2, 0.3, 0.4]])

    recall_k_two_stage_ann_list = []
    recall_k_two_stage_lsh_list = []
    recall_k_one_stage_pg_list = []
    recall_k_one_stage_pg_max_ip_list = []
    recall_k_one_stage_pg_codebook_list = []
    recall_k_one_stage_pg_codebook_multi_list = []
    recall_k_one_stage_rq_vae_codebook_multi_list = []
    recall_k_one_stage_pg_codebook_mips_list = []
    recall_k_one_stage_pg_codebook_query_mips_list = []

    recall_k_two_stage_maxsim_list = []
    recall_k_two_stage_maxmax_list_v1 = []
    cnt = 0
    total_eval_user_cnt = 1000
    for _ in range(total_eval_user_cnt):
        cnt += 1
        # user_query_list, user_item_id_list, target_query, target_item_id = tuple
        # Target Item
        # target_query_emb = lookup_emb([target_query], query_emb_dict, ndim)
        # target_id_emb = lookup_emb([target_item_id], item_id_emb_dict, ndim)
        # target_text_emb = lookup_emb([target_item_id], item_text_emb_dict, ndim)
        # target_image_emb = lookup_emb([target_item_id], item_image_emb_dict, ndim)
        # target_emb = np.stack([target_query_emb, target_id_emb, target_text_emb, target_image_emb], axis = 1)

        if if_dummy:
            if if_random_norm_same:
                ## rand digits [0,1]
                user_query_emb = normalize_tensor(np.random.randn(L, ndim))
                user_id_emb = normalize_tensor(np.random.randn(L, ndim))
                user_text_emb = normalize_tensor(np.random.randn(L, ndim))
                user_image_emb = normalize_tensor(np.random.randn(L, ndim))
                user_emb_mat = np.stack([user_query_emb, user_id_emb, user_text_emb, user_image_emb], axis=1)
                print ("DEBUG: if_random_norm_same %s, Norm Value for user_query_emb, user_id_emb, user_text_emb, user_image_emb is %f,%f,%f,%f"
                       % (str(if_random_norm_same), get_norm_of_matrix_lastdim(user_query_emb), get_norm_of_matrix_lastdim(user_id_emb), get_norm_of_matrix_lastdim(user_text_emb), get_norm_of_matrix_lastdim(user_image_emb)))
                target_query_emb = normalize_tensor(np.random.randn(ndim))
                target_id_emb = normalize_tensor(np.random.randn(ndim))
                target_text_emb = normalize_tensor(np.random.randn(ndim))
                target_image_emb = normalize_tensor(np.random.randn(ndim))
                target_emb = np.stack([target_query_emb, target_id_emb, target_text_emb, target_image_emb], axis = 0)
                print ("DEBUG: shape of user_emb_mat is %s, target_emb is %s" % (str(user_emb_mat.shape), str(target_emb.shape)))

            else:
                ## rand digits [0,1]
                user_query_emb = 1.0 * np.random.randn(L, ndim) + 0.25
                user_id_emb = 1.0 * np.random.randn(L, ndim) + 0.5
                user_text_emb = 1.0 * np.random.randn(L, ndim) + 1.0
                user_image_emb = 1.0 * np.random.randn(L, ndim) + 2.0
                user_emb_mat = np.stack([user_query_emb, user_id_emb, user_text_emb, user_image_emb], axis=1)
                print ("DEBUG: if_random_norm_same %s, Norm Value for user_query_emb, user_id_emb, user_text_emb, user_image_emb is %f,%f,%f,%f"
                       % (str(if_random_norm_same), get_norm_of_matrix_lastdim(user_query_emb), get_norm_of_matrix_lastdim(user_id_emb), get_norm_of_matrix_lastdim(user_text_emb), get_norm_of_matrix_lastdim(user_image_emb)))
                target_query_emb = np.random.randn(ndim) + 0.25
                target_id_emb = np.random.randn(ndim) + 0.5
                target_text_emb = np.random.randn(ndim) + 1.0
                target_image_emb = np.random.randn(ndim) + 2.0
                target_emb = np.stack([target_query_emb, target_id_emb, target_text_emb, target_image_emb], axis = 0)
                print ("DEBUG: shape of user_emb_mat is %s, target_emb is %s" % (str(user_emb_mat.shape), str(target_emb.shape)))

        else:

            ## Evaluate Real World Dataset
            ## Read Local Embedding File
            item_id_emb_dict, item_id_list, item_emb_2d_array = load_embedding_table("./item_id_emb.txt")
            item_text_emb_dict, text_id_list, text_emb_2d_array = load_embedding_table("./item_text_emb.txt")
            item_image_emb_dict, image_id_list, image_emb_2d_array = load_embedding_table("./item_image_emb.txt")
            query_emb_dict, query_id_list, query_emb_2d_array = load_embedding_table("./query_emb.txt")

            # eval_data = load_data("user_seq_eval.txt")
            eval_file = "user_seq_eval_data_demo.txt"
            eval_file = "seminar_sample100.txt"
            eval_data = load_embedding_data_txt(eval_file)

            user_query_emb, user_id_emb, user_text_emb, user_image_emb, target_query_emb, target_id_emb, target_text_emb, target_image_emb = tuple
            user_emb_mat = np.stack([user_query_emb, user_id_emb, user_text_emb, user_image_emb], axis=1)
            target_emb = np.stack([target_query_emb, target_id_emb, target_text_emb, target_image_emb], axis=0)
            print("DEBUG: if_random_norm_same %s, Norm Value for user_query_emb, user_id_emb, user_text_emb, user_image_emb is %f,%f,%f,%f"
                % (str(if_random_norm_same), get_norm_of_matrix_lastdim(user_query_emb),
                   get_norm_of_matrix_lastdim(user_id_emb), get_norm_of_matrix_lastdim(user_text_emb),
                   get_norm_of_matrix_lastdim(user_image_emb)))
            print("DEBUG: shape of user_emb_mat is %s, target_emb is %s" % (str(user_emb_mat.shape), str(target_emb.shape)))

        ## 1.0 HNSW
        hnsw_index_list = build_hnsw_index(user_emb_mat)

        ## 2.0 lsh
        lsh_index_list = build_lsh_index(user_emb_mat)

        ## 3.0 multi-modal PQ
        codebook_emb_multi, user_sequence_pq_multi, target_query_pq_multi = cluster_supervised_multi_modal_faiss([user_query_emb, user_id_emb, user_text_emb, user_image_emb], [np.expand_dims(target_query_emb, 0), np.expand_dims(target_id_emb, 0)
            , np.expand_dims(target_text_emb, 0), np.expand_dims(target_image_emb, 0)], K_centroid, inner_product_loss_enable = False, dist_func="euclDistance")

        ## 4.0 RQ_VAE
        codebook_emb_rq_vae_multi, user_sequence_rq_vae_multi, target_query_rq_vae_multi, rq_vae_multi_user_quantized, rq_vae_multi_target_quantized = cluster_rq_vae([user_query_emb, user_id_emb, user_text_emb, user_image_emb], [np.expand_dims(target_query_emb, 0), np.expand_dims(target_id_emb, 0)
            , np.expand_dims(target_text_emb, 0), np.expand_dims(target_image_emb, 0)], K_centroid, inner_product_loss_enable = False, dist_func="euclDistance")

        ### Evaluation
        ## 0.0 Full Attention
        topk_values, topk_idx = full_attention(user_emb_mat, target_emb, weight_attr, topK)
        # print ("DEBUG: full_attention topk_idx is %s" % str(topk_idx))
        # print ("DEBUG: full_attention topk_values is %s" % str(topk_values))

        ## 1.0
        two_stage_topk_values, two_stage_topk_idx = two_stage_ann_faiss_hnsw(user_emb_mat, target_emb, weight_attr, topK, index_list=hnsw_index_list)
        # print ("DEBUG: two_stage_topk_idx is %s" % str(two_stage_topk_idx))
        # print ("DEBUG: two_stage_topk_values is %s" % str(two_stage_topk_values))
        recall_k_two_stage = calc_recall_k_rate(topk_idx, two_stage_topk_idx)
        print ("DEBUG: two_stage_topk recall@%d is %s" % (topK, str(recall_k_two_stage)))

        ## 2.0
        two_stage_lsh_topk_values, two_stage_lsh_topk_idx = two_stage_ann_faiss_hnsw(user_emb_mat, target_emb, weight_attr, topK, index_list=lsh_index_list)
        # print ("DEBUG: two_stage_topk_idx is %s" % str(two_stage_topk_idx))
        # print ("DEBUG: two_stage_topk_values is %s" % str(two_stage_topk_values))
        recall_k_two_stage_lsh = calc_recall_k_rate(topk_idx, two_stage_lsh_topk_idx)
        print ("DEBUG: recall_k_two_stage_lsh recall@%d is %s" % (topK, str(hitrate_two_stage_lsh)))

        ## 3.0
        pq_coodbook_multi_topk_values, pq_coodbook_multi_topk_idx = one_stage_pq_supervised_distance_faiss_multi(user_sequence_pq_multi, target_query_pq_multi, weight_attr, topK, codebook_emb_multi)
        recall_k_pq_codebook_multi = calc_recall_k_rate(topk_idx, pq_coodbook_multi_topk_idx)
        print("DEBUG: recall_k_codebook_multi recall@%d is %s" % (topK, str(recall_k_pq_codebook_multi)))

        ## 4.0 RQ-VAE
        rq_vae_coodbook_multi_topk_values, rq_vae_coodbook_multi_topk_idx = one_stage_pq_supervised_distance_decoder_quantized_multi(user_sequence_rq_vae_multi, target_query_rq_vae_multi, weight_attr, topK, codebook_emb_rq_vae_multi, rq_vae_multi_user_quantized, rq_vae_multi_target_quantized)
        recall_k_rq_vae_codebook_multi = calc_recall_k_rate(topk_idx, rq_vae_coodbook_multi_topk_idx)
        print("DEBUG: recall_k_rq_vae_codebook_multi recall@%d is %s" % (topK, str(hitrate_rq_vae_codebook_multi)))

        recall_k_two_stage_ann_list.append(recall_k_two_stage)
        recall_k_two_stage_lsh_list.append(recall_k_two_stage_lsh)
        recall_k_one_stage_pg_codebook_multi_list.append(recall_k_pq_codebook_multi)
        recall_k_one_stage_rq_vae_codebook_multi_list.append(recall_k_rq_vae_codebook_multi)

        ## Test
        if cnt % 1 == 0:
            print ("DEBUG:-----------Sequence Length %d-topK %d K_centroid %d Cnt %d if_equal_weight %s, if_same_norm %s----------------" % (L, topK, K_centroid, cnt, str(if_equal_weight), str(if_random_norm_same)))
            print ("DEBUG: recall_k_two_stage_ann_hnsw is %f" % np.mean(np.array(recall_k_two_stage_ann_list)))
            print ("DEBUG: recall_k_two_stage_ann_lsh is %f" % np.mean(np.array(recall_k_hitrate_two_stage_lsh_list)))
            print ("DEBUG: recall_k_one_stage_rq_vae_codebook_multi_list is %f" % np.mean(np.array(recall_k_one_stage_rq_vae_codebook_multi_list)))
            print ("DEBUG: recall_k_one_stage_pg_codebook_multi_list is %f" % np.mean(np.array(recall_k_one_stage_pg_codebook_multi_list)))

def main():
    eval_hitrate()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
