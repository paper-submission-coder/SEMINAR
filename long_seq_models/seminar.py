# -*- coding: utf-8 -*-
# @Author  : Anonymous Period

import torch
import torch.nn as nn
from modules import Encoder, LayerNorm, QueryKeywordsEncoder
import numpy as np
import torch.nn.init as init
from torch.nn.functional import normalize
import torch.nn.functional as F
from long_seq_models.base_model import BasePairSequenceModel, TWIN

class SEMINAR(TWIN):

    def __init__(self, args):
        # parent __init__ method
        super(SEMINAR, self).__init__(args)
        # child __init__ method
        self.temperature = 0.05
        self.alignment_chunks_num = 1000

        # PSU -> GSU,ESU Embedding projection G
        # self.projection_diagonal = True
        self.freeze_psu_projection = args.freeze_psu_projection
        self.G_item_id = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.G_query = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.G_item_text = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.G_item_img = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.freeze_psu_projection = args.freeze_psu_projection
        ## Vit/CLIP emb_dim -> hidden_size
        self.freeze_multi_modal_projection = args.freeze_multi_modal_projection

        if self.freeze_psu_projection:
            self.G_item_id.weight.requires_grad = False
            self.G_query.weight.requires_grad = False
            self.G_item_text.weight.requires_grad = False
            self.G_item_img.weight.requires_grad = False
        if self.freeze_multi_modal_projection:
            self.image_emb_proj.weight.requires_grad = False
            self.text_emb_proj .weight.requires_grad = False

        # GSU
        self.soft_search_type = args.soft_search_type
        self.hash_bits = 16
        self.hash_proj_matrix = nn.Linear(args.hidden_size, self.hash_bits)

        ## multi-modal pq
        self.num_mode = 4
        self.num_centroids = 512
        # notice: centroids_emb * hash_bits = emb_dim
        self.centroids_emb = 4
        # [M, N_bit, N_Centroids, Emb]
        self.codebook_emb = torch.randn(self.num_mode, self.hash_bits, self.num_centroids, self.centroids_emb)
        for param in self.codebook_emb:
            param.requires_grad = False

    def get_restored_emb(self):
        """
            item_embeddings
            attribute_embeddings
            query_keywords_embeddings

            # Need to be freezed, Image and Text Input
            item_image_embedding_layer, image_emb_proj
            item_text_embedding_layer, text_emb_proj
        """
        restored_emb_list = [self.item_embeddings, self.attribute_embeddings, self.query_keywords_embeddings]
        return restored_emb_list

    def hash_emb_layer(self, inputs):
        """
            inputs:  dense embedding of [B, ..., D]
            inputs_proj_hash: int (0/1) embedding of [B, ..., N_Bits], larger distance means similar vectors
        """
        # [B, ..., D] -> [B, ..., N_bits]
        inputs_proj = self.hash_proj_matrix(inputs)
        inputs_proj = torch.unsqueeze(inputs_proj, dim = -1) # [B, N_Bit] -> [B, N_Bit, 1]
        inputs_proj_merge = torch.cat([-1.0 * inputs_proj, inputs_proj], axis=-1)  # [B, N_Bit, 1] -> [B, N_Bit, 2]
        inputs_proj_hash = torch.argmax(inputs_proj_merge, dim=-1)
        return inputs_proj_hash

    def multi_modal_pq_layer(self, inputs):
        """
            inputs: dense embedding of [M, B, ..., D]  first: module,
            inputs_proj_hash: int (0-num_centroids) embedding of [B, ..., N_Bits], larger distance means similar vectors
        """
        num_mode = min(inputs.shape[0], self.codebook_emb.shape[0])

        input_codes_assignment_list = []
        for m in range(num_mode):
            # [B, L, D] -> sub-vectors
            shape_list = list(inputs[m].shape)
            final_dim = int(shape_list[len(shape_list)-1]/self.hash_bits)
            reshape_list = shape_list
            reshape_list[len(reshape_list)-1] = self.hash_bits
            reshape_list += [final_dim]
            # [B, L, D] -> [B, L, N_bit, Final_Dim]
            input_reshape = torch.reshape(inputs[m], shape=reshape_list)
            code_list = []
            for b in range(self.hash_bits):
                # subvector: [B, L, Emb]
                subvector = normalize(input_reshape[:, : , b, :], dim=-1)
                # centroids: [N_Centroids, Emb]
                centroids = normalize(self.codebook_emb[m][b], dim=-1)
                # [B, L]
                codes = torch.argmax(torch.matmul(subvector, centroids.T), dim=-1)
                code_list.append(codes)
            # shape = [B, L, N_Bit]
            input_codes_assignment = torch.transpose(torch.stack(code_list, dim=1), 1, 2)
            input_codes_assignment_list.append(input_codes_assignment)
        # multi_modal pq, [M, B, L, N_Bits]
        input_codes_multi_modal_assignment = torch.stack(input_codes_assignment_list, dim=0)
        return input_codes_multi_modal_assignment

    def hamming_distance(self, query_hashes, keys_hashes):
        """
            query_hashes: [B, 1, N_Bits]
            keys_hashes: [B, L, N_Bits]
            distance: [B, L]
        """
        key_num = keys_hashes.shape[1]
        # [B, 1, N] -> [B, L, N]
        query_hashes_tile = torch.tile(query_hashes, (1, key_num, 1))
        match_buckets = torch.eq(query_hashes_tile, keys_hashes).int()
        distance = torch.sum(match_buckets, dim=-1)
        return distance

    def pretrain_cross_entropy_next_item_loss(self, logits_pos, logits_neg_batch):
        """
            logits_pos: [B, 1]
            logits_neg_group: [B, nun_sample]
            nn.BCELoss() inputs: pred after sigmoid, label 0/1
        """
        ## Some Long Sequence Model, First Stage Embedding Doesn't Participate In the Final Loss
        if logits_pos is None or logits_neg_batch is None:
            return 0.0
        ## [batch_size, num_sample] -> [batch_size * num_sample, 1]
        logits_neg_batch_reshape = torch.reshape(logits_neg_batch, shape=[-1]).unsqueeze(-1)
        logits_merge = torch.cat([logits_pos, logits_neg_batch_reshape], dim=0)
        label_merge = torch.cat([torch.ones_like(logits_pos), torch.zeros_like(logits_neg_batch_reshape)], dim=0)
        pred_merge = nn.Sigmoid()(logits_merge)
        loss = nn.BCELoss()(pred_merge, label_merge)
        return loss

    def next_query_item_prediction_loss(self, input_ids, input_query_keywords_ids, target_item_id_pos, target_query_pos, target_item_id_neg, target_query_neg):
        """
            inputs: [0, L-1] item, query
            prediction: [L] item
        """
        # target_pair_pos, target_pair_neg shape [B, 1, D]
        target_pair_pos = self.fuse_multimodal_embedding_v2(target_item_id_pos, target_query_pos, self.fusion_type)
        target_pair_neg = self.fuse_multimodal_embedding_v2(target_item_id_neg, target_query_neg, self.fusion_type)

        # Negative
        logits_neg_list = []

        psu_output_pos, psu_output_neg = None, None
        if self.soft_search_type == "lsh":
            # ETA style soft search, Fusion First, Retrieval Later
            sequence_emb = self.fuse_multimodal_embedding_v2(input_ids, input_query_keywords_ids, self.fusion_type)
            gsu_output_pos, gsu_merged_pos = self.soft_search_eta(sequence_emb, target_pair_pos, self.retrieve_topK)
            gsu_output_neg, gsu_merged_neg = self.soft_search_eta(sequence_emb, target_pair_neg, self.retrieve_topK)

            psu_output_pos, _ = self.multi_head_target_attention(gsu_output_pos, target_pair_pos, self.params)
            psu_output_neg, _ = self.multi_head_target_attention(gsu_output_neg, target_pair_neg, self.params)

        elif self.soft_search_type == "multi_pq":
            # Later Fusion
            multi_user_seq_emb = self.stack_multimodal_embedding(input_ids, input_query_keywords_ids)
            multi_target_emb_pos = self.stack_multimodal_embedding(target_item_id_pos, target_query_pos)
            multi_target_emb_neg = self.stack_multimodal_embedding(target_item_id_neg, target_query_neg)
            gsu_output_pos, _ = self.soft_search_multi_modal(multi_user_seq_emb, multi_target_emb_pos, self.retrieve_topK)
            gsu_output_neg, _ = self.soft_search_multi_modal(multi_user_seq_emb, multi_target_emb_neg, self.retrieve_topK)
            # PSU Output
            ## 4.0 ESU Search Unit
            psu_output_pos, _ = self.multi_head_target_attention(gsu_output_pos, target_pair_pos, self.params)
            psu_output_neg, _ = self.multi_head_target_attention(gsu_output_neg, target_pair_neg, self.params)

        elif self.soft_search_type == "mhta":
            # Early Fusion
            sequence_emb = self.fuse_multimodal_embedding_v2(input_ids, input_query_keywords_ids, self.fusion_type)
            attention_mask = (input_ids > 0).long()

            # Positive
            gsu_output_pos_list, gsu_merged_pos = self.soft_search_list(sequence_emb, target_pair_pos, attention_mask, self.retrieve_topK)
            esu_output_pos, _ = self.multi_head_target_attention(gsu_output_pos_list, target_pair_pos, self.params)
            logits_pos = self.final_deep_layers(torch.cat([esu_output_pos, torch.squeeze(target_pair_pos)], dim=-1))

            num_negative = target_item_id_neg.shape[-1]
            for i in range(num_negative):
                # [B, D] -> [B, 1, D]
                target_pair_neg_i = target_pair_neg[:, i, :].unsqueeze(1)
                # print ("DEBUG: Processing Negative Sample %d, shape %s" % (i, str(target_pair_neg_i.shape)))
                gsu_output_neg_list_i, _ = self.soft_search_list(sequence_emb, target_pair_neg_i, attention_mask, self.retrieve_topK)
                esu_output_neg_i, _ = self.multi_head_target_attention(gsu_output_neg_list_i, target_pair_neg_i, self.params)
                logits_neg_i = self.final_deep_layers(torch.cat([esu_output_neg_i, torch.squeeze(target_pair_neg_i)], dim=-1))
                logits_neg_list.append(logits_neg_i)
            logits_neg_group = torch.cat(logits_neg_list, dim=-1)

        else:
            sequence_emb = self.fuse_multimodal_embedding_v2(input_ids, input_query_keywords_ids, self.fusion_type)
            attention_mask = (input_ids > 0).long()

            target_pair_pos = self.fuse_multimodal_embedding(target_item_id_pos, target_query_pos, self.fusion_type)
            target_pair_neg = self.fuse_multimodal_embedding(target_item_id_neg, target_query_neg, self.fusion_type)

            ## 3.0 GSU Search Unit, gsu_output_pos_list list of [B, L, D]
            gsu_output_pos_list, gsu_merged_pos = self.soft_search_list(sequence_emb, target_pair_pos, attention_mask, self.retrieve_topK)
            gsu_output_neg_list, gsu_merged_neg = self.soft_search_list(sequence_emb, target_pair_neg, attention_mask, self.retrieve_topK)

            ## 4.0 ESU Search Unit
            psu_output_pos, _ = self.multi_head_target_attention(gsu_output_pos_list, target_pair_pos, self.params)
            psu_output_neg, _ = self.multi_head_target_attention(gsu_output_neg_list, target_pair_neg, self.params)

        next_pair_prediction_loss = self.pretrain_cross_entropy_next_item_loss(logits_pos, logits_neg_group)
        return next_pair_prediction_loss

    def query_item_relevance(self, query_emb, item_emb):
        """
            query_emb  [B, L, D]
            item_emb   [B, L, D]
            # in-batch negative sampling
        """
        ## negative sampling
        num_negative = 5
        batch_size, seq_len, emb_dim = item_emb.shape[0], item_emb.shape[1], item_emb.shape[2]
        # negative_sample_index [L, num_negative]
        negative_sample_index = []
        for index in range(seq_len):
            sample_index = np.random.choice(seq_len, size=num_negative + 1, replace=False, p=None)
            negative_index = list(set(sample_index) - set([index]))[0:num_negative]
            negative_sample_index.append(negative_index)
        item_emb_neg_list = []
        for index in range(seq_len):
            negative_index = negative_sample_index[index]
            negative_emb = item_emb[:, negative_index]  # [B, num_negative]
            item_emb_neg_list.append(negative_emb)
        # [B, seq_len, num_sample, D]
        item_emb_neg = torch.stack(item_emb_neg_list, dim=1)

        ## normlize
        query_emb_norm = normalize(query_emb, dim=-1)
        item_emb_norm = normalize(item_emb, dim=-1)
        item_emb_neg_norm = normalize(item_emb_neg, dim=-1)

        # loss [B, L, D] \times [B, L, D] -> [B, L]
        pos = torch.exp(torch.sum(torch.mul(query_emb_norm, item_emb_norm), dim=-1) / self.temperature)
        # query_batch, [B, L, D] - > [B, L, num_negative, D]
        query_batch = torch.tile(torch.unsqueeze(query_emb_norm, dim=-2), dims=(1, 1, num_negative, 1))
        # [B， L]
        neg = torch.sum(torch.exp(torch.sum(torch.mul(query_batch, item_emb_neg_norm), dim=-1) / self.temperature), dim=-1)
        loss_qi = torch.mean(pos / neg)
        return loss_qi

    def multi_modal_alignment(self, ids_emb, text_emb, img_emb):
        """
            3 channels: ids_emb, text_emb, img_emb
        """
        emb_channel_list = []
        emb_dim = ids_emb.shape[-1]
        # reformat each channel shape from [B, L, D] to [B * L, D]
        if ids_emb is not None:
            emb_channel_list.append(torch.reshape(ids_emb, [-1, emb_dim]))
        if text_emb is not None:
            emb_channel_list.append(torch.reshape(text_emb, [-1, emb_dim]))
        if img_emb is not None:
            emb_channel_list.append(torch.reshape(img_emb, [-1, emb_dim]))
        total_channel = len(emb_channel_list)
        if len(emb_channel_list) <= 1:
            return 0.0
        # n_pair = B * L
        batch_size, seq_len, emb_dim = ids_emb.shape[0], ids_emb.shape[1], ids_emb.shape[2]
        n_pair = emb_channel_list[0].shape[0]
        chunk_size = int(n_pair/self.alignment_chunks_num)
        # print ("DEBUG: multi_modal_alignment batch_size %d, seq_len %d, n_pair %d, alignment_chunks_num %d, chunk_size %d" % (batch_size, seq_len, n_pair, self.alignment_chunks_num, chunk_size))
        loss_list = []
        total_loss = 0.0
        loss_cnt = 0
        for i in range(total_channel):
            for j in range(i+1, total_channel):
                ## inter num_chunks, [L/N, L/N]
                for k in range(self.alignment_chunks_num):
                    channel_norm_i = normalize(emb_channel_list[i][k * chunk_size: (k + 1) * chunk_size], dim=-1)
                    channel_norm_j = normalize(emb_channel_list[j][k * chunk_size: (k + 1) * chunk_size], dim=-1)
                    loss_clip = self.clip_loss(channel_norm_i, channel_norm_j)
                    total_loss += loss_clip
                    loss_cnt += 1
                    loss_list.append(loss_clip)
        ## loss list size: M * (M-1) * N_chunk
        # print ("DEBUG: loss_list size %d" % (len(loss_list)))
        # loss_alignment = torch.mean(torch.cat(loss_list))
        loss_alignment = total_loss/loss_cnt
        return loss_alignment

    def clip_loss(self, image_embeddings, text_embeddings):
        """
            Reference: https://github.com/moein-shariatnia/OpenAI-CLIP

            args:
                image_embeddings, normalized embedding of shape [B, D]
                text_embeddings,  normalized embedding of shape [B, D]

            output:
                scalar, loss
            note:
                another implementation: multiple N-class classification
                clip_loss = nn.CrossEntropyLoss()(logits, torch.arange(batch_size))
                logits: [N, N],
                target: torch.arange(batch_size),  [0, 1, 2, ..., (batch_size-1)]
        """
        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = self.cross_entropy(logits, targets, reduction='none')
        images_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    def pretrain(self, input_ids, input_query_keywords_ids, target_item_id_pos, target_query_pos, target_item_id_neg, target_query_neg):
        """
            Pretraining Methods for SEMINAR,
            Inputs:
                User Sequence Pair (input_ids, input_query_keywords_ids)  # (B, L, D), (B, L, N_keywords, D)
                Target Item Pair-Positive (target_item_id_pos, target_query_pos) # (B, 1, D), (B, 1, N_keywords, D)
                Target Item Pair-Negative (target_item_id_neg, target_query_neg) # (B, 1, D), (B, 1, N_keywords, D)

            Finetuning Params:
                ## mhta
                self.wQ_list = [nn.Linear(args.hidden_size, args.hidden_size) for _ in range(self.num_heads)]
                self.wK_list = [nn.Linear(args.hidden_size, args.hidden_size) for _ in range(self.num_heads)]
                self.wV_list = [nn.Linear(args.hidden_size, args.hidden_size) for _ in range(self.num_heads)]
                self.wO = nn.Linear(self.num_heads * args.hidden_size, args.hidden_size)
        """
        # Query Embedding: [B, L, N_keywords, D] -> [B, L, D]
        psu_query_embeddings = self.query_encoder(self.query_keywords_embeddings(input_query_keywords_ids))
        # Item Embedding: ID Embedding, Image Embedding, Text Embedding
        # ID Embedding: [B, L, D]
        psu_id_embeddings = self.item_embeddings(input_ids)
        # Image Embedding
        psu_image_embeddings = None
        # Text Embedding
        psu_text_embeddings = None
        if self.args.multi_modal_emb_enable:
            # print ("DEBUG: multi_modal_emb_enable|%s" % str(self.args.multi_modal_emb_enable))
            psu_image_embeddings = self.image_emb_proj(self.item_image_embedding_layer(input_ids))
            psu_text_embeddings = self.text_emb_proj(self.item_text_embedding_layer(input_ids))

        ## Pretraining Search Unit Tasks
        # 1.0 Next Query-Item Pair Prediction Loss
        loss_next_pair = self.next_query_item_prediction_loss(input_ids, input_query_keywords_ids, target_item_id_pos, target_query_pos, target_item_id_neg, target_query_neg) if self.args.pretrain_next_pair_enable else 0.0

        # 2.0 Multi Model Alignment
        loss_multimodal_align = self.multi_modal_alignment(psu_id_embeddings, psu_text_embeddings, psu_image_embeddings) if self.args.pretrain_multimodal_align_enable else 0.0

        # 3.0 Query-Item Relevance
        psu_item_embeddings = self.fuse_multi_modal_item_emdbedding(psu_id_embeddings, psu_image_embeddings, psu_text_embeddings)
        loss_qi_relevance = self.query_item_relevance(psu_query_embeddings, psu_item_embeddings) if self.args.pretrain_qi_relevance_enable else 0.0

        return loss_next_pair, loss_multimodal_align, loss_qi_relevance

    def soft_search_eta(self, user_seq_emb, target_emb, K):
        """
            input:
                user_seq_emb: [B, L, D]
                target_emb: [B, 1, D]

            output:
                gsu_out_topk     [B, K, D]
                gsu_merged   [B, D]
                    Stage1 Merged User Representation, Pass the gsu merged tensor to calculate loss of stage 1

            ETA: GSU logits doesn't participate in the final loss
        """
        # Base Implement SoftSearch follows SIM method
        batch_size, sequence_len, emb_dim = user_seq_emb.shape
        query_hashes = self.hash_emb_layer(target_emb)
        keys_hashes = self.hash_emb_layer(user_seq_emb)
        # ETA Caculate Hamming Distance, [B, L]
        qk_hamming_distance = self.hamming_distance(query_hashes, keys_hashes)
        # values: [B, K], indices: [B, K]
        values, indices = torch.topk(qk_hamming_distance, K, dim=-1, largest=True)
        # user_seq_emb: [B, L, D], index= [B, index_length=K]
        gather_index = indices.unsqueeze(-1).expand(-1, -1, emb_dim).to(dtype=torch.int64)
        # user_seq_emb [B, L, D] -> [B, K, D]
        gsu_out_topk = torch.gather(user_seq_emb, dim=1, index=gather_index, out=None)
        # print ("DEBUG: user_seq_emb_topK shape %s" % str(gsu_out_topk.shape))
        ## Generate First Stage Merged Sequence Representation
        # gsu_merged = torch.squeeze(torch.bmm(torch.transpose(user_seq_emb, 1, 2), torch.unsqueeze(qK, axis=-1)), axis=-1)
        return gsu_out_topk, None

    def soft_search_multi_modal(self, user_seq_emb_multi, target_emb_multi, K):
        """
            target_emb  [M, B, 1, D]
            seq_emb [M, B, L, D]
        """
        # [0.5, 0.5]
        multi_module_weight_list = [self.fusion_weight_query, 1 - self.fusion_weight_query]

        num_mode, batch_size, seq_len, emb_dim = user_seq_emb_multi.shape[0], user_seq_emb_multi.shape[1], user_seq_emb_multi.shape[2], user_seq_emb_multi.shape[3]
        query_pq_codes = self.multi_modal_pq_layer(target_emb_multi)
        keys_pq_codes = self.multi_modal_pq_layer(user_seq_emb_multi)

        # print ("DEBUG: soft_search_multi_modal query_pq_codes shape %s" % str(query_pq_codes.shape))
        # print ("DEBUG: soft_search_multi_modal keys_pq_codes shape %s" % str(keys_pq_codes.shape))

        qk_distance_mode_list = []
        for m in range(num_mode):
            # [B, 1, N_bit] \times [B, L, N_bit]
            qk_hamming_distance = self.hamming_distance(query_pq_codes[m], keys_pq_codes[m])
            qk_distance_mode_list.append(qk_hamming_distance)
        qk_distance_fuse = torch.zeros_like(qk_distance_mode_list[0]).float()
        for m in range(num_mode):
            qk_distance_fuse += (qk_distance_mode_list[m] * multi_module_weight_list[m])
        # print ("DEBUG: soft_search_multi_modal qk_distance_fuse shape %s" % str(qk_distance_fuse.shape))

        ## final qk_distance_sum
        # values: [B, K], indices: [B, K]
        values, indices = torch.topk(qk_distance_fuse, K, dim=-1, largest=True)
        # user_seq_emb: [B, L, D], index= [B, index_length=K]
        gather_index = indices.unsqueeze(-1).expand(-1, -1, emb_dim).to(dtype=torch.int64)
        # user_seq_emb [B, L, D] -> [B, K, D]
        # Fuse and Gather
        user_seq_emb = user_seq_emb_multi[0] * multi_module_weight_list[0] + user_seq_emb_multi[1] * multi_module_weight_list[1]
        gsu_out_topk = torch.gather(user_seq_emb, dim=1, index=gather_index, out=None)
        return gsu_out_topk, None

    def get_item_id_emb(self, input_sequence_id):
        psu_id_embeddings = self.item_embeddings(input_sequence_id)
        gsu_id_embeedings = self.G_item_id(psu_id_embeddings)
        return gsu_id_embeedings

    def get_item_image_emb(self, input_sequence_id):
        item_image_embedding_psu = self.image_emb_proj(self.item_image_embedding_layer(input_sequence_id))
        item_image_embedding_gsu = self.G_item_img(item_image_embedding_psu)
        return item_image_embedding_gsu

    def get_item_text_emb(self, input_sequence_id):
        item_text_embedding_psu = self.text_emb_proj(self.item_text_embedding_layer(input_sequence_id))
        item_text_embedding_gsu = self.G_item_text(item_text_embedding_psu)
        return item_text_embedding_gsu

    def get_query_emb(self, input_query_keywords_ids):
        """
            Args:
                input_query_keywords_ids, [B, L, N_keyword, D]
            Output:
                query_embedding [B, L, D]
        """
        psu_query_embeddings = self.query_encoder(self.query_keywords_embeddings(input_query_keywords_ids))
        gsu_query_embeddings = self.G_query(psu_query_embeddings)
        return gsu_query_embeddings

    def fuse_multi_modal_item_emdbedding(self, id_embedding, image_embedding, text_embedding):
        if text_embedding is None or image_embedding is None:
            return id_embedding
        if self.args.multi_modal_emb_enable:
            gating = self.get_multi_modal_fusion_gating()
            # torch.stack(sequence_channels, dim=-1) shape: [B, L, D, M],  gating.T shape [M, 1]
            item_emb_merge = torch.matmul(
                torch.stack([id_embedding, image_embedding, text_embedding], dim=-1), gating.T).squeeze(-1)
            return item_emb_merge
        else:
            return id_embedding

    def get_fused_item_emb(self, input_ids):
        """
            args:
                input_ids, tensor of int [B, ..., ] int id
            output:
                item_emb_merge_gsu
        """
        item_emb_merge_gsu = None
        item_id_embeddings_gsu = self.get_item_id_emb(input_ids)
        if self.args.multi_modal_emb_enable:
            gating = self.get_multi_modal_fusion_gating()
            image_embedding_gsu = self.get_item_image_emb(input_ids)
            text_embedding_gsu = self.get_item_text_emb(input_ids)
            # torch.stack(sequence_channels, dim=-1) shape: [B, L, D, M],  gating.T shape [M, 1]
            item_emb_merge_gsu = torch.matmul(
                torch.stack([item_id_embeddings_gsu, image_embedding_gsu, text_embedding_gsu], dim=-1), gating.T).squeeze(-1)
        else:
            item_emb_merge_gsu = item_id_embeddings_gsu
        return item_emb_merge_gsu

    def finetuneV2(self, input_ids, input_query_keywords_ids, target_item_id_pos, target_query_pos, target_item_id_neg, target_query_neg):
        """
            input:
                input_ids: [B, L-2], historical Id
                input_query_keywords_ids: [B, L-2],  historical Id
                target_item_id_pos: [B, 1], last item of target positive item id sequence
                target_query_pos:   [B, 1], last query of target positive query sequence

                target_item_id_neg: [B, 1], last item of target negative item id sequence
                target_query_neg:   [B, 1], last query of target negative query sequence
                or
                target_item_id_neg: [B, num_negative]   testset negatively sampled
                target_query_neg:   [B, num_negative]
            output:
                logits_pos:  [B, 1]
                logits_neg:  [B, 1] or [B, num_negative]
        """
        ## Restore First Stage PSU Embedding
        ## User Sequence Embedding
        item_emb_merge_gsu = self.get_fused_item_emb(input_ids)
        query_embeddings_gsu = self.get_query_emb(input_query_keywords_ids)
        # Target positive ids
        target_item_pos_emb_merge = self.get_fused_item_emb(target_item_id_pos)
        target_query_pos_emb = self.get_query_emb(target_query_pos)
        # Target negative ids
        target_item_neg_emb_merge = self.get_fused_item_emb(target_item_id_neg)
        target_query_neg_emb = self.get_query_emb(target_query_neg)

        # Fusion Final Inputs to Sequence Model
        sequence_emb = self.get_fused_query_item_pair(query_embeddings_gsu, item_emb_merge_gsu, self.fusion_weight_query)
        target_pair_pos = self.get_fused_query_item_pair(target_query_pos_emb, target_item_pos_emb_merge, self.fusion_weight_query)
        target_pair_neg = self.get_fused_query_item_pair(target_query_neg_emb, target_item_neg_emb_merge, self.fusion_weight_query)

        # esu_output_pos, esu_output_neg = None, None
        logits_pos, logits_neg_group= None, None
        if self.soft_search_type == "lsh":
            # ETA style soft search, Fusion First, Retrieval Later
            # GSU->ESU
            # sequence_emb = self.fuse_multimodal_embedding_v2(input_ids, input_query_keywords_ids, self.fusion_type)
            gsu_output_pos, gsu_merged_pos = self.soft_search_eta(sequence_emb, target_pair_pos, self.retrieve_topK)
            gsu_output_neg, gsu_merged_neg = self.soft_search_eta(sequence_emb, target_pair_neg, self.retrieve_topK)

            ## 4.0 ESU Search Unit
            esu_output_pos, _ = self.multi_head_target_attention(gsu_output_pos, target_pair_pos, self.params)
            esu_output_neg, _ = self.multi_head_target_attention(gsu_output_neg, target_pair_neg, self.params)

        elif self.soft_search_type == "multi_pq":
            # Later Fusion
            multi_user_seq_emb = self.stack_multimodal_embedding(input_ids, input_query_keywords_ids)
            multi_target_emb_pos = self.stack_multimodal_embedding(target_item_id_pos, target_query_pos)
            multi_target_emb_neg = self.stack_multimodal_embedding(target_item_id_neg, target_query_neg)
            print ("DEBUG: multi_user_seq_emb shape %s" % str(multi_user_seq_emb.shape))
            print ("DEBUG: multi_target_emb_pos shape %s" % str(multi_target_emb_pos.shape))

            gsu_output_pos, _ = self.soft_search_multi_modal(multi_user_seq_emb, multi_target_emb_pos, self.retrieve_topK)
            gsu_output_neg, _ = self.soft_search_multi_modal(multi_user_seq_emb, multi_target_emb_neg, self.retrieve_topK)
            ## 4.0 ESU Search Unit
            esu_output_pos, _ = self.multi_head_target_attention(gsu_output_pos, target_pair_pos, self.params)
            esu_output_neg, _ = self.multi_head_target_attention(gsu_output_neg, target_pair_neg, self.params)

            ## 5.0 self.retrieve_topK
            logits_pos = self.final_deep_layers(torch.cat([esu_output_pos, torch.squeeze(target_pair_pos)], dim=-1))
            logits_neg = self.final_deep_layers(torch.cat([esu_output_neg, torch.squeeze(target_pair_neg)], dim=-1))

        elif self.soft_search_type == "mhta":
            # Early Fusion
            attention_mask = (input_ids > 0).long()

            # Positive
            gsu_output_pos_list, _ = self.soft_search_list(sequence_emb, target_pair_pos, attention_mask, self.retrieve_topK)
            esu_output_pos, _ = self.multi_head_target_attention(gsu_output_pos_list, target_pair_pos, self.params)
            logits_pos = self.final_deep_layers(torch.cat([esu_output_pos, torch.squeeze(target_pair_pos)], dim=-1))

            num_negative = target_item_id_neg.shape[-1]
            logits_neg_list = []
            for i in range(num_negative):
                # [B, D] -> [B, 1, D]
                target_pair_neg_i = target_pair_neg[:, i, :].unsqueeze(1)
                # print ("DEBUG: Processing Negative Sample %d, shape %s" % (i, str(target_pair_neg_i.shape)))
                gsu_output_neg_list_i, _ = self.soft_search_list(sequence_emb, target_pair_neg_i, attention_mask, self.retrieve_topK)
                esu_output_neg_i, _ = self.multi_head_target_attention(gsu_output_neg_list_i, target_pair_neg_i, self.params)
                logits_neg_i = self.final_deep_layers(torch.cat([esu_output_neg_i, torch.squeeze(target_pair_neg_i)], dim=-1))
                logits_neg_list.append(logits_neg_i)
            logits_neg_group = torch.cat(logits_neg_list, dim=-1)

        else:
            print ("DEBUG: self.soft_search_type not supported %s" % self.soft_search_type)
            # GSU->ESU
            sequence_emb = self.fuse_multimodal_embedding(input_ids, input_query_keywords_ids, self.fusion_type)
            gsu_output_pos, gsu_merged_pos = self.soft_search(sequence_emb, target_pair_pos, self.retrieve_topK)
            gsu_output_neg, gsu_merged_neg = self.soft_search(sequence_emb, target_pair_neg, self.retrieve_topK)

            ## 4.0 ESU Search Unit
            esu_output_pos, _ = self.multi_head_target_attention(gsu_output_pos, target_pair_pos, self.params)
            esu_output_neg, _ = self.multi_head_target_attention(gsu_output_neg, target_pair_neg, self.params)

            ## 5.0 self.retrieve_topK
            logits_pos = self.final_deep_layers(torch.cat([esu_output_pos, torch.squeeze(target_pair_pos)], dim=-1))
            logits_neg = self.final_deep_layers(torch.cat([esu_output_neg, torch.squeeze(target_pair_neg)], dim=-1))

        # logits_stage1_pos, logits_stage1_neg = None, None
        # if gsu_merged_pos is not None:
        #     logits_stage1_pos = self.gsu_deep_layers(torch.cat([gsu_merged_pos, torch.squeeze(target_pair_pos)], dim=-1))
        # if gsu_merged_neg is not None:
        #     logits_stage1_neg = self.gsu_deep_layers(torch.cat([gsu_merged_neg, torch.squeeze(target_pair_neg)], dim=-1))
        return logits_pos, logits_neg_group, None, None

def multi_modal_pq_layer(inputs, codebook_emb, hash_bits):
        """
            inputs: dense embedding of [M, B, ..., D]  first: module,
            inputs_proj_hash: int (0-num_centroids) embedding of [B, ..., N_Bits], larger distance means similar vectors

            output:
                    shape [mode, batch_size, seq_len, num_bits]
        """
        num_mode = min(inputs.shape[0], codebook_emb.shape[0])

        input_codes_assignment_list = []
        for m in range(num_mode):
            # [B, L, D] -> sub-vectors
            shape_list = list(inputs[m].shape)
            final_dim = int(shape_list[len(shape_list)-1]/hash_bits)
            reshape_list = shape_list
            reshape_list[len(reshape_list)-1] = hash_bits
            reshape_list += [final_dim]
            # [B, L, D] -> [B, L, N_bit, Final_Dim]
            input_reshape = torch.reshape(inputs[m], shape=reshape_list)
            code_list = []
            for b in range(hash_bits):
                # subvector: [B, L, Emb]
                subvector = normalize(input_reshape[:, : , b, :], dim=-1)
                # centroids: [N_Centroids, Emb]
                centroids = normalize(codebook_emb[m][b], dim=-1)
                # [B, L]
                codes = torch.argmax(torch.matmul(subvector, centroids.T), dim=-1)
                code_list.append(codes)
            # shape = [B, L, N_Bit]
            input_codes_assignment = torch.transpose(torch.stack(code_list, dim=1), 1, 2)
            input_codes_assignment_list.append(input_codes_assignment)
        # multi_modal pq, [M, B, L, N_Bits]
        input_codes_multi_modal_assignment = torch.stack(input_codes_assignment_list, dim=0)
        return input_codes_multi_modal_assignment

def test_multi_modal_pq_layer():
    batch_size = 32
    L = 10
    dim = 64
    hash_bits = 16
    num_centroids = 512
    num_mode = 4
    inputs = torch.randn(num_mode, batch_size, L, dim)
    targets = torch.randn(num_mode, 1, L, dim)

    print ("DEBUG: inputs shape %s" % str(inputs.shape))
    codebook_emb = torch.randn(num_mode, hash_bits, num_centroids, int(dim/hash_bits))
    print ("DEBUG: codebook_emb shape %s" % str(codebook_emb.shape))

    input_codes_assignment = multi_modal_pq_layer(inputs, codebook_emb, hash_bits)
    targets_codes_assignment = multi_modal_pq_layer(targets, codebook_emb, hash_bits)
    print("DEBUG: input_codes_assignment shape %s" % str(input_codes_assignment.shape))
    print("DEBUG: targets_codes_assignment shape %s" % str(targets_codes_assignment.shape))
