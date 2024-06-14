# -*- coding: utf-8 -*-
# @Author  : Anonymous Period

import torch
import torch.nn as nn
from modules import Encoder, LayerNorm, QueryKeywordsEncoder
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F

class BasePairSequenceModel(nn.Module):
    """
        Baseline Model for paired Sequence Input of query and item pairs
    """
    def __init__(self, args):
        super(BasePairSequenceModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.attribute_embeddings = nn.Embedding(args.attribute_size, args.hidden_size, padding_idx=0)
        self.query_keywords_embeddings = nn.Embedding(args.query_keywords_vocab_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.query_encoder = QueryKeywordsEncoder()
        self.fusion_type = args.fusion_type
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        self.retrieve_topK = args.retrieve_topK

        ## multi modal embedding initialize by dict
        self.multi_modal_num = 2
        self.multi_modal_emb_dic = args.multi_modal_emb_dic
        self.item_image_embedding_layer = None
        self.item_text_embedding_layer = None
        if self.args.multi_modal_emb_enable:
            self.multi_modal_num = 4
            self.item_image_embedding_layer = nn.Embedding.from_pretrained(self.multi_modal_emb_dic["image"], freeze=True)
            self.item_text_embedding_layer = nn.Embedding.from_pretrained(self.multi_modal_emb_dic["text"], freeze=True)
            ## multi_modal alignment projection
            self.image_emb_proj = nn.Linear(args.image_emb_size, args.hidden_size)
            self.text_emb_proj = nn.Linear(args.text_emb_size, args.hidden_size)
            # channel query, [1, D], randn requires no gradients, need gradient updating
            self.multi_modal_fusion_channel_emb = nn.Linear(args.hidden_size, 1).weight  # shape [1, hidden_size]
            # nn.Embedding(1, args.hidden_size)
            # self.multi_modal_fusion_channel_emb = nn.Embedding(1, args.hidden_size)
            self.multi_modal_fusion_gating_projection = nn.Linear(self.args.hidden_size, self.multi_modal_num-1) # e.g. 4-1=3

        self.fusion_weight_query = args.fusion_weight_query

        ## mhta
        self.params = {}
        self.num_heads = args.num_heads
        self.params["num_heads"] = args.num_heads
        self.wQ_list = [nn.Linear(args.hidden_size, args.hidden_size) for _ in range(self.num_heads)]
        self.wK_list = [nn.Linear(args.hidden_size, args.hidden_size) for _ in range(self.num_heads)]
        self.wV_list = [nn.Linear(args.hidden_size, args.hidden_size) for _ in range(self.num_heads)]
        self.wO = nn.Linear(self.num_heads * args.hidden_size, args.hidden_size)
        # init.xavier_uniform_(self.wQ.weight)
        # init.xavier_uniform_(self.wK.weight)
        # init.xavier_uniform_(self.wV.weight)
        # init.xavier_uniform_(self.wO.weight)

        # add unique dense layer for 4 losses respectively
        self.aap_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.mip_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.map_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.sp_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.criterion = nn.BCELoss(reduction='none')
        self.apply(self.init_weights)

        ## Long Sequence GSU/ESU final output
        self.final_deep_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.gsu_deep_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    # AAP
    def associated_attribute_prediction(self, sequence_output, attribute_embedding):
        '''
        :param sequence_output: [B L H]
        :param attribute_embedding: [arribute_num H]
        :return: scores [B*L tag_num]
        '''
        sequence_output = self.aap_norm(sequence_output) # [B L H]
        sequence_output = sequence_output.view([-1, self.args.hidden_size, 1]) # [B*L H 1]
        # [tag_num H] [B*L H 1] -> [B*L tag_num 1]
        score = torch.matmul(attribute_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1)) # [B*L tag_num]

    # MIP sample neg items
    def masked_item_prediction(self, sequence_output, target_item):
        '''
        :param sequence_output: [B L H]
        :param target_item: [B L H]
        :return: scores [B*L]
        '''
        sequence_output = self.mip_norm(sequence_output.view([-1,self.args.hidden_size])) # [B*L H]
        target_item = target_item.view([-1,self.args.hidden_size]) # [B*L H]
        score = torch.mul(sequence_output, target_item) # [B*L H]
        return torch.sigmoid(torch.sum(score, -1)) # [B*L]

    # MAP
    def masked_attribute_prediction(self, sequence_output, attribute_embedding):
        sequence_output = self.map_norm(sequence_output)  # [B L H]
        sequence_output = sequence_output.view([-1, self.args.hidden_size, 1])  # [B*L H 1]
        # [tag_num H] [B*L H 1] -> [B*L tag_num 1]
        score = torch.matmul(attribute_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1)) # [B*L tag_num]

    # SP sample neg segment
    def segment_prediction(self, context, segment):
        '''
        :param context: [B H]
        :param segment: [B H]
        :return:
        '''
        context = self.sp_norm(context)
        score = torch.mul(context, segment) # [B H]
        return torch.sigmoid(torch.sum(score, dim=-1)) # [B]

    #
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def get_item_id_emb(self, input_sequence_id):
        item_id_embeddings = self.item_embeddings(input_sequence_id)
        return item_id_embeddings

    def get_query_emb(self, input_query_keywords_ids):
        """
            Args:
                input_query_keywords_ids, [B, L, N_keyword, D]
            Output:
                query_embedding [B, L, D]
        """
        query_raw_embeddings = self.query_keywords_embeddings(input_query_keywords_ids)
        query_embedding = self.query_encoder(query_raw_embeddings)
        return query_embedding

    def stack_multimodal_embedding(self, input_sequence_id, input_query_keywords_ids):
        """
            inputs:
            output:
                multi_modal_emb [Query, Attributes, Text, Image]
                [M, B, L, D]
        """
        ## [B, L, D]
        item_id_embeddings = self.get_item_id_emb(input_sequence_id)
        ## [B, L, N_keywords, D] -> [B, L, D]
        query_embedding_encode = self.get_query_emb(input_query_keywords_ids)
        multi_modal_emb = torch.stack([query_embedding_encode, item_id_embeddings], dim=0)
        return multi_modal_emb

    def get_multi_modal_fusion_gating(self):
        """
            output: tensor of shape [1, num_channel]
        """
        # [1, num_modal -1]
        gating_logits = self.multi_modal_fusion_gating_projection(self.multi_modal_fusion_channel_emb)
        gating = F.softmax(gating_logits, dim=-1)
        return gating

    def get_fused_query_item_pair(self, query_emb, item_emb, fusion_weight_query):
        """
            a= fusion_weight_query
            return  a * query_embedding + (1 - a) * item_embedding
        """
        fused_emb = fusion_weight_query * query_emb + (1.0 - fusion_weight_query) * item_emb
        return fused_emb

    def fuse_multimodal_embedding_v2(self, input_sequence_id, input_query_keywords_ids, fusion_type):
        """
            @input_sequence_ids: shape: [B, L] type: int
            @input_query_keywords_ids: shape: [B, L, N_keywords] type: int

            output:
                [B, num_sample, D]
        """
        ## [B, L, N_keywords, D]
        query_embedding_encode = self.get_query_emb(input_query_keywords_ids)

        ## final merged item embeddings
        item_embeddings = None
        if self.args.multi_modal_emb_enable:
            id_embeddings = self.get_item_id_emb(input_sequence_id)
            image_embedding = self.image_emb_proj(self.item_image_embedding_layer(input_sequence_id))
            text_embedding = self.text_emb_proj(self.item_text_embedding_layer(input_sequence_id))
            gating = self.get_multi_modal_fusion_gating() # [1, M]
            # torch.stack(sequence_channels, dim=-1) shape: [B, L, D, M],  gating.T shape [M, 1], reduce last dimension
            item_embeddings = torch.matmul(torch.stack([id_embeddings, image_embedding, text_embedding], dim=-1), gating.T).squeeze(-1)
            # print ("DEBUG: id_embeddings %s, image_embedding %s, text_embedding %s, gating %s, item_embeddings %s"
            #        % (str(id_embeddings.shape), str(image_embedding.shape), str(text_embedding.shape), str(gating.shape), str(item_embeddings.shape)))
        else:
            id_embeddings = self.get_item_id_emb(input_sequence_id)
            item_embeddings = id_embeddings

        ## Item Embedding Fusion
        if fusion_type == "add":
            sequence_emb = item_embeddings + query_embedding_encode
        elif fusion_type == "weighed_average":
            sequence_emb = self.get_fused_query_item_pair(query_embedding_encode, item_embeddings, self.fusion_weight_query)
        elif fusion_type == "fusion_softmax":
            self.fusion_gating_projection = nn.Linear(self.args.hidden_size, self.multi_modal_num)  # [D, M]
            # [B, D] mm [D, M] -> [B, M]
            gating_output = F.softmax(self.fusion_gating_projection(item_embeddings), dim=-1)
            # [B, D, M]
            multi_channel_input = torch.stack([query_embedding_encode, item_embeddings], dim=-1)
            # [B, D, M] \mm [B, M, 1] -> [B, D, 1]
            sequence_emb = torch.bmm(multi_channel_input, gating_output.unsqueeze(-1)).squeeze()
        elif fusion_type == "fusion_gumbel_softmax":
            self.fusion_query_item = nn.Linear(self.args.hidden_size, 2)  # [D, 2]
            # [B, D] mm [D, M] -> [B, M]
            # gumbel_softmax hard=False, reparametrization trick
            gating_output = F.gumbel_softmax(self.fusion_query_item(item_embeddings), tau=1.0, hard=False, dim=-1)
            # [B, D, M]
            multi_channel_input = torch.stack([query_embedding_encode, item_embeddings], dim=-1)
            # [B, D, M] \mm [B, M, 1] -> [B, D, 1]
            sequence_emb = torch.bmm(multi_channel_input, gating_output.unsqueeze(-1)).squeeze()
        else:
            ## merge embedding
            sequence_emb = item_embeddings + query_embedding_encode
        return sequence_emb

    def pretrain(self, attributes, masked_item_sequence, pos_items,  neg_items,
                  masked_segment_sequence, pos_segment, neg_segment):

        # Encode masked sequence
        sequence_emb = self.add_position_embedding(masked_item_sequence)
        sequence_mask = (masked_item_sequence == 0).float() * -1e8
        sequence_mask = torch.unsqueeze(torch.unsqueeze(sequence_mask, 1), 1)

        encoded_layers = self.item_encoder(sequence_emb,
                                          sequence_mask,
                                          output_all_encoded_layers=True)
        # [B L H]
        sequence_output = encoded_layers[-1]

        attribute_embeddings = self.attribute_embeddings.weight
        # AAP
        aap_score = self.associated_attribute_prediction(sequence_output, attribute_embeddings)
        aap_loss = self.criterion(aap_score, attributes.view(-1, self.args.attribute_size).float())
        # only compute loss at non-masked position
        aap_mask = (masked_item_sequence != self.args.mask_id).float() * \
                         (masked_item_sequence != 0).float()
        aap_loss = torch.sum(aap_loss * aap_mask.flatten().unsqueeze(-1))

        # MIP
        pos_item_embs = self.item_embeddings(pos_items)
        neg_item_embs = self.item_embeddings(neg_items)
        pos_score = self.masked_item_prediction(sequence_output, pos_item_embs)
        neg_score = self.masked_item_prediction(sequence_output, neg_item_embs)
        mip_distance = torch.sigmoid(pos_score - neg_score)
        mip_loss = self.criterion(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32))
        mip_mask = (masked_item_sequence == self.args.mask_id).float()
        mip_loss = torch.sum(mip_loss * mip_mask.flatten())

        # MAP
        map_score = self.masked_attribute_prediction(sequence_output, attribute_embeddings)
        map_loss = self.criterion(map_score, attributes.view(-1, self.args.attribute_size).float())
        map_mask = (masked_item_sequence == self.args.mask_id).float()
        map_loss = torch.sum(map_loss * map_mask.flatten().unsqueeze(-1))

        # SP
        # segment context
        segment_context = self.add_position_embedding(masked_segment_sequence)
        segment_mask = (masked_segment_sequence == 0).float() * -1e8
        segment_mask = torch.unsqueeze(torch.unsqueeze(segment_mask, 1), 1)
        segment_encoded_layers = self.item_encoder(segment_context,
                                               segment_mask,
                                               output_all_encoded_layers=True)

        # take the last position hidden as the context
        segment_context = segment_encoded_layers[-1][:, -1, :]# [B H]
        # pos_segment
        pos_segment_emb = self.add_position_embedding(pos_segment)
        pos_segment_mask = (pos_segment == 0).float() * -1e8
        pos_segment_mask = torch.unsqueeze(torch.unsqueeze(pos_segment_mask, 1), 1)
        pos_segment_encoded_layers = self.item_encoder(pos_segment_emb,
                                                   pos_segment_mask,
                                                   output_all_encoded_layers=True)
        pos_segment_emb = pos_segment_encoded_layers[-1][:, -1, :]

        # neg_segment
        neg_segment_emb = self.add_position_embedding(neg_segment)
        neg_segment_mask = (neg_segment == 0).float() * -1e8
        neg_segment_mask = torch.unsqueeze(torch.unsqueeze(neg_segment_mask, 1), 1)
        neg_segment_encoded_layers = self.item_encoder(neg_segment_emb,
                                                       neg_segment_mask,
                                                       output_all_encoded_layers=True)
        neg_segment_emb = neg_segment_encoded_layers[-1][:, -1, :] # [B H]

        pos_segment_score = self.segment_prediction(segment_context, pos_segment_emb)
        neg_segment_score = self.segment_prediction(segment_context, neg_segment_emb)

        sp_distance = torch.sigmoid(pos_segment_score - neg_segment_score)

        sp_loss = torch.sum(self.criterion(sp_distance,
                                           torch.ones_like(sp_distance, dtype=torch.float32)))

        return aap_loss, mip_loss, map_loss, sp_loss

    # Fine tune
    # same as SASRec
    def finetuneV1(self, input_ids, input_query_keywords_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # sequence_emb = self.add_position_embedding(input_ids)
        sequence_emb = self.fuse_multimodal_embedding_v2(input_ids, input_query_keywords_ids, self.fusion_type)
        print ("DEBUG: BasePairSequenceModel sequence_emb input shape input_ids %s| input query_keywords_ids %s |self.fusion_type %s"
               % (str(input_ids.shape), str(input_query_keywords_ids.shape), self.fusion_type))
        print ("DEBUG: BasePairSequenceModel sequence_emb output shape %s" % str(sequence_emb.shape))

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def finetuneV2(self, input_ids, input_query_keywords_ids, target_item_id_pos, target_query_pos, target_item_id_neg, target_query_neg):
        """
            input:
                input_ids: [B, L], historical Id
                input_query_keywords_ids: [B, L, N_keywords],  historical Id
                target_item_id_pos: [B, 1], last item of target positive item id sequence
                target_query_pos:   [B, 1, N_keywords], last query of target positive query sequence
                target_item_id_neg: [B, 1], last item of target negative item id sequence
                target_query_neg:   [B, 1], last query of target negative query sequence

                or
                target_item_id_neg: [B, num_negative], num_negative item of target negative item id sequence
                target_query_neg:   [B, num_negative], num_negative query of target negative query sequence

            output:
                logits_pos:  [B, 1]
                logits_neg:  [B, 1]
        """
        ## 1.0 Inputs Processing Layers  [B, L, D]
        attention_mask = (input_ids > 0).long()

        sequence_emb = self.fuse_multimodal_embedding_v2(input_ids, input_query_keywords_ids, self.fusion_type)
        target_pair_pos = self.fuse_multimodal_embedding_v2(target_item_id_pos, target_query_pos, self.fusion_type)
        target_pair_neg = self.fuse_multimodal_embedding_v2(target_item_id_neg, target_query_neg, self.fusion_type)

        ## Positive
        gsu_output_pos, gsu_merged_pos = self.soft_search(sequence_emb, target_pair_pos, attention_mask, self.retrieve_topK)
        esu_output_pos, _ = self.multi_head_target_attention_single_input(gsu_output_pos, target_pair_pos, self.params)
        logits_pos = self.final_deep_layers(torch.cat([esu_output_pos, torch.squeeze(target_pair_pos)], dim=-1))
        logits_pos_gsu_merged = self.gsu_deep_layers(torch.cat([gsu_merged_pos, torch.squeeze(target_pair_pos)], dim=-1))

        ## Negative
        ## Negative Sample
        ## Negative Sample
        logits_neg_list = []
        logits_neg_gsu_merged_list = []
        num_negative = target_item_id_neg.shape[-1]
        for i in range(num_negative):
            # [B, D] -> [B, 1, D]
            target_pair_neg_i = target_pair_neg[:, i, :].unsqueeze(1)
            # print ("DEBUG: Processing Negative Sample %d, shape %s" % (i, str(target_pair_neg_i.shape)))
            gsu_output_neg_list_i, gsu_merged_neg_i = self.soft_search(sequence_emb, target_pair_neg_i, attention_mask, self.retrieve_topK)
            esu_output_neg_i, _ = self.multi_head_target_attention_single_input(gsu_output_neg_list_i, target_pair_neg_i, self.params)
            logits_neg_i = self.final_deep_layers(torch.cat([esu_output_neg_i, torch.squeeze(target_pair_neg_i)], dim=-1))
            logits_neg_gsu_merged_i = self.gsu_deep_layers(torch.cat([gsu_merged_neg_i, torch.squeeze(target_pair_neg_i)], dim=-1))

            logits_neg_list.append(logits_neg_i)
            logits_neg_gsu_merged_list.append(logits_neg_gsu_merged_i)

        logits_neg_group = torch.cat(logits_neg_list, dim=-1)
        logits_neg_gsu_merged_group = torch.cat(logits_neg_gsu_merged_list, dim=-1)
        # print ("DEBUG: Final logits_neg_group shape %s" % str(logits_neg_group.shape))
        return logits_pos, logits_neg_group, None, None

    def multi_head_target_attention_single_input(self, sequence_emb, target_emb, params):
        """
            input:
                sequence_emb_list: [B, L, D] list size num_head, single input of retrieved topk sequence (v.s. TWIN multi inputs)
                target_emb: [B, 1, D]

            output:
                mhta_output: [B, D]
                attention_mask_list: list of attention_mask [B, L]
        """
        num_heads = params["num_heads"]
        head_list = []
        emb_dim = sequence_emb.shape[-1]

        attention_mask_list = []
        for i in range(num_heads):
            # [B, 1, D]  [B, D, L] ->  [B, 1, L] -> softmax output: [B, L] -> attention_mask [B, L]
            target_emb_proj = self.wQ_list[i](target_emb)
            sequence_emb_proj = self.wK_list[i](sequence_emb)
            value_emb_proj = self.wV_list[i](sequence_emb)
            # [B, L]
            attention_mask = torch.softmax(
                torch.squeeze(torch.bmm(target_emb_proj, torch.transpose(sequence_emb_proj, 1, 2))) / np.sqrt(emb_dim),
                dim=-1)
            attention_mask_list.append(attention_mask)
            # attention_mask: [B,L] value_emb_proj: [B, L, D] ->  [B, D, L]  [B, L, 1] -> [B, D, 1] -> [B, D]
            head_out = torch.squeeze(
                torch.bmm(torch.transpose(value_emb_proj, 1, 2), torch.unsqueeze(attention_mask, -1)))
            # print("DEBUG: The %d-th head shape %s" % (i, str(head_out.shape)))
            head_list.append(head_out)
        mhta_output = self.wO(torch.cat(head_list, dim=-1))
        return mhta_output, attention_mask_list

    def multi_head_target_attention(self, sequence_emb_list, target_emb, params):
        """
            input:
                sequence_emb_list: [B, L, D] list size num_head
                target_emb: [B, 1, D]

            output:
                mhta_output: [B, D]
                attention_mask_list: list of attention_mask [B, L]
        """
        num_heads = params["num_heads"]
        head_list = []
        emb_dim = sequence_emb_list[0].shape[-1]

        attention_mask_list = []
        for i in range(num_heads):
            # [B, 1, D]  [B, D, L] ->  [B, 1, L] -> softmax output: [B, L] -> attention_mask [B, L]
            target_emb_proj = self.wQ_list[i](target_emb)
            sequence_emb_proj = self.wK_list[i](sequence_emb_list[i])
            value_emb_proj = self.wV_list[i](sequence_emb_list[i])
            # [B, L]
            attention_mask = torch.softmax(
                torch.squeeze(torch.bmm(target_emb_proj, torch.transpose(sequence_emb_proj, 1, 2))) / np.sqrt(emb_dim),
                dim=-1)
            attention_mask_list.append(attention_mask)
            # attention_mask: [B,L] value_emb_proj: [B, L, D] ->  [B, D, L]  [B, L, 1] -> [B, D, 1] -> [B, D]
            head_out = torch.squeeze(
                torch.bmm(torch.transpose(value_emb_proj, 1, 2), torch.unsqueeze(attention_mask, -1)))
            # print("DEBUG: The %d-th head shape %s" % (i, str(head_out.shape)))
            head_list.append(head_out)
        mhta_output = self.wO(torch.cat(head_list, dim=-1))
        return mhta_output, attention_mask_list

    def soft_search(self, user_seq_emb, target_emb, attention_mask, K):
        """
            input:
                user_seq_emb: [B, L, D]
                target_emb: [B, 1, D]

            output:
                gsu_out_topk     [B, K, D]
                gsu_merged   [B, D]  Stage1 Merged User Representation, Pass the tensor to stage 1 loss
        """
        # Base Implement SoftSearch follows SIM method
        gsu_out_topk = user_seq_emb[:, -K:, :]
        gsu_merged = torch.mean(user_seq_emb, dim=1)
        return gsu_out_topk, gsu_merged

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class SIM(BasePairSequenceModel):

    def __init__(self, args):
        # 父类init方法
        super(SIM, self).__init__(args)
        # SIM的init方法
        self.wQ = nn.Linear(args.hidden_size, args.hidden_size)
        self.wK = nn.Linear(args.hidden_size, args.hidden_size)

    def soft_search(self, user_seq_emb, target_emb, attention_mask, K):
        """
            input:
                user_seq_emb: [B, L, D]
                target_emb: [B, 1, D]
                attention_mask: [B, L], two types of mask implementation, mask in attention calculation or masked to retrieval emdbedding

            output:
                gsu_out_topk [B, K, D]
                gsu_merged   [B, D]
                Stage1 Merged User Representation, Pass the gsu merged tensor to calculate loss of stage 1
        """
        # Base Implement SoftSearch follows SIM method
        batch_size, sequence_len, emb_dim = user_seq_emb.shape
        target_emb_proj = self.wQ(target_emb)     # [B, 1, D] \times [D, D] -> [B, 1, D]
        user_seq_emb_proj = self.wK(user_seq_emb) # [B, L, D] \times [D, D] -> [B, L, D]
        # SIM Maximum Inner Product
        qK = torch.squeeze(torch.bmm(user_seq_emb_proj, torch.transpose(target_emb_proj, 1, 2)))   # [B,L,D] \times [B, D, 1] -> [B, L, 1] -> [B, L]
        # print ("DEBUG: Model SIM qK shape %s" % str(qK.shape))

        # values: [B, K], indices: [B, K]
        values, indices = torch.topk(qK, K, dim=-1, largest=True)
        # user_seq_emb: [B, L, D], index= [B, index_length=K]
        gather_index = indices.unsqueeze(-1).expand(-1, -1, emb_dim).to(dtype=torch.int64)
        # user_seq_emb [B, L, D] -> [B, K, D]
        # [B, L, D] [B, L, D] -> [B, L, D]
        user_seq_emb_mask = torch.mul(user_seq_emb, torch.tile(attention_mask.unsqueeze(-1), dims=(1, 1, emb_dim)))
        gsu_out_topk = torch.gather(user_seq_emb_mask, dim=1, index=gather_index, out=None)

        ## SIM merge, [B, L, D], [B, L, 1] -> [B, D]
        gsu_out_merge = torch.bmm(torch.transpose(user_seq_emb, 1, 2), torch.unsqueeze(qK, dim=-1)).squeeze()

        return gsu_out_topk, gsu_out_merge

    def finetuneV2(self, input_ids, input_query_keywords_ids, target_item_id_pos, target_query_pos, target_item_id_neg, target_query_neg):
        """
            input:
                input_ids: [B, L], historical Id
                input_query_keywords_ids: [B, L, N_keywords],  historical Id
                target_item_id_pos: [B, 1], last item of target positive item id sequence
                target_query_pos:   [B, 1, N_keywords], last query of target positive query sequence
                target_item_id_neg: [B, 1], last item of target negative item id sequence
                target_query_neg:   [B, 1], last query of target negative query sequence

                or
                target_item_id_neg: [B, num_negative], num_negative item of target negative item id sequence
                target_query_neg:   [B, num_negative], num_negative query of target negative query sequence

            output:
                logits_pos:  [B, 1]
                logits_neg:  [B, 1]
        """
        ## 1.0 Inputs Processing Layers  [B, L, D]
        attention_mask = (input_ids > 0).long()

        sequence_emb = self.fuse_multimodal_embedding_v2(input_ids, input_query_keywords_ids, self.fusion_type)
        target_pair_pos = self.fuse_multimodal_embedding_v2(target_item_id_pos, target_query_pos, self.fusion_type)
        target_pair_neg = self.fuse_multimodal_embedding_v2(target_item_id_neg, target_query_neg, self.fusion_type)

        ## Positive
        gsu_output_pos, gsu_merged_pos = self.soft_search(sequence_emb, target_pair_pos, attention_mask, self.retrieve_topK)
        esu_output_pos, _ = self.multi_head_target_attention_single_input(gsu_output_pos, target_pair_pos, self.params)
        logits_pos = self.final_deep_layers(torch.cat([esu_output_pos, torch.squeeze(target_pair_pos)], dim=-1))
        logits_pos_gsu_merged = self.gsu_deep_layers(torch.cat([gsu_merged_pos, torch.squeeze(target_pair_pos)], dim=-1))

        ## Negative
        ## Negative Sample
        ## Negative Sample
        logits_neg_list = []
        logits_neg_gsu_merged_list = []
        num_negative = target_item_id_neg.shape[-1]
        for i in range(num_negative):
            # [B, D] -> [B, 1, D]
            target_pair_neg_i = target_pair_neg[:, i, :].unsqueeze(1)
            # print ("DEBUG: Processing Negative Sample %d, shape %s" % (i, str(target_pair_neg_i.shape)))
            gsu_output_neg_list_i, gsu_merged_neg_i = self.soft_search(sequence_emb, target_pair_neg_i, attention_mask, self.retrieve_topK)
            esu_output_neg_i, _ = self.multi_head_target_attention_single_input(gsu_output_neg_list_i, target_pair_neg_i, self.params)
            logits_neg_i = self.final_deep_layers(torch.cat([esu_output_neg_i, torch.squeeze(target_pair_neg_i)], dim=-1))
            logits_neg_gsu_merged_i = self.gsu_deep_layers(torch.cat([gsu_merged_neg_i, torch.squeeze(target_pair_neg_i)], dim=-1))

            logits_neg_list.append(logits_neg_i)
            logits_neg_gsu_merged_list.append(logits_neg_gsu_merged_i)

        logits_neg_group = torch.cat(logits_neg_list, dim=-1)
        logits_neg_gsu_merged_group = torch.cat(logits_neg_gsu_merged_list, dim=-1)
        # print ("DEBUG: Final logits_neg_group shape %s" % str(logits_neg_group.shape))
        return logits_pos, logits_neg_group, logits_pos_gsu_merged, logits_neg_gsu_merged_group

class ETA(BasePairSequenceModel):

    def __init__(self, args):
        # parent __init__
        super(ETA, self).__init__(args)
        # child __init__ method
        self.hash_bits = 16
        self.hash_proj_matrix = nn.Linear(args.hidden_size, self.hash_bits)
        # freeze hash proj matrix after initialization
        for param in self.hash_proj_matrix.parameters():
            param.requires_grad = False

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

    def soft_search(self, user_seq_emb, target_emb, attention_mask, K):
        """
            input:
                user_seq_emb: [B, L, D]
                target_emb: [B, 1, D]
                attention_mask: [B, L]

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
        # Applies Attention Mask
        user_seq_emb_mask = torch.mul(user_seq_emb, torch.tile(attention_mask.unsqueeze(-1), dims=(1, 1, emb_dim)))
        # user_seq_emb [B, L, D] -> [B, K, D]
        gsu_out_topk = torch.gather(user_seq_emb_mask, dim=1, index=gather_index, out=None)
        # print ("DEBUG: user_seq_emb_topK shape %s" % str(gsu_out_topk.shape))
        ## Generate First Stage Merged Sequence Representation
        # gsu_merged = torch.squeeze(torch.bmm(torch.transpose(user_seq_emb, 1, 2), torch.unsqueeze(qK, axis=-1)), axis=-1)
        ## SIM merge, [B, L, D], [B, L, 1] -> [B, D]

        attention = F.softmax(qk_hamming_distance.to(dtype=torch.float32), dim=-1)
        gsu_out_merge = torch.bmm(torch.transpose(user_seq_emb, 1, 2), torch.unsqueeze(attention, dim=-1)).squeeze()
        return gsu_out_topk, gsu_out_merge

class TWIN(BasePairSequenceModel):

    def __init__(self, args):
        # parent __init__ method
        super(TWIN, self).__init__(args)
        # child __init__ method
        # self.wQ_gsu = [nn.Linear(args.hidden_size, args.hidden_size) for _ in range(self.num_heads)]
        # self.wK_gsu = [nn.Linear(args.hidden_size, args.hidden_size) for _ in range(self.num_heads)]

    def soft_search_list(self, user_seq_emb, target_emb, attention_mask, K):
        """
            input:
                user_seq_emb: [B, L, D]
                target_emb: [B, 1, D] or [B, num_negative, D]
                attention_mask: [B, L]  0,1
                K: int
            output:
                gsu_out_topk     [B, K, D, H]
                gsu_merged   [B, D, H]

                Stage1 Merged User Representation, Pass the gsu merged tensor to calculate loss of stage 1
                Return Original Sequence Embedding: User
        """
        # Stage 1
        # mhta_out_stage1, attention_mask_list = self.multi_head_target_attention(user_seq_emb, target_emb, self.params)
        num_heads = self.params["num_heads"]
        emb_dim = user_seq_emb.shape[-1]

        attention_list = []
        gsu_output_topK_emb_list = [] # list of [B, K, D]
        for i in range(num_heads):
            # [B, 1, D]  [B, D, L] ->  [B, 1, L] -> softmax output: [B, L] -> attention_mask [B, L]
            target_emb_proj = self.wQ_list[i](target_emb)
            user_seq_emb_proj = self.wK_list[i](user_seq_emb)
            # value_emb_proj = self.wV_list[i](user_seq_emb)
            # [B, L]
            # attention_mask = torch.softmax(
            #     torch.squeeze(torch.bmm(target_emb_proj, torch.transpose(sequence_emb_proj, 1, 2))) / np.sqrt(emb_dim),
            #     dim=-1)
            qK = torch.squeeze(torch.bmm(target_emb_proj, torch.transpose(user_seq_emb_proj, 1, 2)))  # [B,L,D] \times [B, D, 1] -> [B, L, 1] -> [B, L]
            attention = torch.softmax(qK/np.sqrt(emb_dim), dim=-1)
            attention_list.append(attention)
            # Current Attention Mask Cut Top K
            values, indices = torch.topk(qK, K, dim=-1, largest=True)
            # user_seq_emb: [B, L, D], index= [B, index_length=K]
            gather_index = indices.unsqueeze(-1).expand(-1, -1, emb_dim).to(dtype=torch.int64)
            # Applies Attention Mask, PADDING Token [PAD] Embedding to 0
            user_seq_emb_mask = torch.mul(user_seq_emb, torch.tile(attention_mask.unsqueeze(-1), dims=(1, 1, emb_dim)))
            # user_seq_emb [B, L, D] -> [B, K, D]
            gather_topk_emb = torch.gather(user_seq_emb_mask, dim=1, index=gather_index, out=None)
            gsu_output_topK_emb_list.append(gather_topk_emb)
        return gsu_output_topK_emb_list, None

    def multi_head_target_attention(self, sequence_emb_list, target_emb, params):
        """
            input:
                sequence_emb_list: [B, L, D] list size num_head
                target_emb: [B, 1, D]

            output:
                mhta_output: [B, D]
                attention_mask_list: list of attention_mask [B, L]
        """
        num_heads = params["num_heads"]
        head_list = []
        emb_dim = sequence_emb_list[0].shape[-1]

        attention_mask_list = []
        for i in range(num_heads):
            # [B, 1, D]  [B, D, L] ->  [B, 1, L] -> softmax output: [B, L] -> attention_mask [B, L]
            target_emb_proj = self.wQ_list[i](target_emb)
            sequence_emb_proj = self.wK_list[i](sequence_emb_list[i])
            value_emb_proj = self.wV_list[i](sequence_emb_list[i])
            # [B, L]
            attention_mask = torch.softmax(
                torch.squeeze(torch.bmm(target_emb_proj, torch.transpose(sequence_emb_proj, 1, 2))) / np.sqrt(emb_dim),
                dim=-1)
            attention_mask_list.append(attention_mask)
            # attention_mask: [B,L] value_emb_proj: [B, L, D] ->  [B, D, L]  [B, L, 1] -> [B, D, 1] -> [B, D]
            head_out = torch.squeeze(
                torch.bmm(torch.transpose(value_emb_proj, 1, 2), torch.unsqueeze(attention_mask, -1)))
            # print("DEBUG: The %d-th head shape %s" % (i, str(head_out.shape)))
            head_list.append(head_out)
        mhta_output = self.wO(torch.cat(head_list, dim=-1))
        return mhta_output, attention_mask_list

    def finetuneV2(self, input_ids, input_query_keywords_ids, target_item_id_pos, target_query_pos, target_item_id_neg, target_query_neg):
        """
            input:
                input_ids: [B, L], historical Id
                input_query_keywords_ids: [B, L, N_keywords],  historical Id
                target_item_id_pos: [B, 1], last item of target positive item id sequence
                target_query_pos:   [B, N_keywords], last query of target positive query sequence
                target_item_id_neg: [B, num_sample] or [B, num_negative], last item of target negative item id sequence
                target_query_neg:   [B, num_sample, N_keywords], last query of target negative query sequence

            output:
                logits_pos:  [B, 1]
                logits_neg:  [B, 1]
        """
        ## 1.0 Inputs Processing Layers  [B, L, D]
        attention_mask = (input_ids > 0).long()
        # extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        # print ("DEBUG: attention_mask shape: %s" % str(attention_mask.shape))
        # print ("DEBUG: extended_attention_mask shape: %s" % str(extended_attention_mask.shape))

        # sequence_emb = self.add_position_embedding(input_ids)
        sequence_emb = self.fuse_multimodal_embedding_v2(input_ids, input_query_keywords_ids, self.fusion_type)
        # target_pair_pos,target_pair_neg shape: [B, 1, D]
        target_pair_pos = self.fuse_multimodal_embedding_v2(target_item_id_pos, target_query_pos, self.fusion_type)
        gsu_output_pos_list, gsu_merged_pos = self.soft_search_list(sequence_emb, target_pair_pos, attention_mask, self.retrieve_topK)
        esu_output_pos, _ = self.multi_head_target_attention(gsu_output_pos_list, target_pair_pos, self.params)
        logits_pos = self.final_deep_layers(torch.cat([esu_output_pos, torch.squeeze(target_pair_pos)], dim=-1))

        ## Negative Sample
        target_pair_neg = self.fuse_multimodal_embedding_v2(target_item_id_neg, target_query_neg, self.fusion_type)
        logits_neg_list = []
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
        # print ("DEBUG: Final logits_neg_group shape %s" % str(logits_neg_group.shape))
        return logits_pos, logits_neg_group, None, None

class QIN(BasePairSequenceModel):

    def __init__(self, args):
        #
        super(QIN, self).__init__(args)
        #
        self.wQ = nn.Linear(args.hidden_size, args.hidden_size)
        self.wK = nn.Linear(args.hidden_size, args.hidden_size)
        self.retrieve_topK1 = args.retrieve_qin_stage1_topK
        self.retrieve_topK2 = args.retrieve_topK

    def soft_search(self, user_seq_emb, target_emb, attention_mask, K):
        """
            input:
                user_seq_emb: [B, L, D]
                target_emb: [B, 1, D]
                attention_mask: [B, L]
            output:
                gsu_out_topk [B, K, D]
                gsu_merged   [B, D]
                Stage1 Merged User Representation, Pass the gsu merged tensor to calculate loss of stage 1
        """
        # Base Implement SoftSearch follows SIM method
        batch_size, sequence_len, emb_dim = user_seq_emb.shape
        target_emb_proj = self.wQ(target_emb)     # [B, 1, D] \times [D, D] -> [B, 1, D]
        user_seq_emb_proj = self.wK(user_seq_emb) # [B, L, D] \times [D, D] -> [B, L, D]
        # SIM Maximum Inner Product
        qK = torch.squeeze(torch.bmm(user_seq_emb_proj, torch.transpose(target_emb_proj, 1, 2)))   # [B,L,D] \times [B, D, 1] -> [B, L, 1] -> [B, L]
        # print ("DEBUG: Model SIM qK shape %s" % str(qK.shape))

        attn_values, indices = torch.topk(qK, K, dim=-1, largest=True)
        # user_seq_emb: [B, L, D], index= [B, index_length=K]
        gather_index = indices.unsqueeze(-1).expand(-1, -1, emb_dim).to(dtype=torch.int64)
        # user_seq_emb [B, L, D] -> [B, K, D]
        # Applies Attention Mask, PADDING Token [PAD] Embedding to 0
        user_seq_emb_mask = torch.mul(user_seq_emb, torch.tile(attention_mask.unsqueeze(-1), dims=(1, 1, emb_dim)))
        gsu_out_topk = torch.gather(user_seq_emb_mask, dim=1, index=gather_index, out=None)

        # print ("DEBUG: Model SIM gsu_out_topk shape %s" % str(gsu_out_topk.shape))
        ## Generate First Stage Merged Sequence Representation
        gsu_merged = torch.squeeze(torch.bmm(torch.transpose(user_seq_emb, 1, 2), torch.unsqueeze(qK, axis=-1)), axis=-1)
        # print ("DEBUG: Model SIM gsu_merged shape %s" % str(gsu_merged.shape))
        return gsu_out_topk, None

    def finetuneV2(self, input_ids, input_query_keywords_ids, target_item_id_pos, target_query_pos, target_item_id_neg, target_query_neg):
        """
            input:
                input_ids: [B, L-2], historical Id
                input_query_keywords_ids: [B, L-2],  historical Id
                target_item_id_pos: [B, 1], last item of target positive item id sequence
                target_query_pos:   [B, 1], last query of target positive query sequence
                target_item_id_neg: [B, 1], last item of target negative item id sequence
                target_query_neg:   [B, 1], last query of target negative query sequence

            output:
                logits_pos:  [B, 1]
                logits_neg:  [B, 1]
        """
        ## 1.0 Inputs Processing Layers  [B, L, D]
        attention_mask = (input_ids > 0).long()

        seq_id_emb = self.get_item_id_emb(input_ids)
        seq_query_emb = self.get_query_emb(input_query_keywords_ids)
        target_id_emb_pos = self.get_item_id_emb(target_item_id_pos)
        target_query_emb_pos = self.get_query_emb(target_query_pos)
        target_id_emb_neg = self.get_item_id_emb(target_item_id_neg)
        target_query_emb_neg = self.get_query_emb(target_query_neg)

        # ## Stage1-1: Query-Relevant Top K1 Items
        # rsu_output_stage1_pos, _ = self.soft_search(seq_id_emb, target_query_emb_pos, self.retrieve_topK1)
        # rsu_output_stage1_neg, _ = self.soft_search(seq_id_emb, target_query_emb_neg, self.retrieve_topK1)
        #
        # ## Stage1-2: Target Item Top K2 Items
        # rsu_output_stage2_pos, _ = self.soft_search(rsu_output_stage1_pos, target_id_emb_pos, self.retrieve_topK2)
        # rsu_output_stage2_neg, _ = self.soft_search(rsu_output_stage1_neg, target_id_emb_neg, self.retrieve_topK2)
        #
        # ## Stage 2: ESU Search Unit
        # esu_output_pos, _ = self.multi_head_target_attention(rsu_output_stage2_pos, target_id_emb_pos, self.params)
        # esu_output_neg, _ = self.multi_head_target_attention(rsu_output_stage2_neg, target_id_emb_neg, self.params)


        ## Positive Sample
        # Stage1-1: Query-Relevant Top K1 Items,
        # Stage1-2: Target Item Top K2 Items
        # Stage 2: ESU Search Unit
        rsu_output_stage1_pos, _ = self.soft_search(seq_id_emb, target_query_emb_pos, attention_mask, self.retrieve_topK1)
        second_stage_attention_mask_pos = torch.ones((rsu_output_stage1_pos.shape[0], rsu_output_stage1_pos.shape[1]))
        rsu_output_stage2_pos, _ = self.soft_search(rsu_output_stage1_pos, target_id_emb_pos, second_stage_attention_mask_pos, self.retrieve_topK2)
        esu_output_pos, _ = self.multi_head_target_attention_single_input(rsu_output_stage2_pos, target_id_emb_pos, self.params)
        logits_pos = self.final_deep_layers(torch.cat([esu_output_pos, torch.squeeze(target_id_emb_pos)], dim=-1))

        ## Negative Sample
        logits_neg_list = []
        num_negative = target_item_id_neg.shape[-1]
        for i in range(num_negative):
            # [B, D] -> [B, 1, D]
            target_id_emb_neg_i = target_id_emb_neg[:, i, :].unsqueeze(1)
            target_query_emb_neg_i = target_query_emb_neg[:, i, :].unsqueeze(1)
            rsu_output_stage1_neg, _ = self.soft_search(seq_id_emb, target_query_emb_neg_i, attention_mask, self.retrieve_topK1)
            second_stage_attention_mask_neg = torch.ones((rsu_output_stage1_neg.shape[0], rsu_output_stage1_neg.shape[1]))
            rsu_output_stage2_neg, _ = self.soft_search(rsu_output_stage1_neg, target_id_emb_neg_i, second_stage_attention_mask_neg, self.retrieve_topK2)

            esu_output_neg_i, _ = self.multi_head_target_attention_single_input(rsu_output_stage2_neg, target_id_emb_neg_i, self.params)
            logits_neg_i = self.final_deep_layers(torch.cat([esu_output_neg_i, torch.squeeze(target_id_emb_neg_i)], dim=-1))
            logits_neg_list.append(logits_neg_i)
        logits_neg_group = torch.cat(logits_neg_list, dim=-1)
        return logits_pos, logits_neg_group, None, None

def test_soft_search():
    batch_size = 32
    seq_len = 200
    K = 50
    emb_dim = 8
    user_seq_emb = nn.Linear(emb_dim, emb_dim)(torch.randn(batch_size, seq_len, emb_dim))
    target_emb = nn.Linear(emb_dim, emb_dim)(torch.randn(batch_size, 1, emb_dim))


def test_eta_hashes():
    def hamming_distance(query_hashes, keys_hashes):
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

    def hash_emb_layer(inputs):
        """
            inputs:  dense embedding of [B, ..., D]
            inputs_proj_hash: int (0/1) embedding of [B, ..., N_Bits]
        """
        # [B, ..., D] -> [B, ..., N_bits]
        inputs_proj = nn.Linear(hidden_size, hash_bits)(inputs)
        inputs_proj = torch.unsqueeze(inputs_proj, dim = -1) # [B, N_Bit] -> [B, N_Bit, 1]
        inputs_proj_merge = torch.cat([-1.0 * inputs_proj, inputs_proj], axis=-1)  # [B, N_Bit, 1] -> [B, N_Bit, 2]
        inputs_proj_hash = torch.argmax(inputs_proj_merge, dim=-1)
        return inputs_proj_hash

    hidden_size = 4
    hash_bits = 2

    batch_size = 8
    seq_len = 5

    query = torch.randn(batch_size, 1, hidden_size)
    keys = torch.randn(batch_size, seq_len, hidden_size)

    query_hashes = hash_emb_layer(query)
    key_hashes = hash_emb_layer(keys)
    distance = hamming_distance(query_hashes, key_hashes)

def test_gather_emb():
    batch_size = 4
    seq_len = 10
    emb_dim = 8
    attn_indices = torch.Tensor([[1, 2], [4, 3], [7, 2], [5, 8]])
    user_seq_emb = torch.rand((batch_size, seq_len, emb_dim))
    # indics [B, N] -> [B, N, D]
    gather_index = torch.tensor(attn_indices.unsqueeze(-1).expand(-1, -1, emb_dim), dtype=torch.int64)
    # user_seq_emb [B, L, D]
    gather_topk_emb = torch.gather(user_seq_emb, dim=1, index=gather_index, out=None)
