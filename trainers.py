# -*- coding: utf-8 -*-
# @Author  : Anonymous Period

import traceback
import numpy as np
import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import Adam

from utils import recall_at_k, ndcg_k, get_metric


class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.criterion = nn.BCELoss()

        self.bce_loss = nn.BCELoss()
        # self.ce_loss = nn.CrossEntropyLoss(reduction="mean")

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        HIT_50, NDCG_50, MRR = get_metric(pred_list, 50)

        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "HIT@50": '{:.4f}'.format(HIT_50), "NDCG@50": '{:.4f}'.format(NDCG_50),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR, HIT_50, NDCG_50], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 50]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@50": '{:.4f}'.format(recall[3]), "NDCG@50": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name, exclude_key_list=[]):
        pretrained_state_dict = torch.load(file_name)
        state_dict_filtered = {k: v for k, v in pretrained_state_dict.items() if k not in exclude_key_list}
        self.model.load_state_dict(state_dict_filtered)

    def cross_entropy_next_item_loss(self, logits_pos, logits_neg_batch):
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

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

class PretrainTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(PretrainTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def pretrain(self, epoch, pretrain_dataloader):

        desc = f'AAP-{self.args.aap_weight}-' \
               f'MIP-{self.args.mip_weight}-' \
               f'MAP-{self.args.map_weight}-' \
               f'SP-{self.args.sp_weight}'

        pretrain_data_iter = tqdm.tqdm(enumerate(pretrain_dataloader),
                                       desc=f"{self.args.model_name}-{self.args.data_name} Epoch:{epoch}",
                                       total=len(pretrain_dataloader),
                                       bar_format="{l_bar}{r_bar}")

        self.model.train()
        aap_loss_avg = 0.0
        mip_loss_avg = 0.0
        map_loss_avg = 0.0
        sp_loss_avg = 0.0

        for i, batch in pretrain_data_iter:
            # 0. batch_data will be sent into the device(GPU or CPU)
            batch = tuple(t.to(self.device) for t in batch)
            attributes, masked_item_sequence, pos_items, neg_items, \
            masked_segment_sequence, pos_segment, neg_segment = batch

            aap_loss, mip_loss, map_loss, sp_loss = self.model.pretrain(attributes,
                                            masked_item_sequence, pos_items, neg_items,
                                            masked_segment_sequence, pos_segment, neg_segment)

            joint_loss = self.args.aap_weight * aap_loss + \
                         self.args.mip_weight * mip_loss + \
                         self.args.map_weight * map_loss + \
                         self.args.sp_weight * sp_loss

            self.optim.zero_grad()
            joint_loss.backward()
            self.optim.step()

            aap_loss_avg += aap_loss.item()
            mip_loss_avg += mip_loss.item()
            map_loss_avg += map_loss.item()
            sp_loss_avg += sp_loss.item()

        num = len(pretrain_data_iter) * self.args.pre_batch_size
        post_fix = {
            "epoch": epoch,
            "aap_loss_avg": '{:.4f}'.format(aap_loss_avg /num),
            "mip_loss_avg": '{:.4f}'.format(mip_loss_avg /num),
            "map_loss_avg": '{:.4f}'.format(map_loss_avg / num),
            "sp_loss_avg": '{:.4f}'.format(sp_loss_avg / num),
        }
        print(desc)
        print(str(post_fix))
        with open(self.args.log_file, 'a') as f:
            f.write(str(desc) + '\n')
            f.write(str(post_fix) + '\n')

class FinetuneTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(FinetuneTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, target_pos, target_neg, _ = batch
                # Binary cross_entropy
                sequence_output = self.model.finetune(input_ids)
                loss = self.cross_entropy(sequence_output, target_pos, target_neg)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model.finetune(input_ids)

                    recommend_output = recommend_output[:, -1, :]

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)

class MultiModalFinetuneTrainer(Trainer):
    """
        Support MultiModal Inputs Training
    """
    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(MultiModalFinetuneTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0

            try:
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or CPU)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_id, input_ids, input_queries, target_pos, target_queries_pos, target_neg, target_queries_neg, answer, answer_query = batch

                    ## train
                    # input: (input_ids, input_queries),
                    # output: (target_pos, target_queries_pos)

                    ## process last item prediction
                    target_pos_last = torch.unsqueeze(target_pos[:, -1], -1)
                    target_queries_pos_last = torch.unsqueeze(target_queries_pos[:, -1], -2)
                    target_neg_last = torch.unsqueeze(target_neg[:, -1], -1)
                    target_queries_neg_last = torch.unsqueeze(target_queries_neg[:, -1], -2)

                    ## network
                    logits_pos, logits_neg_group, logits_stage1_pos, logits_stage1_neg = self.model.finetuneV2(input_ids, input_queries, target_pos_last, target_queries_pos_last,
                               target_neg_last, target_queries_neg_last)

                    loss_gsu = self.cross_entropy_next_item_loss(logits_stage1_pos, logits_stage1_neg)
                    loss_esu = self.cross_entropy_next_item_loss(logits_pos, logits_neg_group)
                    loss = loss_gsu + loss_esu

                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    rec_avg_loss += loss.item()
                    rec_cur_loss = loss.item()

                    if i % self.args.log_epoch_step == 0:
                        print ("DEBUG: Epoch %d, Step %d, loss_gsu is %f, loss_esu is %f" % (epoch, i, loss_gsu, loss_esu))

            except Exception as e:
                print ("DEBUG: Error Failed to Iter Data...")
                traceback.print_exc()
                print (e)

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)

                    user_ids, input_ids, input_queries, target_pos, target_queries_pos, target_neg, target_queries_neg, answers, answer_query = batch
                    recommend_output = self.model.finetune(input_ids, input_queries)

                    recommend_output = recommend_output[:, -1, :]

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)

                    ## Test/Eval
                    # input: (input_ids, input_queries),
                    # output:
                    #   positive (answers, answer_query)
                    #   negative (sample_negs, sample_neg_queries)

                    # finetune - v2
                    user_id, input_ids, input_queries, target_pos, target_queries_pos, target_neg, target_queries_neg, answers, answer_query, sample_negs, sample_neg_queries = batch
                    num_negative = sample_negs.shape[-1] # [batch_size, num_negative]
                    # [B, 1, N_keywords] -> [B, num_negative, N_keywords]
                    # sample_negs [B, num_neg] negatively sampled items
                    logits_pos, logits_neg_group, _, _ = self.model.finetuneV2(input_ids, input_queries, answers, answer_query, sample_negs, sample_neg_queries)

                    # [B, 1+num_sample] -> [B, 100]
                    test_logits = torch.cat([logits_pos, logits_neg_group], dim=-1).cpu().detach().numpy().copy()
                    # print ("DEBUG: test_logits shape %s" % str(test_logits.shape))
                    # print ("DEBUG: test_logits %s" % str(test_logits))

                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)
                ##
                # print ("DEBUG: Final Test pred_list.shape: pred_list %s" % str(pred_list.shape))
                return self.get_sample_scores(epoch, pred_list)

class MultiModalPretrainTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(MultiModalPretrainTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def pretrain(self, epoch, pretrain_dataloader):

        desc = f'loss_next_pair_weight-{self.args.loss_next_pair_weight}-' \
               f'loss_multimodal_align_weight-{self.args.loss_multimodal_align_weight}-' \
               f'loss_qi_relevance_weight-{self.args.loss_qi_relevance_weight}-'

        pretrain_data_iter = tqdm.tqdm(enumerate(pretrain_dataloader),
                                       desc=f"{self.args.model_name}-{self.args.data_name} Epoch:{epoch}",
                                       total=len(pretrain_dataloader),
                                       bar_format="{l_bar}{r_bar}")

        self.model.train()
        loss_next_pair_avg = 0.0
        loss_multimodal_align_avg = 0.0
        loss_qi_relevance_avg = 0.0

        for i, batch in pretrain_data_iter:
            batch = tuple(t.to(self.device) for t in batch)
            user_id, input_ids, input_queries, target_pos, target_queries_pos, target_neg, target_queries_neg, answer, answer_query = batch

            # Binary cross_entropy
            # Next Item Prediction Batch
            target_pos_last = torch.unsqueeze(target_pos[:, -1], -1)
            target_queries_pos_last = torch.unsqueeze(target_queries_pos[:, -1], -2)
            target_neg_last = torch.unsqueeze(target_neg[:, -1], -1)
            target_queries_neg_last = torch.unsqueeze(target_queries_neg[:, -1], -2)

            ## pretrain
            loss_next_pair, loss_multimodal_align, loss_qi_relevance = self.model.pretrain(input_ids, input_queries, target_pos_last, target_queries_pos_last
                                                                        , target_neg_last, target_queries_neg_last)

            joint_loss = self.args.loss_next_pair_weight * loss_next_pair + \
                         self.args.loss_multimodal_align_weight * loss_multimodal_align + \
                         self.args.loss_qi_relevance_weight * loss_qi_relevance

            self.optim.zero_grad()
            joint_loss.backward()
            self.optim.step()

            loss_next_pair_avg += loss_next_pair
            loss_multimodal_align_avg += loss_multimodal_align
            loss_qi_relevance_avg += loss_qi_relevance

            if i % self.args.log_epoch_step == 0:
                print("DEBUG: Epoch %d, Batch %d, joint_loss %f, loss_next_pair %f, loss_multimodal_align %f, loss_qi_relevance %f"
                      % (epoch, i, joint_loss, loss_next_pair, loss_multimodal_align, loss_qi_relevance))

        # num = len(pretrain_data_iter) * self.args.pre_batch_size
        num = len(pretrain_data_iter)
        post_fix = {
            "epoch": epoch,
            "loss_next_pair_avg": '{:.4f}'.format(loss_next_pair_avg /num),
            "loss_multimodal_align_avg": '{:.4f}'.format(loss_multimodal_align_avg /num),
            "loss_qi_relevance_weight": '{:.4f}'.format(loss_qi_relevance_avg / num),
        }
        print(desc)
        print(str(post_fix))
        with open(self.args.log_file, 'a') as f:
            f.write(str(desc) + '\n')
            f.write(str(post_fix) + '\n')
