# -*- coding: utf-8 -*-
# @Author  : Anonymous Period

import os
import numpy as np
import random
import torch
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import SASRecDataset, MultiModalSARDataset
from trainers import FinetuneTrainer, MultiModalFinetuneTrainer, MultiModalPretrainTrainer
from long_seq_models.base_model import BasePairSequenceModel, SIM, TWIN, ETA, QIN
from long_seq_models.seminar import SEMINAR
from utils import EarlyStopping, get_user_seqs_and_sample, get_user_id_query_pair_seqs_and_sample, \
    get_item2attribute_json, get_item2attribute_json_simplified, check_path, set_seed

def load_multi_modal_id2emb_ckpt(data_path, data_name):
    """
        item_image_embedding, dict ,key: original asin id, value: tensor
        item_title_embedding, dict ,key: original asin id, value: tensor
    """
    image_emb_file = os.path.join(data_path, data_name + '_id2image_emb.pth')
    title_emb_file = os.path.join(data_path, data_name + '_id2title_emb.pth')

    id2image_emb_dic = torch.load(image_emb_file)
    id2title_emb_dic = torch.load(title_emb_file)
    return id2image_emb_dic, id2title_emb_dic

def init_item_multi_modal_embedding_tensor(item_size, emb_dim, embedding_dic):
    """
        init a tensor shape  [item_size, emb_dim]
        embedding_dic: key: string of iid. e.g. '1', '2'
    """
    item_embedding = np.random.randn(item_size, emb_dim)
    for idx in range(item_size):
        key = str(idx)
        pretrain_emb = embedding_dic[key].numpy() if key in embedding_dic else None
        item_embedding[idx] = pretrain_emb
    item_embedding_tensor = torch.tensor(item_embedding, dtype=torch.float32)
    return item_embedding_tensor

def parse_bool_args(args):
    """
        args:
            "true" or "True", return True
            else: return False
    """
    return args.lower() == "true"

def filter_dataset_to_max_size(data, max_size = None):
    """
        data: dict, key: [0-N]
    """
    if max_size is None or len(data) <= max_size:
        return data
    data_filter_map = {}
    for i in range(max_size):
        data_filter_map[i] = data[i]
    return data_filter_map

def construct_model(args):
    """
        args:
            model_type
    """
    model = None
    if args.model_type == "base":
        model = BasePairSequenceModel(args)
    elif args.model_type == "sim":
        model = SIM(args)
    elif args.model_type == "eta":
        model = ETA(args)
    elif args.model_type == "twin":
        model = TWIN(args)
    elif args.model_type == "qin":
        model = QIN(args)
    elif args.model_type == "seminar":
        model = SEMINAR(args)
    else:
        model = BasePairSequenceModel(args)
    return model

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='Beauty', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--run_mode', default='train', type=str)
    parser.add_argument('--ckp', default=10, type=int, help="pretrain epochs 10, 20, 30...")

    # multi modal sequence input
    parser.add_argument("--query_keywords_vocab_size", type=int, default=200000, help="size of query keywords vocab")
    parser.add_argument("--query_max_token_num", type=int, default=5, help="size of query keywords vocab")
    parser.add_argument("--fusion_type", default='weighed_average', type=str)
    parser.add_argument("--fusion_weight_query", type=float, default=0.5, help="Fusion Weight of Query alpha, Weight of ID 1-alpha")
    parser.add_argument("--soft_search_type", default='mhta', type=str)
    parser.add_argument("--image_emb_size", type=int, default=512, help="size of input image embedding size")
    parser.add_argument("--text_emb_size", type=int, default=512, help="size of input text embedding size")
    parser.add_argument("--multi_modal_emb_enable", type=str, default="false", help="Whether to load multi_modal_emb")

    # long sequence retrieval
    parser.add_argument("--model_type", default='base', type=str)
    parser.add_argument("--retrieve_qin_stage1_topK", type=int, default=500, help="size of retrieval stage1 topK items")
    parser.add_argument("--retrieve_topK", type=int, default=200, help="size of retrieval topk item")
    parser.add_argument("--num_heads", type=int, default=4, help="the number of heads of multi-head attention")

    # model args
    parser.add_argument("--model_name", default='Finetune_sample', type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--pre_batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--pretrain_epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_epoch_step", type=int, default=1000, help="per step in epoch print res")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    # pretrain weight
    parser.add_argument("--loss_next_pair_weight", type=float, default=1.0, help="weight of loss_next_pair")
    parser.add_argument("--loss_multimodal_align_weight", type=float, default=1.0, help="weight loss_multimodal_align")
    parser.add_argument("--loss_qi_relevance_weight", type=float, default=1.0, help="weight of loss_qi_relevance")
    parser.add_argument("--pretrain_next_pair_enable", type=str, default="true", help="If task next pair prediction task is enable..")
    parser.add_argument("--pretrain_multimodal_align_enable", type=str, default="true", help="If task multimodal_align task is enable..")
    parser.add_argument("--pretrain_qi_relevance_enable", type=str, default="true", help="If task qi_relevance task is enable..")

    parser.add_argument("--freeze_psu_emb", type=str, default="false", help="If Restored Embeddings from PSU are freezed...")
    parser.add_argument("--freeze_psu_projection", type=str, default="false", help="If Restored Embeddings Projection from PSU are freezed...")
    parser.add_argument("--freeze_multi_modal_projection", type=str, default="false", help="If Freeze the multi-modal projection...")
    parser.add_argument("--pretrain_max_sample_size", type=int, default=100000000, help="maximum pretraining sample size")

    # test args
    parser.add_argument('--test_set_mode', default='recommendation', type=str)

    args = parser.parse_args()
    ## modify bool args params
    args.multi_modal_emb_enable = parse_bool_args(args.multi_modal_emb_enable)
    args.freeze_psu_emb = parse_bool_args(args.freeze_psu_emb)
    args.freeze_psu_projection = parse_bool_args(args.freeze_psu_projection)
    args.freeze_multi_modal_projection = parse_bool_args(args.freeze_multi_modal_projection)

    args.pretrain_next_pair_enable = parse_bool_args(args.pretrain_next_pair_enable)
    args.pretrain_multimodal_align_enable = parse_bool_args(args.pretrain_multimodal_align_enable)
    args.pretrain_qi_relevance_enable = parse_bool_args(args.pretrain_qi_relevance_enable)

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + args.data_name + '.txt'
    args.queries_file = args.data_dir + args.data_name + '_queries.txt'
    args.sample_file = args.data_dir + args.data_name + '_sample.txt'
    args.sample_neg_queries_file = args.data_dir + args.data_name + '_sample_queries.txt'
    item2attribute_file = args.data_dir + args.data_name + '_item2attributes.json'

    ## multimodal embedding file
    # args.text_emb_file = args.data_dir + args.data_name + '_text.npy'
    # args.image_emb_file = args.data_dir + args.data_name + '_image.npy'

    # user_seq: positive interacted items, sample_seq: negatively sampled items
    # user_seq, max_item, sample_seq = \
    #     get_user_seqs_and_sample(args.data_file, args.sample_file)

    # user_seq: [L], user_queries_seq: [L, N_keywords]
    user_seq, user_queries_seq, max_item, sample_seq, sample_neg_queries_seq = get_user_id_query_pair_seqs_and_sample(args.data_file,
                                                                                              args.queries_file,
                                                                                              args.sample_file,
                                                                                              args.sample_neg_queries_file,
                                                                                              query_max_token_num=args.query_max_token_num)
    print("DEBUG: User 0 Item Sequence Length %d, Query Sequence Length %d" % (
    len(user_seq[0]), len(user_queries_seq[0])))

    item2attribute, attribute_size = get_item2attribute_json_simplified(item2attribute_file,
                                                                        default_attribute_size=150000)
    print("DEBUG: Finish Reading get_item2attribute_json attribute_size size %d, Item 1 Sparse Key Value %s"
          % (attribute_size, str(item2attribute["1"])))

    # item index [0,1,2,...,item_size, <EOS>]: item_size + special token [0] for padding, index:0, token <EOS>, index: [item_size+1]
    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    # save model args
    args_str = f'{args.model_name}-{args.model_type}-{args.data_name}-{args.ckp}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    args.item2attribute = item2attribute

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)
    ## pretraining path
    pretrain_file = f'Pretrain-{args.model_name}-{args.model_type}-{args.data_name}-{args.ckp}.pt'
    pretrained_model_path = os.path.join(args.output_dir, pretrain_file)
    args.pretrained_model_path = pretrained_model_path

    # max_pretrain_dataset_size = 20000
    pretrain_dataset = MultiModalSARDataset(args, filter_dataset_to_max_size(user_seq, args.pretrain_max_sample_size)
                                            , filter_dataset_to_max_size(user_queries_seq, args.pretrain_max_sample_size)
                                            , data_type='pretrain')
    pretrain_sampler = RandomSampler(pretrain_dataset)
    pretrain_dataloader = DataLoader(pretrain_dataset, sampler=pretrain_sampler, batch_size=args.pre_batch_size)

    # train_dataset = SASRecDataset(args, user_seq, data_type='train')
    train_dataset = MultiModalSARDataset(args, user_seq, user_queries_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    ## Todo: test and evaluation dataset negative samples should be separated, sample_neg_queries_seq Different
    eval_dataset = MultiModalSARDataset(args, user_seq, user_queries_seq, test_neg_items=sample_seq, test_neg_queries=sample_neg_queries_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    # test_dataset = SASRecDataset(args, user_seq, test_neg_items=sample_seq, data_type='test')
    test_dataset = MultiModalSARDataset(args, user_seq, user_queries_seq, test_neg_items=sample_seq, test_neg_queries=sample_neg_queries_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    ## load multi_modal item embedding
    multi_modal_emb_dic = None
    if args.multi_modal_emb_enable:
        id2image_emb_dic, id2title_emb_dic = load_multi_modal_id2emb_ckpt(args.data_dir, args.data_name)
        image_emb = init_item_multi_modal_embedding_tensor(args.item_size, args.image_emb_size, id2image_emb_dic)
        text_emb = init_item_multi_modal_embedding_tensor(args.item_size, args.image_emb_size, id2title_emb_dic)
        print ("DEBUG: Finish Loadding Pytorch id2image_emb_dic size %d, image_emb shape %s" % (len(id2image_emb_dic), image_emb.shape))
        print ("DEBUG: Finish Loadding Pytorch id2title_emb_dic size %d, text_emb shape %s" % (len(id2title_emb_dic), text_emb.shape))
        multi_modal_emb_dic = {
            "image": image_emb,
            "text": text_emb
        }
    args.multi_modal_emb_dic = multi_modal_emb_dic

    # construct models
    model = construct_model(args)
    trainer = MultiModalFinetuneTrainer(model, train_dataloader, eval_dataloader, test_dataloader, args)
    pretrain_trainer = MultiModalPretrainTrainer(model, train_dataloader, eval_dataloader, test_dataloader, args)

    ## Runing Mode
    if args.run_mode == "pretrain":
        print(f'##### Starting Pretraining...')
        ## Runing Pretrain Model
        for epoch in range(args.pretrain_epochs):
            pretrain_trainer.pretrain(epoch, pretrain_dataloader)
        torch.save(model.state_dict(), args.pretrained_model_path)
        print(f'Finish Saving model to path {args.pretrained_model_path}...')

    elif args.run_mode == "eval":
        print(f'##### Starting Evaluation...')
        #
        with torch.no_grad():
            # if args.do_eval:
            # trainer.load(args.checkpoint_path)
            trainer.model.load_state_dict(torch.load(args.checkpoint_path))

            print(f'Load model from {args.checkpoint_path} for test!')
            scores, result_info = trainer.test(0, full_sort=False)
            print(args_str)
            print(result_info)
            with open(args.log_file, 'a') as f:
                f.write(args_str + '\n')
                f.write(result_info + '\n')

    else:
        print(f'##### Starting Training...')
        try:
            print(f'### Starting Loading Pretrained Checkpoint from {args.pretrained_model_path}')
            trainer.load(args.pretrained_model_path)
            print(f'### End Loading Checkpoint From {args.pretrained_model_path}!')
            if args.freeze_psu_emb:
                print ("DEBUG: freeze_psu_emb true...")
                for embedding in trainer.model.get_restored_emb():
                    embedding.weight.requires_grad = False
                    # for param in tensor:
                    #     param.requires_grad = False
                print(f'### Finished Freezing PSU parameters!!!')
            else:
                print ("DEBUG: freeze_psu_emb false...")

        except FileNotFoundError:
            print(f'{args.pretrained_model_path} Not Found! The Model is same as Base MultiModal SAR Sequence Model...')

        print(f'##### Starting Training...')
        # save checkpoint in EarlyStopping, can't skip
        early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            scores, _ = trainer.valid(epoch, full_sort=False)
            # # evaluate on
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
               print("Early stopping")
               break
        # Save Model Checkpoint At End of Traning Epoch
        early_stopping.save_checkpoint(0.0, trainer.model)

        print('---------------Sample 99 results-------------------')

        with torch.no_grad():
            # load the best model
            trainer.model.load_state_dict(torch.load(args.checkpoint_path))

            # start evaluation
            scores, result_info = trainer.test(0, full_sort=False)

            print(args_str)
            print(result_info)
            with open(args.log_file, 'a') as f:
                f.write(args_str + '\n')
                f.write(result_info + '\n')

main()
