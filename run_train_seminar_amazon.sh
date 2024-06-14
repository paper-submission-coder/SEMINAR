### Run Amaon Review Dataset
#Total User: 297377, Avg User: 11.4623, Min Len: 5, Max Len: 3508
#Total Item: 59925, Avg Item: 56.8813, Min Inter: 5, Max Inter: 7194
#Iteraction Num: 3408612, Sparsity: 99.98%
#before delete, attribute num:22511
#attributes len, Min:1, Max:102, Avg.:46.7052
# DEBUG: keywords_set size 49787

## Loading Multi-Modal Pretrained Embedding Encoder from .pt file
# DEBUG: Finish Loadding Pytorch id2image_emb_dic size 59925, image_emb shape torch.Size([59927, 512])
# DEBUG: Finish Loadding Pytorch id2title_emb_dic size 59925, text_emb shape torch.Size([59927, 512])
# DEBUG: Total Item Cnt 59925, Match Image Ratio 0.184097, Match Title Ratio 0.184097

# Dataset Preparation
## generate Amazon Movies_and_TV
python3 data_process.py

## generate multi-modal embedding data
python3 generate_multimodal_embedding.py

## Generate Sample data
python3 generate_test.py


#### Train
# SIM
python3 run_seminar_model.py --data_name Movies_and_TV --epochs 1 --model_type sim --model_name amazon_sim_model_mm --multi_modal_emb_enable true --soft_search_type mhta --max_seq_length 100 --retrieve_topK 50 --query_keywords_vocab_size 3000000 --log_epoch_step 1
# ETA
python3 run_seminar_model.py --data_name Movies_and_TV --epochs 1 --model_type eta --model_name amazon_eta_model_mm --multi_modal_emb_enable true --soft_search_type mhta --max_seq_length 100 --retrieve_topK 50 --query_keywords_vocab_size 3000000 --log_epoch_step 1
# QIN
python3 run_seminar_model.py --data_name Movies_and_TV --epochs 1 --model_type qin --model_name amazon_qin_model_mm --multi_modal_emb_enable true --soft_search_type mhta --max_seq_length 100 --retrieve_qin_stage1_topK 80 --retrieve_topK 50 --query_keywords_vocab_size 3000000 --log_epoch_step 1
# TWIN
python3 run_seminar_model.py --data_name Movies_and_TV --epochs 1 --model_type twin --model_name amazon_twin_model_mm --multi_modal_emb_enable true --soft_search_type mhta --max_seq_length 100 --retrieve_topK 50 --query_keywords_vocab_size 3000000 --log_epoch_step 1
# SEMNIAR
python3 run_seminar_model.py --run_mode pretrain --soft_search_type mhta --data_name Movies_and_TV --pretrain_epochs 1 --model_type seminar --model_name amazon_seminar_pretrain_1_train_1 --multi_modal_emb_enable true --fusion_weight_query 0.5 --max_seq_length 100 --retrieve_topK 50 --query_keywords_vocab_size 3000000 --loss_multimodal_align_weight 0.1 --log_epoch_step 1
python3 run_seminar_model.py --run_mode train --soft_search_type mhta --freeze_psu_emb false --freeze_psu_projection true --freeze_multi_modal_projection false --data_name Movies_and_TV --epochs 1 --model_type seminar --model_name amazon_seminar_pretrain_1_train_1 --multi_modal_emb_enable true --fusion_weight_query 0.5 --max_seq_length 100 --retrieve_topK 50 --query_keywords_vocab_size 3000000 --log_epoch_step 1
