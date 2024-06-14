## Evaluation

## SIM
python3 run_seminar_model.py --run_mode eval --soft_search_type mhta --data_name kuaisar --model_type sim --model_name kuaisar_sim --multi_modal_emb_enable false --fusion_weight_query 0.5 --max_seq_length 1000 --retrieve_topK 200 --query_keywords_vocab_size 3000000 --log_epoch_step 1
## ETA
python3 run_seminar_model.py --run_mode eval --soft_search_type mhta --data_name kuaisar --model_type eta --model_name kuaisar_eta --multi_modal_emb_enable false --fusion_weight_query 0.5 --max_seq_length 1000 --retrieve_topK 200 --query_keywords_vocab_size 3000000 --log_epoch_step 1
## QIN
python3 run_seminar_model.py --run_mode eval --soft_search_type mhta --data_name kuaisar --model_type qin --model_name kuaisar_qin --multi_modal_emb_enable false --fusion_weight_query 0.5 --max_seq_length 1000 --retrieve_topK 200 --query_keywords_vocab_size 3000000 --log_epoch_step 1
## TWIN
python3 run_seminar_model.py --run_mode eval --soft_search_type mhta --data_name kuaisar --model_type twin --model_name kuaisar_twin --multi_modal_emb_enable false --fusion_weight_query 0.5 --max_seq_length 1000 --retrieve_topK 200 --query_keywords_vocab_size 3000000 --log_epoch_step 1
## SEMINAR
python3 run_seminar_model.py --run_mode eval --soft_search_type mhta --freeze_psu_emb False --data_name kuaisar --model_type seminar --model_name seminar_pretrain_5_train_1_a05_v2 --multi_modal_emb_enable false --fusion_weight_query 0.5 --max_seq_length 1000 --retrieve_topK 200 --query_keywords_vocab_size 3000000 --log_epoch_step 1
