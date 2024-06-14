#### Reproduce Results
## Evaluate Dataset as Recommendation (test_set_mode=recommendation, Query set to 0).
# SIM
python3 run_seminar_model.py --run_mode eval --test_set_mode recommendation --data_name Movies_and_TV --epochs 1 --model_type sim --model_name amazon_sim_model_mm --multi_modal_emb_enable true --soft_search_type mhta --max_seq_length 100 --retrieve_topK 50 --query_keywords_vocab_size 3000000 --log_epoch_step 1
# ETA
python3 run_seminar_model.py --run_mode eval --test_set_mode recommendation --data_name Movies_and_TV --epochs 1 --model_type eta --model_name amazon_eta_model_mm --multi_modal_emb_enable true --soft_search_type mhta --max_seq_length 100 --retrieve_topK 50 --query_keywords_vocab_size 3000000 --log_epoch_step 1
# QIN
python3 run_seminar_model.py --run_mode eval --test_set_mode recommendation --data_name Movies_and_TV --epochs 1 --model_type qin --model_name amazon_qin_model_mm --multi_modal_emb_enable true --soft_search_type mhta --max_seq_length 100 --retrieve_qin_stage1_topK 80 --retrieve_topK 50 --query_keywords_vocab_size 3000000 --log_epoch_step 1
# TWIN
python3 run_seminar_model.py --run_mode eval --test_set_mode recommendation --data_name Movies_and_TV --epochs 1 --model_type twin --model_name amazon_twin_model_mm --multi_modal_emb_enable true --soft_search_type mhta --max_seq_length 100 --retrieve_topK 50 --query_keywords_vocab_size 3000000 --log_epoch_step 1
# SEMNIAR
python3 run_seminar_model.py --run_mode eval --test_set_mode recommendation --soft_search_type mhta --freeze_psu_emb false --freeze_psu_projection true --freeze_multi_modal_projection false --data_name Movies_and_TV --epochs 1 --model_type seminar --model_name amazon_seminar_pretrain_1_train_1 --multi_modal_emb_enable true --fusion_weight_query 0.5 --max_seq_length 100 --retrieve_topK 50 --query_keywords_vocab_size 3000000 --log_epoch_step 1


## Evaluate Dataset as Search (test_set_mode=search, set query of all examples to the same query of the positive pair given the same query).
# SIM
python3 run_seminar_model.py --run_mode eval --test_set_mode search --data_name Movies_and_TV --epochs 1 --model_type sim --model_name amazon_sim_model_mm --multi_modal_emb_enable true --soft_search_type mhta --max_seq_length 100 --retrieve_topK 50 --query_keywords_vocab_size 3000000 --log_epoch_step 1
# ETA
python3 run_seminar_model.py --run_mode eval --test_set_mode search --data_name Movies_and_TV --epochs 1 --model_type eta --model_name amazon_eta_model_mm --multi_modal_emb_enable true --soft_search_type mhta --max_seq_length 100 --retrieve_topK 50 --query_keywords_vocab_size 3000000 --log_epoch_step 1
# QIN
python3 run_seminar_model.py --run_mode eval --test_set_mode search --data_name Movies_and_TV --epochs 1 --model_type qin --model_name amazon_qin_model_mm --multi_modal_emb_enable true --soft_search_type mhta --max_seq_length 100 --retrieve_qin_stage1_topK 80 --retrieve_topK 50 --query_keywords_vocab_size 3000000 --log_epoch_step 1
# TWIN
python3 run_seminar_model.py --run_mode eval --test_set_mode search --data_name Movies_and_TV --epochs 1 --model_type twin --model_name amazon_twin_model_mm --multi_modal_emb_enable true --soft_search_type mhta --max_seq_length 100 --retrieve_topK 50 --query_keywords_vocab_size 3000000 --log_epoch_step 1
# SEMNIAR
python3 run_seminar_model.py --run_mode eval --test_set_mode search --soft_search_type mhta --freeze_psu_emb false --freeze_psu_projection true --freeze_multi_modal_projection false --data_name Movies_and_TV --epochs 1 --model_type seminar --model_name amazon_seminar_pretrain_1_train_1_a05 --multi_modal_emb_enable true --fusion_weight_query 0.5 --max_seq_length 100 --retrieve_topK 50 --query_keywords_vocab_size 3000000 --log_epoch_step 1

