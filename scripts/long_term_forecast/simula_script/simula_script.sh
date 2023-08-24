export CUDA_VISIBLE_DEVICES=0,1
model_name=Autoformer

python -u run_debug.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/simula/ \
  --data_path team_a_complete.csv \
  --model_id team_a_complete_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 9 \
  --dec_in 9 \
  --c_out 9 \
  --des 'Exp' \
  --target 'readiness'\
  --itr 1
