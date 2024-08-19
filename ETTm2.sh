if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/TSMixer" ]; then
    mkdir ./logs/LongForecasting/TSMixer
fi
<<<<<<< HEAD
seq_len=512
=======
seq_len=336
>>>>>>> e9653bc9d66b38d0b89d5b0a4cf149caa6566eef
model_name=TSMixer
dataset=ETTm2
num_channels=7

#Best configuration for ETTm2 and 96 frames horizon
pred_len=96
python3 -u run_longExp.py \
  --activation 'relu' \
  --dropout 0.9\
<<<<<<< HEAD
  --hidden_size 256\
  --num_blocks 8 \
=======
  --hidden_size 64\
  --num_blocks 2 \
>>>>>>> e9653bc9d66b38d0b89d5b0a4cf149caa6566eef
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id $dataset'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $num_channels \
  --des 'Exp' \
  --patience 5 \
  --train_epochs 100 \
  --itr 1 --batch_size 32 --learning_rate 0.001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log 

#Best configuration for ETTm2 and 192 frames horizon
pred_len=192
python3 -u run_longExp.py \
  --activation 'relu' \
  --dropout 0.9\
<<<<<<< HEAD
  --hidden_size 256\
=======
  --hidden_size 32\
>>>>>>> e9653bc9d66b38d0b89d5b0a4cf149caa6566eef
  --num_blocks 1 \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id $dataset'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $num_channels \
  --des 'Exp' \
  --patience 5 \
  --train_epochs 100 \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log 

#Best configuration for ETTm2 and 336 frames horizon
pred_len=336
python3 -u run_longExp.py \
  --activation 'relu' \
  --dropout 0.9\
<<<<<<< HEAD
  --hidden_size 512\
  --num_blocks 8 \
=======
  --hidden_size 32\
  --num_blocks 1 \
>>>>>>> e9653bc9d66b38d0b89d5b0a4cf149caa6566eef
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id $dataset'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $num_channels \
  --des 'Exp' \
  --patience 5 \
  --train_epochs 100 \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log 

#Best configuration for ETTm2 and 720 frames horizon
pred_len=720
python3 -u run_longExp.py \
  --activation 'relu' \
  --dropout 0.1\
<<<<<<< HEAD
  --hidden_size 256\
  --num_blocks 8 \
=======
  --hidden_size 64\
  --num_blocks 2 \
>>>>>>> e9653bc9d66b38d0b89d5b0a4cf149caa6566eef
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id $dataset'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $num_channels \
  --des 'Exp' \
  --patience 5 \
  --train_epochs 100 \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log 