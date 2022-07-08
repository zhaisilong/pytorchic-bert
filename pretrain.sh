
CUDA_LAUNCH_BLOCKING=1
python pretrain.py \
  --data_file="/data/datasets/glue/MRPC/train.tsv" \
  --vocab="/data/models/huggingface/bert-base-uncased/vocab.txt" \
  --save_dir='models' \
  --log_dir='logs' \
  --max_len=256