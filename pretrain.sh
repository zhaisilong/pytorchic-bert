DIR="/hy-tmp"

python pretrain.py \
  --data_file="$DIR/datasets/MRPC/train.tsv" \
  --vocab="$DIR/models/bert-base-uncased/vocab.txt" \
  --save_dir='models' \
  --log_dir='logs'