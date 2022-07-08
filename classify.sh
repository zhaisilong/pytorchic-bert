python classify.py \
  --task='mrpc' \
  --train_cfg='config/train_mrpc.json' \
  --model_cfg='config/bert_base.json' \
  --data_file="/data/datasets/glue/MRPC/train.tsv" \
  --pretrain_file="/data/models/pytorchic-bert/model_steps_125.pt" \
  --vocab="/data/models/huggingface/bert-base-uncased/vocab.txt" \
  --save_dir='models' \
  --mode='eval'