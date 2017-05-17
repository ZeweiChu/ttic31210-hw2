
model_dir="att_seq2seq"
if [ ! -d $model_dir ]; then
	mkdir $model_dir
fi

python main_att.py --train_file data/bobsue.seq2seq.train.tsv --dev_file data/bobsue.seq2seq.dev.tsv --test_file data/bobsue.seq2seq.test.tsv --batch_size 128 --num_epoches 20 --model_file $att_seq2seq/model_seq2seq_bi.th --model AttentionEncoderDecoderModel --criterion LanguageModelCriterion --learning_rate 0.01 --embedding_size 256 --hidden_size 256 --eval_epoch 1 
