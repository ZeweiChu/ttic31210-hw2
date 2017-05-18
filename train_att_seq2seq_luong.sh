
checkpoint_path="att_seq2seq_luong"
if [ ! -d $checkpoint_path ]; then
	mkdir $checkpoint_path
fi

python main_att.py --train_file data/bobsue.seq2seq.train.tsv --dev_file data/bobsue.seq2seq.dev.tsv --test_file data/bobsue.seq2seq.test.tsv --batch_size 128 --num_epoches 20 --checkpoint_path $checkpoint_path --model AttentionEncoderDecoderModel --criterion LanguageModelCriterion --learning_rate 0.01 --embedding_size 200 --hidden_size 200 --eval_epoch 1 --num_layers 1 --start_from $checkpoint_path 
