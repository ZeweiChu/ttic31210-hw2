model_dir="seq2seq"
if [ ! -d $model_dir ]; then
	mkdir $model_dir
fi
python main_seq2seq.py --train_file data/bobsue.seq2seq.train.tsv --dev_file data/bobsue.seq2seq.dev.tsv --test_file data/bobsue.seq2seq.test.tsv --batch_size 128 --num_epoches 20 --model_file $model_dir/model_seq2seq.th --model EncoderDecoderModel --criterion LanguageModelCriterion --learning_rate 0.01 --embedding_size 200 --hidden_size 200 --eval_epoch 1 --log_file $model_dir/log.tx --log_file $model_dir/log.txt 
