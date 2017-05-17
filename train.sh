
model_dir="lm_logloss"
if [ ! -d $model_dir ]; then
	mkdir $model_dir
fi

python main.py --train_file data/bobsue.lm.train.txt --dev_file data/bobsue.lm.dev.txt --test_file data/bobsue.lm.test.txt --batch_size 128 --num_epoches 1 --model_file $model_dir/model_lstm.th --criterion LanguageModelCriterion --model LSTMModel --learning_rate 0.01 --log_file $model_dir/log.txt --embedding_size 200 --hidden_size 200
