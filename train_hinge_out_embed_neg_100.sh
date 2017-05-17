model_dir="lm_hinge_out_emb_neg_100"
if [ ! -d $model_dir ]; then
	mkdir $model_dir
fi

python main_hinge.py --train_file data/bobsue.lm.train.txt --dev_file data/bobsue.lm.dev.txt --test_file data/bobsue.lm.test.txt --batch_size 128 --num_epoches 20 --model_file $model_dir/model_hinge_out_embed_neg.th --criterion HingeModelCriterion --model LSTMHingeOutEmbNegModel --learning_rate 0.01 --embedding_size 200 --hidden_size 200 --num_sampled 100 --log_file $model_dir/log.txt 
