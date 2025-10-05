for num_epochs in '20'
do
	for lr in '1e-5'
	do
		for warmup_ratio in '0.2'
		do
			for seed in '2025'
			do
				for batch_size in '64'
				do
					for max_seq in '64'
					do
					  for DR_step in '4'
					  do
					    for weight_diff in '0'
					    do
                    echo ${num_epochs}
                    echo ${lr}
                    echo ${warmup_ratio}
                    echo ${seed}
                    echo ${batch_size}
                    echo ${max_seq}
                    echo ${DR_step}
                    echo ${weight_diff}
                    CUDA_VISIBLE_DEVICES=1 python run.py  \
                    --num_epochs ${num_epochs} \
                    --lr ${lr} \
                    --warmup_ratio ${warmup_ratio} \
                    --seed ${seed} \
                    --batch_size ${batch_size} \
                    --max_seq ${max_seq} \
                    --DR_step ${DR_step} \
                    --weight_diff ${weight_diff}
                  done
                done
					done
				done
			done
		done
	done
done