#!/bin/bash


########################################
########################################
########################################
##### Mean (Not robust with fw > 0) ####
########################################
########################################
########################################

for fw in 0 # 1 2 3
do
	ps_names="servers"
	worker_names="workers"
	num_wrk=1000      #The upper bound on the number of workers in any deployment
	r_col=8
	lambda=0
	fps=0
	iter=2605     #100000
	optimizer='sgd'
	batch=64
	loss='cross-entropy'
  dataset="cifar10"
	lr=0.2            #for resnet, lr=0.2 for cifar. I reduced it to 0.1 for tiny imagenet.
	port=29400
	master=''
	num_ps=0
	savedir="average/"
	augmenteddataset="none"
  augmentedfolder="none"
  attack="random"
	gar="average"
  lr_update_freq=15

	while read p; do
		num_ps=$((num_ps+1))
		if [ $num_ps -eq  1 ]
		then
			master=$p
		fi
	done < $ps_names


	num_workers=0
	while read p; do
		num_workers=$((num_workers+1))
		if [ $num_workers -eq $num_wrk ]
		then
			break
		fi
	done < $worker_names

	pwd=`pwd`

	common="python3 $PWD/trainer.py --master $master --num_iter $iter --dataset $dataset --batch $batch --loss $loss"
	common="$common --optimizer $optimizer --opt_args '{\"lr\":\"$lr\",\"momentum\":\"0.9\",\"weight_decay\":\"0.0005\"}' --num_ps $num_ps --num_workers $num_workers --fps $fps"
	common="$common --port $port --r_col $r_col --lambda $lambda --augmenteddataset $augmenteddataset --augmentedfolder $augmentedfolder --lr_update_freq $lr_update_freq --savedir $savedir"

	for seed in 2021 2022 2023 #2021 2022 2023
	do
		for model in "resnet18" #"resnet18" "resnet50"
		do
			sleep 5;
			i=0
			while read p;
			do
				cmd="export PYTHONPATH=/home/miniconda3/envs/garfield/lib/python3.8/site-packages;source /home/$USER/miniconda3/bin/activate;conda activate garfield;ulimit -n 102400;$common --model $model --attack $attack --fw $fw --gar $gar --rank $i --seed $seed"
				ssh $p "$cmd" < /dev/tty &
				echo "Running $cmd on $p"
				i=$((i+1))
			done < $ps_names

			count_workers=0
			while read p;
			do
				cmd="export PYTHONPATH=/home/miniconda3/envs/garfield/lib/python3.8/site-packages;source /home/$USER/miniconda3/bin/activate;conda activate garfield;ulimit -n 102400;$common --model $model --attack $attack --fw $fw --gar $gar --rank $i --seed $seed"
				ssh $p "$cmd" < /dev/tty &
				echo "Running $cmd on $p"
				i=$((i+1))
				count_workers=$((count_workers+1))
				if [ $count_workers -eq $num_wrk ]
				then
						break
				fi
			done < $worker_names

			echo "Waiting for the current combination to finish..."
			wait
		done
	done
done
