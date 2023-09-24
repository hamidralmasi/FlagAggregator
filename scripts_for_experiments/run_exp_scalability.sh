#!/bin/bash

########################
########################
########################
####   Scalability  ####
########################
########################
########################

for worker_names in "workers" "workers30" "workers60"
do
	ps_names="servers"
	num_wrk=1000      #The upper bound on the number of workers in any deployment
	r_col=8
	lambda=0
	fps=0
	iter=625     #100000
	dataset='mnist'    #tinyimagenet cifar10 mnist
	optimizer='sgd'
	batch=128
	loss='cross-entropy'
	lr=0.2
	port=29400
	master=''
	num_ps=0
	savedir="Scalability/"
	augmenteddataset="mnistnoisy"
	augmentedfolder="train_lv"
	fw=3

	if [ "$worker_names" = "workers30" ];
	then
  		iter=313
		fw=6
		r_col=16
	fi

	if [ "$worker_names" = "workers60" ];
	then
  		iter=157
		fw=12
		r_col=32
	fi

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
	common="$common --port $port --r_col $r_col --lambda $lambda --augmenteddataset $augmenteddataset --augmentedfolder $augmentedfolder --savedir $savedir"

	for seed in 2021 2022 2023 #2021 2022 2023
	do
		for model in "convnet" #"resnet18" "resnet50"
		do
			for attack in "random" #"random" "drop"
			do
				for gar in "flag" #"flag" "bulyan" "krum" "median" "average" "trmean" "phocas" "meamed"
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
	done
done
