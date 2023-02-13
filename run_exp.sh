#!/bin/bash

ps_names="servers"		# Hostfile for parameter server
worker_names="workers"	# Hostfile for workers
num_wrk=1000			# The upper bound on the number of workers in any deployment
fw=1					# Number of byzantine workers
r_col=3					# Y is m=min(r, p) left singular vectors of G when [Y] to be the flag median of [G_i], 1 <= i <= p 
lambda=0.5				# Regularization parameter
fps=0
iter=500				# Number of training iterations to execute
dataset='cifar10'
model='resnet18'
optimizer='sgd'
batch=200				# Batch size
loss='cross-entropy'
lr=0.2					# Learning rate
gar='flag_median'		# Gradient Aggregation Rule: 'average' 'krum' 'bulyan' 'flag_median'
attack='random' 		# 'random' 'reverse' 'drop' 'lie' 'empire'
port=29200
master=''
num_ps=0
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
#CUDAHOSTCXX=/usr/bin/gcc-5
common="python3 $pwd/trainer.py --master $master --num_iter $iter --dataset $dataset --model $model --batch $batch --loss $loss"
common="$common --optimizer $optimizer --opt_args '{\"lr\":\"$lr\",\"momentum\":\"0.9\",\"weight_decay\":\"0.0005\"}' --num_ps $num_ps --num_workers $num_workers --fw $fw --fps $fps --gar $gar"
common="$common --attack $attack --port $port --r_col $r_col --lambda $lambda"
i=0
while read p; do
	cmd="export PATH=/usr/local/cuda/bin:$PATH;source /home/evl/$USER/miniconda3/bin/activate;conda activate tensorflow1.10.1;$common --rank $i"
	ssh $p "$cmd" < /dev/tty &
        echo "running $cmd on $p"
	i=$((i+1))
done < $ps_names

count_workers=0
while read p; do
        cmd="export PATH=/usr/local/cuda/bin:$PATH;source /home/evl/$USER/miniconda3/bin/activate;conda activate tensorflow1.10.1;$common --rank $i"
        ssh $p "$cmd" < /dev/tty &
        echo "running $cmd on $p"
        i=$((i+1))
	count_workers=$((count_workers+1))
        if [ $count_workers -eq $num_wrk ]
        then
                break
        fi
done < $worker_names
