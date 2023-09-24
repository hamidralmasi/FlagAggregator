#!/bin/bash

########################
########################
########################
######   Lambdas  ######
########################
########################
########################

ps_names="servers"
worker_names="workers7"
num_wrk=1000      #The upper bound on the number of workers in any deployment
r_col=4
lambda=0
fps=0
iter=1120     #100000
dataset='cifar10'    #tinyimagenet cifar10 fmnist
optimizer='sgd'
batch=128
loss='cross-entropy'
lr=0.2            #for resnet, lr=0.2 for cifar. I reduced it to 0.1 for tiny imagenet.
port=29400
master=''
num_ps=0
savedir="lambdas/"
augmenteddataset="none"
augmentedfolder="none"

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
common="$common --port $port --r_col $r_col --augmenteddataset $augmenteddataset  --augmentedfolder $augmentedfolder --savedir $savedir"

for lambda in 64 32 16 8 4 2 1 0 #64 32 16 8 4 2 1 0
do
  for model in "resnet18" #"resnet18"
  do
      for attack in "random" #"random"
      do
          for fw in 1 #0 1 2 3
          do
              for gar in "flag" #"flag" "bulyan" "krum" "median" "meamed" "phocas" "trmean" "average"
              do
                  for seed in 2021 2022 2023 #2021 2022 2023
                  do
                      sleep 5;
                      i=0
                      while read p;
                      do
                          cmd="export PYTHONPATH=/home/miniconda3/envs/garfield/lib/python3.8/site-packages;source /home/$USER/miniconda3/bin/activate;conda activate garfield;ulimit -n 102400;$common --model $model --attack $attack --fw $fw --gar $gar --rank $i --lambda $lambda --seed $seed"
                          ssh $p "$cmd" < /dev/tty &
                          echo "Running $cmd on $p"
                          i=$((i+1))
                      done < $ps_names

                      count_workers=0
                      while read p;
                      do
                          cmd="export PYTHONPATH=/home/miniconda3/envs/garfield/lib/python3.8/site-packages;source /home/$USER/miniconda3/bin/activate;conda activate garfield;ulimit -n 102400;$common --model $model --attack $attack --fw $fw --gar $gar --rank $i --lambda $lambda --seed $seed"
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
done
