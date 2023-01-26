#!/bin/bash

function in_array {
    ARRAY=$2
    for e in ${ARRAY[*]}
    do
        if [[ "$e" == "$1" ]]
        then
            return 0
        fi
    done
    return 1
}

# parameter setting
task="pouring_skill"
curr_dir=$(pwd)
task_dir=$curr_dir/$task

# *********************************************
# batch folder generate
# *********************************************
cd $task_dir
batch_num_list=$(ls | grep batch | cut -c 6-)

# batch num list to array
ar=("")
for num in $batch_num_list
do
  ar+=($num)
done
echo "batch num array: ${ar[@]}"

# create batch folders
n_batch_folders=3
for i in $(seq 1 1 $n_batch_folders)
do
  if in_array $i "${ar[*]}"
  then
      echo "batch$i already exists!"
  else
      mkdir "batch"$i
  fi
done

# print result
batch_list=$(ls | grep batch)
echo $batch_list

# *********************************************
# copy rollout files
# *********************************************
src_batch="batch1"
src_rollout="rollout_6.h5"
src=$task_dir/$src_batch/$src_rollout
n_rollouts=300

for batch in $batch_list
do
  for i in $(seq 0 1 $n_rollouts)
  do
    dst=$task_dir/$batch/"rollout_"$i".h5"
    if [ -f "$dst" ]; then
      echo "$dst exists"
    else
      cp $src $dst
    fi
  done
  echo "$batch complete!"
done