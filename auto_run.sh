#!/bin/bash
number=$1
dir_name=$2
log_dir=$3
shift
shift
shift
count=0
while [ $count -lt $number ]
do
	for name in "$@" 
	do
		tmp=`basename $name`
		arrIN=(${tmp//./ })
		tmp=${arrIN[0]}
		tmp="${tmp}_${count}_.log"
		python enzyme_classifier_model_generator.py ${dir_name}${name} | tee ${log_dir}/${tmp}
        if [ $? -ne 0 ]
        then
            exit 1
        fi
	done
	count=$(( count + 1 ))
done

