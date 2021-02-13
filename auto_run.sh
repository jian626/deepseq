#!/bin/bash
number=$1
dir_name=$2
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
		if [ ! -d ${dir_name}${name} ]
		then
			python enzyme_classifier_model_generator.py ${dir_name}${name} | tee $tmp
		fi
	done
	count=$(( count + 1 ))
done

