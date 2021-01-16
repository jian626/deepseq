#!/bin/bash
number=$1
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
		`python enzyme_classifier_model_generator.py $name | tee $tmp`
	done
	count=$(( count + 1 ))
done

