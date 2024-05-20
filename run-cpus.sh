#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <number_of_iterations>"
    exit 1
fi

# Number of iterations
n=$1

# Loop n times
for ((i=0; i<n; i++))
do
    bsub < run-cpu-01-RJ.sh
    echo launched RJ
    sleep 5
    bsub < run-cpu-01-RS.sh
    echo launched RS
    sleep 5
done
