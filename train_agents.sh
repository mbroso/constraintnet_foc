#!/bin/bash

# Use six different random seeds for training unconstrained FOC and ConstraintNet FOC.
# Experiment ids for unconstrained FOC are starting at 0.
# Experiment ids for clipped unconstrained FOC are starting at 50.
# Experiment ids for ConstraintNet FOC are starting at 100.

for ((i=0;i<6;i+=1))
do 
	echo 
    echo "Training unconstrained FOC with seed $i and experiment id $i:"
    python train.py \
	--config ./options/config_unconstrained.yaml \
    --experiment_id $i \
	--seed $i 

    echo
    echo "Training clipped unconstrained FOC with seed $i and experiment id $(($i + 50)):"
	python train.py \
	--config ./options/config_unconstrained_clipped.yaml \
	--experiment_id $(($i + 50)) \
	--seed $i 
    
    echo
    echo "Training ConstraintNet FOC with seed $i and experiment id $(($i + 100)):"
	python train.py \
	--config ./options/config_ConstraintNet.yaml \
	--experiment_id $(($i + 100)) \
	--seed $i 

done
