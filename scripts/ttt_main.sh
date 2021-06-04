#!/bin/bash
GPUS=1
CPUS=4
JOBNAME="ttt_main"

OUTPUT_PATH=/atlas/u/alekxos/domain_adaptation/mettta/ttt_cifar_release/logs
ERROR_PATH=/atlas/u/alekxos/domain_adaptation/mettta/ttt_cifar_release/logs

cd /atlas/u/alekxos/domain_adaptation/mettta/ttt_cifar_release

echo "Running test-time-training main..."
CHANGE_DIR="cd /atlas/u/alekxos/domain_adaptation/mettta/ttt_cifar_release"
SOURCE=". /atlas/u/alekxos/miniconda3/etc/profile.d/conda.sh"
ENV="conda activate mettta"
WRAP="bash /atlas/u/alekxos/domain_adaptation/mettta/ttt_cifar_release/script_meta.sh"

sbatch --output=$OUTPUT_PATH/ttt_main_%j.out --error=$ERROR_PATH/ttt_main_%j.err \
			  --nodes=1 --ntasks-per-node=1 --mem=8G \
			  --partition=tibet --cpus-per-task=${CPUS} \
				--gres=gpu:titanx:${GPUS} --job-name=${JOBNAME} --wrap="${CHANGE_DIR} && ${SOURCE} && ${ENV} && ${WRAP}"
bash

# done
echo "Done"
