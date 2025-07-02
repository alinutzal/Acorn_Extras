#!/bin/bash
#SBATCH -A m4439_g
#SBATCH -C gpu&hbm80g
#SBATCH -N 1
#SBATCH -q regular
#SBATCH -c 32
#SBATCH --gpu-bind=none
#SBATCH --requeue
#SBATCH -o logs/%x-%j.out
#SBATCH -J pipe_80g
#SBATCH --mail-type=ALL
#SBATCH --signal=SIGUSR1@90
#SBATCH --time=05:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --module=gpu,nccl-2.18

export SLURM_CPU_BIND="cores"
export CUDA_LAUNCH_BLOCKING=1
export FI_MR_CACHE_MONITOR=userfaultfd
export NCCL_NET_GDR_LEVEL=PHB

# Setup software
module load python
conda activate /pscratch/sd/a/alazar/conda/acorn

# Create a logs directory if it doesn't exist
mkdir -p logs

gpu="80g"
data_set=("meanrms" "minmax")
inf_options=("base" "amp" "tc" "amp_tc")
# inf_options=("amp" "amp_tc")

template="/global/homes/a/alazar/acorn_yaml/examples/v9/pipe_infer_template.yaml"
gnn_template="/global/homes/a/alazar/acorn_yaml/examples/v9/gnn/gnn_infer_template.yaml"
tmp_yaml="tmp_pipe_infer.yaml"
gnn_tmp_yaml="tmp_gnn_infer.yaml"

# Loop over hyperparameters and run the script
for ds in "${data_set[@]}"
do
    for op in "${inf_options[@]}"
    do
        echo "Running acorn inference for: $gpu $ds $op"
        case $op in
            base)
                AMP="False"
                TC="False"
                ;;
            amp)
                AMP="True"
                TC="False"
                ;;
            tc)
                AMP="False"
                TC="True"
                ;;
            amp_tc)
                AMP="True"
                TC="True"
                ;;
        esac
        sed -e "s/{{DATASET}}/$ds/g" "$gnn_template" > "$gnn_tmp_yaml"
        sed -e "s/{{GPU}}/$gpu/g" \
            -e "s/{{DATASET}}/$ds/g" \
            -e "s/{{OPTION}}/$op/g" \
            -e "s/{{AMP}}/$AMP/g" \
            -e "s/{{TC}}/$TC/g" \
            -e "s|{{GNN_CONFIG}}|$gnn_tmp_yaml|g" \
            "$template" > "$tmp_yaml"
        srun -u acorn infer "$tmp_yaml" > logs/${gpu}/${ds}_ignn2/infer_${gpu}_${ds}_${op}.log 2>&1
    done
done

echo "All experiments completed!"