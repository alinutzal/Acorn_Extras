#!/bin/bash
#SBATCH -A m4439_g
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -q regular
#SBATCH -c 32
#SBATCH --gpu-bind=none
#SBATCH --requeue
#SBATCH -o logs/%x-%j.out
#SBATCH -J pipe_40g
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

gpu="40g"
data_set=("meanrms" "minmax")
# data_set=("meanrms")
inf_options=("base" "amp" "tc" "amp_tc")
# inf_options=("base")

pipe_template="examples/v9/pipe/pipe_infer_template.yaml"
infer_template="examples/v9/gnn/gnn_infer_template.yaml"
eval_template="examples/v9/gnn/gnn_eval_template.yaml"
pipe_yaml="tmp/tmp_pipe_infer.yaml"
infer_yaml="tmp/tmp_gnn_infer.yaml"
eval_yaml="tmp/tmp_gnn_eval.yaml"

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
        sed -e "s/{{DATASET}}/$ds/g" "$infer_template" > "$infer_yaml"
        sed -e "s/{{DATASET}}/$ds/g" \
            -e "s/{{GPU}}/$gpu/g" \
            -e "s/{{OPTION}}/$op/g" \
            "$eval_template" > "$eval_yaml"
        sed -e "s/{{GPU}}/$gpu/g" \
            -e "s/{{DATASET}}/$ds/g" \
            -e "s/{{OPTION}}/$op/g" \
            -e "s/{{AMP}}/$AMP/g" \
            -e "s/{{TC}}/$TC/g" \
            -e "s|{{GNN_CONFIG}}|$infer_yaml|g" \
            "$pipe_template" > "$pipe_yaml"
        srun -u acorn infer "$pipe_yaml" > logs/${gpu}/${ds}_ignn2/infer_${gpu}_${ds}_${op}.log 2>&1
        srun -u acorn eval "$eval_yaml" -c "/global/cfs/cdirs/m4439/acorn_model_store/rel_24_v9/gnn/MM_${ds}_ignn2.ckpt"> logs/${gpu}/${ds}_ignn2/eval_${gpu}_${ds}_${op}.log 2>&1
    done
done

echo "All experiments completed!"