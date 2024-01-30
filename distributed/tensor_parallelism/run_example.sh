#!/usr/bin/env bash

# To run samples:
# bash run_example.sh {file_to_run.py} {num_gpus}
# where file_to_run = example to launch.  Default = 'fsdp_tp_example.py'
# num_gpus = num local gpus to use (must be at least 2). Default = 4

# samples to run include:
# sequence_parallel_example.py
# tensor_parallel_example.py
# fsdp_tp_example.py

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Launching ${1:-fsdp_tp_example.py} with ${2:-$NUM_GPUS} gpus"
torchrun --nnodes=1 --nproc_per_node=${2:-4} --rdzv_id=101 --rdzv_endpoint="localhost:5972" ${1:-fsdp_tp_example.py}
