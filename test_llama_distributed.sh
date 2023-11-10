#!/bin/bash
export OMP_NUM_THREADS=16
torchrun --nproc_per_node=2 test_llama_distributed.py NousResearch/LLama-2-7b-hf
