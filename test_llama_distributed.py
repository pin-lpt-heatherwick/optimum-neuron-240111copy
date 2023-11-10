import argparse
import os
import time
import torch

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from optimum.neuron.distributed import lazy_load_for_parallelism, ParallelizersManager
from neuronx_distributed.utils.model_utils import move_model_to_device
from neuronx_distributed.parallel_layers.parallel_state import initialize_model_parallel
from optimum.neuron.utils.training_utils import patch_generation_mixin_to_neuron_generation_mixin


def generate(model, tokenizer, prompts, length):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    start = time.time()
    with torch.no_grad():
        sample_output = model.generate(
            input_ids=inputs.input_ids.to("xla"),
            attention_mask=inputs.attention_mask.to("xla"),
            min_length=length,
            max_length=length,
        )
    end = time.time()
    outputs = [tokenizer.decode(tok) for tok in sample_output]
    return outputs, (end - start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="The HF Hub model id or a local directory.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size.",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=2,
        help="The tensor parallelism size.",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=128,
        help="The sequence_length.",
    )
    args = parser.parse_args()

    # Initialize process (note that the group is initialized only once)
    if os.environ.get("TORCHELASTIC_RUN_ID"):
        import torch_xla.distributed.xla_backend as xbn

        if not isinstance(torch.distributed.group.WORLD, xbn.ProcessGroupXla):
            torch.distributed.init_process_group(backend="xla")
    initialize_model_parallel(tensor_model_parallel_size=args.tp_size)

    # Prepare the model for parallelization (it will be compiled only at inference)
    with lazy_load_for_parallelism(tensor_parallel_size=args.tp_size):
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype='auto',
            low_cpu_mem_usage=True,
        )
    patch_generation_mixin_to_neuron_generation_mixin(model)
    model = ParallelizersManager.parallelizer_for_model(model).parallelize(
          model,
          parallelize_embeddings=False,
          sequence_parallel_enabled=False,
    )
    move_model_to_device(model, "xla")

    # Instantiate a tokenizer and configure its padding for open-ended generation
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Generate (will trigger the compilation)
    prompts = ["One of my fondest memory is",] * args.batch_size
    outputs, latency = generate(model, tokenizer, prompts, args.sequence_length)
    print(outputs)
    print(f"{len(outputs)} outputs generated using Neuron model in {latency:.4f} s")


if __name__ == "__main__":
    main()
