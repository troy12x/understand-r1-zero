# understand-r1-zero

[![PyPI - Version](https://img.shields.io/pypi/v/understand-r1-zero.svg)](https://pypi.org/project/understand-r1-zero)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/understand-r1-zero.svg)](https://pypi.org/project/understand-r1-zero)

-----

## Table of Contents

- [understand-r1-zero](#understand-r1-zero)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [License](#license)

## Installation

```console
# Install dependencies required by oat, the LLM RL training framework we used.
pip install vllm==0.7.2 && pip install oat-llm==0.0.7

# Install this package locally to use the math grader.
pip install -e .
```

## Training

```diff
# Patch LD_LIBRARY_PATH to avoid dependency errors:
export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH

# Run the experiment (tested on 8 x A100-40G):
python train_zero_math.py \
    --critic_type drppo \
    --gpus 8 \
    --enable_prefix_caching \
    --collocate \
    --vllm_sleep \
    --vllm_gpu_ratio 0.35 \
    --gradient-checkpointing \
    --flash-attn \
    --bf16 \
    --rnd-seed \
    --learning_rate 0.000001 \
    --lr_scheduler constant \
    --num_ppo_epochs 1 \
    --beta 0 \
    --oracle_type reward \
    --oracle math \
    --pretrain Qwen/Qwen2.5-Math-1.5B \
    --prompt_template r1 \
    --zero-stage 2 \
    --ref_offload \
    --prompt_data ./datasets/train/math_12k \
    --train_split train \
    --input_key problem \
    --output_key answer \
    --max-train 9999999 \
    --num_prompt_epoch 20 \
    --prompt_max_length 1024 \
    --num_samples 8 \
    --temperature 1 \
    --top_p 1 \
    --generate_max_length 3000 \
    --save_steps -1 \
    --train_batch_size 128 \
    --rollout_batch_size 128 \
    --rollout_batch_size_per_device 16 \
    --pi_buffer_maxlen_per_device 128 \
    --eval_batch_size 200 \
    --eval_steps 16 \
    --eval_temperature 0 \
    --eval_generate_max_length 3000 \
    --eval_data ./datasets/evaluation_suite \
    --eval_input_key input \
    --use-wb \
    --wb-run-name qwen-2.5-Math-1.5b-oss_test-r1template \
    --wb_project oat-zero
```

## Evaluation
```diff
# Evaluate our models:
python evaluate_model.py --model_name sail/Qwen2.5-Math-7B-Oat-Zero
python evaluate_model.py --model_name sail/Qwen2.5-Math-1.5B-Oat-Zero
python evaluate_model.py --model_name sail/Llama-3.2-3B-Oat-Zero --template r1

# Evaluate baseline models:
python evaluate_model.py --model_name Qwen/Qwen2.5-Math-1.5B
python evaluate_model.py --model_name Qwen/Qwen2.5-Math-7B
python evaluate_model.py --model_name hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zero
python evaluate_model.py --model_name PRIME-RL/Eurus-2-7B-PRIME-Zero
python evaluate_model.py --model_name Open-Reasoner-Zero/Open-Reasoner-Zero-7B
```


## License

`understand-r1-zero` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
