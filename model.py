"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
from unsloth import FastLanguageModel
def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    model,tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length, 
        load_in_4bit = True)
    return model,tokenizer

# Step 2 - count_total_parameters
def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Step 3 - is_model_4bit_quantized
from bitsandbytes.nn import Linear4bit 
def is_model_4bit_quantized(model):
    return any(isinstance(m,Linear4bit) for m in model.modules())

# Step 4 - ensure_pad_token
def ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

# Step 5 - get_lora_target_modules
def get_lora_target_modules():
    return ['q_proj', 'k_proj', 'v_proj', 'o_proj']

# Step 6 - attach_lora_adapters
from unsloth import FastLanguageModel
def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
   
    if target_modules is None:
        target_modules = get_lora_target_modules()
    
    
    peft_model = FastLanguageModel.get_peft_model(model,
    target_modules = target_modules,
    r=r,
    lora_alpha=lora_alpha,
    lora_dropout=0,
    bias='none')

    return peft_model

# Step 7 - count_trainable_parameters
def count_trainable_parameters(model):
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Step 8 - trainable_fraction
def trainable_fraction(trainable_count, total_count):
    return float(trainable_count/total_count)

# Step 9 - build_instruction_examples
def build_instruction_examples():
    return [
    {'instruction':'What language is spoken in Türkiye?',
    'response': 'The spoken language in Türkiye is Turkish'},
    {'instruction': 'What is 3x3?',
    'response': 'It equals 9'},
    {'instruction': 'What does the fox say?',
    'response': 'Din ding ding ding, ahi ahi ahiho'}
    ]

# Step 10 - format_instruction_example
def format_instruction_example(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"

# Step 11 - format_all_examples
def format_all_examples(examples):
    result=[]
    for ex in examples:
        result.append(format_instruction_example(ex))
    
    return result

# Step 12 - build_text_dataset
from datasets import Dataset
def build_text_dataset(texts):
    return Dataset.from_dict({'text':texts})

# Step 13 - tokenize_text
def tokenize_text(tokenizer, text):
    return tokenizer.encode(text)

# Step 14 - count_tokens
def count_tokens(input_ids):
    return len(input_ids)

# Step 15 - build_training_arguments
from transformers import TrainingArguments 
import torch
def build_training_arguments(output_dir='./sft_out', max_steps=5, learning_rate=2e-4):
    use_bf16 = torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(output_dir=output_dir,
    max_steps=max_steps,
    per_device_train_batch_size = 1,
    learning_rate = learning_rate,
    gradient_accumulation_steps = 1,
    bf16= use_bf16,
    fp16 = not use_bf16,
    optim='adamw_8bit',
    logging_steps = 1
    )
    return training_args

# Step 16 - build_sft_trainer
from trl import SFTTrainer
def build_sft_trainer(model, tokenizer, dataset, training_args, max_seq_length=256):
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        dataset_text_field = 'text',
        train_dataset = dataset,
        args = training_args,
        max_seq_length = max_seq_length,
        packing=False
    )
    return trainer

# Step 17 - run_sft_training (not yet solved)
# TODO: implement

# Step 18 - switch_to_inference_mode (not yet solved)
# TODO: implement

# Step 19 - build_chat_prompt (not yet solved)
# TODO: implement

# Step 20 - generate_reply (not yet solved)
# TODO: implement

