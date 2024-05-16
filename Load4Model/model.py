import os
import torch
from datasets import load_dataset,Dataset,concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel,prepare_model_for_kbit_training,get_peft_model,AutoPeftModelForCausalLM
from trl import SFTTrainer
from .config import *





def load_model_and_tokenizer(model_name):
  compute_dtype = getattr(torch, "float16")

  bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                                                    #config for Qunatized model
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=False,
  )

  model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
  model.config.use_cache = False
  model.config.pretraining_tp = 1

  # Load LLaMA tokenizer
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"

  return model,tokenizer



def intialialize_lora_configration(model):
  model = prepare_model_for_kbit_training(model)
  peft_config = LoraConfig(
      lora_alpha=16,
      lora_dropout=0.1,
        target_modules=[
  "q_proj",
  "k_proj",
      "v_proj",
    "o_proj",
  "gate_proj",
  "up_proj",
  "down_proj",
  "lm_head",
    ],

    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

  model = get_peft_model(model, peft_config)

  return model,peft_config



# Set training parameters

def set_the_trainning(model,tokenizer,peft_config,train_data,valid_data,output_dir):
  training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

# Set supervised fine-tuning parameters
  trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=valid_data,
    dataset_text_field='formatted_prompt',
    peft_config=peft_config,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# Train model
  trainer.train()
  return trainer