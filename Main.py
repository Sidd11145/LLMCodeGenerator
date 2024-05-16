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
from codebleu import calc_codebleu
import streamlit as st
from datasets_extraction_and_modeification.datasets_Ex_mod import Custom_Prompt_for_Code_Gen,Loading_and_split
from Load4Model.model import *
from functools import partial

# Name of The Model That will be used as Pretrained model
model_name = "NousResearch/Llama-2-7b-hf"
# The Benchmark datasets
dataset_name = "Muennighoff/mbpp"


valid_train_dataset,test_benchmark_data=Loading_and_split(dataset_name)
train_data=valid_train_dataset['train'].shuffle(seed=42).map(Custom_Prompt_for_Code_Gen).remove_columns(['task_id', 'text', 'code', 'test_list', 'test_setup_code', 'challenge_test_list'])
valid_data=valid_train_dataset['test'].shuffle(seed=42).map(Custom_Prompt_for_Code_Gen).remove_columns(['task_id', 'text', 'code', 'test_list', 'test_setup_code', 'challenge_test_list'])

model_A,tokenizer=load_model_and_tokenizer()
conf = model_A.config
print(conf)

#######################################MODEL_B############################################
model_A,tokenizer=load_model_and_tokenizer()
model,peft_config=intialialize_lora_configration(model_A)
trainer=set_the_trainning(model,tokenizer,peft_config,train_data,valid_data,"MODEL_B_RUN")
new_model_name_B = "MODEL_B"
trainer.model.save_pretrained(new_model_name_B)
tokenizer.save_pretrained(new_model_name_B)
tokenizer = AutoTokenizer.from_pretrained(new_model_name_B)
temp_function=partial(new_output,model=Model_B,tokenizer=tokenizer)
test_benchmark_data_code_Model_B=test_benchmark_data.map(temp_function)
#############################################################################################