"""Tiny random CPU model checks dependency APIs, masked loss, adapter save and merge.

This is not MiniCPM training, BF16 validation, or a GPU benchmark.
"""
import argparse
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import LlamaConfig,LlamaForCausalLM,Trainer,TrainingArguments,set_seed
from peft import LoraConfig,get_peft_model
from camera_sft.common import new_dir,write_json

p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=new_dir(a.out)
set_seed(42)
model=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
    num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=64,use_cache=False))
model=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,task_type='CAUSAL_LM',target_modules=['q_proj','v_proj']))
rows=[{'input_ids':[1,2,3,4,5],'attention_mask':[1]*5,'labels':[-100,-100,-100,4,5]}]*2
def collate(batch):return {k:torch.tensor([r[k] for r in batch]) for k in rows[0]}
trainer=Trainer(model=model,args=TrainingArguments(output_dir=str(out/'checkpoints'),use_cpu=True,max_steps=1,
    per_device_train_batch_size=1,gradient_accumulation_steps=1,learning_rate=1e-3,report_to='none',
    gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},save_strategy='no',logging_steps=1),
    train_dataset=rows,data_collator=collate)
result=trainer.train();assert math.isfinite(result.training_loss)
assert any(torch.count_nonzero(p).item()>0 for n,p in model.named_parameters() if 'lora_B' in n)
model.save_pretrained(out/'adapter')
model.eval();batch=collate(rows[:1]);batch.pop('labels')
with torch.no_grad():before=model(**batch).logits
merged=model.merge_and_unload(safe_merge=True);merged.save_pretrained(out/'merged')
restored=LlamaForCausalLM.from_pretrained(out/'merged',local_files_only=True).eval()
with torch.no_grad():after=restored(**batch).logits
assert torch.allclose(before,after,atol=1e-5,rtol=1e-5)
write_json(out/'result.json',{'status':'pass','scope':'tiny random CPU API probe only','train_loss':result.training_loss,
    'max_logit_difference':float((before-after).abs().max()),'cuda_validated':False,'minicpm_weights_used':False})
print('PASS: tiny CPU Trainer/PEFT/save/merge/reload; no MiniCPM GPU claims.')
