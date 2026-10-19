"""Cloud-only BF16 training/evaluation CLI. No cloud credentials or board operations."""
import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from .common import ROOT, read_json, read_jsonl, write_json, write_jsonl, sha, digest, contract, new_dir
from .data import verify_release, review_report
from .protocol import training_examples, render_prefix, score_first, exact_answer

CFG=read_json(ROOT/'configs/base.json')


def runtime_info():
    packages={}
    for name in ['torch','transformers','peft','accelerate','tokenizers','safetensors']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    commit=None;dirty=None;source_kind='unversioned_copy'
    if (ROOT/'.git').exists():
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip());source_kind='git'
    elif (ROOT/'SOURCE_COMMIT').is_file():
        commit=(ROOT/'SOURCE_COMMIT').read_text(encoding='utf-8').strip();source_kind='git_archive'
    return {'python':platform.python_version(),'packages':packages,'git_commit':commit,
        'git_dirty':dirty,'source_kind':source_kind,
        'code_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'camera_sft').glob('*.py'))}}


def fetch(a):
    from huggingface_hub import snapshot_download
    out=new_dir(a.model_dir)
    patterns=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja']
    if not a.tokenizer_only:patterns+=['*.safetensors','*.safetensors.index.json']
    snapshot_download(CFG['model_id'],revision=CFG['revision'],local_dir=str(out),allow_patterns=patterns)
    files={str(p.relative_to(out)).replace('\\','/'):sha(p) for p in sorted(out.glob('*')) if p.is_file()}
    if not a.tokenizer_only and not any(k.endswith('.safetensors') for k in files):raise ValueError('No checkpoint weights downloaded')
    for name in ['config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja']:
        if name not in files:raise ValueError('Missing model asset: '+name)
    write_json(out/'snapshot_manifest.json',{'model_id':CFG['model_id'],'revision':CFG['revision'],
        'tokenizer_only':a.tokenizer_only,'files':files,'status':'downloaded_not_runtime_validated'})
    print(f'Snapshot saved to {out}; tokenizer_only={a.tokenizer_only}')


def check_model(path, weights=False):
    p=Path(path);m=read_json(p/'snapshot_manifest.json')
    if m['model_id']!=CFG['model_id'] or m['revision']!=CFG['revision']:raise ValueError('base identity mismatch')
    if weights and m['tokenizer_only']:raise ValueError('Tokenizer-only snapshot cannot train/evaluate')
    for name,value in m['files'].items():
        if Path(name).name!=name or sha(p/name)!=value:raise ValueError('model file changed: '+name)
    return m


def tokenizer(path):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    if tok.pad_token_id is None:raise ValueError('Missing native pad token; do not grow vocabulary')
    return tok


def prepare(a):
    verify_release(a.data);check_model(a.model_dir)
    review=review_report(a.data,a.reviews,require_independent=not a.smoke) if a.reviews else None
    if review is not None and not review['ok']:raise ValueError('Review incomplete/rejected or independence requirement not met')
    if not a.smoke and review is None:raise ValueError('Independent semantic review required; use --smoke only for pipeline validation')
    out=new_dir(a.out);tok=tokenizer(a.model_dir);counts={};lengths=[];examples=[]
    for split in ['train','dev']:
        features=[]
        for row in read_jsonl(Path(a.data)/f'{split}.jsonl'):
            for feat,evidence in training_examples(tok,row,a.max_length):
                features.append(feat);lengths.append(evidence['input_tokens'])
                # Store all boundaries and token IDs, but only a few decoded examples.
                examples.append({**{k:v for k,v in evidence.items() if k not in ['prefix_text','target_text']},
                    'split':split,'prefix_sha256':digest(evidence['prefix_text']),
                    'target_text':evidence['target_text'], 'masked_prefix_matches_inference':True})
        write_jsonl(out/f'{split}.jsonl',features);counts[split]=len(features)
    write_jsonl(out/'template-audit.jsonl',examples)
    # All frozen 50 prompts must also fit; gold never enters their inference prefixes.
    for r in read_json(ROOT/'data/frozen/cases50.json'):
        n=len(tok.encode(render_prefix(tok,r['request']['messages'],r['request']['tools']),add_special_tokens=False))
        if n+CFG['max_new_tokens']>a.max_length:raise ValueError('Frozen prompt exceeds context: '+r['id'])
    write_json(out/'manifest.json',{'status':'prepared_not_trained','smoke_only':a.smoke,'max_length':a.max_length,
        'model_id':CFG['model_id'],'revision':CFG['revision'],'data_manifest_sha256':sha(Path(a.data)/'manifest.json'),
        'snapshot_manifest_sha256':sha(Path(a.model_dir)/'snapshot_manifest.json'),
        'counts':counts,'max_observed_length':max(lengths),'files':{f:sha(out/f) for f in ['train.jsonl','dev.jsonl','template-audit.jsonl']},
        'reviews_sha256':sha(a.reviews) if a.reviews else None,
        'review_summary':review,
        'loss':'assistant target + EOS only; each assistant turn is one example; all prompt/tool result tokens masked',
        'fixed_50_prompt_fit':'pass'})
    print(json.dumps({'counts':counts,'max_length':max(lengths),'smoke_only':a.smoke},indent=2))


def gpu():
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 is required; no silent CPU/FP32 fallback')
    return torch


def load_base(path):
    check_model(path,weights=True)
    torch=gpu()
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,attn_implementation='sdpa')
    if next(model.parameters()).dtype!=torch.bfloat16:raise ValueError('Model is not BF16')
    return model


def train(a):
    torch=gpu();p=Path(a.prepared);m=read_json(p/'manifest.json');check_model(a.model_dir,True)
    if m['smoke_only'] and not a.smoke:raise ValueError('Prepared data is smoke-only; acknowledge with --smoke')
    if m['snapshot_manifest_sha256']!=sha(Path(a.model_dir)/'snapshot_manifest.json'):raise ValueError('model/prepare snapshot mismatch')
    for f,value in m['files'].items():
        if sha(p/f)!=value:raise ValueError('Prepared data changed: '+f)
    # MVP deliberately single GPU. Multi-GPU adds scheduling/accumulation variables.
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('MVP train is single-process; do not torchrun it')
    out=new_dir(a.out);tok=tokenizer(a.model_dir);model=load_base(a.model_dir)
    from peft import LoraConfig,get_peft_model
    from transformers import Trainer,TrainingArguments,TrainerCallback,set_seed
    set_seed(a.seed)
    model=get_peft_model(model,LoraConfig(r=CFG['lora_rank'],lora_alpha=CFG['lora_alpha'],lora_dropout=CFG['lora_dropout'],
        bias='none',task_type='CAUSAL_LM',target_modules=CFG['target_modules']))
    model.config.use_cache=False
    class Rows(torch.utils.data.Dataset):
        def __init__(self,f):self.rows=read_jsonl(f)
        def __len__(self):return len(self.rows)
        def __getitem__(self,i):return self.rows[i]
    def collate(rows):
        n=max(len(r['input_ids']) for r in rows)
        return {k:torch.tensor([r[k]+[(-100 if k=='labels' else tok.pad_token_id if k=='input_ids' else 0)]*(n-len(r[k])) for r in rows]) for k in ['input_ids','attention_mask','labels']}
    args=TrainingArguments(output_dir=str(out/'checkpoints'),num_train_epochs=a.epochs,max_steps=a.max_steps,
        per_device_train_batch_size=CFG['micro_batch'],per_device_eval_batch_size=1,
        gradient_accumulation_steps=CFG['gradient_accumulation_steps'],learning_rate=CFG['learning_rate'],
        warmup_ratio=0.03,lr_scheduler_type='cosine',bf16=True,fp16=False,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},eval_strategy='epoch',save_strategy='epoch',
        save_total_limit=2,logging_steps=1,logging_nan_inf_filter=False,report_to='none',seed=a.seed,data_seed=a.seed,dataloader_num_workers=0)
    initial={'status':'running','config':CFG,'seed':a.seed,'epochs':a.epochs,'max_steps':a.max_steps,
        'prepared_manifest_sha256':sha(p/'manifest.json'),'smoke_only':m['smoke_only'],
        'runtime':runtime_info(),'device':torch.cuda.get_device_name(0)}
    write_json(out/'run.json',initial)
    class FiniteLoss(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            for key,value in (logs or {}).items():
                if key in ('loss','eval_loss','grad_norm') and isinstance(value,(int,float)) and not math.isfinite(value):
                    raise ValueError('Non-finite training metric: '+key)
    trainer=Trainer(model=model,args=args,train_dataset=Rows(p/'train.jsonl'),eval_dataset=Rows(p/'dev.jsonl'),
        data_collator=collate,processing_class=tok,callbacks=[FiniteLoss()])
    try:
        result=trainer.train()
        if not math.isfinite(result.training_loss):raise ValueError('Non-finite final loss')
        trainer.save_model(str(out/'adapter'));tok.save_pretrained(out/'adapter')
        write_json(out/'run.json',{**initial,'status':'complete','metrics':result.metrics,'peak_memory_bytes':torch.cuda.max_memory_allocated()})
    except Exception as e:
        write_json(out/'run.json',{**initial,'status':'failed','error':str(e)});raise


def identity(a):
    result={'base_snapshot_manifest_sha256':sha(Path(a.model_dir)/'snapshot_manifest.json')}
    if getattr(a,'adapter',None):
        p=Path(a.adapter)
        result['adapter_files']={f.name:sha(f) for f in sorted(p.glob('*')) if f.suffix in ['.json','.safetensors']}
    return result


def evaluate(a):
    verify_release(a.data);out=new_dir(a.out);torch=gpu();model=load_base(a.model_dir)
    if a.adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,a.adapter)
    model.to('cuda').eval();tok=tokenizer(a.model_dir)
    eos=read_json(Path(a.model_dir)/'config.json')['eos_token_id']
    if not isinstance(eos,list):eos=[eos]
    eos=list(set(eos+[tok.convert_tokens_to_ids('<|im_end|>')]))
    frozen=read_json(ROOT/'data/frozen/cases50.json')
    data=read_jsonl(Path(a.data)/'dev.jsonl') if a.suite=='dev' else frozen
    def run(messages,tools):
        text=render_prefix(tok,messages,tools);batch=tok(text,return_tensors='pt',add_special_tokens=False).to('cuda')
        n=batch['input_ids'].shape[1]
        if n+CFG['max_new_tokens']>CFG['max_length']:raise ValueError('context budget exceeded; no truncation')
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():res=model.generate(**batch,max_new_tokens=CFG['max_new_tokens'],do_sample=False,
            eos_token_id=eos,pad_token_id=tok.pad_token_id,use_cache=True)
        torch.cuda.synchronize();tokens=res[0,n:].tolist();ended=bool(tokens and tokens[-1] in eos)
        body=tokens[:-1] if ended else tokens
        return {'text':tok.decode(body,skip_special_tokens=False),'tokens':tokens,'truncated':not ended,
                'prompt_token_ids':batch['input_ids'][0].tolist(),
                'seconds':time.perf_counter()-start,'prompt_sha256':digest(text)}
    manifest={'status':'running','suite':a.suite,'precision':'bfloat16','enable_thinking':False,
        'max_new_tokens':CFG['max_new_tokens'],'do_sample':False,'retry':0,'concurrency':1,
        'identity':identity(a),'runtime':runtime_info(),'dataset_sha256':sha(Path(a.data)/'dev.jsonl') if a.suite=='dev' else sha(ROOT/'data/frozen/cases50.json'),
        'scope':'first-call regression' if a.suite=='frozen50' else 'simulated tool fixture with conservative exact-answer check; not hardware acceptance'}
    write_json(out/'manifest.json',manifest);results=[]
    try:
        for row in data:
            if a.suite=='frozen50':
                messages=row['request']['messages'];tools=row['request']['tools'];expected=row['gold'];name=expected['tool_id'];params=expected['params']
            else:
                messages=row['messages'][:2];tools=contract()['tools'];expected=row['expected'];name=expected['tool'];params=expected['args']
            raw=run(messages,tools);score=score_first(raw['text'],name,params,raw['truncated'])
            item={'id':row['id'],'input_messages':messages,'tools_sha256':digest(tools),'first':raw,'routing':score}
            if a.suite=='dev':
                if name and score['pass']:
                    # Dispatch only a correct fixture call, never real hardware; previous generated text is retained semantically.
                    call={'role':'assistant','content':'','tool_calls':[{'id':'evaluation-call','type':'function',
                        'function':{'name':score['calls'][0]['name'],'arguments':score['calls'][0]['arguments']}}]}
                    result={'role':'tool','tool_call_id':'evaluation-call','content':json.dumps(expected['result'],ensure_ascii=False,separators=(',',':'))}
                    second=run(messages+[call,result],tools);item['final']=second
                    item['simulated_dispatch']={'call':call,'result':result,'source':'simulated_fixture'}
                    item['final_exact']=not second['truncated'] and exact_answer(second['text'],expected['answer'])
                else:item['final_exact']=not name and score['pass'] and exact_answer(raw['text'],expected['answer'])
                item['simulated_exact_chain']=score['pass'] and item['final_exact']
            results.append(item)
            with (out/'predictions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
        summary={'n':len(results),'routing_pass':sum(x['routing']['pass'] for x in results),
            'simulated_exact_chain':sum(x.get('simulated_exact_chain',False) for x in results) if a.suite=='dev' else None}
        write_json(out/'summary.json',summary);write_json(out/'manifest.json',{**manifest,'status':'complete','summary':summary})
        print(json.dumps(summary,indent=2))
    except Exception as e:
        write_json(out/'manifest.json',{**manifest,'status':'failed','completed':len(results),'error':str(e)});raise


def merge(a):
    out=new_dir(a.out);model=load_base(a.model_dir)
    from peft import PeftModel
    model=PeftModel.from_pretrained(model,a.adapter).merge_and_unload(safe_merge=True)
    model.save_pretrained(out,safe_serialization=True);tokenizer(a.model_dir).save_pretrained(out)
    manifest=read_json(Path(a.model_dir)/'snapshot_manifest.json')
    manifest['files']={p.name:sha(p) for p in out.glob('*') if p.is_file()}
    manifest['derived']=identity(a);manifest['status']='merged_not_validated'
    write_json(out/'snapshot_manifest.json',manifest)
    write_json(out/'merge.json',{'status':'merged_not_validated','source':identity(a),'note':'Run evaluate on merged model and compare with adapter outputs. This is HF, not OM.'})


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    f=s.add_parser('fetch');f.add_argument('--model-dir',required=True);f.add_argument('--tokenizer-only',action='store_true')
    f=s.add_parser('prepare');f.add_argument('--model-dir',required=True);f.add_argument('--data',default='data/mvp');f.add_argument('--out',required=True);f.add_argument('--smoke',action='store_true');f.add_argument('--reviews');f.add_argument('--max-length',type=int,default=CFG['max_length'])
    f=s.add_parser('train');f.add_argument('--model-dir',required=True);f.add_argument('--prepared',required=True);f.add_argument('--out',required=True);f.add_argument('--smoke',action='store_true');f.add_argument('--epochs',type=float,default=CFG['epochs']);f.add_argument('--max-steps',type=int,default=-1);f.add_argument('--seed',type=int,default=CFG['seed'])
    f=s.add_parser('evaluate');f.add_argument('--model-dir',required=True);f.add_argument('--adapter');f.add_argument('--suite',choices=['dev','frozen50'],required=True);f.add_argument('--data',default='data/mvp');f.add_argument('--out',required=True)
    f=s.add_parser('merge');f.add_argument('--model-dir',required=True);f.add_argument('--adapter',required=True);f.add_argument('--out',required=True)
    a=p.parse_args();globals()[a.action](a)


if __name__=='__main__':main()
