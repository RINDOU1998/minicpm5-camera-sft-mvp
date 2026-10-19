"""Compare complete paired runs; never equates simulated exact answers to real-tool acceptance."""
import argparse
from pathlib import Path
from .common import read_json, read_jsonl, write_json


def compare(before,after):
    before,after=Path(before),Path(after)
    a=read_json(before/'manifest.json');b=read_json(after/'manifest.json')
    if a['status']!='complete' or b['status']!='complete':raise ValueError('both runs must be complete')
    for field in ['suite','precision','enable_thinking','max_new_tokens','do_sample','retry','concurrency','dataset_sha256']:
        if a[field]!=b[field]:raise ValueError('non-comparable setting: '+field)
    ar=read_jsonl(before/'predictions.jsonl');br=read_jsonl(after/'predictions.jsonl')
    ids=[r['id'] for r in ar]
    if len(set(ids))!=len(ids) or ids!=[r['id'] for r in br]:raise ValueError('case IDs/order mismatch')
    expected=50 if a['suite']=='frozen50' else 60
    if len(ids)!=expected:raise ValueError('incomplete count')
    for x,y in zip(ar,br):
        # Equal dataset hashes alone do not prove equal rendered model inputs.
        for field in ['input_messages','tools_sha256']:
            if x[field]!=y[field]:raise ValueError('non-comparable input: '+field+' at '+x['id'])
        for field in ['prompt_sha256','prompt_token_ids']:
            if x['first'][field]!=y['first'][field]:raise ValueError('non-comparable prompt: '+field+' at '+x['id'])
    out={'suite':a['suite'],'n':len(ids),'metrics':{},'scope':a['scope']}
    for key in ['routing']+(['simulated_exact_chain'] if a['suite']=='dev' else []):
        def good(r):return r[key]['pass'] if key=='routing' else r[key]
        out['metrics'][key]={'before':sum(good(x) for x in ar),'after':sum(good(x) for x in br),
            'gains':[x['id'] for x,y in zip(ar,br) if not good(x) and good(y)],
            'losses':[x['id'] for x,y in zip(ar,br) if good(x) and not good(y)]}
    out['raw_first_output_changes']=[x['id'] for x,y in zip(ar,br) if x['first']['tokens']!=y['first']['tokens']]
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--before',required=True);p.add_argument('--after',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    result=compare(a.before,a.after);write_json(a.out,result);print(result)


if __name__=='__main__':main()
