"""Generate, validate, freeze and review data. Standard library only; never calls a model."""
import argparse
import collections
import difflib
import json
import re
from pathlib import Path
from .common import ROOT, contract, digest, read_json, read_jsonl, sha, write_json, write_jsonl, new_dir

DEFAULT_SEEDS = 'data/seeds/seed_bank.json'


def seed_path(value=None):
    p=Path(value or DEFAULT_SEEDS)
    p=(p if p.is_absolute() else ROOT/p).resolve()
    if not p.is_relative_to(ROOT.resolve()):raise ValueError('seed bank must be inside repository')
    return p


def normalize(text):
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


def user_text(row):
    text = row['messages'][1]['content']
    try:
        obj = json.loads(text)
        return obj.get('current_query', text) if isinstance(obj, dict) else text
    except ValueError:
        return text


def call_message(seed, call_id):
    return {'role':'assistant', 'content':'', 'tool_calls':[{'id':call_id, 'type':'function',
        'function':{'name':seed['tool'], 'arguments':seed['args']}}]}


def generate(seed_bank=None):
    bank=read_json(seed_path(seed_bank)); c=contract(); rows=[]
    for seed in bank['seeds']:
        # Identical semantic tool/argument families cannot cross train/dev, even for failures.
        family = {'tool':seed['tool'],'args':seed['args']} if seed['tool'] else {'seed_id':seed['seed_id']}
        group = 'g-'+digest(family)[:16]
        for v,form in enumerate(bank['surface_forms']):
            sid=f"mvp-{seed['seed_id']}-{v+1}"
            query=form.format(query=seed['query'])
            content=query
            if seed['previous_tool_result']:
                content=json.dumps({'previous_tool_result':seed['previous_tool_result'],'current_query':query},ensure_ascii=False,separators=(',',':'))
            messages=[{'role':'system','content':c['system']},{'role':'user','content':content}]
            if seed['tool']:
                cid='call-'+sid
                messages += [call_message(seed,cid),{'role':'tool','tool_call_id':cid,
                    'content':json.dumps(seed['result'],ensure_ascii=False,separators=(',',':'))}]
            messages += [{'role':'assistant','content':seed['answer']}]
            rows.append({'id':sid,'seed_id':seed['seed_id'],'group_id':group,'split':seed['split'],
              'contract_id':c['id'],'category':seed['category'],'source':'agent_authored_seed_controlled_expansion',
              'result_source':'simulated' if seed['tool'] else 'not_applicable',
              'review_status':'independent_semantic_review_pending','parameter_evidence':seed['evidence'],
              'expected':{'tool':seed['tool'],'args':seed['args'],'result':seed['result'],'answer':seed['answer']},
              'messages':messages})
    return rows


def validate_args(name, args, c=None):
    c=c or contract(); catalog={t['function']['name']:t['function']['parameters'] for t in c['tools']}
    if name not in catalog:raise ValueError('unknown tool: '+str(name))
    schema=catalog[name]
    if not isinstance(args,dict) or set(args)!=set(schema['required']):raise ValueError('missing/extra arguments')
    for key,value in args.items():
        prop=schema['properties'][key]
        if not isinstance(value,str) or value not in prop['enum']:raise ValueError('invalid enum/type: '+key)


def validate(rows, expected_count=300, seed_bank=None):
    c=contract(); bank=read_json(seed_path(seed_bank)); seeds={s['seed_id']:s for s in bank['seeds']}
    frozen=read_json(ROOT/'data/frozen/cases50.json')
    errors=[]; warnings=[]; ids=set(); groups={}; full_seen={}; query_seen={}; cats=collections.Counter()
    def err(sid,msg):errors.append({'id':sid,'error':msg})
    if sha(ROOT/'data/frozen/cases50.json')!=c['source_sha256']:err('contract','frozen source changed')
    if c['tools']!=frozen[0]['request']['tools'] or c['system']!=frozen[0]['request']['messages'][0]['content']:err('contract','tools/system differ from frozen source')
    if len(rows)!=expected_count:err('dataset',f'expected {expected_count}, got {len(rows)}')
    for row in rows:
        sid=row.get('id','MISSING')
        try:
            if sid in ids:raise ValueError('duplicate sample id')
            ids.add(sid)
            seed=seeds[row['seed_id']]; m=row['messages']; tool=seed['tool']; args=seed['args']
            planned_ids={f"mvp-{seed['seed_id']}-{v+1}" for v in range(len(bank['surface_forms']))}
            if sid not in planned_ids:raise ValueError('sample id not in seed plan')
            if row['split']!=seed['split'] or row['category']!=seed['category']:raise ValueError('split/category changed from plan')
            if row['split'] not in ('train','dev'):raise ValueError('unknown split')
            if row['contract_id']!=c['id'] or m[0]!={'role':'system','content':c['system']}:raise ValueError('contract/system mismatch')
            if m[1]['role']!='user' or not isinstance(m[1]['content'],str):raise ValueError('missing initial user')
            family={'tool':tool,'args':args} if tool else {'seed_id':seed['seed_id']}
            if row['group_id']!='g-'+digest(family)[:16]:raise ValueError('group id not bound to semantic family')
            if row['group_id'] in groups and groups[row['group_id']]!=row['split']:raise ValueError('group crosses split')
            groups[row['group_id']]=row['split']
            if row['expected']!={'tool':tool,'args':args,'result':seed['result'],'answer':seed['answer']}:raise ValueError('expected label changed from seed plan')
            if row['parameter_evidence']!=seed['evidence']:raise ValueError('parameter evidence changed')
            if row['review_status']!='independent_semantic_review_pending':raise ValueError('reviews must be recorded separately, not self-declared')
            q=user_text(row)
            if not q.strip():raise ValueError('empty query')
            expected_roles=['system','user','assistant','tool','assistant'] if tool else ['system','user','assistant']
            if [x['role'] for x in m]!=expected_roles:raise ValueError('invalid trajectory role sequence')
            if m[-1]!={'role':'assistant','content':seed['answer']}:raise ValueError('final answer differs from simulator/reference')
            if tool:
                validate_args(tool,args,c)
                calls=m[2].get('tool_calls',[])
                if len(calls)!=1 or m[2].get('content')!='':raise ValueError('need exactly one pure tool call')
                f=calls[0]['function']
                if f!={'name':tool,'arguments':args} or calls[0].get('type')!='function':raise ValueError('wrong tool or arguments')
                if not calls[0].get('id') or m[3].get('tool_call_id')!=calls[0]['id']:raise ValueError('tool_call_id mismatch')
                if json.loads(m[3]['content'])!=seed['result'] or row['result_source']!='simulated':raise ValueError('tool result or provenance mismatch')
                for key,value in args.items():
                    lexkey=('mode_privacy' if tool=='set_privacy' else 'mode_monitor') if key=='mode' else key
                    phrase=bank['lexicon'][lexkey][value]
                    if phrase not in q:raise ValueError('parameter lacks lexical evidence: '+key)
                if tool=='stop_monitor':
                    obj=json.loads(m[1]['content'])
                    if obj.get('previous_tool_result')!=seed['previous_tool_result']:raise ValueError('stop task state changed')
                    active=[a for a in obj['previous_tool_result']['active_monitors'] if a['camera']==args['camera']]
                    if not active or (args['scope']=='current' and len(active)!=1):raise ValueError('no unique stop task')
                if row['category']=='tool_failure' and seed['result'].get('ok') is not False:raise ValueError('failure fixture is not a failure')
            elif any(x.get('tool_calls') for x in m) or row['result_source']!='not_applicable':raise ValueError('no-tool sample contains calls/results')
            # Hash full semantic trajectory without arbitrary call IDs.
            fp=digest({'query':m[1]['content'],'expected':row['expected']})
            if fp in full_seen:raise ValueError('duplicate trajectory: '+full_seen[fp])
            full_seen[fp]=sid
            nq=normalize(q)
            if nq in query_seen and query_seen[nq][0]!=row['split']:raise ValueError('same query crosses split')
            query_seen[nq]=(row['split'],sid)
            best=(0,None)
            for case in frozen:
                fq=case['query']
                try:fq=json.loads(fq).get('current_query',fq)
                except (ValueError,AttributeError):pass
                n=normalize(fq)
                if nq==n:raise ValueError('exact frozen-test contamination: '+case['id'])
                ratio=difflib.SequenceMatcher(None,nq,n).ratio()
                if ratio>best[0]:best=(ratio,case['id'])
            if best[0]>=0.72:warnings.append({'id':sid,'warning':'frozen lexical similarity needs semantic review','reference':best[1],'ratio':round(best[0],4)})
            cats[(row['split'],row['category'])]+=1
        except (KeyError,TypeError,ValueError,IndexError) as e:err(sid,str(e))
    splits=collections.Counter(r.get('split') for r in rows)
    if expected_count==300 and splits!={'train':240,'dev':60}:err('dataset','split counts must be 240/60')
    return {'ok':not errors,'count':len(rows),'splits':dict(splits),'groups':len(groups),
        'categories':{f'{a}/{b}':n for (a,b),n in sorted(cats.items())},'errors':errors,'warnings':warnings,
        'semantic_status':'NOT independently reviewed; lexical checks do not prove intent or answer correctness',
        'scope':'rules and simulated fixtures only; no actual camera execution'}


def artifacts(out, rows, seed_bank=None):
    bank_name=seed_path(seed_bank).relative_to(ROOT).as_posix()
    report=validate(rows,seed_bank=bank_name)
    if not report['ok']:raise ValueError(json.dumps(report['errors'],ensure_ascii=False))
    write_jsonl(out/'candidates.jsonl',rows)
    for split in ['train','dev']:write_jsonl(out/f'{split}.jsonl',[r for r in rows if r['split']==split])
    write_json(out/'validation.json',report)
    write_jsonl(out/'review.jsonl',[{'id':r['id'],'sample_sha256':digest(r),'verdict':'pending','reviewer':'','independent':False,'notes':''} for r in rows])
    tracked=['candidates.jsonl','train.jsonl','dev.jsonl','validation.json','review.jsonl']
    sources=['contracts/camera_v1.json','data/frozen/cases50.json',bank_name,'configs/base.json']
    version={'version':1} if bank_name==DEFAULT_SEEDS else {'version':2,'seed_bank':bank_name}
    write_json(out/'manifest.json',{**version,'status':'rules_passed_semantic_review_pending','count':len(rows),'splits':report['splits'],
        'files':{f:sha(out/f) for f in tracked},'sources':{f:sha(ROOT/f) for f in sources},
        'provenance':'75 agent-authored seed records expanded into 300 controlled trajectories; no external teacher requests',
        'limitations':['not 300 independent semantic tasks','no blind test or real hardware','not a 95-percent acceptance dataset']})
    return report


def verify_release(path):
    path=Path(path); manifest=read_json(path/'manifest.json')
    for name,value in manifest['files'].items():
        target=(path/name).resolve()
        if target.parent!=path.resolve() or sha(target)!=value:raise ValueError('release file hash mismatch: '+name)
    for name,value in manifest['sources'].items():
        target=(ROOT/name).resolve()
        if not target.is_relative_to(ROOT.resolve()) or sha(target)!=value:raise ValueError('source hash mismatch: '+name)
    bank_name=manifest.get('seed_bank',DEFAULT_SEEDS)
    if bank_name not in manifest['sources']:raise ValueError('seed bank missing from source hashes')
    rows=read_jsonl(path/'candidates.jsonl');report=validate(rows,seed_bank=bank_name)
    if not report['ok']:raise ValueError(str(report['errors']))
    for split in ['train','dev']:
        if read_jsonl(path/f'{split}.jsonl')!=[r for r in rows if r['split']==split]:raise ValueError('split content mismatch')
    return report


def review_report(data_dir, reviews_path, require_independent=True):
    verify_release(data_dir)
    rows=read_jsonl(Path(data_dir)/'candidates.jsonl');reviews=read_jsonl(reviews_path)
    byid={x['id']:x for x in reviews}
    if len(byid)!=len(reviews) or set(byid)!={x['id'] for x in rows}:raise ValueError('review IDs missing/duplicated/extra')
    pending=[];non_independent=[]
    for row in rows:
        v=byid[row['id']]
        if v['sample_sha256']!=digest(row):raise ValueError('stale review: '+row['id'])
        if v.get('independent') is not True:non_independent.append(row['id'])
        if (v['verdict']!='accept' or not v.get('reviewer','').strip() or not v.get('notes','').strip()
                or (require_independent and v.get('independent') is not True)):pending.append(row['id'])
    return {'ok':not pending,'pending_or_rejected':pending,'count':len(rows),
            'require_independent':require_independent,'non_independent_count':len(non_independent),
            'independent_acceptance':not pending and not non_independent,
            'note':'Reviewer identity and independence are recorded attestations, not automatically proven.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    g=sub.add_parser('build');g.add_argument('--out',required=True);g.add_argument('--candidates',help='Optional agent-produced JSONL; same fixed seed labels and schema required')
    g.add_argument('--seeds',default=DEFAULT_SEEDS)
    v=sub.add_parser('verify');v.add_argument('--data',default='data/mvp')
    v=sub.add_parser('validate');v.add_argument('--candidates',required=True);v.add_argument('--out',required=True);v.add_argument('--expected-count',type=int,default=300)
    v.add_argument('--seeds',default=DEFAULT_SEEDS)
    r=sub.add_parser('review');r.add_argument('--data',default='data/mvp');r.add_argument('--reviews',required=True);r.add_argument('--out',required=True)
    r.add_argument('--allow-author-review',action='store_true',help='Check author review completeness only; does not satisfy independent review')
    g=sub.add_parser('plan');g.add_argument('--out',required=True)
    g.add_argument('--seeds',default=DEFAULT_SEEDS)
    a=p.parse_args()
    if a.action=='build':
        rows=read_jsonl(a.candidates) if a.candidates else generate(a.seeds)
        report=validate(rows,seed_bank=a.seeds)
        if not report['ok']:raise ValueError(json.dumps(report['errors'],ensure_ascii=False))
        out=new_dir(a.out);report=artifacts(out,rows,a.seeds)
    elif a.action=='verify':report=verify_release(a.data)
    elif a.action=='validate':report=validate(read_jsonl(a.candidates),expected_count=a.expected_count,seed_bank=a.seeds);write_json(a.out,report)
    elif a.action=='review':report=review_report(a.data,a.reviews,require_independent=not a.allow_author_review);write_json(a.out,report)
    else:
        out=new_dir(a.out)
        rows=generate(a.seeds);write_jsonl(out/'plans.jsonl',rows)
        for i in range(0,len(rows),20):write_jsonl(out/f'batch-{i//20+1:02d}.jsonl',rows[i:i+20])
        write_json(out/'contract.json',contract())
        print(f'{len(rows)} editable candidates in {out}; see docs/AGENT_GENERATION.md');return
    print(json.dumps({k:v for k,v in report.items() if k not in ['warnings','categories']},ensure_ascii=False,indent=2))
    if not report['ok']:raise SystemExit(1)


if __name__=='__main__':main()
