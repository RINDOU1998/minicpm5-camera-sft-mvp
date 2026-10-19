"""Materialize recorded assistant decisions; this script is NOT a semantic judge."""
import argparse
import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from camera_sft.common import ROOT,read_json,read_jsonl,write_json,write_jsonl,sha,digest,new_dir
from camera_sft.data import generate,artifacts,user_text,verify_release,review_report


def build(out, audit_out):
    source=ROOT/'data/mvp'
    verify_release(source)
    decision_path=ROOT/'data/reviews/author-v1/decisions.json'
    decision=read_json(decision_path)
    if sha(source/'candidates.jsonl')!=decision['source_candidates_sha256']:
        raise ValueError('Decisions are stale for original data')
    original=read_jsonl(source/'candidates.jsonl')
    bank=read_json(ROOT/'data/seeds/seed_bank.json')
    seeds={s['seed_id']:s for s in bank['seeds']}
    decisions={d['seed_id']:d for d in decision['decisions']}
    if len(decisions)!=75 or set(decisions)!=set(seeds):raise ValueError('Incomplete decisions')
    # Inspect actual variants, not merely assuming they came from the generator.
    for row in original:
        variant=int(row['id'].rsplit('-',1)[1])-1
        expected=bank['surface_forms'][variant].format(query=seeds[row['seed_id']]['query'])
        if user_text(row)!=expected:raise ValueError('Unreviewed surface variation: '+row['id'])
    corrected=copy.deepcopy(bank)
    for seed in corrected['seeds']:
        d=decisions[seed['seed_id']]
        if d['final_verdict']!='accept':raise ValueError('Unresolved decision: '+seed['seed_id'])
        if 'correction' in d:
            seed['result']=d['correction']['result'];seed['answer']=d['correction']['answer']
    corrected['semantic_review']='Same-author assistant review recorded in data/reviews/author-v1; independent review still pending.'
    seed_name='data/seeds/seed_bank-reviewed-v1.json'
    seed_file=ROOT/seed_name
    if seed_file.exists():
        if read_json(seed_file)!=corrected:raise ValueError('Existing reviewed seed bank differs')
    else:write_json(seed_file,corrected)
    rows=generate(seed_name)
    out=new_dir(out);audit_out=new_dir(audit_out)
    artifacts(out,rows,seed_name)
    before={r['id']:r for r in original}
    old_reviews=[];new_reviews=[];changes=[]
    for row in rows:
        d=decisions[row['seed_id']];old=before[row['id']]
        shared={'id':row['id'],'reviewer':decision['reviewer'],'reviewer_kind':'assistant',
                'independent':False,'review_date':decision['review_date'],'scope':'pipeline_smoke_only',
                'seed_id':row['seed_id'],'decision_source_sha256':sha(decision_path),
                'method':'seed semantic review + exact surface-form expansion check'}
        old_reviews.append({**shared,'sample_sha256':digest(old),'verdict':d['original_verdict'],'notes':d['notes']})
        new_reviews.append({**shared,'sample_sha256':digest(row),'verdict':d['final_verdict'],
                            'notes':d.get('correction',{}).get('notes',d['notes'])})
        if old!=row:
            # Corrections affect the fixture and final answer only; inputs/splits are frozen.
            if any(old[k]!=row[k] for k in ['id','seed_id','split','group_id','category']):raise ValueError('Unexpected identity change')
            if old['messages'][:3]!=row['messages'][:3]:raise ValueError('Unexpected request/call change')
            changes.append({'id':row['id'],'before_sha256':digest(old),'after_sha256':digest(row),
                            'before':old['expected'],'after':row['expected'],'reason':d['notes']})
    write_jsonl(audit_out/'original-reviews.jsonl',old_reviews)
    write_jsonl(audit_out/'reviews.jsonl',new_reviews)
    write_jsonl(audit_out/'changes.jsonl',changes)
    report=review_report(out,audit_out/'reviews.jsonl',require_independent=False)
    strict=review_report(out,audit_out/'reviews.jsonl')
    if not report['ok'] or strict['ok'] or len(changes)!=8:raise ValueError('Unexpected review outcome')
    write_json(audit_out/'summary.json',{'source_manifest_sha256':sha(source/'manifest.json'),
        'reviewed_manifest_sha256':sha(out/'manifest.json'),'decision_source_sha256':sha(decision_path),
        'reviewed_seed_meanings':75,'checked_surface_rows':300,'original_accepted':292,'original_rejected':8,
        'corrected':len(changes),'after_accepted_for_smoke':300,'independent_review_pass':False,
        'review_check':report,'files':{f:sha(audit_out/f) for f in ['original-reviews.jsonl','reviews.jsonl','changes.jsonl']},
        'limitations':['same author assistant, not human or independent review','75 meanings and four prefixes, not 300 independent judgments',
                      'simulated fixtures, sparse result diversity, no actual camera execution','no blind generalization or 95-percent acceptance claim']})
    print('Reviewed release created: 300 rows, 8 corrected, author-reviewed smoke only.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='data/mvp-reviewed-v1')
    p.add_argument('--audit-out',default='data/reviews/author-v1/results')
    a=p.parse_args();build(a.out,a.audit_out)
