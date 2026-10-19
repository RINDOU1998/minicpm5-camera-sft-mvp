"""Build the standalone HUMAN review UI; all human decisions initially pending."""
import html
import json
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from camera_sft.common import ROOT,read_jsonl,read_json,write_json,sha,digest,contract

rows=read_jsonl(ROOT/'data/mvp-reviewed-v1/candidates.jsonl')
reviews={r['id']:r for r in read_jsonl(ROOT/'data/reviews/author-v1/results/reviews.jsonl')}
changes={r['id']:r for r in read_jsonl(ROOT/'data/reviews/author-v1/results/changes.jsonl')}
groups=defaultdict(list)
for r in rows:groups[r['seed_id']].append(r)
def esc(x):return html.escape(str(x),quote=True)
def pretty(x):return esc(json.dumps(x,ensure_ascii=False,indent=2))
cards=[]
for number,(seed,rs) in enumerate(groups.items(),1):
    r=rs[0];e=r['expected']
    assert len(rs)==4 and all(x['expected']==e and x['split']==r['split'] for x in rs)
    changed=r['id'] in changes
    old=''
    if changed:
        old='<details class="diff"><summary>已修正：查看修改前后（本组 4 条）</summary><p>原问题：查询设置只返回 normal，没有提供设置内容。</p><div class="grid"><div><b>修改前</b><pre>'+pretty(changes[r['id']]['before'])+'</pre></div><div><b>修改后</b><pre>'+pretty(e)+'</pre></div></div></details>'
    variants=[]
    for row in rs:
        content=row['messages'][1]['content']
        try:content=json.dumps(json.loads(content),ensure_ascii=False,indent=2)
        except ValueError:pass
        variants.append(f'''<article class="variant" data-id="{esc(row['id'])}">
<div class="rowhead"><code>{esc(row['id'])}</code><strong class="status">待审核</strong></div>
<pre class="query">{esc(content)}</pre>
<div class="actions"><button data-action="accept" type="button">通过</button><button data-action="reject" type="button">退回</button><button data-action="pending" type="button">待审核</button></div>
<label>审核意见（退回必填）<textarea rows="2" aria-label="{esc(row['id'])} 审核意见" placeholder="问题在哪里、应如何修正；通过时可留空使用明确的人工确认说明"></textarea></label></article>''')
    search=esc(' '.join([seed,r['category'],r['split']]+[x['messages'][1]['content'] for x in rs]).lower())
    cards.append(f'''<section class="card" data-seed="{esc(seed)}" data-search="{search}" data-changed="{str(changed).lower()}">
<h2>{number:02d} · {esc(seed)} <span class="tag">{esc(r['split'])}</span> <span class="tag">{esc(r['category'])}</span>{'<span class="changed">本组已修正</span>' if changed else ''}</h2>
<p class="assistant-note"><b>Assistant 复核意见：</b>{esc(reviews[r['id']]['notes'])}<br><small>同作者复核，仅供参考；下方人工状态尚未通过。</small></p>
{old}<div class="grid"><div><h3>本组共享工具与参数</h3><pre>{pretty({'tool':e['tool'],'arguments':e['args']})}</pre></div>
<div><h3>共享模拟返回（非实机）</h3><pre>{pretty(e['result'])}</pre></div></div>
<div class="answer"><b>最终参考答复</b><p>{esc(e['answer'])}</p></div>
<p>以下是本组全部 4 条实际输入。它们共享上述标签和返回，请逐条核对表达是否改变意图。</p>
<button class="group-accept" type="button">本组待审项通过</button><small> 仅在你已读完四条后使用；已退回项不会被批量覆盖。</small>
{''.join(variants)}</section>''')
manifest_hash=sha(ROOT/'data/mvp-reviewed-v1/manifest.json')
payload={'manifest_sha256':manifest_hash,'rows':[{'id':r['id'],'sample_sha256':digest(r),'seed_id':r['seed_id']} for r in rows]}
payload_text=json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')
tools=''.join('<tr><td><code>'+esc(t['function']['name'])+'</code></td><td>'+esc(t['function']['description'])+'</td><td>'+esc('; '.join(k+' = '+', '.join(v['enum']) for k,v in t['function']['parameters']['properties'].items()))+'</td></tr>' for t in contract()['tools'])
css='''*{box-sizing:border-box}code{overflow-wrap:anywhere;word-break:break-word}.grid>div{min-width:0}body{margin:0;background:#f3f6fb;color:#20314a;font:15px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1120px;margin:24px auto;padding:0 20px}header,.card,.intro{background:white;padding:24px;border:1px solid #d9e2ef;border-radius:12px;margin:16px 0}h1{font-size:28px;color:#143d6a}h2{font-size:20px;margin:0 0 12px;overflow-wrap:anywhere}h3{font-size:16px;margin:8px 0}.toolbar{position:sticky;top:0;z-index:5;background:#eaf1fc;border:1px solid #b6c9e6;border-radius:8px;padding:12px;box-shadow:0 3px 10px #16304d18}.toolbar>div{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.toolbar p{margin:8px 0 0}input,select,textarea,button{font:inherit}input,select,textarea{border:1px solid #aabbd0;border-radius:5px;padding:7px;background:white;max-width:100%}input[type=search]{flex:1;min-width:180px}button,.file-button{border:1px solid #abc0db;background:#fff;border-radius:5px;padding:5px 12px;cursor:pointer;color:#193e71}button:hover{background:#dceafb}button.active{background:#234f87;color:white}button:focus-visible,a:focus-visible{outline:3px solid #e49a20;outline-offset:2px}button.primary{background:#244f86;color:white}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.65 Consolas,monospace;background:#f2f5f9;padding:12px;border-radius:6px;margin:8px 0}.variant{margin:15px 0 0;padding:16px;border:1px solid #c7d4e6;border-radius:8px}.rowhead{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}.status{color:#716115}.variant[data-verdict=accept]{border-left:5px solid #29824b}.variant[data-verdict=reject]{border-left:5px solid #bc3f38}.variant[data-verdict=accept] .status{color:#21723e}.variant[data-verdict=reject] .status{color:#ab312d}.variant textarea{width:100%;display:block;margin-top:5px}.actions{display:flex;gap:7px;margin:8px 0}small{color:#5e6f87}.tag,.changed{display:inline-block;font-size:12px;padding:2px 7px;border-radius:5px;background:#e8eef7;margin-left:5px}.changed{background:#fff0ce;color:#83500c}.assistant-note{background:#eff4fb;padding:12px}.diff{padding:12px;background:#fff6df}.answer{padding:12px;background:#eaf6ef}.answer p{margin:6px 0}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;text-align:left;border:1px solid #d4dfed;vertical-align:top}summary{cursor:pointer;font-weight:600}.notice{color:#8a4d0b}#message{white-space:pre-wrap;color:#244f86}a{color:#1760b0}[hidden]{display:none!important}@media(max-width:700px){main{padding:0 10px}header,.intro,.card{padding:16px}.grid{grid-template-columns:1fr}h1{font-size:23px}.toolbar{position:static}.variant{padding:10px}h2{font-size:17px}}'''
js=r'''
const DATA=JSON.parse(document.getElementById('dataset').textContent);
const KEY='minicpm5-human-review:'+DATA.manifest_sha256;
const allowed=new Set(['pending','accept','reject']);
const byId=new Map(DATA.rows.map(r=>[r.id,r]));
const state=new Map(DATA.rows.map(r=>[r.id,{verdict:'pending',notes:''}]));
const labels={pending:'待审核',accept:'通过',reject:'退回'};
const $=s=>document.querySelector(s);
const message=t=>$('#message').textContent=t;
function validateImport(rows){
 if(rows.length!==DATA.rows.length)throw Error('记录必须包含本版全部 300 个 ID；可包含 pending。');
 const seen=new Set();
 for(const r of rows){
  const ref=byId.get(r.id);
  if(!ref||seen.has(r.id)||r.sample_sha256!==ref.sample_sha256||!allowed.has(r.verdict))throw Error('版本/hash/ID/结论不匹配：'+r.id);
  if(typeof r.notes!=='string'||typeof r.reviewer!=='string')throw Error('审核记录缺少文字字段：'+r.id);
  if(r.verdict!=='pending'&&(r.reviewer_kind!=='human'||r.independent!==true||!r.reviewer.trim()))throw Error('请导入你导出的人工记录，不能将 assistant 记录转换成人工通过。');
  seen.add(r.id);
 }
}
function exported(){
 const reviewer=$('#reviewer').value.trim();
 return DATA.rows.map(r=>{
  const s=state.get(r.id);
  return {id:r.id,sample_sha256:r.sample_sha256,verdict:s.verdict,reviewer,
   reviewer_kind:'human',independent:s.verdict!=='pending',
   notes:s.notes.trim()||(s.verdict==='accept'?'本人已核对本条请求、工具参数、模拟结果与最终答复，确认通过。':''),
   review_date:new Date().toISOString(),source_manifest_sha256:DATA.manifest_sha256};
 });
}
function save(){
 try{localStorage.setItem(KEY,JSON.stringify({reviewer:$('#reviewer').value,records:DATA.rows.map(r=>({...r,...state.get(r.id)}))}));}
 catch(e){message('浏览器无法保存本地草稿，请及时导出 JSONL。');}
}
function refresh(){
 const counts={pending:0,accept:0,reject:0};
 for(const el of document.querySelectorAll('.variant')){
  const s=state.get(el.dataset.id);counts[s.verdict]++;
  el.dataset.verdict=s.verdict;el.querySelector('.status').textContent=labels[s.verdict];
  for(const b of el.querySelectorAll('[data-action]'))b.classList.toggle('active',b.dataset.action===s.verdict);
  const ta=el.querySelector('textarea');if(document.activeElement!==ta)ta.value=s.notes;
 }
 $('#progress').textContent=`300 条 · 待审核 ${counts.pending} · 通过 ${counts.accept} · 退回 ${counts.reject}`;
 $('#readiness').textContent=counts.accept===300?'全部已标记通过；请导出并在服务器做 hash 与完整性验证。':'尚不能作为全通过人工审核文件进入正式 prepare；可随时导出草稿。';
 filter();
}
function filter(){
 const mode=$('#filter').value,q=$('#search').value.trim().toLowerCase();let visible=0;
 for(const card of document.querySelectorAll('.card')){
  const statusMatch=mode==='all'||(mode==='changed'?card.dataset.changed==='true':[...card.querySelectorAll('.variant')].some(e=>state.get(e.dataset.id).verdict===mode));
  card.hidden=!statusMatch||!card.dataset.search.includes(q);if(!card.hidden)visible++;
 }
 $('#visible').textContent=`显示 ${visible}/75 个场景组（命中一条时显示整组供对照）`;
}
document.addEventListener('click',e=>{
 const action=e.target.closest('button[data-action]');
 if(action){const el=action.closest('.variant');state.get(el.dataset.id).verdict=action.dataset.action;save();refresh();}
 const group=e.target.closest('.group-accept');
 if(group){for(const el of group.closest('.card').querySelectorAll('.variant')){const s=state.get(el.dataset.id);if(s.verdict==='pending')s.verdict='accept';}save();refresh();}
});
document.addEventListener('input',e=>{
 if(e.target.matches('.variant textarea')){state.get(e.target.closest('.variant').dataset.id).notes=e.target.value;save();}
});
$('#reviewer').addEventListener('input',save);$('#search').addEventListener('input',filter);$('#filter').addEventListener('change',filter);
$('#export').addEventListener('click',()=>{
 if(!$('#reviewer').value.trim())return message('请先填写你的审核者标识。');
 if(!$('#attest').checked)return message('请确认你亲自审核了已标记的条目；未审核项请保留待审核。');
 const rs=exported(),missing=rs.filter(r=>r.verdict==='reject'&&!r.notes);
 if(missing.length)return message('以下退回项需要填写理由：'+missing.map(r=>r.id).join('、'));
 const blob=new Blob([rs.map(r=>JSON.stringify(r)).join('\n')+'\n'],{type:'application/x-ndjson;charset=utf-8'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='human-reviewed-v1.jsonl';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
 message('已导出审核文件。包含待审核/退回时仅作为草稿，服务器严格检查不会通过。');
});
$('#import').addEventListener('change',async e=>{
 try{
  const f=e.target.files[0];if(!f)return;
  const rs=(await f.text()).split(/\r?\n/).filter(x=>x.trim()).map(x=>JSON.parse(x));validateImport(rs);
  const reviewers=[...new Set(rs.filter(r=>r.verdict!=='pending').map(r=>r.reviewer))];
  if(reviewers.length>1)throw Error('页面草稿只支持一个人工审核者，不合并多人记录。');
  for(const r of rs)state.set(r.id,{verdict:r.verdict,notes:r.notes});
  $('#reviewer').value=reviewers[0]||rs[0].reviewer;$('#attest').checked=false;save();refresh();message('已导入同版本人工审核草稿，请继续核对。');
 }catch(err){message('导入失败，当前审核未改变：'+err.message);}finally{e.target.value='';}
});
$('#reset').addEventListener('click',()=>{
 if(!confirm('清空本页本地草稿？已导出的文件不受影响。'))return;
 for(const r of DATA.rows)state.set(r.id,{verdict:'pending',notes:''});
 $('#reviewer').value='';$('#attest').checked=false;save();refresh();message('已恢复全部待审核。');
});
try{
 const saved=JSON.parse(localStorage.getItem(KEY)||'null');
 if(saved){
  if(saved.records.length!==300)throw Error('缓存数量不匹配');
  const seen=new Set();
  for(const r of saved.records){if(!byId.has(r.id)||seen.has(r.id)||r.sample_sha256!==byId.get(r.id).sample_sha256||!allowed.has(r.verdict)||typeof r.notes!=='string')throw Error('缓存无效');seen.add(r.id);}
  for(const r of saved.records)state.set(r.id,{verdict:r.verdict,notes:r.notes});$('#reviewer').value=saved.reviewer||'';
 }
}catch(e){message('本地缓存不可用或不匹配；请使用导出的 JSONL 恢复。');}
refresh();
'''
page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>数据生成与验证：300条审核记录</title><style>{css}</style></head><body><main>
<header><h1>Camera Agent · 300 条人工审核</h1><p>75 个场景主体 × 4 种受控表达 · 240 train / 60 dev · assistant 已修正 8 条</p>
<p><b>你的人工审核从待审核开始。</b>Assistant 意见没有替你签核。全部数据均为合成场景与模拟结果，非真实摄像头执行。</p>
<p><a href="DATA_REVIEW.md">生成器与验证器说明</a> · <a href="../README.md">服务器操作指南</a></p>
<p><small>数据版本：mvp-reviewed-v1 · manifest SHA256：<code>{manifest_hash}</code></small></p></header>
<section class="intro"><h2>如何审核</h2><ol><li>先读下面的 system 与工具契约；核对每组四条请求。</li><li>判断该不该调用、参数是否唯一、答复是否仅依据模拟事实；点击通过或退回。</li><li>填写审核者标识，导出 JSONL。未完成也可导出草稿，下次导入继续。</li><li>300 条全部通过后，将文件放到 <code>data/reviews/human-reviewed-v1.jsonl</code>，与整个仓库一起上传服务器。</li></ol>
<p class="notice">本页不编辑训练数据。发现错误请退回并说明；修正数据后须重新绑定新 hash 审核。浏览器缓存可能丢失，导出文件才是备份。本页无网络请求。</p>
<details><summary>固定 system 与 10 工具契约</summary><pre>{esc(contract()['system'])}</pre><div class="table"><table><thead><tr><th>工具</th><th>用途</th><th>全部必填枚举参数</th></tr></thead><tbody>{tools}</tbody></table></div></details>
<details><summary>本次复核与保留的 MVP 局限</summary><p>75 个场景主体经同作者 assistant 复核，并核对全部 300 条前缀扩展。292 条保留；设置查询 8 条原先只返回 normal，现改为逐项报告具体模拟设置。</p><p>同一主体的四条不算独立任务。历史事件全为空、存在性结果全为 false、停止 all 只有一个活动任务、失败只有 device_unavailable；适合跑通链路，不能证明泛化或 95% 达标。</p></details></section>
<div class="toolbar"><div><strong id="progress"></strong><span id="visible"></span></div><div><input id="search" type="search" placeholder="搜索工具、ID、请求或 train/dev" aria-label="搜索样本"><select id="filter" aria-label="筛选状态"><option value="all">全部</option><option value="pending">待审核</option><option value="reject">已退回</option><option value="accept">已通过</option><option value="changed">本次修正的 8 条</option></select></div><p id="readiness"></p></div>
<section class="intro"><label>审核者标识 <input id="reviewer" placeholder="填写你的姓名或稳定代号" autocomplete="off"></label><p><label><input id="attest" type="checkbox"> 我亲自审核了已标记为通过或退回的条目，未审核的保持待审核。</label></p><button id="export" class="primary" type="button">导出人工审核 JSONL</button> <label class="file-button">导入审核草稿 <input id="import" type="file" accept=".jsonl,.txt" hidden></label> <button id="reset" type="button">清空本页草稿</button><p id="message" role="status" aria-live="polite"></p></section>
{''.join(cards)}<footer><p>数据与审核记录分离保存。页面上的人工状态不会修改候选数据或 assistant 记录。</p></footer></main><script id="dataset" type="application/json">{payload_text}</script><script>{js}</script></body></html>\n'''
out=ROOT/'docs/数据生成与验证：300条审核记录.html'
out.write_text(page,encoding='utf-8',newline='\n')
Path(str(out)+'.sha256').write_text(sha(out)+'  '+out.name+'\n',encoding='utf-8',newline='\n')
print(out)
