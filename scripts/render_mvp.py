"""Render the current MVP spec as a self-contained HTML; optional docs dependency."""
import hashlib
from pathlib import Path
from markdown_it import MarkdownIt

root=Path(__file__).resolve().parents[1]
source=root/'docs/MVP.md'
content=MarkdownIt('commonmark',{'html':False}).enable('table').render(source.read_text(encoding='utf-8'))
content=content.replace('<table>','<div class="table"><table>').replace('</table>','</table></div>')
css='''*{box-sizing:border-box}body{margin:0;background:#f0f4f9;color:#172d45;font:16px/1.85 "Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1050px;margin:35px auto;padding:32px 40px;background:white;border-radius:16px}h1{font-size:30px;line-height:1.5;color:#103d6b}h2{margin-top:34px;padding-top:14px;border-top:1px solid #dbe3ed;font-size:22px}a{color:#1760bd}strong{color:#113a64}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px 12px;border-bottom:1px solid #dbe3ed;text-align:left;vertical-align:top}th{background:#eef3fa}.table{overflow:auto}code{font-family:Consolas,monospace;font-size:0.9em;overflow-wrap:anywhere}li{margin:6px 0}pre{padding:15px;background:#edf2f8;overflow:auto}footer{font-size:13px;color:#526780;margin-top:35px}@media(max-width:700px){main{margin:12px;padding:20px}h1{font-size:25px}h2{font-size:21px}td,th{min-width:90px;padding:8px}}'''
text='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Camera Agent 数据生产 MVP：300 条链路方案</title><style>'+css+'</style></head><body><main>'+content+'<footer>源码：docs/MVP.md。云端命令统一维护在 README.md；本页不是已完成 GPU 实验的报告。</footer></main></body></html>\n'
out=root/'docs/MVP.html';out.write_text(text,encoding='utf-8',newline='\n')
(root/'docs/MVP.html.sha256').write_text(hashlib.sha256(out.read_bytes()).hexdigest()+'  MVP.html\n',encoding='utf-8',newline='\n')
print(out)
