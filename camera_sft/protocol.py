"""Strict MiniCPM XML interpretation and exact official-template training prefixes."""
import json
import re
import xml.etree.ElementTree as ET
from .common import contract
from .data import validate_args

EMPTY_THINK='<think>\n\n</think>\n\n'


def parse_output(text, truncated=False):
    if truncated:raise ValueError('generation reached token budget')
    text=text.strip()
    if '<think>' in text or '</think>' in text:raise ValueError('unexpected reasoning markers in no-think output')
    if not text:raise ValueError('empty output')
    if '<function' not in text:
        if any(x in text for x in ['<param','</function>','<tool_call','<|']):raise ValueError('malformed tool/special marker')
        return []
    # A pure function XML must consume the entire output. No permissive regex extraction.
    if not text.startswith('<function'):raise ValueError('text before function')
    try:root=ET.fromstring('<root>'+text+'</root>')
    except ET.ParseError as e:raise ValueError('invalid XML') from e
    if root.text and root.text.strip():raise ValueError('extra text')
    calls=[]
    for node in root:
        if node.tag!='function' or set(node.attrib)!={'name'}:raise ValueError('invalid function node')
        if node.tail and node.tail.strip():raise ValueError('extra text after call')
        if node.text and node.text.strip():raise ValueError('unexpected function text')
        args={}
        for param in node:
            if param.tag!='param' or set(param.attrib)!={'name'} or len(param):raise ValueError('invalid param')
            key=param.attrib['name']
            if key in args:raise ValueError('duplicate parameter')
            if param.tail and param.tail.strip():raise ValueError('text after param')
            args[key]=param.text or ''
        validate_args(node.attrib['name'],args)
        calls.append({'name':node.attrib['name'],'arguments':args})
    if len(calls)!=1:raise ValueError('expected at most one tool call')
    return calls


def render_prefix(tok, messages, tools=None):
    return tok.apply_chat_template(messages, tools=tools if tools is not None else contract()['tools'],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def encode_turn(tok, history, target, max_length):
    tools=contract()['tools']
    # Original official template emits assistant targets without an empty think block.
    # We use its complete assistant serialization, and prepend the exact no-think
    # inference prefix. No simplified ChatML or custom XML formatter is substituted.
    plain=tok.apply_chat_template(history,tools=tools,tokenize=False,add_generation_prompt=True)
    full=tok.apply_chat_template(history+[target],tools=tools,tokenize=False,add_generation_prompt=False)
    prefix=render_prefix(tok,history,tools)
    if prefix!=plain+EMPTY_THINK or not full.startswith(plain):raise ValueError('official template prefix mismatch')
    suffix=full[len(plain):]
    if not suffix.endswith('<|im_end|>\n'):raise ValueError('missing target EOS')
    prefix_ids=tok.encode(prefix,add_special_tokens=False)
    ids=tok.encode(prefix+suffix,add_special_tokens=False)
    if ids[:len(prefix_ids)]!=prefix_ids:raise ValueError('token boundary changed when appending target')
    if len(ids)>max_length:raise ValueError(f'No truncation allowed: {len(ids)} > {max_length}')
    labels=[-100]*len(prefix_ids)+ids[len(prefix_ids):]
    if all(x==-100 for x in labels):raise ValueError('empty loss mask')
    return {'input_ids':ids,'attention_mask':[1]*len(ids),'labels':labels}, {
        'prefix_text':prefix,'target_text':suffix,'input_tokens':len(ids),
        'prompt_tokens':len(prefix_ids),'supervised_tokens':len(ids)-len(prefix_ids)}


def training_examples(tok,row,max_length):
    for i,message in enumerate(row['messages']):
        if message['role']=='assistant':
            feature,evidence=encode_turn(tok,row['messages'][:i],message,max_length)
            yield feature, {'sample_id':row['id'],'assistant_index':i,**evidence}


def score_first(text, expected_tool, expected_args, truncated=False):
    try:
        calls=parse_output(text,truncated)
        want=[] if expected_tool is None else [{'name':expected_tool,'arguments':expected_args}]
        return {'pass':calls==want,'calls':calls,'error':None}
    except ValueError as e:return {'pass':False,'calls':[],'error':str(e)}


def exact_answer(text, reference):
    # Conservative, deterministic smoke metric. Not a semantic production judge.
    return re.sub(r'\s+','',text)==re.sub(r'\s+','',reference)
