import copy
import json
import tempfile
import unittest
from pathlib import Path
from camera_sft.data import generate,validate,validate_args,artifacts,verify_release,review_report
from camera_sft.common import ROOT,write_jsonl,read_jsonl,read_json,write_json
from camera_sft.compare import compare
from camera_sft.protocol import parse_output,score_first,exact_answer


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows=generate()

    def mutate(self,fn):
        rows=copy.deepcopy(self.rows);fn(rows)
        result=validate(rows)
        self.assertFalse(result['ok'],result)
        return str(result['errors'])

    def test_complete_dataset(self):
        r=validate(self.rows);self.assertTrue(r['ok'],r['errors']);self.assertEqual(r['splits'],{'train':240,'dev':60})
        self.assertEqual(len({r['seed_id'] for r in self.rows}),75)
        self.assertIn('NOT independently reviewed',r['semantic_status'])

    def test_extra_and_missing_arguments(self):
        for args in [{'camera':'rear','quality':'high','format':'png','action':'capture'},{'camera':'rear'}]:
            with self.assertRaisesRegex(ValueError,'missing/extra'):validate_args('camera_capture',args)

    def test_enum_type(self):
        with self.assertRaisesRegex(ValueError,'enum/type'):validate_args('list_events',{'camera':'rear','target':'pet','window':'24h','limit':20})

    def test_tool_call_binding(self):
        self.assertIn('tool_call_id',self.mutate(lambda r:r[0]['messages'][3].update(tool_call_id='other')))

    def test_fabricated_tool_result(self):
        self.assertIn('result',self.mutate(lambda r:r[0]['messages'][3].update(content='{"ok":true,"photo_id":"fake"}')))

    def test_unjustified_parameter(self):
        self.assertIn('lexical evidence',self.mutate(lambda r:r[0]['messages'][1].update(content='请拍照。')))

    def test_plan_cannot_be_relabelled(self):
        self.assertIn('label',self.mutate(lambda r:r[0]['expected'].update(tool='camera_status')))

    def test_no_synthetic_human_review(self):
        self.assertIn('self-declared',self.mutate(lambda r:r[0].update(review_status='human_approved')))

    def test_group_and_split_locked(self):
        self.assertIn('split',self.mutate(lambda r:r[0].update(split='dev')))
        self.assertIn('group',self.mutate(lambda r:r[0].update(group_id='fake')))

    def test_no_tool_cannot_call(self):
        def alter(rows):
            r=next(x for x in rows if not x['expected']['tool']);r['messages'][-1]['tool_calls']=[{'function':{'name':'camera_capture'}}]
        self.assertIn('final answer',self.mutate(alter))

    def test_frozen_question_leak(self):
        def alter(rows):
            r=next(x for x in rows if not x['expected']['tool'])
            source=next(x for x in read_json(ROOT/'data/frozen/cases50.json') if not x['gold']['need_tool'])
            r['messages'][1]['content']=source['query']
        self.assertIn('contamination',self.mutate(alter))

    def test_hash_tamper_and_stale_review(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);artifacts(p,self.rows);self.assertTrue(verify_release(p)['ok'])
            self.assertFalse(review_report(p,p/'review.jsonl')['ok'])
            reviews=read_jsonl(p/'review.jsonl');reviews[0]['sample_sha256']='wrong';write_jsonl(p/'review-edited.jsonl',reviews)
            with self.assertRaisesRegex(ValueError,'stale review'):review_report(p,p/'review-edited.jsonl')
            with (p/'train.jsonl').open('a',encoding='utf-8') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):verify_release(p)


class ProtocolTests(unittest.TestCase):
    XML='<function name="set_privacy"><param name="camera">rear</param><param name="mode">enabled</param></function>'

    def test_valid_and_no_tool(self):
        self.assertTrue(score_first(self.XML,'set_privacy',{'camera':'rear','mode':'enabled'})['pass'])
        self.assertTrue(score_first('这只是知识问答。',None,{})['pass'])
        self.assertFalse(score_first(self.XML,None,{})['pass'])

    def test_truncation_cannot_pass(self):
        self.assertFalse(score_first(self.XML,'set_privacy',{'camera':'rear','mode':'enabled'},True)['pass'])

    def test_malformed_and_extra_calls(self):
        for x in [self.XML+self.XML,self.XML+'已完成',self.XML[:-3],'<function name="unknown"/>',self.XML.replace('</function>','<param name="camera">front</param></function>'),'<think>thinking</think>hello','']:
            with self.subTest(x=x):
                with self.assertRaises(ValueError):parse_output(x)

    def test_cdata_and_string_enum(self):
        x=self.XML.replace('>rear<','><![CDATA[rear]]><')
        self.assertEqual(parse_output(x)[0]['arguments']['camera'],'rear')

    def test_conservative_final_answer(self):
        self.assertTrue(exact_answer('共有 3 人。','共有3人。'))
        self.assertFalse(exact_answer('不是3人。','共有3人。'))


class ComparisonTests(unittest.TestCase):
    def test_incomplete_and_changed_decoding_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            a=Path(td)/'a';b=Path(td)/'b';a.mkdir();b.mkdir()
            m={'status':'complete','suite':'frozen50','precision':'bfloat16','enable_thinking':False,
               'max_new_tokens':128,'do_sample':False,'retry':0,'concurrency':1,'dataset_sha256':'same','scope':'first call'}
            write_json(a/'manifest.json',m);write_json(b/'manifest.json',m)
            rows=[{'id':str(i),'input_messages':[{'role':'user','content':'same'}],'tools_sha256':'same',
                   'routing':{'pass':True},'first':{'tokens':[1],'prompt_sha256':'same','prompt_token_ids':[2,3]}} for i in range(50)]
            write_jsonl(a/'predictions.jsonl',rows);write_jsonl(b/'predictions.jsonl',rows)
            self.assertEqual(compare(a,b)['metrics']['routing']['after'],50)
            changed=copy.deepcopy(rows);changed[0]['first']['prompt_token_ids']=[2,4]
            write_jsonl(b/'predictions.jsonl',changed)
            with self.assertRaisesRegex(ValueError,'non-comparable prompt'):compare(a,b)
            write_jsonl(b/'predictions.jsonl',rows)
            write_json(b/'manifest.json',{**m,'max_new_tokens':256})
            with self.assertRaisesRegex(ValueError,'non-comparable'):compare(a,b)
            write_json(b/'manifest.json',{**m,'status':'failed'})
            with self.assertRaisesRegex(ValueError,'complete'):compare(a,b)


if __name__=='__main__':unittest.main()
