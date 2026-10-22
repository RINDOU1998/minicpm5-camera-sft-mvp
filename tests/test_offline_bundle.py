import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from camera_sft.common import sha,write_json
from camera_sft.cloud import CFG
from scripts.offline_bundle import safe_relative,verify_tree,verify_archive


class OfflineBundleTests(unittest.TestCase):
    def test_streaming_file_hash(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'weights';body=b'abc123'*(1024*1024)+b'end';p.write_bytes(body)
            self.assertEqual(sha(p),hashlib.sha256(body).hexdigest())

    def test_missing_or_tampered_file_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);f=p/'weight';f.write_bytes(b'original')
            write_json(p/'OFFLINE_MANIFEST.json',{'model_id':CFG['model_id'],'revision':CFG['revision'],
                'source_commit':'fixture','smoke_only':True,'files':{'weight':{'bytes':8,'sha256':sha(f)}}})
            self.assertTrue(verify_tree(p)['ok'])
            f.write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):verify_tree(p)
            f.unlink()
            with self.assertRaisesRegex(ValueError,'Missing'):verify_tree(p)

    def test_unsafe_paths_rejected(self):
        for p in ['../weight','/weight','D:/weight','models\\weight','models/../weight','./weight','']:
            with self.subTest(path=p),self.assertRaises(ValueError):safe_relative(p)

    def test_archive_incomplete_or_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'fixture.tar.gz';body=b'weight'
            expected={'model':{'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}}
            def create(count):
                with tarfile.open(p,'w:gz') as t:
                    for _ in range(count):
                        m=tarfile.TarInfo('minicpm5-sft/model');m.size=len(body);t.addfile(m,io.BytesIO(body))
            create(1);verify_archive(p,expected)
            create(0)
            with self.assertRaisesRegex(ValueError,'Incomplete'):verify_archive(p,expected)
            create(2)
            with self.assertRaisesRegex(ValueError,'duplicate'):verify_archive(p,expected)


if __name__=='__main__':unittest.main()
