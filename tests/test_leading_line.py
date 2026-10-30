import json
import tempfile
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from rescore_leading_line import leading_label, read_jsonl

class LeadingLineTests(unittest.TestCase):
    def test_jsonl_preserves_unicode_line_separators(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            item = {"text": "A\u0085B\u2028C\u2029D"}
            path.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(read_jsonl(path), [item])

    def test_label_then_explanation(self):
        self.assertEqual(leading_label('Positive.\n\n### Analysis:\nnegative'), 'positive')
    def test_multiword_label_is_preserved(self):
        self.assertEqual(leading_label('very positive\n### Explanation:'), 'very positive')
    def test_no_label_search_in_prose(self):
        self.assertEqual(leading_label('The answer is negative\npositive'), 'the answer is negative')
    def test_empty_and_leading_blank_lines(self):
        self.assertEqual(leading_label(''), '')
        self.assertEqual(leading_label('\n  \nnon-offensive\nComment'), 'non-offensive')
    def test_no_silent_header_skip(self):
        self.assertEqual(leading_label('### Explanation:\npositive'), '### explanation')

if __name__=='__main__':unittest.main()
