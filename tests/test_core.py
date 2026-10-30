from pathlib import Path
import json
import tempfile
import unittest
from intentbench import Classifier, evaluate, tokens

ROWS=[{'text':'book visit appointment','label':'book'},{'text':'schedule appointment','label':'book'},
      {'text':'cancel remove booking','label':'cancel'},{'text':'cancel my reservation','label':'cancel'}]
class MLTests(unittest.TestCase):
    def setUp(self): self.model=Classifier().fit(ROWS)
    def test_tokenizer(self): self.assertEqual(tokens('BOOK, visit!'),['book','visit'])
    def test_prediction(self): self.assertEqual(self.model.predict('book appointment')['label'],'book')
    def test_unknown(self): self.assertEqual(self.model.predict('galactic aardvark')['label'],'unknown')
    def test_threshold_rejects(self): self.assertEqual(self.model.predict('book',1)['label'],'unknown')
    def test_probabilities(self): self.assertAlmostEqual(sum(self.model.predict('cancel')['probabilities'].values()),1)
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.json'; self.model.save(path)
            self.assertEqual(Classifier.load(path).predict('book'),self.model.predict('book'))
    def test_unfitted(self):
        with self.assertRaises(ValueError): Classifier().predict('book')
    def test_invalid_training(self):
        with self.assertRaises(ValueError): Classifier().fit([])
        with self.assertRaises(ValueError): Classifier(alpha=0)
    def test_metrics_include_rejection(self):
        result=evaluate(self.model,[{'text':'unseen','label':'book'}])
        self.assertEqual(result['coverage'],0)
        self.assertEqual(result['accuracy'],0)
        self.assertEqual(result['macro_f1'],0)
    def test_bad_model_version(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'x.json'; path.write_text('{"version":99}')
            with self.assertRaises(ValueError): Classifier.load(path)
    def test_refit_resets_counts(self):
        self.model.fit(ROWS); self.assertEqual(sum(self.model.documents.values()),4)
