import unittest

import torch

from metrics.bleu import bleu_seq, bleu_tensor


class TestBLEUScore(unittest.TestCase):
    def setUp(self):
        # Créer des exemples de tenseurs pour les tests
        self.reference_tensor = torch.tensor([[1, 2, 3, 4, 8], [4, 5, 7, 10, 5]])
        self.candidate_tensor = torch.tensor([[1, 2, 3, 4, 8], [5, 5, 7, 10, 5]])

        self.reference_sequences = ["this is test", "this is another test"]
        self.candidate_sequences = ["this is a test", "this is another test"]

    def test_bleu_score_tensors(self):
        result = bleu_tensor(self.reference_tensor, self.candidate_tensor, 2)

        self.assertIsInstance(result, torch.Tensor)
        scores = torch.tensor([1.0000, 0.75])
        for i, r in enumerate(result):
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)
            self.assertAlmostEqual(r.item(), scores[i].item(), places=4)

    def test_bleu_score_sequences(self):
        result = bleu_seq(self.reference_sequences, self.candidate_sequences, 2)

        self.assertIsInstance(result, torch.Tensor)
        scores = torch.tensor([0.3333, 1.0])
        for i, r in enumerate(result):
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)
            self.assertAlmostEqual(r.item(), scores[i].item(), places=4)


if __name__ == "__main__":
    unittest.main()
