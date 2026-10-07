"""Small CPU unit tests; no models downloaded and no GPU work performed."""
import os, unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from study import LoRALinear, mean_pool, metrics, require_slurm_gpu, identity

class StudyTests(unittest.TestCase):
    def test_lora_is_initially_exact_base(self):
        torch.manual_seed(42)
        base=nn.Linear(4,3); layer=LoRALinear(base,2,4,0.)
        x=torch.randn(5,4)
        torch.testing.assert_close(layer(x),base(x),rtol=0,atol=0)
    def test_lora_updates_without_changing_base(self):
        torch.manual_seed(42)
        base=nn.Linear(4,3); old=base.weight.detach().clone()
        layer=LoRALinear(base,2,4,0.); opt=torch.optim.AdamW([layer.A,layer.B],lr=.01)
        loss=layer(torch.randn(5,4)).square().mean();loss.backward();opt.step()
        self.assertGreater(float(layer.B.abs().sum()),0)
        torch.testing.assert_close(base.weight,old,rtol=0,atol=0)
        self.assertIsNone(base.weight.grad)
    def test_pool_excludes_special_tokens(self):
        h=torch.tensor([[[100.],[2.],[4.],[200.]]]); ids=torch.tensor([[0,5,6,1]])
        torch.testing.assert_close(mean_pool(h,ids,torch.tensor([5,6])),torch.tensor([[3.]]))
    def test_empty_sequence_rejected(self):
        with self.assertRaises(ValueError):mean_pool(torch.ones(1,2,3),torch.tensor([[0,1]]),torch.tensor([5,6]))
    def test_perfect_metrics(self):
        result=metrics([0,0,1,1],[.1,.2,.9,.8])
        self.assertEqual(result['average_precision'],1.)
        self.assertEqual(result['enrichment_at_10_percent'],2.)
    def test_constant_ap_equals_prevalence(self):
        self.assertAlmostEqual(metrics([0,0,0,1],[.1]*4)['average_precision'],.25)
    def test_identity(self):
        self.assertEqual(identity('ACDE','ACDE'),1.)
        self.assertEqual(identity('ACDE','ACDF'),.75)
    def test_gpu_requires_slurm(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(RuntimeError):require_slurm_gpu()
if __name__=='__main__':unittest.main(verbosity=2)
