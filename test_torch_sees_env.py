#!/usr/bin/env python3
"""Verify torch sees the environment variable."""

import sys
import os

sys.argv = ['cellbender', 'remove-background', '--input', 'x.h5', '--output', 'y.h5', '--mps']

print("Setting env var as base_cli.main() does...")
if '--mps' in sys.argv:
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

print(f"ENV VAR: {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}")

print("\nImporting torch...")
import torch
print(f"Torch imported")

print(f"\nTesting Gamma on MPS...")
from torch.distributions import Gamma
conc = torch.tensor([2.0, 3.0]).to('mps')
rate = torch.tensor([1.0, 1.0]).to('mps')
gamma = Gamma(conc, rate)
sample = gamma.rsample()
print(f"✅ SUCCESS! Sample: {sample}")
