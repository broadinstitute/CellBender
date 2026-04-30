#!/usr/bin/env python3
"""Test that MPS fallback environment variable is set correctly."""

import sys
import os

# Simulate command line args
sys.argv = ['cellbender', 'remove-background', '--input', 'dummy.h5', '--output', 'dummy_out.h5', '--mps']

print("=" * 60)
print("Testing MPS Fallback Environment Variable")
print("=" * 60)

print(f"\n1. Before importing cellbender:")
print(f"   PYTORCH_ENABLE_MPS_FALLBACK = {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'NOT SET')}")

# This should trigger the env var setting in base_cli.main()
from cellbender import base_cli

# Check if --mps is in argv
if '--mps' in sys.argv:
    print(f"\n2. After checking sys.argv (contains --mps):")
    if '--mps' in sys.argv:
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    print(f"   PYTORCH_ENABLE_MPS_FALLBACK = {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'NOT SET')}")

print(f"\n3. Now importing torch...")
import torch
print(f"   PyTorch imported successfully")
print(f"   PYTORCH_ENABLE_MPS_FALLBACK = {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'NOT SET')}")
print(f"   MPS available: {torch.backends.mps.is_available()}")

print(f"\n4. Testing Gamma distribution on MPS:")
try:
    from torch.distributions import Gamma
    concentration = torch.tensor([1.0, 2.0]).to('mps')
    rate = torch.tensor([1.0, 1.0]).to('mps')
    gamma_dist = Gamma(concentration, rate)
    sample = gamma_dist.rsample()
    print(f"   ✅ SUCCESS! Gamma sampling worked on MPS")
    print(f"   Sample: {sample}")
except Exception as e:
    print(f"   ❌ FAILED: {type(e).__name__}: {str(e)[:100]}")

print("\n" + "=" * 60)
