#!/usr/bin/env python3
"""Test the actual cellbender command flow."""

import sys
import os

# Simulate calling: cellbender remove-background --mps (with required args)
sys.argv = [
    'cellbender', 
    'remove-background',
    '--input', '/tmp/dummy.h5',
    '--output', '/tmp/dummy_out.h5', 
    '--mps'
]

print("Command line:", ' '.join(sys.argv))
print(f"ENV before main(): {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'NOT SET')}\n")

# This is what happens when you run `cellbender` command
from cellbender.base_cli import main

try:
    main()
except Exception as e:
    print(f"\nExpected error (no input file): {type(e).__name__}")
    print(f"ENV after main() started: {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'NOT SET')}")
