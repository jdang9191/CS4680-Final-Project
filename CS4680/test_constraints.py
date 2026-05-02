#!/usr/bin/env python3
"""Quick test of constraint extraction."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from csv_two_agent import _extract_prompt_constraints, _extract_impute_constraints
from mcp_csv_server.mcp_server import profile_dataset, FileInput

# Profile the test dataset
test_input = Path("mcp_csv_server/test_data/inputs/sample_dataset_100.csv")
profile = profile_dataset(FileInput(file_path=str(test_input)))

# Your exact prompt
prompt = "clean this dataset. drop the 'age' column. impute null values in 'city' column to Los Angeles."

# Extract constraints
constraints = _extract_prompt_constraints(prompt, profile)

print("=" * 60)
print(f"Prompt: {prompt}")
print("=" * 60)
print(f"\nDrop columns requested:  {constraints.get('drop_columns_requested')}")
print(f"Protected columns:       {constraints.get('protected_columns')}")
print(f"Impute constraints:      {constraints.get('impute_constraints')}")
print()

# Verify city is NOT in drop list
if 'city' in constraints.get('drop_columns_requested', []):
    print("❌ FAIL: city should NOT be in drop list")
else:
    print("✓ PASS: city is NOT in drop list")

# Verify city IS in protected list
if 'city' in constraints.get('protected_columns', []):
    print("✓ PASS: city is protected")
else:
    print("❌ FAIL: city should be protected")

# Verify age IS in drop list
if 'age' in constraints.get('drop_columns_requested', []):
    print("✓ PASS: age is in drop list")
else:
    print("❌ FAIL: age should be in drop list")
