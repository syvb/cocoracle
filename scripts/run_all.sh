#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "=== Step 1: Generate Data ==="
python3 scripts/01_generate_data.py

echo "=== Step 2: Train Coconut ==="
python3 scripts/02_train_coconut.py

echo "=== Step 3: Collect Activations ==="
python3 scripts/03_collect_activations.py

echo "=== Step 4: Train Activation Oracle ==="
python3 scripts/04_train_oracle.py

echo "=== Step 5: Train Linear Probes ==="
python3 scripts/05_train_probes.py

echo "=== Step 6: Evaluate ==="
python3 scripts/06_evaluate.py

echo "=== All done! ==="
