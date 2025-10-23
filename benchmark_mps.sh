#!/bin/bash
#
# Script de benchmark pour comparer les performances CPU vs MPS
# Usage: ./benchmark_mps.sh /path/to/input.h5
#

set -e

# Couleurs pour l'output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Vérifier les arguments
if [ "$#" -lt 1 ]; then
    echo -e "${RED}Usage: $0 <input_h5_file> [expected_cells] [total_droplets] [epochs]${NC}"
    echo "Example: $0 data/raw_feature_bc_matrix.h5 5000 15000 50"
    exit 1
fi

INPUT_FILE="$1"
EXPECTED_CELLS="${2:-1000}"
TOTAL_DROPLETS="${3:-3000}"
EPOCHS="${4:-10}"
OUTPUT_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"

echo -e "${BLUE}=== CellBender MPS Benchmark ===${NC}"
echo -e "Input file: ${GREEN}$INPUT_FILE${NC}"
echo -e "Expected cells: ${GREEN}$EXPECTED_CELLS${NC}"
echo -e "Total droplets: ${GREEN}$TOTAL_DROPLETS${NC}"
echo -e "Epochs: ${GREEN}$EPOCHS${NC}"
echo ""

# Créer le dossier de sortie
mkdir -p "$OUTPUT_DIR"

# Vérifier que MPS est disponible
echo -e "${YELLOW}Checking MPS availability...${NC}"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if torch.cuda.is_available():
    print(f'CUDA available: True')
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
else:
    print(f'CUDA available: False')
"
echo ""

# Fonction pour mesurer le temps et la mémoire
run_benchmark() {
    local mode=$1
    local flag=$2
    local output_prefix=$3
    
    echo -e "${BLUE}=== Running with $mode ===${NC}"
    
    # Mesurer le temps et la mémoire
    /usr/bin/time -l cellbender remove-background \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_DIR/${output_prefix}_output.h5" \
        --expected-cells "$EXPECTED_CELLS" \
        --total-droplets-included "$TOTAL_DROPLETS" \
        --epochs "$EPOCHS" \
        $flag \
        2>&1 | tee "$OUTPUT_DIR/${output_prefix}_log.txt"
    
    echo -e "${GREEN}$mode benchmark complete!${NC}"
    echo ""
}

# Benchmark CPU
echo -e "${YELLOW}Starting CPU benchmark...${NC}"
run_benchmark "CPU" "" "cpu"

# Benchmark MPS (si disponible)
if python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    echo -e "${YELLOW}Starting MPS benchmark...${NC}"
    run_benchmark "MPS" "--mps" "mps"
else
    echo -e "${RED}MPS not available, skipping MPS benchmark${NC}"
fi

# Benchmark CUDA (si disponible)
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo -e "${YELLOW}Starting CUDA benchmark...${NC}"
    run_benchmark "CUDA" "--cuda" "cuda"
else
    echo -e "${YELLOW}CUDA not available, skipping CUDA benchmark${NC}"
fi

# Résumé des résultats
echo -e "${BLUE}=== Benchmark Summary ===${NC}"
echo ""

for mode in cpu mps cuda; do
    log_file="$OUTPUT_DIR/${mode}_log.txt"
    if [ -f "$log_file" ]; then
        echo -e "${GREEN}$mode:${NC}"
        
        # Extraire le temps total
        total_time=$(grep "real" "$log_file" | tail -1 | awk '{print $2}')
        if [ -n "$total_time" ]; then
            echo "  Total time: $total_time"
        fi
        
        # Extraire la mémoire maximale
        max_mem=$(grep "maximum resident set size" "$log_file" | awk '{print $1}')
        if [ -n "$max_mem" ]; then
            max_mem_gb=$(echo "scale=2; $max_mem / 1024 / 1024 / 1024" | bc)
            echo "  Peak memory: ${max_mem_gb} GB"
        fi
        
        # Temps par epoch
        epoch_time=$(grep "seconds per epoch" "$log_file" | head -1 | awk '{print $(NF-3)}')
        if [ -n "$epoch_time" ]; then
            echo "  Time per epoch: ${epoch_time}s"
        fi
        
        echo ""
    fi
done

echo -e "${BLUE}Results saved in: ${GREEN}$OUTPUT_DIR${NC}"
echo ""
echo -e "${YELLOW}Files generated:${NC}"
ls -lh "$OUTPUT_DIR"

echo ""
echo -e "${GREEN}Benchmark complete!${NC}"
