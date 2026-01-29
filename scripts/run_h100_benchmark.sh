#!/bin/bash
# =============================================================================
# SIMM CPU vs GPU Benchmark for H100 System
# =============================================================================
#
# Target Hardware:
#   - 8x NVIDIA H100 80GB HBM3
#   - 2x Intel Xeon Platinum 8568Y+ (48 cores each = 96 total)
#
# Usage:
#   ./run_h100_benchmark.sh              # Default (medium scale)
#   ./run_h100_benchmark.sh small        # Quick test
#   ./run_h100_benchmark.sh large        # Large scale
#   ./run_h100_benchmark.sh xlarge       # 1M portfolios
#   ./run_h100_benchmark.sh stress       # 10M portfolios (stress test)
#   ./run_h100_benchmark.sh all          # Run all configurations
#   ./run_h100_benchmark.sh custom 50000 200  # Custom: 50K portfolios, 200 factors
#
# =============================================================================

set -e

# Configuration for your H100 system
NUM_CPUS=96          # 2x Xeon 8568Y+ = 96 cores
NUM_GPUS=8           # 8x H100
RESULTS_DIR="benchmark_results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create results directory
mkdir -p "$RESULTS_DIR"

print_header() {
    echo ""
    echo -e "${BLUE}=================================================================${NC}"
    echo -e "${BLUE}  SIMM Benchmark: CPU vs GPU Performance${NC}"
    echo -e "${BLUE}=================================================================${NC}"
    echo ""
}

print_system_info() {
    echo -e "${YELLOW}System Configuration:${NC}"
    echo "  CPUs: Dual Intel Xeon Platinum 8568Y+ (${NUM_CPUS} cores total)"
    echo "  GPUs: ${NUM_GPUS}x NVIDIA H100 80GB"
    echo ""

    # Check for NVIDIA GPUs
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}GPU Status:${NC}"
        nvidia-smi --query-gpu=index,name,memory.total,memory.free,temperature.gpu \
                   --format=csv,noheader,nounits | \
        while IFS=, read -r idx name mem_total mem_free temp; do
            echo "  GPU $idx: $name | Memory: $mem_free/$mem_total MB free | Temp: ${temp}°C"
        done
        echo ""
    fi

    # CPU info
    echo -e "${YELLOW}CPU Info:${NC}"
    if [ -f /proc/cpuinfo ]; then
        model=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
        cores=$(grep -c "processor" /proc/cpuinfo)
        echo "  Model: $model"
        echo "  Cores: $cores"
    fi
    echo ""
}

run_benchmark() {
    local scale=$1
    local output_file="${RESULTS_DIR}/benchmark_${scale}_$(date +%Y%m%d_%H%M%S).json"

    echo -e "${GREEN}Running benchmark: ${scale}${NC}"
    echo "  Output: ${output_file}"
    echo ""

    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi

    # Set optimal CPU affinity and memory policy
    export OMP_NUM_THREADS=${NUM_CPUS}
    export MKL_NUM_THREADS=${NUM_CPUS}
    export NUMBA_NUM_THREADS=${NUM_CPUS}
    export OPENBLAS_NUM_THREADS=${NUM_CPUS}

    # Run benchmark
    python benchmark_cpu_vs_gpu.py \
        --scale "$scale" \
        --cpu-threads ${NUM_CPUS} \
        --gpus ${NUM_GPUS} \
        --output "$output_file"

    echo ""
    echo -e "${GREEN}Results saved to: ${output_file}${NC}"
}

run_custom() {
    local portfolios=$1
    local factors=$2
    local iterations=${3:-10}
    local output_file="${RESULTS_DIR}/benchmark_custom_${portfolios}p_${factors}f_$(date +%Y%m%d_%H%M%S).json"

    echo -e "${GREEN}Running custom benchmark:${NC}"
    echo "  Portfolios: ${portfolios}"
    echo "  Factors: ${factors}"
    echo "  Iterations: ${iterations}"
    echo "  Output: ${output_file}"
    echo ""

    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi

    # Set optimal threading
    export OMP_NUM_THREADS=${NUM_CPUS}
    export MKL_NUM_THREADS=${NUM_CPUS}
    export NUMBA_NUM_THREADS=${NUM_CPUS}

    python benchmark_cpu_vs_gpu.py \
        --portfolios "$portfolios" \
        --factors "$factors" \
        --iterations "$iterations" \
        --cpu-threads ${NUM_CPUS} \
        --gpus ${NUM_GPUS} \
        --output "$output_file"
}

run_all() {
    local output_file="${RESULTS_DIR}/benchmark_all_$(date +%Y%m%d_%H%M%S).json"

    echo -e "${GREEN}Running all benchmark configurations...${NC}"
    echo ""

    # Activate virtual environment
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi

    export OMP_NUM_THREADS=${NUM_CPUS}
    export MKL_NUM_THREADS=${NUM_CPUS}
    export NUMBA_NUM_THREADS=${NUM_CPUS}

    python benchmark_cpu_vs_gpu.py \
        --all \
        --cpu-threads ${NUM_CPUS} \
        --gpus ${NUM_GPUS} \
        --output "$output_file"
}

# =============================================================================
# Main
# =============================================================================

print_header
print_system_info

case "${1:-medium}" in
    small)
        run_benchmark "small"
        ;;
    medium)
        run_benchmark "medium"
        ;;
    large)
        run_benchmark "large"
        ;;
    xlarge)
        run_benchmark "xlarge"
        ;;
    stress)
        echo -e "${RED}WARNING: Stress test with 10M portfolios${NC}"
        echo "This may take several minutes and use significant memory."
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            run_benchmark "stress"
        fi
        ;;
    all)
        run_all
        ;;
    custom)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo -e "${RED}Error: custom requires portfolios and factors${NC}"
            echo "Usage: $0 custom <portfolios> <factors> [iterations]"
            exit 1
        fi
        run_custom "$2" "$3" "${4:-10}"
        ;;
    help|--help|-h)
        echo "Usage: $0 [scale|custom]"
        echo ""
        echo "Scales:"
        echo "  small   - 1,000 portfolios (quick test)"
        echo "  medium  - 10,000 portfolios (default)"
        echo "  large   - 100,000 portfolios"
        echo "  xlarge  - 1,000,000 portfolios"
        echo "  stress  - 10,000,000 portfolios"
        echo "  all     - Run all configurations"
        echo ""
        echo "Custom:"
        echo "  custom <portfolios> <factors> [iterations]"
        echo ""
        echo "Examples:"
        echo "  $0                      # Run medium scale"
        echo "  $0 large                # Run large scale"
        echo "  $0 custom 50000 200 5   # 50K portfolios, 200 factors, 5 iterations"
        ;;
    *)
        echo -e "${RED}Unknown option: $1${NC}"
        echo "Run '$0 help' for usage"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Benchmark complete!${NC}"
echo "Results are in: ${RESULTS_DIR}/"
