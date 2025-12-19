#!/bin/bash

# =============================================================================
# Toy Data LVMOGP SSVI Configuration Folder Runner (SLURM Batch Version)
# =============================================================================
# 
# This script submits toy data LVMOGP SSVI experiments as SLURM batch jobs
# using configurations from a specified subfolder within the configs directory.
#
# Usage examples:
#   # Submit all configs in configs/toy_data_grid subfolder
#   ./run_toy_data_configs_folder.sh configs/toy_data_grid
#
#   # Submit specific config file from configs/toy_data_grid subfolder
#   ./run_toy_data_configs_folder.sh configs/toy_data_grid n_ind_32_arch_192_iter_15000_b_324_local_lr_0001_global_lr_0001_len_10_sig_10_P_32_s_1.yaml
#
#   # Submit with custom training iterations and SLURM resources
#   ./run_toy_data_configs_folder.sh configs/toy_data_grid --training-iterations 5000 --cpus 8 --mem 32G
#
# =============================================================================

# Set script directory (lvmogp_ruby folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values for experiment
DEVICE="cuda"  # Default to CUDA since we're using GPU partition
TRAINING_ITERATIONS=""
BATCH_SIZE=""
LOCAL_LR=""
GLOBAL_LR=""
CONFIG_SUBFOLDER=""
SPECIFIC_CONFIG=""

# SLURM defaults - GPU partition with 1 GPU, 12 hours
SLURM_PARTITION="gpu_p"
SLURM_QOS="gpu_normal"
SLURM_CPUS=8
SLURM_MEM="32G"
SLURM_TIME="12:00:00"
SLURM_GPUS=1
SLURM_NICE=10000
SLURM_JOB_NAME=""

# Submission delay (seconds between job submissions)
SUBMIT_DELAY=5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --training-iterations)
            TRAINING_ITERATIONS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --local-lr)
            LOCAL_LR="$2"
            shift 2
            ;;
        --global-lr)
            GLOBAL_LR="$2"
            shift 2
            ;;
        --partition)
            SLURM_PARTITION="$2"
            shift 2
            ;;
        --qos)
            SLURM_QOS="$2"
            shift 2
            ;;
        --cpus)
            SLURM_CPUS="$2"
            shift 2
            ;;
        --mem)
            SLURM_MEM="$2"
            shift 2
            ;;
        --time)
            SLURM_TIME="$2"
            shift 2
            ;;
        --gpus)
            SLURM_GPUS="$2"
            shift 2
            ;;
        --nice)
            SLURM_NICE="$2"
            shift 2
            ;;
        --job-name)
            SLURM_JOB_NAME="$2"
            shift 2
            ;;
        --delay)
            SUBMIT_DELAY="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 <config_path> [config_file] [options]"
            echo ""
            echo "Arguments:"
            echo "  config_path              Path to config subfolder (e.g., configs/toy_data_grid) (required)"
            echo "  config_file              Specific config file to run (optional, supports wildcards)"
            echo ""
            echo "Experiment Options:"
            echo "  --device DEVICE          Device to use (auto, cpu, cuda) [default: cuda]"
            echo "  --training-iterations N  Override training iterations"
            echo "  --batch-size N           Override batch size"
            echo "  --local-lr LR            Override local learning rate"
            echo "  --global-lr LR           Override global learning rate"
            echo ""
            echo "SLURM Options:"
            echo "  --partition PARTITION    SLURM partition [default: gpu_p]"
            echo "  --qos QOS               SLURM QoS [default: gpu_normal]"
            echo "  --cpus N                CPUs per task [default: 8]"
            echo "  --mem SIZE              Memory allocation [default: 32G]"
            echo "  --time HH:MM:SS         Time limit [default: 12:00:00]"
            echo "  --gpus N                Number of GPUs [default: 1]"
            echo "  --nice N                Job priority (higher=lower priority) [default: 10000]"
            echo "  --job-name NAME         Custom job name prefix [default: from config filename]"
            echo "  --delay SECONDS         Delay between job submissions [default: 5]"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Submit all configs with default GPU settings (1 GPU, 12h)"
            echo "  $0 configs/toy_data_grid"
            echo ""
            echo "  # Submit specific config with more resources"
            echo "  $0 configs/toy_data_grid n_ind_32_*.yaml --cpus 16 --mem 64G --time 24:00:00"
            echo ""
            echo "  # Submit with experiment overrides"
            echo "  $0 configs/toy_data_grid --training-iterations 5000 --batch-size 512"
            echo ""
            echo "  # Submit with longer delay between jobs"
            echo "  $0 configs/toy_data_grid --delay 10"
            exit 0
            ;;
        -*)
            echo "Unknown option $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            if [[ -z "$CONFIG_SUBFOLDER" ]]; then
                CONFIG_SUBFOLDER="$1"
            elif [[ -z "$SPECIFIC_CONFIG" ]]; then
                SPECIFIC_CONFIG="$1"
            else
                echo "Multiple config files specified. Only one allowed."
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if config path is provided
if [[ -z "$CONFIG_SUBFOLDER" ]]; then
    echo "Error: Config path is required."
    echo "Usage: $0 <config_path> [config_file] [options]"
    echo "Use --help for more information."
    exit 1
fi

# Set config directory
CONFIG_DIR="$SCRIPT_DIR/$CONFIG_SUBFOLDER"

# Check if config directory exists
if [[ ! -d "$CONFIG_DIR" ]]; then
    echo "Error: Config directory does not exist: $CONFIG_DIR"
    echo "Available config paths:"
    if [[ -d "$SCRIPT_DIR/configs" ]]; then
        find "$SCRIPT_DIR/configs" -type d -maxdepth 1 | sed "s|$SCRIPT_DIR/||" | grep -v "^configs$" | sed 's/^/  - /'
    else
        echo "  (configs directory not found)"
    fi
    exit 1
fi

# Create logs directory if it doesn't exist
LOGS_DIR="$SCRIPT_DIR/slurm_logs"
mkdir -p "$LOGS_DIR"

# Function to submit a single config as a batch job
submit_config() {
    local config_file="$1"
    local config_path="$CONFIG_DIR/$config_file"
    
    # Extract config name from filename (remove .yaml extension)
    local config_name=$(basename "$config_file" .yaml)
    
    # Set job name
    local job_name="${SLURM_JOB_NAME:-toy_data_${config_name}}"
    
    # Create temporary sbatch script
    local sbatch_script="${LOGS_DIR}/sbatch_${config_name}.sh"
    
    cat > "$sbatch_script" << EOF
#!/bin/bash

#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --qos=${SLURM_QOS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --time=${SLURM_TIME}
#SBATCH --ntasks=1
#SBATCH --gres=gpu:${SLURM_GPUS}
#SBATCH --nice=${SLURM_NICE}

#SBATCH --job-name=${job_name}
#SBATCH --output=${LOGS_DIR}/%x_%j_output.txt
#SBATCH --error=${LOGS_DIR}/%x_%j_error.txt

# Print job information
echo "=========================================="
echo "Job started at: \$(date)"
echo "Running on node: \$(hostname)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Job name: \$SLURM_JOB_NAME"
echo "Config: ${config_file}"
echo "GPUs allocated: \$SLURM_GPUS_ON_NODE"
echo "=========================================="
echo ""

# Change to working directory
cd ${SCRIPT_DIR} || exit 1

# Build Python command
CMD="python toy_data_LVMOGP_SSVI.py --config \"${config_path}\" --device ${DEVICE}"

EOF

    # Add optional parameters to the Python command
    if [[ -n "$TRAINING_ITERATIONS" ]]; then
        echo "CMD=\"\$CMD --training-iterations ${TRAINING_ITERATIONS}\"" >> "$sbatch_script"
    fi
    
    if [[ -n "$BATCH_SIZE" ]]; then
        echo "CMD=\"\$CMD --batch-size ${BATCH_SIZE}\"" >> "$sbatch_script"
    fi
    
    if [[ -n "$LOCAL_LR" ]]; then
        echo "CMD=\"\$CMD --local-lr ${LOCAL_LR}\"" >> "$sbatch_script"
    fi
    
    if [[ -n "$GLOBAL_LR" ]]; then
        echo "CMD=\"\$CMD --global-lr ${GLOBAL_LR}\"" >> "$sbatch_script"
    fi

    # Add execution and completion logging
    cat >> "$sbatch_script" << EOF

echo "Executing: \$CMD"
echo ""

# Execute command
eval "\$CMD"

EXIT_CODE=\$?

echo ""
echo "=========================================="
echo "Job finished at: \$(date)"
echo "Exit code: \$EXIT_CODE"
echo "=========================================="

exit \$EXIT_CODE
EOF

    # Submit the job
    echo "Submitting job for config: $config_file"
    echo "  Job name: $job_name"
    echo "  Partition: $SLURM_PARTITION"
    echo "  QoS: $SLURM_QOS"
    echo "  CPUs: $SLURM_CPUS"
    echo "  Memory: $SLURM_MEM"
    echo "  Time: $SLURM_TIME"
    echo "  GPUs: $SLURM_GPUS"
    echo "  Device: $DEVICE"
    echo "  Output: ${LOGS_DIR}/${job_name}_%j_output.txt"
    echo "  Error: ${LOGS_DIR}/${job_name}_%j_error.txt"
    echo "  Script: $sbatch_script"
    
    sbatch "$sbatch_script"
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        echo "[OK] Successfully submitted: $config_file"
    else
        echo "[FAILED] Failed to submit: $config_file (exit code: $exit_code)"
        return $exit_code
    fi
    
    echo ""
    return 0
}

# Main execution
echo "Toy Data LVMOGP SSVI Configuration Folder Runner (SLURM Batch - GPU)"
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Config directory: $CONFIG_DIR"
echo "Logs directory: $LOGS_DIR"
echo "Default GPU config: ${SLURM_GPUS} GPU(s), ${SLURM_TIME}, ${SLURM_MEM} memory"
echo "Submission delay: ${SUBMIT_DELAY}s between jobs"
echo ""

# Get list of YAML config files
if [[ -n "$SPECIFIC_CONFIG" ]]; then
    # Run specific config (support wildcards)
    config_files=($(find "$CONFIG_DIR" -name "$SPECIFIC_CONFIG" -type f -exec basename {} \; | sort))
    
    if [[ ${#config_files[@]} -eq 0 ]]; then
        echo "Error: No config file(s) found matching: $SPECIFIC_CONFIG"
        echo "Available config files in $CONFIG_SUBFOLDER/:"
        find "$CONFIG_DIR" -name "*.yaml" -type f -exec basename {} \; | sed 's/^/  - /' | head -20
        if [[ $(find "$CONFIG_DIR" -name "*.yaml" -type f | wc -l) -gt 20 ]]; then
            echo "  ... and more"
        fi
        exit 1
    fi
else
    # Run all config files
    config_files=($(find "$CONFIG_DIR" -name "*.yaml" -type f -exec basename {} \; | sort))
    
    if [[ ${#config_files[@]} -eq 0 ]]; then
        echo "No YAML config files found in: $CONFIG_DIR"
        echo "Please generate config files first using:"
        echo "  python toy_data_generate_config_grid.py"
        exit 1
    fi
fi

echo "Found ${#config_files[@]} config file(s) to submit:"
for config in "${config_files[@]}"; do
    echo "  - $config"
done
echo ""

# Ask for confirmation if submitting many jobs
if [[ ${#config_files[@]} -gt 10 ]]; then
    echo "WARNING: You are about to submit ${#config_files[@]} jobs."
    read -p "Do you want to continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Submission cancelled."
        exit 0
    fi
fi

# Submit each config
failed_submissions=()
submitted_jobs=0

for config in "${config_files[@]}"; do
    submit_config "$config"
    if [[ $? -eq 0 ]]; then
        ((submitted_jobs++))
    else
        failed_submissions+=("$config")
    fi
    
    # Sleep between submissions to avoid race conditions
    if [[ $submitted_jobs -lt ${#config_files[@]} ]]; then
        echo "Waiting ${SUBMIT_DELAY}s before next submission..."
        sleep $SUBMIT_DELAY
    fi
done

# Summary
echo "=========================================="
echo "Submission Summary"
echo "=========================================="
echo "Config path: $CONFIG_SUBFOLDER"
echo "Total configs: ${#config_files[@]}"
echo "Successfully submitted: $submitted_jobs"
echo "Failed submissions: ${#failed_submissions[@]}"

if [[ ${#failed_submissions[@]} -gt 0 ]]; then
    echo "Failed config files:"
    for config in "${failed_submissions[@]}"; do
        echo "  - $config"
    done
fi

echo ""
echo "Monitor your jobs with:"
echo "  squeue --me"
echo ""
echo "Check job outputs in:"
echo "  $LOGS_DIR/"
echo ""
echo "Cancel all jobs with:"
echo "  scancel -u \$USER"

if [[ ${#failed_submissions[@]} -gt 0 ]]; then
    exit 1
else
    echo "All jobs submitted successfully!"
fi