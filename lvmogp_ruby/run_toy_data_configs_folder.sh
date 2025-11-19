#!/bin/bash

# =============================================================================
# Toy Data LVMOGP SSVI Configuration Folder Runner
# =============================================================================
# 
# This script runs toy data LVMOGP SSVI experiments using configurations from 
# a specified subfolder within the configs directory.
#
# Usage examples:
#   # Run all configs in configs/toy_data_grid subfolder
#   ./run_toy_data_configs_folder.sh configs/toy_data_grid
#
#   # Run specific config file from configs/toy_data_grid subfolder
#   ./run_toy_data_configs_folder.sh configs/toy_data_grid n_ind_32_arch_192_iter_15000_b_324_local_lr_0001_global_lr_0001_len_10_sig_10_P_32_s_1.yaml
#
#   # Run with custom device
#   ./run_toy_data_configs_folder.sh configs/toy_data_grid --device cuda
#
#   # Run with custom training iterations
#   ./run_toy_data_configs_folder.sh configs/toy_data_grid --training-iterations 5000
#
#   # Run specific config with overrides
#   ./run_toy_data_configs_folder.sh configs/toy_data_grid n_ind_32_arch_192_iter_15000_b_324_local_lr_0001_global_lr_0001_len_10_sig_10_P_32_s_1.yaml --device cuda --training-iterations 5000
#
# =============================================================================

# Set script directory (lvmogp_ruby folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
DEVICE=""  # Empty means use config file setting
TRAINING_ITERATIONS=""
BATCH_SIZE=""
LOCAL_LR=""
GLOBAL_LR=""
CONFIG_SUBFOLDER=""
SPECIFIC_CONFIG=""

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
        --help|-h)
            echo "Usage: $0 <config_path> [config_file] [options]"
            echo ""
            echo "Arguments:"
            echo "  config_path              Path to config subfolder (e.g., configs/toy_data_grid) (required)"
            echo "  config_file              Specific config file to run (optional)"
            echo ""
            echo "Options:"
            echo "  --device DEVICE          Device to use (auto, cpu, cuda) [default: from config]"
            echo "  --training-iterations N  Override training iterations"
            echo "  --batch-size N           Override batch size"
            echo "  --local-lr LR            Override local learning rate"
            echo "  --global-lr LR           Override global learning rate"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 configs/toy_data_grid                                                    # Run all configs in subfolder"
            echo "  $0 configs/toy_data_grid n_ind_32_arch_192_iter_15000_b_324_*.yaml         # Run specific config"
            echo "  $0 configs/toy_data_grid --device cuda                                      # Run all configs on GPU"
            echo "  $0 configs/toy_data_grid --device cuda --training-iterations 5000"
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

# Change to lvmogp_ruby directory so relative paths in configs resolve correctly
cd "$SCRIPT_DIR" || exit 1

# Function to run a single config
run_config() {
    local config_file="$1"
    local config_path="$CONFIG_DIR/$config_file"
    
    echo "=========================================="
    echo "Running Toy Data LVMOGP SSVI with config: $config_file"
    echo "Config path: $config_path"
    if [[ -n "$DEVICE" ]]; then
        echo "Device: $DEVICE (override)"
    else
        echo "Device: Using config file setting"
    fi
    echo "=========================================="
    
    # Build command (running from lvmogp_ruby directory)
    local cmd="python toy_data_LVMOGP_SSVI.py --config \"$config_path\""
    
    # Only add device parameter if explicitly specified
    if [[ -n "$DEVICE" ]]; then
        cmd="$cmd --device \"$DEVICE\""
    fi
    
    # Add optional parameters
    if [[ -n "$TRAINING_ITERATIONS" ]]; then
        cmd="$cmd --training-iterations \"$TRAINING_ITERATIONS\""
    fi
    
    if [[ -n "$BATCH_SIZE" ]]; then
        cmd="$cmd --batch-size \"$BATCH_SIZE\""
    fi
    
    if [[ -n "$LOCAL_LR" ]]; then
        cmd="$cmd --local-lr \"$LOCAL_LR\""
    fi
    
    if [[ -n "$GLOBAL_LR" ]]; then
        cmd="$cmd --global-lr \"$GLOBAL_LR\""
    fi
    
    echo "Executing: $cmd"
    echo ""
    
    # Execute command
    eval "$cmd"
    
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo "[OK] Successfully completed: $config_file"
    else
        echo "[FAILED] Failed to run: $config_file (exit code: $exit_code)"
    fi
    
    echo ""
    return $exit_code
}

# Main execution
echo "Toy Data LVMOGP SSVI Configuration Folder Runner"
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Config directory: $CONFIG_DIR"
if [[ -n "$DEVICE" ]]; then
    echo "Device: $DEVICE (override)"
else
    echo "Device: Using config file setting"
fi
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
    
    echo "Found ${#config_files[@]} config file(s) in $CONFIG_SUBFOLDER/:"
    for config in "${config_files[@]}"; do
        echo "  - $config"
    done
    echo ""
fi

# Run each config
failed_configs=()
for config in "${config_files[@]}"; do
    run_config "$config"
    if [[ $? -ne 0 ]]; then
        failed_configs+=("$config")
    fi
done

# Summary
echo "=========================================="
echo "Execution Summary"
echo "=========================================="
echo "Config path: $CONFIG_SUBFOLDER"
echo "Total configs: ${#config_files[@]}"
echo "Failed configs: ${#failed_configs[@]}"

if [[ ${#failed_configs[@]} -gt 0 ]]; then
    echo "Failed config files:"
    for config in "${failed_configs[@]}"; do
        echo "  - $config"
    done
    exit 1
else
    echo "All configs completed successfully!"
fi

