#!/bin/bash

# =============================================================================
# Original GP-LVM Configuration Folder Runner
# =============================================================================
# 
# This script runs Original GP-LVM experiments using configurations from a 
# specified subfolder within the original_gplvm_configs directory.
#
# Usage examples:
#   # Run all configs in base_config_many_seeds subfolder
#   ./run_original_gplvm_configs_folder.sh original_gplvm_configs/base_config_many_seeds
#
#   # Run all configs in missing_data_configs subfolder
#   ./run_original_gplvm_configs_folder.sh original_gplvm_configs/missing_data_configs
#
#   # Run specific config file from a subfolder
#   ./run_original_gplvm_configs_folder.sh original_gplvm_configs/base_config_many_seeds original_gplvm_config_001.yaml
#
#   # Run with custom device
#   ./run_original_gplvm_configs_folder.sh original_gplvm_configs/base_config_many_seeds --device cuda
#
#   # Run with custom training steps
#   ./run_original_gplvm_configs_folder.sh original_gplvm_configs/base_config_many_seeds --training-steps 200
#
#   # Run specific config with overrides
#   ./run_original_gplvm_configs_folder.sh original_gplvm_configs/base_config_many_seeds original_gplvm_config_001.yaml --device cuda --training-steps 150
#
# =============================================================================

# Set script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
DEVICE=""  # Empty means use config file setting
TRAINING_STEPS=""
CONFIG_NAME=""
CONFIG_SUBFOLDER=""
SPECIFIC_CONFIG=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --training-steps)
            TRAINING_STEPS="$2"
            shift 2
            ;;
        --config-name)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 <config_path> [config_file] [options]"
            echo ""
            echo "Arguments:"
            echo "  config_path              Full path to config subfolder (e.g., original_gplvm_configs/random_grid_hypers_configs) (required)"
            echo "  config_file              Specific config file to run (optional)"
            echo ""
            echo "Options:"
            echo "  --device DEVICE          Device to use (cpu, cuda) [default: cpu]"
            echo "  --training-steps N       Override training steps"
            echo "  --config-name NAME       Custom config name for results"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 original_gplvm_configs/random_grid_hypers_configs                                    # Run all configs in subfolder"
            echo "  $0 original_gplvm_configs/missing_data_configs                                        # Run all configs in subfolder"
            echo "  $0 original_gplvm_configs/random_grid_hypers_configs original_gplvm_config_001.yaml  # Run specific config"
            echo "  $0 original_gplvm_configs/random_grid_hypers_configs --device cuda                    # Run all configs on GPU"
            echo "  $0 original_gplvm_configs/random_grid_hypers_configs original_gplvm_config_001.yaml --device cuda --training-steps 200"
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
    if [[ -d "$SCRIPT_DIR/original_gplvm_configs" ]]; then
        find "$SCRIPT_DIR/original_gplvm_configs" -type d -name "*" | sed "s|$SCRIPT_DIR/||" | grep -v "^original_gplvm_configs$" | sed 's/^/  - /'
    else
        echo "  (original_gplvm_configs directory not found)"
    fi
    exit 1
fi

# Change to project root directory
cd "$PROJECT_ROOT" || exit 1

# Function to run a single config
run_config() {
    local config_file="$1"
    local config_path="$CONFIG_DIR/$config_file"
    
    echo "=========================================="
    echo "Running Original GP-LVM with config: $config_file"
    echo "Config path: $config_path"
    if [[ -n "$DEVICE" ]]; then
        echo "Device: $DEVICE (override)"
    else
        echo "Device: Using config file setting"
    fi
    echo "=========================================="
    
    # Build command
    local cmd="python ssvi_scripts_experiments/oil_flow_GPLVM.py --config \"$config_path\""
    
    # Only add device parameter if explicitly specified
    if [[ -n "$DEVICE" ]]; then
        cmd="$cmd --device \"$DEVICE\""
    fi
    
    # Add optional parameters
    if [[ -n "$TRAINING_STEPS" ]]; then
        cmd="$cmd --training-steps \"$TRAINING_STEPS\""
    fi
    
    if [[ -n "$CONFIG_NAME" ]]; then
        cmd="$cmd --config-name \"$CONFIG_NAME\""
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
echo "Original GP-LVM Configuration Folder Runner"
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
    # Run specific config
    if [[ -f "$CONFIG_DIR/$SPECIFIC_CONFIG" ]]; then
        run_config "$SPECIFIC_CONFIG"
    else
        echo "Error: Config file not found: $CONFIG_DIR/$SPECIFIC_CONFIG"
        echo "Available config files in $CONFIG_SUBFOLDER/:"
        find "$CONFIG_DIR" -name "*.yaml" -type f -exec basename {} \; | sed 's/^/  - /'
        exit 1
    fi
else
    # Run all config files
    config_files=($(find "$CONFIG_DIR" -name "*.yaml" -type f -exec basename {} \; | sort))
    
    if [[ ${#config_files[@]} -eq 0 ]]; then
        echo "No YAML config files found in: $CONFIG_DIR"
        echo "Please add your config files to this directory."
        exit 1
    fi
    
    echo "Found ${#config_files[@]} config file(s) in $CONFIG_SUBFOLDER/:"
    for config in "${config_files[@]}"; do
        echo "  - $config"
    done
    echo ""
    
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
fi
