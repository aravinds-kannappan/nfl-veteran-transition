"""
main.py - MASTER CALLER SCRIPT
NFL Veterans Team Change Analysis - Complete Pipeline

Executes all Python modules in the correct sequence:
1. data_collection.py - Load and validate NFL data
2. preprocessing.py - Clean and standardize data
3. feature_engineering.py - Create ML-ready features
4. modeling.py - Train predictive models and generate insights

This script orchestrates the entire analysis pipeline from raw data to final outputs.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import traceback
import json

# Configuration
PYTHON_SCRIPTS = [
    ("data_collection.py", "Data Collection & Validation"),
    ("preprocessing.py", "Data Preprocessing & Standardization"),
    ("feature_engineering.py", "Feature Engineering"),
    ("modeling.py", "Predictive Modeling & Analysis")
]

PYTHON_DIR = Path("python")
LOG_FILE = Path("outputs") / "pipeline_execution.log"
SUMMARY_FILE = Path("outputs") / "pipeline_summary.json"

# Ensure output directory exists
Path("outputs").mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def print_header():
    """Print pipeline header."""
    header = """
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║        NFL VETERANS TEAM CHANGE ANALYSIS - COMPLETE PIPELINE              ║
║                                                                            ║
║  Processing: nfl_panel_for_python.csv                                     ║
║  Pipeline: Collection → Preprocessing → Feature Engineering → Modeling    ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """
    print(header)

def log_message(message, level="INFO"):
    """Log message to both console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}"
    
    print(log_entry)
    
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry + "\n")

def check_dependencies():
    """Check if all required packages are available."""
    log_message("Checking dependencies...")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost (optional)',
        'lightgbm': 'lightgbm (optional)',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing_packages = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            log_message(f"  ✓ {package_name}")
        except ImportError:
            if 'optional' in package_name:
                log_message(f"  ⚠ {package_name} (optional - will skip if not available)", "WARNING")
            else:
                log_message(f"  ✗ {package_name} (REQUIRED)", "ERROR")
                missing_packages.append(package_name)
    
    if missing_packages:
        log_message(f"Missing required packages: {', '.join(missing_packages)}", "ERROR")
        return False
    
    return True

def check_input_data():
    """Check if input data file exists."""
    log_message("Checking input data...")
    
    input_file = Path("data") / "nfl_panel_for_python.csv"
    
    if not input_file.exists():
        log_message(f"Input file not found: {input_file}", "ERROR")
        return False
    
    # Check file size and basic properties
    file_size = input_file.stat().st_size / (1024 * 1024)
    log_message(f"  ✓ Input file found: {input_file} ({file_size:.2f} MB)")
    
    return True

def run_script(script_name, script_description, idx, total):
    """Run a single Python script."""
    script_path = PYTHON_DIR / script_name
    
    log_message("=" * 80, "INFO")
    log_message(f"SCRIPT {idx}/{total}: {script_description}", "INFO")
    log_message("=" * 80, "INFO")
    
    if not script_path.exists():
        log_message(f"Script not found: {script_path}", "ERROR")
        return False, 0
    
    print(f"\n{'=' * 80}")
    print(f"Running: {script_description}")
    print(f"Script: {script_path}")
    print(f"{'=' * 80}\n")
    
    start_time = datetime.now()
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per script
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Log output
        if result.stdout:
            print(result.stdout)
            with open(LOG_FILE, 'a') as f:
                f.write(result.stdout + "\n")
        
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
            with open(LOG_FILE, 'a') as f:
                f.write(f"STDERR:\n{result.stderr}\n")
        
        if result.returncode == 0:
            log_message(
                f"✓ {script_description} completed successfully in {elapsed:.2f}s",
                "SUCCESS"
            )
            return True, elapsed
        else:
            log_message(
                f"✗ {script_description} failed with return code {result.returncode}",
                "ERROR"
            )
            return False, elapsed
    
    except subprocess.TimeoutExpired:
        log_message(f"✗ {script_description} timed out after 1 hour", "ERROR")
        return False, 3600
    
    except Exception as e:
        log_message(f"✗ {script_description} encountered error: {str(e)}", "ERROR")
        log_message(traceback.format_exc(), "ERROR")
        return False, 0

def print_summary(results, execution_times, total_time):
    """Print final summary."""
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    
    log_message("=" * 80, "INFO")
    log_message("PIPELINE EXECUTION SUMMARY", "INFO")
    log_message("=" * 80, "INFO")
    
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    
    # Successful scripts
    print(f"\n✓ Successful: {len(successful)}/{len(results)}")
    log_message(f"Successful: {len(successful)}/{len(results)}", "INFO")
    
    for script_name, success, description in successful:
        elapsed = execution_times.get(script_name, 0)
        msg = f"  • {description} ({elapsed:.2f}s)"
        print(msg)
        log_message(f"  ✓ {description} ({elapsed:.2f}s)", "SUCCESS")
    
    # Failed scripts
    if failed:
        print(f"\n✗ Failed: {len(failed)}/{len(results)}")
        log_message(f"Failed: {len(failed)}/{len(results)}", "ERROR")
        
        for script_name, success, description in failed:
            msg = f"  ✗ {description}"
            print(msg)
            log_message(f"  ✗ {description}", "ERROR")
    
    # Timing summary
    print(f"\nExecution Times:")
    print(f"  Total Pipeline Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    log_message(f"Total Pipeline Time: {total_time:.2f}s ({total_time/60:.2f} minutes)", "INFO")
    
    for script_name, elapsed in execution_times.items():
        pct = (elapsed / total_time * 100) if total_time > 0 else 0
        msg = f"    • {script_name}: {elapsed:.2f}s ({pct:.1f}%)"
        print(msg)
        log_message(msg, "INFO")
    
    # Overall status
    print("\n" + "=" * 80)
    if len(failed) == 0:
        status_msg = "🎉 PIPELINE COMPLETE - ALL SCRIPTS EXECUTED SUCCESSFULLY!"
        print(status_msg)
        log_message(status_msg, "SUCCESS")
    else:
        status_msg = f"⚠️  PIPELINE COMPLETE WITH {len(failed)} FAILURE(S)"
        print(status_msg)
        log_message(status_msg, "WARNING")
    
    print("=" * 80 + "\n")

def generate_outputs_summary():
    """Generate a summary of all outputs created."""
    print("\nOUTPUT LOCATIONS:")
    print("=" * 80)
    
    outputs = {
        "Enriched Data": "data/enriched/",
        "Processed Data": "data/processed/",
        "ML Features": "data/ml_features/",
        "Models": "outputs/models/",
        "Analysis": "outputs/analysis/",
        "Figures": "outputs/figures/",
        "Logs": "outputs/pipeline_execution.log"
    }
    
    for label, path in outputs.items():
        p = Path(path)
        if p.exists():
            if p.is_dir():
                file_count = len(list(p.glob('*')))
                print(f"  ✓ {label}: {path} ({file_count} files)")
                log_message(f"✓ {label}: {path} ({file_count} files)", "INFO")
            else:
                print(f"  ✓ {label}: {path}")
                log_message(f"✓ {label}: {path}", "INFO")
        else:
            print(f"  ⚠ {label}: {path} (not created)")
            log_message(f"⚠ {label}: {path} (not created)", "WARNING")
    
    print("=" * 80)

def save_execution_summary(results, execution_times, total_time, start_time, end_time):
    """Save execution summary as JSON."""
    summary = {
        'timestamp_start': start_time.isoformat(),
        'timestamp_end': end_time.isoformat(),
        'total_execution_time_seconds': total_time,
        'total_execution_time_minutes': total_time / 60,
        'pipeline_status': 'SUCCESS' if all(r[1] for r in results) else 'FAILED',
        'scripts_total': len(results),
        'scripts_successful': sum(1 for r in results if r[1]),
        'scripts_failed': sum(1 for r in results if not r[1]),
        'execution_details': [
            {
                'script': result[0],
                'description': result[2],
                'success': result[1],
                'execution_time_seconds': execution_times.get(result[0], 0)
            }
            for result in results
        ],
        'output_directories': {
            'enriched_data': 'data/enriched/',
            'processed_data': 'data/processed/',
            'ml_features': 'data/ml_features/',
            'models': 'outputs/models/',
            'analysis': 'outputs/analysis/',
            'figures': 'outputs/figures/',
            'logs': 'outputs/pipeline_execution.log'
        }
    }
    
    with open(SUMMARY_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    
    log_message(f"Execution summary saved to: {SUMMARY_FILE}", "INFO")

def main():
    """Main pipeline execution."""
    # Print header
    print_header()
    
    # Start logging
    log_message("Pipeline started", "INFO")
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check dependencies
    if not check_dependencies():
        log_message("Dependency check failed. Exiting.", "ERROR")
        return False
    
    print("\n")
    
    # Check input data
    if not check_input_data():
        log_message("Input data check failed. Exiting.", "ERROR")
        return False
    
    print("\n")
    
    # Execute scripts
    results = []
    execution_times = {}
    
    for idx, (script_name, description) in enumerate(PYTHON_SCRIPTS, 1):
        success, elapsed = run_script(script_name, description, idx, len(PYTHON_SCRIPTS))
        results.append((script_name, success, description))
        execution_times[script_name] = elapsed
        
        if not success:
            log_message(f"Stopping pipeline due to {script_name} failure", "ERROR")
            break
    
    # Calculate total time
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    # Print summary
    print_summary(results, execution_times, total_time)
    
    # Generate outputs summary
    generate_outputs_summary()
    
    # Save execution summary
    save_execution_summary(results, execution_times, total_time, start_time, end_time)
    
    # Final status
    all_successful = all(r[1] for r in results)
    
    if all_successful:
        print("\n PIPELINE EXECUTION SUCCESSFUL")
        print(f"   Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"   Log file: {LOG_FILE}")
        print(f"   Summary file: {SUMMARY_FILE}")
        log_message("Pipeline completed successfully", "SUCCESS")
        return True
    else:
        print("\n PIPELINE EXECUTION FAILED")
        print(f"   Check log file for details: {LOG_FILE}")
        log_message("Pipeline failed", "ERROR")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
