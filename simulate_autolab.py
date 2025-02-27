#!/usr/bin/env python3
import os
import shutil
import subprocess
import argparse
from pathlib import Path


"""
This script simulates Autolab's autograding process.
# First create autograde.tar
make create_autograde

# Create a copy of your Makefile as autograde-Makefile
cp Makefile autograde-Makefile

# Run the simulation on a submission
python simulate_autolab.py handin.tar

# Or keep the simulation directory for debugging
python simulate_autolab.py handin.tar --keep
"""

def simulate_autolab(submission_path: str, autograde_dir: str = "autograde_simulation", keep: bool = False):
    """
    Simulate Autolab's autograding process.
    
    Args:
        submission_path: Path to student's submission file
        autograde_dir: Directory to use for autograding simulation
        keep: Whether to keep the simulation directory after completion
    """
    # Convert paths to Path objects
    submission_path = Path(submission_path)
    autograde_dir = Path(autograde_dir)
    
    print(f"\n{'='*80}")
    print(f"Starting Autolab simulation for: {submission_path}")
    print(f"{'='*80}\n")

    # Step 1: Create empty autograding directory
    print("Setting up autograding environment...")
    if autograde_dir.exists():
        shutil.rmtree(autograde_dir)
    autograde_dir.mkdir(parents=True)

    # Step 2: Copy required files to autograding directory
    print("Copying autograding files...")
    try:
        # Copy student submission
        shutil.copy2(submission_path, autograde_dir / "handin.tar")
        
        # Copy autograde.tar
        if not Path("autograde.tar").exists():
            raise FileNotFoundError("autograde.tar not found. Run 'make create_autograde' first.")
        shutil.copy2("autograde.tar", autograde_dir / "autograde.tar")
        
        # Copy and rename Makefile
        if not Path("autograde-Makefile").exists():
            raise FileNotFoundError("autograde-Makefile not found")
        shutil.copy2("autograde-Makefile", autograde_dir / "Makefile")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Step 3: Change to autograding directory
    original_dir = os.getcwd()
    os.chdir(autograde_dir)
    
    try:
        # Step 4: Execute make command
        print("\nExecuting make command...")
        print("-" * 40)
        result = subprocess.run(["make"], capture_output=True, text=True)
        
        # Print stdout and stderr
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        # Print return code
        print(f"\nMake process returned with code: {result.returncode}")
        
    finally:
        # Step 5: Change back to original directory
        os.chdir(original_dir)
        
        # Step 6: Cleanup (unless --keep is specified)
        if not keep and result.returncode == 0:
            print("\nCleaning up simulation directory...")
            shutil.rmtree(autograde_dir)
        else:
            print(f"\nSimulation files preserved in: {autograde_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Autolab's autograding process")
    parser.add_argument("submission", help="Path to student's submission file (handin.tar)")
    parser.add_argument("--keep", action="store_true", help="Keep simulation directory after completion")
    parser.add_argument("--dir", default="autograde_simulation", help="Directory to use for simulation")
    
    args = parser.parse_args()
    simulate_autolab(args.submission, args.dir, args.keep) 