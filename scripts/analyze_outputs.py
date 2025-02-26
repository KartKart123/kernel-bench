import os
import json
import pandas as pd
from collections import defaultdict
import glob
from pathlib import Path
import argparse
import re

def get_task_name(task_id, level):
    """
    Extract task name from KernelBench folder structure based on task_id.
    
    Args:
        task_id: The task ID
        level: The level number (1, 2, etc.)
        
    Returns:
        Task name extracted from the file naming convention
    """
    try:
        # Look for the task file in the KernelBench directory
        kernel_bench_dir = Path("KernelBench")
        level_dir = kernel_bench_dir / f"level{level}"
        
        if not level_dir.exists():
            return f"Unknown-{task_id}"
            
        # Find files that start with the task_id followed by underscore
        pattern = f"{task_id}_*.py"
        matching_files = list(level_dir.glob(pattern))
        
        if matching_files:
            # Extract task name from filename (part between first _ and .py)
            filename = matching_files[0].name
            match = re.match(r'\d+_([^.]+)\.py', filename)
            if match:
                return match.group(1)
        
        return f"Unknown-{task_id}"
    except Exception as e:
        print(f"Error getting task name for {task_id}: {e}")
        return f"Unknown-{task_id}"

def process_training_data(level_dir, level_num):
    """
    Process training data from multiple devices across training steps for a specific level.
    
    Args:
        level_dir: Path to the level directory containing step folders
        level_num: The level number
    
    Returns:
        DataFrame with aggregated metrics by task and step
    """
    # Store data organized by task and step
    task_step_data = defaultdict(lambda: defaultdict(lambda: {
        'compiled': 0, 'run': 0, 'correct': 0, 'total': 0
    }))
    
    # Get all step directories sorted numerically
    step_dirs = sorted([d for d in os.listdir(level_dir) if os.path.isdir(os.path.join(level_dir, d))],
                      key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else float('inf'))
    
    for step_dir in step_dirs:
        step_path = os.path.join(level_dir, step_dir)
        
        # Extract step number from directory name
        try:
            step_num = int(step_dir.split('_')[-1])
        except ValueError:
            print(f"Skipping directory with non-numeric step: {step_dir}")
            continue
        
        # Find all device JSON files in this step directory
        device_files = glob.glob(os.path.join(step_path, "device_*.json"))
        
        for device_file in device_files:
            try:
                with open(device_file, 'r') as f:
                    device_data = json.load(f)
                
                # Process each task in the device data
                task_id = None
                for task_data in device_data:
                    task_id = task_data.get('task_id', 'unknown')
                    
                    # Aggregate metrics for this task and step
                    task_step_data[task_id][step_num]['compiled'] += task_data.get('compiled', 0)
                    task_step_data[task_id][step_num]['run'] += task_data.get('run', 0)
                    task_step_data[task_id][step_num]['correct'] += task_data.get('correctness', 0)
                    task_step_data[task_id][step_num]['total'] += 1
                if task_id == 'unknown':
                    print("Warning: task_id is unknown")
                    
            except Exception as e:
                print(f"Error processing {device_file}: {e}")
    
    # Convert nested dictionary to a flat list for DataFrame
    all_data = []
    for task_id, step_dict in task_step_data.items():
        # Get task name from KernelBench directory
        task_name = get_task_name(task_id, level_num)
        
        for step_num, metrics in step_dict.items():
            # Calculate rates
            total = metrics['total']
            if total > 0:
                compile_rate = metrics['compiled'] / total
                run_rate = metrics['run'] / total
                success_rate = metrics['correct'] / total
            else:
                compile_rate = run_rate = success_rate = 0
            
            all_data.append({
                'task_id': task_id,
                'task_name': task_name,
                'level': level_num,
                'step': step_num,
                'compiled': metrics['compiled'],
                'run': metrics['run'],
                'correct': metrics['correct'],
                'total': total,
                'compile_rate': compile_rate,
                'run_rate': run_rate,
                'success_rate': success_rate
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    return df

def process_all_levels(base_dir):
    """
    Process all level directories found in the base directory.
    
    Args:
        base_dir: Path to the base directory containing level folders
    
    Returns:
        DataFrame with aggregated metrics for all levels
    """
    all_dfs = []
    
    # Find all level directories
    level_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('level_')]
    
    if not level_dirs:
        print(f"No level directories found in {base_dir}")
        return pd.DataFrame()
    
    print(f"Found level directories: {', '.join(level_dirs)}")
    
    for level_dir in sorted(level_dirs):
        level_path = os.path.join(base_dir, level_dir)
        level_match = re.search(r'level_(\d+)', level_dir)
        
        if level_match:
            level_num = int(level_match.group(1))
            print(f"Processing level {level_num} from {level_path}...")
            
            df = process_training_data(level_path, level_num)
            
            if not df.empty:
                all_dfs.append(df)
            else:
                print(f"No data found for level {level_num}")
        else:
            print(f"Skipping directory with invalid level format: {level_dir}")
    
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def generate_summary_report(df, output_dir):
    """Generate a summary report of the analysis"""
    report_path = os.path.join(output_dir, "analysis_report.txt")
    
    # Convert 'task_id' to numeric for proper sorting
    df['task_id'] = pd.to_numeric(df['task_id'], errors='coerce')  # Convert to numeric, coercing errors to NaN
    df = df.sort_values(by=['level', 'task_id'])  # Sort by level and task_id
    
    with open(report_path, 'w') as f:
        f.write("=== Training Performance Analysis Report ===\n\n")
        
        # Overall statistics
        f.write("Overall Statistics:\n")
        f.write(f"Total training steps analyzed: {df['step'].nunique()}\n")
        f.write(f"Total tasks analyzed: {df['task_id'].nunique()}\n")
        f.write(f"Total levels analyzed: {df['level'].nunique()}\n")
        f.write(f"Total generations analyzed: {df['total'].sum()}\n")
        f.write("\n")
        
        # Add a section for step-by-step performance for each task
        print("\n\nDetailed Step-by-Step Performance:")
        f.write("\n\nDetailed Step-by-Step Performance:\n")
        
        # Group by level first
        for level in sorted(df['level'].unique()):
            level_df = df[df['level'] == level]
            
            print(f"\n=== Level {level} ===")
            f.write(f"\n=== Level {level} ===\n")
            
            for task_id in level_df['task_id'].unique():
                task_df = level_df[level_df['task_id'] == task_id].sort_values('step')
                task_name = task_df['task_name'].iloc[0]
                
                print(f"\nTask {task_id}: {task_name}")
                f.write(f"\nTask {task_id}: {task_name}\n")
                print("  Step | Compile Rate | Run Rate | Correct Rate | Correct/Total")
                f.write("  Step | Compile Rate | Run Rate | Correct Rate | Correct/Total\n")
                print("  --------------------------------------------------------")
                f.write("  --------------------------------------------------------\n")
                
                for _, row in task_df.iterrows():
                    # Format the row with left-aligned fields for alignment
                    line = f"  {int(row['step']):<4} | {row['compile_rate'] * 100:<12.2f} | {row['run_rate'] * 100:<9.2f} | {row['success_rate'] * 100:<12.2f} | {row['correct']}/{row['total']}\n"
                    print(line, end='')  # Print to CLI without adding an extra newline
                    f.write(line)  # Write to file
        
        f.write("\n=== End of Report ===\n")

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Analyze KernelBench training outputs')
    parser.add_argument('output_dir', type=str, help='Path to the base directory containing level folders')
    args = parser.parse_args()
    
    base_dir = args.output_dir
    
    print(f"Processing training data from {base_dir}...")
    df = process_all_levels(base_dir)
    
    if df.empty:
        print("No data found. Please check the directory path and file structure.")
        return
    
    print("Generating summary report...")
    generate_summary_report(df, base_dir)
    
    # Save processed data to CSV for further analysis
    df.to_csv(os.path.join(base_dir, "task_performance_by_step.csv"), index=False)
    
    print(f"Analysis complete. Results saved to {base_dir}")

if __name__ == "__main__":
    main()