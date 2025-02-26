from pydra import Config
import pydra
import os
import json
import numpy as np
from collections import defaultdict

class AnalyzeConfig(Config):
    def __init__(self):
        self.level = 1
        self.eval_dir = f"reasoning_traces/level_{self.level}/eval"

def analyze_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    total_generations = len(data['generations'])
    compiled_count = 0
    correct_count = 0
    speedups = []
    
    for gen in data['generations']:
        if 'eval_result' not in gen or gen['eval_result'] is None:
            continue
            
        result = gen['eval_result']
        if result['compiled']:
            compiled_count += 1
        if result.get('correctness', False):
            correct_count += 1
        
        # Calculate speedup if both runtimes are available
        if result['compiled'] and result.get('correctness', False):
            if result.get('runtime') and result.get('runtime_original'):
                speedup = result['runtime_original'] / result['runtime']
                speedups.append(speedup)
    
    return {
        'total': total_generations,
        'compiled': compiled_count,
        'correct': correct_count,
        'max_speedup': max(speedups) if speedups else 0,
        'speedups': speedups
    }

@pydra.main(base=AnalyzeConfig)
def main(config: AnalyzeConfig):
    print(f"Analyzing evaluation results in {config.eval_dir}...")
    
    if not os.path.exists(config.eval_dir):
        print(f"Error: Directory {config.eval_dir} does not exist!")
        return
        
    files = sorted(os.listdir(config.eval_dir))
    if not files:
        print("No evaluation files found!")
        return
        
    total_stats = defaultdict(int)
    all_speedups = []
    file_stats = {}
    
    # Analyze each file
    for file in files:
        file_path = os.path.join(config.eval_dir, file)
        stats = analyze_file(file_path)
        file_stats[file] = stats
        
        total_stats['total_generations'] += stats['total']
        total_stats['total_compiled'] += stats['compiled']
        total_stats['total_correct'] += stats['correct']
        all_speedups.extend(stats['speedups'])
    
    # Sort file_stats by the leading number in the filename so it goes from 1 to 100
    file_stats = dict(sorted(file_stats.items(), 
                        key=lambda x: int(x[0].split('_')[0])))
    
    # Print per-file statistics
    print("\nPer-file Statistics:")
    print("-" * 80)
    print(f"{'File':<40} {'Compiled %':>10} {'Correct %':>10} {'Max Speedup':>12}")
    print("-" * 80)
    
    for file, stats in file_stats.items():
        compiled_pct = (stats['compiled'] / stats['total'] * 100) if stats['total'] > 0 else 0
        correct_pct = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"{file[:38]:<40} {compiled_pct:>9.1f}% {correct_pct:>9.1f}% {stats['max_speedup']:>11.2f}x")
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print("-" * 40)
    total_gens = total_stats['total_generations']
    compiled_pct = (total_stats['total_compiled'] / total_gens * 100) if total_gens > 0 else 0
    correct_pct = (total_stats['total_correct'] / total_gens * 100) if total_gens > 0 else 0
    
    print(f"Total generations analyzed: {total_gens}")
    print(f"Overall compiled rate: {compiled_pct:.1f}%")
    print(f"Overall correctness rate: {correct_pct:.1f}%")
    print(f"Maximum speedup achieved: {max(all_speedups):.2f}x")
    
    if all_speedups:
        print(f"Average speedup (correct solutions): {np.mean(all_speedups):.2f}x")
        print(f"Median speedup (correct solutions): {np.median(all_speedups):.2f}x")
        print(f"Speedup std dev (correct solutions): {np.std(all_speedups):.2f}")
        
        # Print speedup distribution
        print("\nSpeedup Distribution:")
        print("-" * 40)
        percentiles = [25, 50, 75, 90, 95, 99]
        for p in percentiles:
            print(f"{p}th percentile: {np.percentile(all_speedups, p):.2f}x")

if __name__ == "__main__":
    main() 