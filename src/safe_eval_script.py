from src.safe_eval import eval_kernel_against_ref, KernelExecResult
import sys
import json

def result_to_json_dict(result: KernelExecResult) -> dict:
    """
    Convert KernelExecResult to a JSON-serializable dictionary
    """
    def make_serializable(obj):
        """Recursively convert values to JSON serializable types"""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    return {
        'compiled': result.compiled,
        'run': result.run,
        'correctness': result.correctness,
        'metadata': make_serializable(result.metadata),
        'runtime': float(result.runtime),
        'runtime_stats': make_serializable(result.runtime_stats),
        'runtime_original': float(result.runtime_original),
        'runtime_stats_original': make_serializable(result.runtime_stats_original)
    }

def main():
    # Read input from files specified in command line arguments
    ref_path = sys.argv[1]
    custom_path = sys.argv[2]
    process_index = int(sys.argv[3])
    eval_cache_path = sys.argv[4]

    with open(ref_path, 'r') as f:
        ref_arch = f.read()
    with open(custom_path, 'r') as f:
        custom_cuda = f.read()

    result = eval_kernel_against_ref(
        process_index=process_index,
        original_model_src=ref_arch,
        custom_model_src=custom_cuda,
        measure_performance=True,
        verbose=True
    )

    result = result_to_json_dict(result)
    with open(eval_cache_path, 'w') as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()