import json
from typing import Optional

def get_only(s):
    if len(s) == 1:
        return s[0]
    raise ValueError("FUCK! Error in get_only")

def fetch_speedups_from_problem_id(problem_id: tuple[int, int], leaderboard_data: str) -> list[float]:
    with open(leaderboard_data, 'r') as f:
        data = json.load(f)
        problem_key_prefix = f"Level {problem_id[0]}: {problem_id[1]}_"
        match = get_only([key for key in data if key.startswith(problem_key_prefix)])
        return [entry["torch_speedup"] for entry in data[match]]

def get_relative_speedup_wrt_leaderboard(speedup : float, problem_id: tuple[int, int], leaderboard_data: str = "/home/ubuntu/kernel-bench-pietro/kernel-bench/KernelBenchLeaderBoard/assets/data.json") -> float:
    leaderboard_speedups = fetch_speedups_from_problem_id(problem_id, leaderboard_data)
    if len(leaderboard_speedups) == 0:
        raise ValueError("I guess this means that we solved a previously unsolved kernel, WTF sick")
        # return float('inf')
    relative_speedup = speedup / max(leaderboard_speedups)
    if relative_speedup > 1:
        print(f"Improvement over leaderboard by {relative_speedup} for problem with ID {problem_id}")
    return relative_speedup

def test_evaluation():
    speedup = 0.16857142857142854 * 4 # 4x the best one in the leaderboard
    problem_id = (1,1)
    relative_speedup = get_relative_speedup_wrt_leaderboard(speedup, problem_id)
    assert relative_speedup == 4

if __name__ == '__main__':
    test_evaluation()
