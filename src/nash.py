import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog


def _solve_zero_sum_game(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square payoff matrix, got shape {matrix.shape}")

    num_strategies = matrix.shape[0]

    c_row = np.zeros(num_strategies + 1)
    c_row[-1] = -1.0
    a_ub_row = np.hstack((-matrix.T, np.ones((num_strategies, 1))))
    b_ub_row = np.zeros(num_strategies)
    a_eq_row = np.append(np.ones(num_strategies), 0.0)[None, :]
    bounds_row = [(0.0, None)] * num_strategies + [(None, None)]

    row_result = linprog(c_row, A_ub=a_ub_row, b_ub=b_ub_row, A_eq=a_eq_row, b_eq=[1.0], bounds=bounds_row, method="highs")
    if not row_result.success:
        raise RuntimeError(f"Row player LP failed: {row_result.message}")

    c_col = np.zeros(num_strategies + 1)
    c_col[-1] = 1.0
    a_ub_col = np.hstack((matrix, -np.ones((num_strategies, 1))))
    b_ub_col = np.zeros(num_strategies)
    a_eq_col = np.append(np.ones(num_strategies), 0.0)[None, :]
    bounds_col = [(0.0, None)] * num_strategies + [(None, None)]

    col_result = linprog(c_col, A_ub=a_ub_col, b_ub=b_ub_col, A_eq=a_eq_col, b_eq=[1.0], bounds=bounds_col, method="highs")
    if not col_result.success:
        raise RuntimeError(f"Column player LP failed: {col_result.message}")

    row_strategy = np.clip(row_result.x[:-1], 0.0, 1.0)
    col_strategy = np.clip(col_result.x[:-1], 0.0, 1.0)
    row_sum = row_strategy.sum()
    col_sum = col_strategy.sum()
    if row_sum <= 0 or col_sum <= 0:
        raise RuntimeError("LP solver returned a degenerate strategy distribution")
    row_strategy /= row_sum
    col_strategy /= col_sum

    return row_strategy, col_strategy, float(row_result.x[-1])

def calculate_nash_equilibria(
    matrix_path: str,
    output_path: str
) -> None:
    
    """
    Reads the Delta matrix from a directory, calculates the Nash Equilibria
    for each epoch, and saves the results in a CSV file (table format).

    Args:
        matrix_path (str): The path to the file containing the payoff matrices.
        output_path (str): The path to the output CSV file. 

    Returns:
        None (The results are saved directly to a CSV file in the same directory).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading matrices from {matrix_path}...", flush=True)
    matrices_delta = np.load(matrix_path)
    num_epochs = matrices_delta.shape[0]
    num_strategies = matrices_delta.shape[1]
    print(f"Loaded matrices: {num_epochs} epochs, {num_strategies} strategies", flush=True)
    
    results_table = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}: starting computation...", flush=True)
        A = matrices_delta[epoch]
        prob_attacker, prob_defender, game_value = _solve_zero_sum_game(A)
        print(f"Epoch {epoch + 1}, Equilibrium 1: game_value={game_value:.6f}", flush=True)

        p_att_clean = np.round(np.where(prob_attacker < 1e-5, 0, prob_attacker), 4)
        p_def_clean = np.round(np.where(prob_defender < 1e-5, 0, prob_defender), 4)

        row = {
            'Epoch': epoch + 1,
            'Equilibrium_Num': 1,
            'Relative_Loss': round(game_value, 4)
        }

        for j in range(num_strategies): row[f'A{j+1}'] = p_att_clean[j]
        for j in range(num_strategies): row[f'D{j+1}'] = p_def_clean[j]

        results_table.append(row)
            
    df = pd.DataFrame(results_table)
    df.to_csv(output_path, sep=';', index=False)
    print(f"Saved Nash equilibria to {output_path}", flush=True)

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    tasks = [
        {"size": 10, "eps": 0.1},
        {"size": 10, "eps": 0.2},
        {"size": 27, "eps": 0.2},
    ]

    for task in tasks:
        size = task["size"]
        eps = task["eps"]
        
        in_file = os.path.join(project_root, f"output/matrices/matrices_{size}x{size}_eps_{eps}/payoff_matrices_relative_loss.npy")
        out_file = os.path.join(project_root, f"output/nash_equilibria/nash_{size}x{size}_eps_{eps}/nash_{size}x{size}_eps_{eps}.csv")
        
        if os.path.exists(in_file):
            print(f"\n{'='*60}")
            print(f"Processing {size}x{size}, epsilon={eps}")
            print(f"{'='*60}")
            calculate_nash_equilibria(in_file, out_file)
        else:
            print(f"\n⚠️ Skipping {size}x{size}, epsilon={eps} (file not found: {in_file})")
