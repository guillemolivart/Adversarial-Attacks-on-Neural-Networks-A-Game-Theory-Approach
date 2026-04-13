import os
import numpy as np
import nashpy as nash
import pandas as pd

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
        
    matrices_delta = np.load(matrix_path)
    num_epochs = matrices_delta.shape[0]
    
    results_table = []
    
    for epoch in range(num_epochs):
        A = matrices_delta[epoch]
        game = nash.Game(A)

        # Obtain the Nash equilibria for the current epoch
        equilibria = list(game.vertex_enumeration())
        
        for i, eq in enumerate(equilibria):
            prob_attacker = eq[0]
            prob_defender = eq[1]
            
            # Value of the game at the equilibrium
            game_value = game[prob_attacker, prob_defender][0] 
            
            p_att_clean = np.round(np.where(prob_attacker < 1e-5, 0, prob_attacker), 4)
            p_def_clean = np.round(np.where(prob_defender < 1e-5, 0, prob_defender), 4)
            
            row = {
                'Epoch': epoch + 1,
                'Equilibrium_Num': i + 1,
                'Delta_Value': round(game_value, 4)
            }
            
            for j in range(10): row[f'A{j+1}'] = p_att_clean[j]
            for j in range(10): row[f'D{j+1}'] = p_def_clean[j]
                
            results_table.append(row)
            
    df = pd.DataFrame(results_table)
    df.to_csv(output_path, sep=';', index=False)

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    eps = [0.1, 0.2]
    size = 10

    tasks = [
        {
            "input": f"output/matrices/matrices_{size}x{size}_eps_{eps[0]}/payoff_matrices_delta.npy",
            "output": f"output/nash_equilibria/nash_{size}x{size}_eps_{eps[0]}/nash_{size}x{size}_eps_{eps[0]}.csv"
        },
        {
            "input": f"output/matrices/matrices_{size}x{size}_eps_{eps[1]}/payoff_matrices_delta.npy",
            "output": f"output/nash_equilibria/nash_{size}x{size}_eps_{eps[1]}/nash_{size}x{size}_eps_{eps[1]}.csv"
        }
    ]
    
    for task in tasks:
        in_file = os.path.join(project_root, task["input"])
        out_file = os.path.join(project_root, task["output"])
        
        if os.path.exists(in_file):
            calculate_nash_equilibria(in_file, out_file)
