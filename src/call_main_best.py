import pandas as pd
import subprocess
import os
import tqdm

if __name__ == "__main__":

    best_params = pd.read_csv('src/data/outputs/results/best_params.csv')

    python_interpreter = os.path.join(r"C:\Projects\Mestrado\speech-acts-classification\venv_mestrado", "Scripts", "python.exe")
    best_params = best_params[(best_params['use_pos']==True) & (best_params['use_context']==False) & (best_params['use_trans']==True)]
    for idx, row in best_params.iterrows():
        # Construindo a lista de argumentos
        print(row['use_pos'], row['use_context'], row['use_trans'], row['resample_perc'], row["lr_rate"], row["use_weights"])
        args = ["--use_pos", str(row['use_pos']), "--use_context", str(row['use_context']), 
                "--use_trans", str(row['use_trans']), "--lr_rate", str(row["lr_rate"]), "--resample_perc", str(row['resample_perc']),
                "--use_weights", str(row["use_weights"]), "--join_train_val", str(False), "--do_validation", str(True), "--path_to_results", 
                'src/data/outputs/results/best_model.csv']


        # Executando script1.py com os argumentos específicos
        try:
            subprocess.run([python_interpreter, r"src/main.py"] + args, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            # Imprimindo a saída de erro do script1.py
            print("Ocorreu um erro ao executar script1.py")
            print("Código de retorno:", e.returncode)
            print("Saída padrão (stdout):", e.stdout)
            print("Saída de erro (stderr):", e.stderr)
