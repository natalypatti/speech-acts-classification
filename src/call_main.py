import itertools
import subprocess
import os
import tqdm

if __name__ == "__main__":

    list_resample_perc = [0.1, 0.05, 0.01]
    list_lr_rate = [5e-5, 1e-5, 5e-6]
    list_pos = [True, False]
    list_context = [True, False]
    list_trans = [True, False]
    list_use_weights = [True, False]

    all_combinations = itertools.product(*[list_pos, list_context, list_trans, list_resample_perc, list_lr_rate, list_use_weights])
    python_interpreter = os.path.join(r"C:\Projects\Mestrado\speech-acts-classification\venv_mestrado", "Scripts", "python.exe")
    
    train_pseudo = True
    for pos, con, trans, resample, lr_rate, use_weights in tqdm.tqdm(all_combinations):
        # Construindo a lista de argumentos
        print(pos, con, trans, resample, lr_rate, use_weights)
        args = ["--use_pos", str(pos), "--use_context", str(con), 
                "--use_trans", str(trans), "--lr_rate", str(lr_rate), "--resample_perc", str(resample),
                "--use_weights", str(use_weights), "--need_to_train_trans", str(train_pseudo)]
        # Executando script1.py com os argumentos específicos
        try:
            subprocess.run([python_interpreter, r"src/main.py"] + args, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            # Imprimindo a saída de erro do script1.py
            print("Ocorreu um erro ao executar script1.py")
            print("Código de retorno:", e.returncode)
            print("Saída padrão (stdout):", e.stdout)
            print("Saída de erro (stderr):", e.stderr)
        
        if trans: train_pseudo=False
