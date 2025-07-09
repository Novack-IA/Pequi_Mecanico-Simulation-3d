import os
from scripts.commons.Script import Script

# 1. Inicializa o Script aqui, no escopo global.
#    Isso garante que os módulos C++ sejam compilados ANTES de qualquer outra coisa.
script = Script()

# 2. Agora que a compilação está garantida, podemos importar o nosso Gym.
from scripts.gyms.self_play_gym import Train as TrainSelfPlay

def main():
    """
    Script principal para o treinamento de self-play por N gerações.
    """
    generations = 100
    total_timesteps_per_generation = 100000  # Ajuste conforme necessário

    # Caminho do melhor modelo da geração anterior
    last_best_model = None

    for gen in range(1, generations + 1):
        print(f"=========================================")
        print(f" INICIANDO GERAÇÃO DE TREINAMENTO Nº {gen} ")
        print(f"=========================================")

        # A instância do 'script' já foi criada no escopo global.
        # Instancia nosso treinador customizado, passando o script já inicializado.
        trainer = TrainSelfPlay(script)

        # Inicia o treinamento
        # O adversário será o melhor modelo da geração anterior
        trainer.train(
            total_timesteps=total_timesteps_per_generation,
            opponent_model_path=last_best_model,
            save_path_prefix=f"gen_{gen}"
        )

        # Define o melhor modelo desta geração como o adversário da próxima
        last_best_model = f"./models/gen_{gen}/best_model.zip"

        # Verifica se o arquivo do melhor modelo realmente existe
        if not os.path.exists(last_best_model):
            print(f"AVISO: O arquivo {last_best_model} não foi encontrado. A próxima geração usará um adversário aleatório.")
            last_best_model = None
            
        print(f"\nFIM DA GERAÇÃO {gen}.\n")

if __name__ == "__main__":
    main()