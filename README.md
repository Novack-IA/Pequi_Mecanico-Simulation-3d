# Pequi Mecânico - Simulação 3D

Este é o repositório oficial do time de Simulação 3D de Futebol da Universidade Federal de Goiás (UFG), uma colaboração com o time Pequi Mecânico. O projeto é uma plataforma completa para o desenvolvimento e teste de estratégias de futebol robótico no ambiente SimSpark, com foco em modularidade, desempenho e pesquisa em Inteligência Artificial.

## Destaques e Funcionalidades

* **Arquitetura Híbrida Python/C++:** Combina a agilidade do Python para a lógica de alto nível dos agentes com o desempenho do C++ para tarefas computacionalmente intensivas.
* **Localização 6D Avançada:** Utiliza um módulo de localização probabilística em C++ que funde dados de visão, giroscópio e acelerômetro para uma estimativa precisa da pose do robô no espaço.
* **Planejamento de Caminho com A*:** Integra um planejador de caminho A* otimizado para o ambiente do futebol, capaz de desviar de obstáculos dinâmicos e estáticos.
* **Predição de Trajetória da Bola:** Usa um módulo C++ para prever o movimento da bola, permitindo interceptações e um planejamento mais inteligente.
* **Framework de Aprendizado por Reforço:** Totalmente integrado com ambientes **OpenAI Gym** e **Stable Baselines3** para o treinamento de comportamentos complexos como andar, chutar e driblar.
* **Sistema de Comportamentos Modulares:** Permite a criação de comportamentos simples baseados em keyframes (XML) e comportamentos complexos e reativos em Python.
* **Utilitários de Desenvolvimento:** Inclui um menu de utilitários para depuração, testes de cinemática, controle de juntas e configuração fácil do servidor.

## Instalação

Siga estes três passos para configurar seu ambiente.

### 1. Pré-requisitos do Sistema

* **Compilador C++ e Make:** Necessário para compilar os módulos C++.
    ```bash
    # Em sistemas Debian/Ubuntu
    sudo apt-get install build-essential
    ```
* **Java 17 (ou superior):** Requerido para executar o RoboViz.

### 2. Servidor e Visualizador

* **SimSpark (Servidor):**
    ```bash
    sudo apt-get update
    sudo apt-get install simspark
    ```
* **RoboViz (Visualizador):**
    1.  Faça o download dos binários pré-compilados na [página de releases do RoboViz](https://github.com/magmaOffenburg/RoboViz/releases).
    2.  Descompacte o arquivo em um local de sua preferência.

### 3. Ambiente Python e Dependências

1.  **Crie e ative um ambiente virtual (altamente recomendado):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
2.  **Instale as bibliotecas Python:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Compilação dos Módulos C++:**
    Na primeira vez que você executar um agente (ex: `./start.sh`), o projeto tentará compilar automaticamente os módulos C++ localizados na pasta `/cpp`. O script `scripts/commons/Script.py` gerencia esse processo. Se a compilação falhar, verifique se você tem os pré-requisitos do sistema instalados.

## Como Rodar a Simulação

Para rodar uma partida, você precisa iniciar três componentes em terminais separados: o **servidor**, o **visualizador** e os **agentes** (jogadores).

**Passo 1: Iniciar o Servidor SimSpark**
Abra um terminal e inicie o servidor de simulação.
```bash
rcssserver3d
```

**Passo 2: Iniciar o RoboViz**
Em um segundo terminal, execute o visualizador.
```bash
# Navegue até a pasta onde você descompactou o RoboViz
./roboviz.sh
```

**Passo 3: Iniciar os Agentes**
Em um terceiro terminal, use os scripts `*.sh` para iniciar os jogadores em diferentes cenários.

* **Partida Completa (11 vs 11):** Inicia um time de 11 jogadores.
    ```bash
    ./start.sh
    ```

* **Disputa de Pênaltis:** Inicia um goleiro e um batedor para treino de pênaltis.
    ```bash
    ./start_penalty.sh
    ```

* **Modo de Depuração:** Inicia os jogadores com logs e desenhos de depuração ativados no RoboViz.
    ```bash
    ./start_debug.sh
    ```

**Passo 4: Encerrar a Simulação**
Para parar todos os processos dos agentes de uma vez, use o script `kill.sh`.
```bash
./kill.sh
```

## Treinamento de Modelos (Aprendizado por Reforço)

O framework permite treinar novos comportamentos usando aprendizado por reforço. Para iniciar o processo, use o menu de utilitários.

1.  **Execute o Menu de Utilitários:**
    ```bash
    python3 Run_Utils.py
    ```
2.  **Selecione a opção "Gyms"** e, em seguida, escolha o ambiente que deseja treinar ou testar.

### Configuração do Treinamento
Os principais parâmetros de treinamento são definidos diretamente no método `train` da classe `Train` de cada ambiente (ex: `scripts/gyms/Fast_Dribble.py`).

* **Parâmetros Editáveis:**
    * `n_envs`: Número de ambientes paralelos para treinamento.
    * `n_steps_per_env`: Número de passos por ambiente antes de cada atualização do modelo.
    * `total_steps`: Número total de passos de simulação para o treinamento.
    * `learning_rate`: A taxa de aprendizado do otimizador.
    * `folder_name`: Nome da pasta onde os logs e modelos serão salvos.

### Configuração das Portas para Treinamento
Para evitar conflitos durante o treinamento com múltiplos ambientes (`n_envs > 1`), o sistema gerencia as portas do servidor e do monitor automaticamente.

* **Porta Base dos Agentes (`-p`):** A porta inicial para os agentes é definida pelo argumento `-p` (padrão: 3100). Cada novo ambiente de treinamento usará uma porta incrementada: `porta_base`, `porta_base + 1`, `porta_base + 2`, e assim por diante.
* **Porta Base do Monitor (`-m`):** Para evitar conflitos com a porta do agente, a porta base do monitor é deslocada por um valor fixo (1000). Se a porta `-m` for 3200, os ambientes de treino usarão `3200 + 1000`, `3200 + 1001`, etc.

Você pode alterar as portas base ao iniciar o treinamento através de `Run_Utils.py`, por exemplo:
```bash
# Inicia o treinamento com a porta do agente base em 4100 e do monitor em 4200
python3 Run_Utils.py -p 4100 -m 4200
```
O sistema então gerenciará as portas incrementais a partir desses novos valores base.

## Desenvolvimento e Utilitários

### Scripts de Utilitários
Execute `Run_Utils.py` para acessar um menu interativo com diversas ferramentas de desenvolvimento, incluindo:
* Controle individual de juntas.
* Demonstrações de cinemática direta e inversa.
* Testes do planejador de caminho.
* Configurações do servidor SimSpark.

### Configuração Geral
* **Formação do Time:** As posições iniciais e os tipos de robô para cada jogador são definidos em `config/formation.json`.
* **Argumentos de Linha de Comando:** O script `scripts/commons/Script.py` gerencia a leitura de argumentos da linha de comando e de um arquivo `config.json`, permitindo uma configuração flexível.