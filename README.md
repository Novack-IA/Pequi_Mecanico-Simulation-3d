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
    '''bash
    # Em sistemas Debian/Ubuntu
    sudo apt-get install build-essential
    '''
* **Java 17 (ou superior):** Requerido para executar o RoboViz.

### 2. Servidor e Visualizador

* **SimSpark (Servidor):**
    '''bash
    sudo apt-get update
    sudo apt-get install simspark
    '''
* **RoboViz (Visualizador):**
    1.  Faça o download dos binários pré-compilados na [página de releases do RoboViz](https://github.com/magmaOffenburg/RoboViz/releases).
    2.  Descompacte o arquivo em um local de sua preferência.

### 3. Ambiente Python e Dependências

1.  **Crie e ative um ambiente virtual:**
    '''bash
    python3 -m venv venv
    source venv/bin/activate
    '''
2.  **Instale as bibliotecas Python:**
    '''bash
    pip install -r requirements.txt
    '''
3.  **Compilação dos Módulos C++:**
    Na primeira vez que você executar um agente (ex: `./start.sh`), o projeto tentará compilar automaticamente os módulos C++ localizados na pasta `/cpp`. O script `scripts/commons/Script.py` gerencia esse processo. Se a compilação falhar, verifique se você tem os pré-requisitos do sistema instalados.

## Como Rodar a Simulação

Para iniciar uma partida, você precisa executar o servidor, o visualizador e os agentes, cada um em seu próprio terminal.

**Passo 1: Iniciar o Servidor SimSpark**
Abra um terminal e inicie o servidor de simulação.
'''bash
rcssserver3d
'''

**Passo 2: Iniciar o RoboViz**
Em um segundo terminal, execute o visualizador.
'''bash
# Navegue até a pasta onde você descompactou o RoboViz
./roboviz.sh
'''

**Passo 3: Iniciar os Agentes**
Em um terceiro terminal, use os scripts `*.sh` para iniciar os jogadores em diferentes cenários.

* **Partida Completa (11 vs 11):** Inicia um time de 11 jogadores.
    '''bash
    ./start.sh
    '''

* **Disputa de Pênaltis:** Inicia um goleiro e um batedor para treino de pênaltis.
    '''bash
    ./start_penalty.sh
    '''

* **Modo de Depuração:** Inicia os jogadores com logs detalhados e desenhos de depuração ativados no RoboViz.
    '''bash
    ./start_debug.sh
    '''

**Passo 4: Encerrar a Simulação**
Para parar todos os processos dos agentes de uma vez:
'''bash
./kill.sh
'''

## Argumentos de Linha de Comando

Você pode personalizar a execução dos agentes usando argumentos. O script `Run_Player.py` aceita as seguintes flags:

* `-i <IP>`: Endereço do servidor (padrão: `localhost`).
* `-p <PORTA>`: Porta do servidor para os agentes (padrão: `3100`).
* `-m <PORTA>`: Porta do monitor (padrão: `3200`).
* `-t <NOME>`: Nome do time (padrão: `FCPortugal`).
* `-u <NUM>`: Número do uniforme do jogador (1-11).
* `-r <TIPO>`: Tipo do robô (0-4), definindo suas características físicas.
* `-P <0|1>`: Ativa o modo de disputa de pênaltis.
* `-D <0|1>`: Ativa o modo de depuração.
* `-F <0|1>`: Ativa o modo `magmaFatProxy`.

## Desenvolvimento e Pesquisa

### Utilitários
Execute `Run_Utils.py` para acessar um menu interativo com diversas ferramentas de desenvolvimento, incluindo:
* Controle individual de juntas.
* Demonstrações de cinemática direta e inversa.
* Testes do planejador de caminho.
* Configurações do servidor.

'''bash
python3 Run_Utils.py
'''

### Treinamento com Aprendizado por Reforço
O projeto está pronto para pesquisa em RL. A pasta `scripts/gyms` contém ambientes Gym para treinar diferentes habilidades.

* Para **treinar um novo modelo** ou **testar um existente**, selecione a opção `Gyms` no menu de `Run_Utils.py`.
* A classe `Train` dentro de cada arquivo de ambiente gerencia o processo de treinamento e teste usando `Stable Baselines3`.

## Configuração

* **Formação do Time:** As posições iniciais e os tipos de robô para cada jogador são definidos em `config/formation.json`.
* **Configurações do Servidor:** Use o utilitário `Server` (`Run_Utils.py` -> `Server`) para modificar facilmente as configurações do SimSpark, como modo síncrono, tempo real e ruído, sem precisar editar os arquivos de configuração manualmente.