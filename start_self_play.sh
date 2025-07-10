#!/bin/bash
echo "Iniciando o script de auto-jogo (time vs time)..."
export OMP_NUM_THREADS=1

# Define o host e as portas para cada time
host=${1:-localhost}
port_time_A=${2:-3100} # Porta para o primeiro time
port_time_B=${2:-3100} # Porta para o segundo time (diferente da primeira)

# --- Inicia o Time A ---
echo "Iniciando Time A ('Pequi-Mecanico-A') na porta base $port_time_A..."
for i in {1..11}; do
  python3 ./Run_teamCEIA.py -i $host -p $port_time_A -u $i -t "Pequi-Mecanico-A" -P 0 -D 0 &
done

# --- Inicia o Time B ---
echo "Iniciando Time B ('Pequi-Mecanico-B') na porta base $port_time_B..."
for i in {1..11}; do
  # É crucial usar uma porta e um nome de time diferentes
  python3 ./Run_teamCEIA2.py -i $host -p $port_time_B -u $i -t "Pequi-Mecanico-B" -P 0 -D 0 &
done

echo "Times A e B iniciados com sucesso!"