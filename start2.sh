#!/bin/bash
echo "Iniciando o script start.sh..."
export OMP_NUM_THREADS=1

host=${1:-localhost}
port=${2:-3100}

for i in {1..11}; do
  python3 ./Run_teamCEIA2.py -i $host -p $port -u $i -t Pequi-Mecânico2 -P 0 -D 0 &
done