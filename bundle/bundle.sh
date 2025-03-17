#!/bin/bash

# Call this script from any directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# cd to main folder
cd "${SCRIPT_DIR}/.."

rm -rf ./bundle/build
rm -rf ./bundle/dist

onefile="--onefile"

# bundle app, dependencies, and data files into a single executable
pyinstaller \
--onefile \
--add-binary '/usr/lib/x86_64-linux-gnu/libm.so.6:lib/' \
--add-data './behaviors/slot/common:behaviors/slot/common' \
--add-data './behaviors/slot/r0:behaviors/slot/r0' \
--add-data './behaviors/slot/r1:behaviors/slot/r1' \
--add-data './behaviors/slot/r2:behaviors/slot/r2' \
--add-data './behaviors/slot/r3:behaviors/slot/r3' \
--add-data './behaviors/slot/r4:behaviors/slot/r4' \
--add-data './behaviors/custom/Dribble/*.pkl:behaviors/custom/Dribble' \
--add-data './behaviors/custom/Walk/*.pkl:behaviors/custom/Walk' \
--add-data './behaviors/custom/Fall/*.pkl:behaviors/custom/Fall' \
${onefile} --distpath ./bundle/dist/ --workpath ./bundle/build/ --noconfirm --name peq Run_teamCEIA.py

# start.sh
cat > ./bundle/dist/start.sh << EOF
#!/bin/bash
export OMP_NUM_THREADS=1

host=\${1:-localhost}
port=\${2:-3100}

for i in {1..11}; do
  ./peq -i \$host -p \$port -u \$i -t Pequi_Mecânico &
done
EOF

# start_penalty.sh
cat > ./bundle/dist/start_penalty.sh << EOF
#!/bin/bash
export OMP_NUM_THREADS=1

host=\${1:-localhost}
port=\${2:-3100}

./peq -i \$host -p \$port -u 1  -t Pequi_Mecânico -P 1 &
./peq -i \$host -p \$port -u 11 -t Pequi_Mecânico -P 1 &
EOF

# start_fat_proxy.sh
cat > ./bundle/dist/start_fat_proxy.sh << EOF
#!/bin/bash
export OMP_NUM_THREADS=1

host=\${1:-localhost}
port=\${2:-3100}

for i in {1..11}; do
  ./peq -i \$host -p \$port -u \$i -t Pequi_Mecânico -F 1 &
done
EOF

# kill.sh
cat > ./bundle/dist/kill.sh << EOF
#!/bin/bash
pkill -9 -e peq
EOF

# execution permission
chmod a+x ./bundle/dist/start.sh
chmod a+x ./bundle/dist/start_penalty.sh
chmod a+x ./bundle/dist/start_fat_proxy.sh
chmod a+x ./bundle/dist/kill.sh