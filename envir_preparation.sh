# 1. Create the speechain environment if there is no environment named speechain
speechain_envir=$(conda env list | grep speechain)
if [ -z "${speechain_envir}" ]; then
  conda env create -f environment.yaml
  speechain_envir=$(conda env list | grep speechain)
fi

# 2. Get the environment root path by conda
read -ra speechain_envir <<< "${speechain_envir}"
envir_path="${speechain_envir[$((${#speechain_envir[*]}-1))]}"

# 3. Get the python compiler path in the environment root
pycompiler_path="$(ls "${envir_path}"/bin/python?.?)"
# add the python compiler path to the environmental variables
if ! grep -q "export SPEECHAIN_PYTHON" ~/.bashrc; then
  echo "export SPEECHAIN_PYTHON=${pycompiler_path}" >> ~/.bashrc
fi
export SPEECHAIN_PYTHON=${pycompiler_path}

# 4. Add the current path to the environmental variables as the toolkit root path
if ! grep -q "export SPEECHAIN_ROOT" ~/.bashrc; then
  echo "export SPEECHAIN_ROOT=${PWD}" >> ~/.bashrc
fi
export SPEECHAIN_ROOT=${PWD}

# 5. Install the local development packages to conda environment
conda activate speechain
${SPEECHAIN_PYTHON} -m pip install -e "${SPEECHAIN_ROOT}"
