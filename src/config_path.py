from os.path import join, realpath, dirname

BASE_PATH = ".."
print(f"Base path: {BASE_PATH}")
DATA_PATH = join(BASE_PATH, 'data')
MS_DATA_PATH = join(DATA_PATH, 'ms')
REACTOME_DATA_PATH = join(DATA_PATH, 'reactome')
PLOTS_PATH = join(BASE_PATH, 'plots')