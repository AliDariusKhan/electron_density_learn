import os

DEFAULT_FILTER_CUTOFF = 3.5
DATA_DIR = 'data'
PDB_IDS_FILE = os.path.join(DATA_DIR, 'pdb_ids.txt')
MTZ_DIR = os.path.join(DATA_DIR, 'mtz_files')
FILTERED_MTZ_DIR = os.path.join(DATA_DIR, 'filtered_mtz_files')
CONTENTS_DIR = os.path.join(DATA_DIR, 'contents_files')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
PDB_DIR = os.path.join(DATA_DIR, 'pdb_files')
TRAINING_DATA_PATH = os.path.join(MODELS_DIR, 'training_data.h5')
MTZ_DOWNLOAD_URL = "https://edmaps.rcsb.org/coefficients"
PDB_DOWNLOAD_URL = "https://files.rcsb.org/download"
JARVIS_PDB_PATH = "/vault/pdb"
MODELCRAFT_ARGS = {
    "--cycles": "1",
    "--phases": "PHWT,FOM"
}
MODELCRAFT_FLAGS = [
    "--overwrite-directory",
    "--basic",
    "--disable-sheetbend",
    "--disable-pruning",
    "--disable-parrot",
    "--disable-dummy-atoms",
    "--disable-waters",
    "--disable-side-chain-fixing",
]
DENSITY_MAP_NAMES = "FWT,PHWT DELFWT,PHDELWT FC_ALL_LS,PHIC_ALL_LS"
DEFAULT_GRID_SIZE = 32
DEFAULT_SPACING = 0.75
DEFAULT_DENSITY_CUTOFF = 15
CNN_DIR = os.path.join(DATA_DIR, 'cnn')
DENSITY_DATA_NAME = "map_values"
REFINEMENT_VEC_NAME = "refinement_vecs"