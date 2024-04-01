import os
import requests
import gemmi
import numpy as np
import argparse
import subprocess
import config
import concurrent.futures

def fetch_mtz(pdb_id):
    mtz_path = os.path.join(config.MTZ_DIR, f"{pdb_id}.mtz")
    if os.path.exists(mtz_path):
        print(f"MTZ for {pdb_id} already downloaded")
        return mtz_path
    mtz_url = f"{config.MTZ_DOWNLOAD_URL}/{pdb_id}.mtz"
    try:
        response = requests.get(mtz_url, stream=True)
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        print(f"HTTPError: Unable to download MTZ for {pdb_id}")
        return None
    with open(mtz_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded MTZ for {pdb_id}")
    return mtz_path

def add_free_r_column(mtz_path):
    new_mtz_path = mtz_path.replace('.mtz', '_free.mtz')
    cmd = [
        'freerflag',
        'HKLIN', mtz_path,
        'HKLOUT', new_mtz_path,
    ]
    process = subprocess.Popen(
        cmd, 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True)
    process.communicate("END\n")
    print(f"Added free R column to {os.path.basename(mtz_path)}")
    return new_mtz_path

def filter_mtz(mtz_path, resolution_cutoff):
    mtz_name = os.path.basename(mtz_path)
    resolution_str = str(resolution_cutoff).replace('.', '_')
    filtered_mtz_filename = f"{mtz_name.split('.')[0]},{resolution_str}.mtz"
    filtered_mtz_path = os.path.join(config.FILTERED_MTZ_DIR, filtered_mtz_filename)
    if os.path.exists(filtered_mtz_path):
        return filtered_mtz_path
    mtz = gemmi.read_mtz_file(mtz_path)

    if 'FREE' not in mtz.column_labels():
        mtz = gemmi.read_mtz_file(add_free_r_column(mtz_path))

    data = np.array(mtz, copy=False)
    mtz.set_data(data[mtz.make_d_array() >= resolution_cutoff])
    mtz.write_to_file(filtered_mtz_path)
    return filtered_mtz_path

def generate_contents_json(pdb_id):
    contents_json_path = os.path.join(config.CONTENTS_DIR, f"{pdb_id}_contents.json")
    if os.path.exists(contents_json_path):
        return contents_json_path

    cmd = ["modelcraft-contents", pdb_id, contents_json_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return contents_json_path

def execute_modelcraft(pdb_id, filtered_mtz_path):
    contents_json_path = generate_contents_json(pdb_id)

    model_name = os.path.basename(filtered_mtz_path).split('.')[0]
    model_path = os.path.join(config.MODELS_DIR, model_name)
    expected_files = [os.path.join(model_path, f"modelcraft.{ext}") for ext in ["cif", "mtz"]]
    if all(os.path.exists(file_path) for file_path in expected_files):
        print(f"Skipped modelcraft for {pdb_id}: files already present")
        return
    os.makedirs(model_path, exist_ok=True)

    cmd = [
        "modelcraft", "xray",
        "--contents", contents_json_path,
        "--data", filtered_mtz_path,
        "--directory", model_path,
        *[key_or_value for key_and_value in config.MODELCRAFT_ARGS.items() for key_or_value in key_and_value],
        *config.MODELCRAFT_FLAGS,
    ]
    print(f"Running modelcraft for {pdb_id}")
    result = subprocess.run(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True)
    print(f"Finished modelcraft for {pdb_id}")
    print(result.stdout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Download mtz files, filter them and generate models')
    parser.add_argument(
        '--pdb_ids_file',
        type=str,
        default=config.PDB_IDS_FILE,
        help=f'Path to a file containing a comma-separated list of PDB IDs to process. Default is at {config.PDB_IDS_FILE}.')
    parser.add_argument(
        '--resolution_cutoff',
        default=config.DEFAULT_FILTER_CUTOFF,
        type=float,
        help=f'Filter mtz data such that we remove reflections with resolution better than this value (in angstroms). Default is {config.DEFAULT_FILTER_CUTOFF} angstroms.')
    args = parser.parse_args()
    resolution_cutoff = args.resolution_cutoff
    pdb_ids_file = args.pdb_ids_file

    for directory in [config.MTZ_DIR, config.FILTERED_MTZ_DIR, config.CONTENTS_DIR]:
        os.makedirs(directory, exist_ok=True)

    with open(pdb_ids_file, 'r') as file:
        pdb_ids = [pdb_id.strip() for pdb_id in file.read().lower().split(',')]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for pdb_id in pdb_ids:
            if not (mtz := fetch_mtz(pdb_id)):
                continue
            try:
                filtered_mtz = filter_mtz(mtz, resolution_cutoff)
            except RuntimeError as e:
                print(f"Error when filtering MTZ data for {pdb_id}: {str(e)}")
                continue
            executor.submit(execute_modelcraft, pdb_id, filtered_mtz)
