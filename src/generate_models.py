import os
import requests
import gemmi
import numpy as np
import argparse
import subprocess
import sys
import config


def fetch_mtz(pdb_id):
    mtz_path = os.path.join(config.MTZ_DIR, f"{pdb_id}.mtz")
    if os.path.exists(mtz_path):
        return mtz_path
    mtz_url = f"{config.MTZ_DOWNLOAD_URL}/{pdb_id}.mtz"
    response = requests.get(mtz_url, stream=True)
    response.raise_for_status()
    with open(mtz_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    return mtz_path

def filter_mtz(mtz_path, resolution_cutoff):
    mtz_name = os.path.basename(mtz_path)
    resolution_str = str(resolution_cutoff).replace('.', '_')
    filtered_mtz_filename = f"{mtz_name.split('.')[0]},{resolution_str}.mtz"
    filtered_mtz_path = os.path.join(config.FILTERED_MTZ_DIR, filtered_mtz_filename)
    if os.path.exists(filtered_mtz_path):
        return filtered_mtz_path
    mtz = gemmi.read_mtz_file(mtz_path)
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
        *[item[idx] for item in config.MODELCRAFT_ARGS.items() for idx in [0, 1]],
        *config.MODELCRAFT_FLAGS,
    ]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        if process.returncode != 0:
            for line in process.stderr:
                print(line.strip())
            print(f"Failed to execute modelcraft for {pdb_id}': {process.stderr.read().strip()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download mtz files, filter them and generate models')
    parser.add_argument(
        '--pdb_ids_file', 
        type=str, 
        default=config.DEFAULT_FILTER_CUTOFF, 
        help=f'Filter mtz data such that we remove reflections with resolution better than this value (in angstroms). Default is {config.DEFAULT_FILTER_CUTOFF} angstroms.')
    parser.add_argument(
        '--resolution_cutoff', 
        default=config.PDB_IDS_FILE, 
        help=f'Filter mtz data such that we remove reflections with resolution better than this value (in angstroms). Default is {config.DEFAULT_FILTER_CUTOFF} angstroms.')
    args = parser.parse_args()
    resolution_cutoff = args.resolution_cutoff

    for directory in [config.MTZ_DIR, config.FILTERED_MTZ_DIR, config.CONTENTS_DIR]:
        os.makedirs(directory, exist_ok=True)

    with open(config.PDB_IDS_FILE, 'r') as file:
        pdb_ids = file.read().lower().split(',')

    for pdb_id in pdb_ids:
        mtz = fetch_mtz(pdb_id)
        filtered_mtz = filter_mtz(mtz, resolution_cutoff)
        execute_modelcraft(pdb_id, filtered_mtz)
