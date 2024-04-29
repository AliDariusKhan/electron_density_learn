import argparse
import os
import gemmi
import numpy as np
import requests
import config
import h5py
import cProfile
import pstats
import io
import subprocess

def reference_structure_path(pdb_id):
    path = os.path.join(config.PDB_DIR, f"{pdb_id}.pdb")
    jarvis_path = os.path.join(config.JARVIS_PDB_PATH, f'pdb{pdb_id}.ent')
    if os.path.exists(jarvis_path):
        print(f"PDB for {pdb_id} found on Jarvis")
        return jarvis_path
    elif os.path.exists(path):
        print(f"PDB for {pdb_id} already downloaded")
        return path
    else:
        base_url = config.PDB_DOWNLOAD_URL
        pdb_url = f"{base_url}/{pdb_id}.pdb"
        try:
            response = requests.get(pdb_url, stream=True)
            response.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"PDB ID {pdb_id} not found on RCSB. Skipping...")
                return None
            else:
                raise e
    return path

def gemmi_position_to_np_array(gemmi_pos):
    return np.array([getattr(gemmi_pos, coord) for coord in ['x', 'y', 'z']])

def get_rotation_matrix(basis_vecs):
    return np.column_stack(basis_vecs)

def get_grid_basis(residue):
    Ca = gemmi_position_to_np_array(residue.find_atom("CA", "\0").pos)
    N = gemmi_position_to_np_array(residue.find_atom("N", "\0").pos)
    C = gemmi_position_to_np_array(residue.find_atom("C", "\0").pos)
    x = C - Ca
    x_norm = x / np.linalg.norm(x)
    Ca_to_N = N - Ca
    y = np.cross(x, Ca_to_N)
    y_norm = y / np.linalg.norm(y)
    z_norm = np.cross(x_norm, y_norm)
    return Ca, (x_norm, y_norm, z_norm)

def density_map_grid(density_map, origin, basis_vecs, rotation_matrix, grid_size, spacing):
    grid_corner = origin - ((grid_size - 1) * spacing / 2) * sum(basis_vecs)
    density_map_grid_values = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

    transform = gemmi.Transform()
    transform.mat.fromlist(spacing * rotation_matrix)
    transform.vec.fromlist(grid_corner)

    density_map.interpolate_values(density_map_grid_values, transform)
    return density_map_grid_values

def standard_position(position, unit_cell):
    return np.array([
        position.x % unit_cell.a,
        position.y % unit_cell.b,
        position.z % unit_cell.c])

def align_reference_to_model(ref_structure_path, model_structure_path, pdb_id, model_dir):
    aligned_file_path = os.path.join(model_dir, f"{pdb_id}_aligned.pdb")
    csymmatch_process = subprocess.run(
        f'csymmatch -pdbin-ref {model_structure_path} -pdbin {ref_structure_path} -pdbout {aligned_file_path}', 
        shell=True)
    if csymmatch_process.returncode != 0:
        raise RuntimeError(f"csymmatch failed with error: {csymmatch_process.stderr}")
    return aligned_file_path

def process_model(
        model_dir, 
        density_map_names, 
        grid_size, 
        spacing, 
        cutoff, 
        normalize_density,
        align):
    model_name = os.path.basename(model_dir)
    pdb_id = model_name[:4]
    if (ref_structure_path := reference_structure_path(pdb_id)) is None:
        print(f"Reference structure not found for PDB ID {pdb_id}. Skipping...")
        return 0
    model_structure_path = os.path.join(model_dir, 'modelcraft.cif')
    if align:
        ref_structure_path = align_reference_to_model(
            ref_structure_path, 
            model_structure_path,
            pdb_id,
            model_dir)
    ref_structure = gemmi.read_structure(ref_structure_path)
    
    ref_neighbor_search = gemmi.NeighborSearch(
        ref_structure[0], 
        ref_structure.cell, 
        max_radius=cutoff)
    for chain_idx, chain in enumerate(ref_structure[0]):
        for res_idx, res in enumerate(chain):
            for atom_idx, atom in enumerate(res):
                if atom.name == 'CA':
                    ref_neighbor_search.add_atom(atom, chain_idx, res_idx, atom_idx)

    mtz = gemmi.read_mtz_file(os.path.join(model_dir, 'modelcraft.mtz'))
    density_maps = {map_name : mtz.transform_f_phi_to_map(*map_name.split(',')) 
                    for map_name in density_map_names}
    if normalize_density:
        for density_map in density_maps.values():
            density_map.normalize()

    model_map_values = []
    model_refinement_vecs = []
    model_structure = gemmi.read_structure(model_structure_path)
    residue_count = 0
    for chain in model_structure[0]:
        for residue in chain:
            if not gemmi.find_tabulated_residue(residue.name).is_amino_acid():
                continue
            origin, basis_vecs = get_grid_basis(residue)
            rotation_matrix = get_rotation_matrix(basis_vecs)
            inverse_rotation_matrix = np.transpose(rotation_matrix)

            residue_map_values = []
            for density_map in density_maps.values():
                map_values = density_map_grid(
                    density_map,
                    origin,
                    basis_vecs, 
                    rotation_matrix,
                    grid_size,
                    spacing)
                residue_map_values.append(map_values)
            
            model_CA = residue.find_atom("CA", "\0")
            ref_CA = ref_neighbor_search.find_nearest_atom(model_CA.pos)
            if not ref_CA:
                continue
            model_CA_pos = standard_position(model_CA.pos, model_structure.cell)
            try:
                ref_CA_pos = standard_position(
                    ref_CA.pos(), 
                    ref_structure.cell)
            except TypeError:
                ref_CA_pos = standard_position(
                    ref_CA.pos, 
                    ref_structure.cell)
            model_to_ref = ref_CA_pos - model_CA_pos
            model_to_ref_residue_basis = inverse_rotation_matrix.dot(model_to_ref)

            model_map_values.append(np.stack(residue_map_values))
            model_refinement_vecs.append(model_to_ref_residue_basis)
            residue_count += 1

    if not model_refinement_vecs:
        return 0

    with h5py.File(os.path.join(config.MODELS_DIR, 'training_data.h5'), 'a') as f:
        dataset = f.create_group(model_name)
        dataset.create_dataset(config.DENSITY_DATA_NAME, data=np.stack(model_map_values))
        dataset.create_dataset(config.REFINEMENT_VEC_NAME, data=np.stack(model_refinement_vecs))
    return residue_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate training data based on models generated by the generate_models.py script.')
    parser.add_argument(
        '--density_map_names', 
        default=config.DENSITY_MAP_NAMES, 
        help=f'Space delimited list of density maps to be sampled. Default is {config.DENSITY_MAP_NAMES}.')
    parser.add_argument(
        '--grid_size', 
        default=config.DEFAULT_GRID_SIZE, 
        type=int, 
        help=f'If this is N, the script will sample electron density at the points in an NxNxN grid centred on the residue. Default is {config.DEFAULT_GRID_SIZE}.')
    parser.add_argument(
        '--spacing', 
        default=config.DEFAULT_SPACING, 
        type=float, 
        help=f'The spacing between points in the grid in angstroms. Default is {config.DEFAULT_SPACING} angstroms.')
    parser.add_argument(
        '--cutoff', 
        default=config.DEFAULT_DENSITY_CUTOFF, 
        type=float, 
        help=f'The script will not consider residues that are more than this distance from the nearest residue in the reference structure. Default is {config.DEFAULT_DENSITY_CUTOFF} angstroms.')
    parser.add_argument(
        '--disable_mtz_normalise', 
        action='store_true', 
        help='Disable MTZ normalisation if this flag is set.')
    parser.add_argument(
        '--align_models', 
        action='store_true', 
        default=False,
        help='Disable MTZ normalisation if this flag is set.')
    
    args = parser.parse_args()
    grid_size = args.grid_size
    spacing = args.spacing
    cutoff = args.cutoff
    density_map_names = args.density_map_names.split()
    normalize_density = not args.disable_mtz_normalise
    align = args.align_models

    os.makedirs(config.PDB_DIR, exist_ok=True)

    pr = cProfile.Profile()
    pr.enable()

    with h5py.File(os.path.join(config.MODELS_DIR, 'training_data.h5'), 'w') as f:
        f.attrs['density_map_names'] = density_map_names
        f.attrs['grid_size'] = grid_size
        f.attrs['spacing'] = spacing
        f.attrs['cutoff'] = cutoff
        f.attrs['normalize_density'] = normalize_density

    residue_count = 0
    for root, _, files in os.walk(config.MODELS_DIR):
        if not all (data in files for data in ["modelcraft.cif", "modelcraft.mtz"]):
            continue
        residue_count += process_model(
            root, 
            density_map_names, 
            grid_size, 
            spacing, 
            cutoff, 
            normalize_density,
            align)
        print("Residue count:", residue_count)

    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sortby)
    ps.print_stats('extract_training_data')
    print(s.getvalue())
