## based on https://downloads.ccdc.cam.ac.uk/documentation/API/descriptive_docs/docking.html#essential-steps
## & https://downloads.ccdc.cam.ac.uk/documentation/API/descriptive_docs/docking.html#interactive-docking

import argparse
from ccdc.docking import Docker

from ccdc.io import MoleculeReader, EntryReader
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--ligand', type=str,
                    help="ligand input file")
parser.add_argument('--out', type=str,
                    help="flag for output file")
parser.add_argument('--config', type=str,
                    help="gold config used as base for docking")
parser.add_argument('--ndocks', type=int, default=10,
                    help="number of docking attempts for each ligand")

if __name__ == "__main__":
    args = parser.parse_args()

    # load settings from config file
    conf_file = args.config

    settings = Docker.Settings.from_file(conf_file)
    docker = Docker(settings=settings)

    # specify ligand file 
    settings.clear_ligand_files()
    lig_file = args.ligand
    settings.add_ligand_file(lig_file, 1) #file_name, ndocks=1, start=0, finish=0)

    # write outputs to temp dir
    batch_tempd = tempfile.mkdtemp()
    settings.output_directory = batch_tempd
    settings.write_options = ['NO_LINK_FILES', 'NO_RNK_FILES', 'NO_BESTRANKING_LST_FILE']

    # launch docker
    #TODO to stop writing gold api file to wd, use settings file, need to copy all files to tempd 
    # settings.write(f'{batch_tempd}/gold.conf')
    results = docker.dock() #file_name=f'{batch_tempd}/gold.conf')    

    # write poses and scores to sdf with score field
    try:
        ligands = results.ligands
        with open(args.out, "w") as f:
            for ligand in ligands:
                ligand.attributes.update({'docking_score': ligand.fitness(settings.fitness_function)})
                sdf_str = ligand.to_string(format='sdf')
                f.write(sdf_str)
    except RuntimeError:
        print("No gold solution files")
