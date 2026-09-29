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

if __name__ == "__main__":
    args = parser.parse_args()

    # load settings from config file
    conf_file = args.config

    settings = Docker.Settings.from_file(conf_file)
    docker = Docker(settings=settings)
    settings.fitness_function = 'plp'

    settings.autoscale = 10.

    settings.early_termination = False

    # specify ligand file 
    #TODO remove any existing ligand files in config
    lig_file = args.ligand
    settings.add_ligand_file(lig_file, 1) #file_name, ndocks=1, start=0, finish=0)

    # write outputs to temp dir
    batch_tempd = tempfile.mkdtemp()
    settings.output_directory = batch_tempd

    # launch docker
    results = docker.dock()    

    # write poses and scores to sdf with score field
    ligands = results.ligands
    
    with open(args.out, "w") as f:
        for ligand in ligands:
            ligand.attributes.update({'docking_score': ligand.fitness(settings.fitness_function)})
            sdf_str = ligand.to_string(format='sdf')
            f.write(sdf_str)
