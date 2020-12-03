import os
from openbabel import pybel
from .preparation import BindingSite, Ligand, LigandFinder, Mapper, PDBParser, PLInteraction
from plip.basic.supplemental import start_pymol
import numpy as np
from parmed.formats import PDBFile
import tempfile
from collections import namedtuple

import io

import numpy as np

from ..basic import config, logger
from ..basic.supplemental import centroid, extract_pdbid, readmol, start_pymol, tilde_expansion, vecangle, vector, euclidean3d, projection
from ..basic.supplemental import whichresnumber, whichrestype, whichchain

logger = logger.get_logger()

standard_aa_names = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]


def is_standard_residue(name):
    return (name in standard_aa_names)


def coords(atom):
    return np.array([atom.xx, atom.xy, atom.xz], dtype=float)


class TrajComplex:
    """Contains a collection of objects associated with a traj complex, i.e. one or several ligands and their binding
    sites as well as information about the pliprofiler between them. Provides functions to load and prepare input files
    such as PDB files.
    """

    def __init__(self):
        self.interaction_sets = {}  # Dictionary with site identifiers as keys and object as value
        self.protcomplex = None
        self.filetype = None
        self.atoms = {}  # Dictionary of Pybel atoms, accessible by their idx
        self.sourcefiles = {}
        self.information = {}
        self.corrected_pdb = ''
        self._output_path = tempfile.gettempdir()
        self.pymol_name = None
        self.modres = set()
        self.resis = []
        self.altconf = []  # Atom idx of atoms with alternate conformations
        self.covalent = []  # Covalent linkages between ligands and protein residues/other ligands
        self.excluded = []  # Excluded ligands
        self.Mapper = Mapper()
        self.ligands = []

    def __str__(self):
        formatted_lig_names = [":".join([x.hetid, x.chain, str(x.position)]) for x in self.ligands]
        return "Protein structure %s with ligands:\n" % (self.pymol_name) + "\n".join(
        [lig for lig in formatted_lig_names])

    def load_traj(self, protocomp):
        """Loads a pdb file with protein AND ligand(s), separates and prepares them.
        If specified 'as_string', the input is a PDB string instead of a path."""
        
        as_string=True
    
        # Parse PDB file to find errors and get additional data manually here
        # #@todo Refactor and rename here
        
        self.Mapper.proteinmap = {i: i for i, _ in enumerate(protocomp.atoms)}
        self.Mapper.reversed_proteinmap = {v: k for k, v in self.Mapper.proteinmap.items()}
        self.modres = []
        self.covalent = []
        self.altconf = []

        buf = io.StringIO()
        filep = PDBFile(buf)
        filep.write(protocomp, buf)

        self.protcomplex, self.filetype = readmol(buf.getvalue(), as_string=True)

        # Update the model in the Mapper class instance
        self.Mapper.original_structure = self.protcomplex.OBMol
        logger.info('PDB structure successfully read')

        # Determine (temporary) PyMOL Name from Filename
        self.pymol_name = 'Target_Protein'
        # Replace characters causing problems in PyMOL
        self.pymol_name = self.pymol_name.replace(' ', '').replace('(', '').replace(')', '').replace('-', '_')
        # But if possible, name it after PDBID in Header
        # if 'HEADER' in self.protcomplex.data:  # If the PDB file has a proper header
        #     potential_name = self.protcomplex.data['HEADER'][56:60].lower()
        # if extract_pdbid(potential_name) != 'UnknownProtein':
        #     self.pymol_name = potential_name
        # logger.debug(f'PyMOL name set as: {self.pymol_name}')

        # Extract and prepare ligands
        ligandfinder = LigandFinder(self.protcomplex, self.altconf, self.modres, self.covalent, self.Mapper)
        self.ligands = ligandfinder.ligands
        self.excluded = ligandfinder.excluded

        # decide whether to add polar hydrogens
        # if not config.NOHYDRO:
        #     if not as_string:
        #         basename = os.path.basename(pdbpath).split('.')[0]
        #     else:
        #         basename = "from_stdin"
        self.protcomplex.OBMol.AddPolarHydrogens()
        #     output_path = os.path.join(self._output_path, f'{basename}_protonated.pdb')
        #     self.protcomplex.write('pdb', output_path, overwrite=True)
        #     logger.info(f'protonated structure written to {output_path}')
        # else:
        #     logger.warning('no polar hydrogens will be assigned (make sure your structure contains hydrogens)')

        for atm in self.protcomplex:
            self.atoms[atm.idx] = atm

        if len(self.excluded) != 0:
            logger.info(f'excluded molecules as ligands: {self.excluded}')

        if config.DNARECEPTOR:
            self.resis = [obres for obres in pybel.ob.OBResidueIter(
                self.protcomplex.OBMol) if obres.GetName() in config.DNA + config.RNA]
        else:
            self.resis = [obres for obres in pybel.ob.OBResidueIter(
                self.protcomplex.OBMol) if obres.GetResidueProperty(0)]

        num_ligs = len(self.ligands)
        if num_ligs == 1:
            logger.info('analyzing one ligand')
        elif num_ligs > 1:
            logger.info(f'analyzing {num_ligs} ligands')
        else:
            logger.info(f'structure contains no ligands')

    def analyze(self):
        """Triggers analysis of all complexes in structure"""
        for ligand in self.ligands:
            self.characterize_complex(ligand)

    def characterize_complex(self, ligand):
        """Handles all basic functions for characterizing the interactions for one ligand"""

        single_sites = []
        for member in ligand.members:
            single_sites.append(':'.join([str(x) for x in member]))
        site = ' + '.join(single_sites)
        site = site if not len(site) > 20 else site[:20] + '...'
        longname = ligand.longname if not len(ligand.longname) > 20 else ligand.longname[:20] + '...'
        ligtype = 'unspecified type' if ligand.type == 'UNSPECIFIED' else ligand.type
        ligtext = f'{longname} [{ligtype}] -- {site}'
        logger.info(f'processing ligand {ligtext}')
        if ligtype == 'PEPTIDE':
            logger.info(f'chain {ligand.chain} will be processed in [PEPTIDE / INTER-CHAIN] mode')
        if ligtype == 'INTRA':
            logger.info(f'chain {ligand.chain} will be processed in [INTRA-CHAIN] mode')
        any_in_biolip = len(set([x[0] for x in ligand.members]).intersection(config.biolip_list)) != 0

        if ligtype not in ['POLYMER', 'DNA', 'ION', 'DNA+ION', 'RNA+ION', 'SMALLMOLECULE+ION'] and any_in_biolip:
            logger.info('may be biologically irrelevant')

        lig_obj = Ligand(self, ligand)
        cutoff = lig_obj.max_dist_to_center + config.BS_DIST
        bs_res = self.extract_bs(cutoff, lig_obj.centroid, self.resis)
        # Get a list of all atoms belonging to the binding site, search by idx
        bs_atoms = [self.atoms[idx] for idx in [i for i in self.atoms.keys()
                                            if self.atoms[i].OBAtom.GetResidue().GetIdx() in bs_res]
                if idx in self.Mapper.proteinmap and self.Mapper.mapid(idx, mtype='protein') not in self.altconf]
        if ligand.type == 'PEPTIDE':
        # If peptide, don't consider the peptide chain as part of the protein binding site
            bs_atoms = [a for a in bs_atoms if a.OBAtom.GetResidue().GetChain() != lig_obj.chain]
        if ligand.type == 'INTRA':
        # Interactions within the chain
            bs_atoms = [a for a in bs_atoms if a.OBAtom.GetResidue().GetChain() == lig_obj.chain]
        bs_atoms_refined = []

        # Create hash with BSRES -> (MINDIST_TO_LIG, AA_TYPE)
        # and refine binding site atom selection with exact threshold
        min_dist = {}
        for r in bs_atoms:
            bs_res_id = ''.join([str(whichresnumber(r)), whichchain(r)])
            for l in ligand.mol.atoms:
                distance = euclidean3d(r.coords, l.coords)
                if bs_res_id not in min_dist:
                    min_dist[bs_res_id] = (distance, whichrestype(r))
                elif min_dist[bs_res_id][0] > distance:
                    min_dist[bs_res_id] = (distance, whichrestype(r))
                if distance <= config.BS_DIST and r not in bs_atoms_refined:
                    bs_atoms_refined.append(r)
        num_bs_atoms = len(bs_atoms_refined)
        logger.info(f'binding site atoms in vicinity ({config.BS_DIST} A max. dist: {num_bs_atoms})')

        bs_obj = BindingSite(bs_atoms_refined, self.protcomplex, self, self.altconf, min_dist, self.Mapper)
        pli_obj = PLInteraction(lig_obj, bs_obj, self)
        self.interaction_sets[ligand.mol.title] = pli_obj

    def extract_bs(self, cutoff, ligcentroid, resis):
        """Return list of ids from residues belonging to the binding site"""
        return [obres.GetIdx() for obres in resis if self.res_belongs_to_bs(obres, cutoff, ligcentroid)]

    @staticmethod
    def res_belongs_to_bs(res, cutoff, ligcentroid):
        """Check for each residue if its centroid is within a certain distance to the ligand centroid.
        Additionally checks if a residue belongs to a chain restricted by the user (e.g. by defining a peptide chain)"""
        rescentroid = centroid([(atm.x(), atm.y(), atm.z()) for atm in pybel.ob.OBResidueAtomIter(res)])
        # Check geometry
        near_enough = True if euclidean3d(rescentroid, ligcentroid) < cutoff else False
        # Check chain membership
        restricted_chain = True if res.GetChain() in config.PEPTIDES else False
        return (near_enough and not restricted_chain)

    def get_atom(self, idx):
        return self.atoms[idx]

    @property
    def output_path(self):
        return self._output_path

    @output_path.setter
    def output_path(self, path):
        self._output_path = tilde_expansion(path)