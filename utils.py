import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import warnings

# Define SMARTS patterns for common functional groups
FUNCTIONAL_GROUPS = {
    'hydroxyl': '[OX2H]',
    'carboxyl': '[CX3](=O)[OX2H1]',
    'carbonyl': '[CX3]=[OX1]',
    'amine_primary': '[NX3;H2;!$(NC=O)]',
    'amine_secondary': '[NX3;H1;!$(NC=O)]',
    'amine_tertiary': '[NX3;H0;!$(NC=O)]',
    'amide': '[NX3][CX3](=[OX1])',
    'nitro': '[NX3](=[OX1])=[OX1]',
    'nitrile': '[NX1]#[CX2]',
    'thiol': '[SX2H]',
    'thioether': '[#16X2][#6]',
    'sulfoxide': '[#16X3](=[OX1])',
    'sulfonyl': '[#16X4](=[OX1])(=[OX1])',
    'ester': '[CX3](=[OX1])[OX2][CX4]',
    'ether': '[OX2]([CX4])[CX4]',
    'halogen': '[F,Cl,Br,I]',
    'alkene': '[CX3]=[CX3]',
    'alkyne': '[CX2]#[CX2]',
    'aromatic': 'c1ccccc1',
    'heterocyclic': '[a;!c]',
    'phosphate': '[PX4](=[OX1])([OX2][#6])',
}

def safe_parse_smiles(smiles):
    """
    Safely parse a SMILES string to an RDKit molecule with error handling.
    
    Args:
        smiles (str): SMILES string
        
    Returns:
        rdkit.Chem.rdchem.Mol or None: RDKit molecule object or None if parsing fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            warnings.warn(f"Failed to parse SMILES: {smiles}")
            return None
        
        # Try to compute 2D coordinates for visualization
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            mol = Chem.RemoveHs(mol)
        except Exception as e:
            # Embedding failed, but molecule is still valid
            warnings.warn(f"Failed to compute 3D coordinates for {smiles}: {e}")
        
        return mol
    except Exception as e:
        warnings.warn(f"Error parsing SMILES {smiles}: {e}")
        return None

def safe_compute_descriptors(mol):
    """
    Safely compute molecular descriptors with error handling.
    
    Args:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule object
        
    Returns:
        dict: Dictionary of molecular descriptors
    """
    if mol is None:
        return default_descriptors()
    
    descriptors = {}
    try:
        descriptors['MolWt'] = Descriptors.MolWt(mol)
    except:
        descriptors['MolWt'] = 0.0
    
    try:
        descriptors['LogP'] = Descriptors.MolLogP(mol)
    except:
        descriptors['LogP'] = 0.0
    
    try:
        descriptors['NumHDonors'] = Descriptors.NumHDonors(mol)
    except:
        descriptors['NumHDonors'] = 0
    
    try:
        descriptors['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
    except:
        descriptors['NumHAcceptors'] = 0
    
    try:
        descriptors['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
    except:
        descriptors['NumRotatableBonds'] = 0
    
    try:
        descriptors['NumHeavyAtoms'] = Descriptors.HeavyAtomCount(mol)
    except:
        descriptors['NumHeavyAtoms'] = 0
    
    try:
        descriptors['NumRings'] = Descriptors.RingCount(mol)
    except:
        descriptors['NumRings'] = 0
    
    try:
        descriptors['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
    except:
        descriptors['NumAromaticRings'] = 0
    
    try:
        descriptors['TPSA'] = Descriptors.TPSA(mol)
    except:
        descriptors['TPSA'] = 0.0
    
    try:
        descriptors['Fsp3'] = Descriptors.FractionCSP3(mol)
    except:
        descriptors['Fsp3'] = 0.0
    
    try:
        descriptors['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
    except:
        descriptors['NumAliphaticRings'] = 0
    
    try:
        descriptors['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
    except:
        descriptors['NumSaturatedRings'] = 0
    
    try:
        descriptors['NumAromaticHeterocycles'] = Descriptors.NumAromaticHeterocycles(mol)
    except:
        descriptors['NumAromaticHeterocycles'] = 0
    
    try:
        descriptors['NumSaturatedHeterocycles'] = Descriptors.NumSaturatedHeterocycles(mol)
    except:
        descriptors['NumSaturatedHeterocycles'] = 0
    
    try:
        descriptors['NumAliphaticHeterocycles'] = Descriptors.NumAliphaticHeterocycles(mol)
    except:
        descriptors['NumAliphaticHeterocycles'] = 0
    
    try:
        descriptors['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
    except:
        descriptors['NumHeteroatoms'] = 0
    
    try:
        descriptors['LabuteASA'] = Descriptors.LabuteASA(mol) if mol.GetNumConformers() > 0 else 0
    except:
        descriptors['LabuteASA'] = 0.0
    
    return descriptors

def default_descriptors():
    """
    Return a dictionary with default values for molecular descriptors.
    
    Returns:
        dict: Dictionary of default molecular descriptors
    """
    return {
        'MolWt': 0.0,
        'LogP': 0.0,
        'NumHDonors': 0,
        'NumHAcceptors': 0,
        'NumRotatableBonds': 0,
        'NumHeavyAtoms': 0,
        'NumRings': 0,
        'NumAromaticRings': 0,
        'TPSA': 0.0,
        'Fsp3': 0.0,
        'NumAliphaticRings': 0,
        'NumSaturatedRings': 0,
        'NumAromaticHeterocycles': 0,
        'NumSaturatedHeterocycles': 0,
        'NumAliphaticHeterocycles': 0,
        'NumHeteroatoms': 0,
        'LabuteASA': 0.0,
    }

def create_empty_heterograph():
    """
    Create an empty heterogeneous graph with the necessary structure.
    
    Returns:
        dgl.DGLHeteroGraph: Empty heterogeneous graph
    """
    import dgl
    
    g = dgl.heterograph({
        ('atom', 'bond', 'atom'): [],
        ('atom', 'part_of', 'group'): [],
        ('group', 'has', 'atom'): [],
        ('atom', 'in_clique', 'clique'): [],
        ('clique', 'contains', 'atom'): []
    })
    
    # Add minimum required features
    g.nodes['atom'].data['feat'] = torch.zeros((0, 22), dtype=torch.float)  # 22 is the atom feature dimension
    g.nodes['group'].data['feat'] = torch.zeros((0, len(FUNCTIONAL_GROUPS)), dtype=torch.float)
    g.nodes['clique'].data['feat'] = torch.zeros((0, 22), dtype=torch.float)  # Same as atom features
    
    # Add empty descriptors
    g.global_descriptors = default_descriptors()
    
    return g

def get_feature_dimensions(sample_graph):
    """
    Extract feature dimensions from a sample graph.
    
    Args:
        sample_graph (dgl.DGLHeteroGraph): Sample graph to extract dimensions from
        
    Returns:
        dict: Dictionary of feature dimensions by node type
    """
    feature_dims = {}
    
    # Extract atom feature dimension
    if 'atom' in sample_graph.ntypes and sample_graph.number_of_nodes('atom') > 0:
        if 'feat' in sample_graph.nodes['atom'].data:
            feature_dims['atom'] = sample_graph.nodes['atom'].data['feat'].shape[1]
        else:
            feature_dims['atom'] = 22  # Default atom feature dimension
    else:
        feature_dims['atom'] = 22  # Default atom feature dimension
    
    # Extract group feature dimension
    if 'group' in sample_graph.ntypes and sample_graph.number_of_nodes('group') > 0:
        if 'feat' in sample_graph.nodes['group'].data:
            feature_dims['group'] = sample_graph.nodes['group'].data['feat'].shape[1]
        else:
            feature_dims['group'] = len(FUNCTIONAL_GROUPS)  # Default group feature dimension
    else:
        feature_dims['group'] = len(FUNCTIONAL_GROUPS)  # Default group feature dimension
    
    # Extract clique feature dimension
    if 'clique' in sample_graph.ntypes and sample_graph.number_of_nodes('clique') > 0:
        if 'feat' in sample_graph.nodes['clique'].data:
            feature_dims['clique'] = sample_graph.nodes['clique'].data['feat'].shape[1]
        else:
            feature_dims['clique'] = feature_dims['atom']  # Default to atom feature dimension
    else:
        feature_dims['clique'] = feature_dims['atom']  # Default to atom feature dimension
    
    return feature_dims
