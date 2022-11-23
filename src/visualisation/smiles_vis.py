from src.features import smiles_bio


def visualise_molecule(df, i):
    """

    :param df: (pandas.DataFrame) dataframe of molecules data
    :param i: (int) index of molecule to visualise
    :return: (rdkit.Chem.rdchem.Mol) - molecule visualisation
    """
    print(f"Name:\t{df.name[i]}\nSMILES:\t{df.smiles[i]}\nBBBP:\t{df.p_np[i]}")
    molecule = smiles_bio.molecule_from_smiles(df.iloc[100].smiles)
    print("Molecule:")
    return molecule
