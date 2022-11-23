from tensorflow import keras
import pandas as pd


def load_csv_bbbp():
    """
    Blood-Brain Barrier Penetration
    Information about the dataset can be found in
    A Bayesian Approach to in Silico Blood-Brain Barrier Penetration Modeling https://pubs.acs.org/doi/10.1021/ci300124c
    and
    MoleculeNet: A Benchmark for Molecular Machine Learning https://arxiv.org/abs/1703.00564.
    The dataset will be downloaded from https://moleculenet.org/datasets-1.
    :return: pandas.DataFrame
    """
    csv_path = keras.utils.get_file(
        "BBBP.csv", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
    )
    df = pd.read_csv(csv_path, usecols=[1, 2, 3])
    return df
