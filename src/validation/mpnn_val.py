from src.features.smiles_bio import molecule_from_smiles
import pandas as pd
import tensorflow as tf
from rdkit.Chem.Draw import MolsToGridImage
from src.models.mpnn_trainer import MPNNTrainer


def look_up(model_trainer: MPNNTrainer,
            df: pd.DataFrame,
            dataset: tf.data.Dataset,
            indices: list) -> list[list]:
    molecules = [molecule_from_smiles(df.smiles.values[index]) for index in indices]
    y_true = [df.p_np.values[index] for index in indices]
    y_pred = model_trainer.predict(dataset)

    legends = [f"y_true/y_pred = {y_true[i]}/{y_pred[i]:.2f}" for i in range(len(y_true))]
    grid = MolsToGridImage(molecules, molsPerRow=4, legends=legends, returnPNG=False)
    return [molecules, legends, grid]
