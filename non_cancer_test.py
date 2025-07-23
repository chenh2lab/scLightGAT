import sys, importlib
import scLightGAT.pipeline

# Reload to pick up any local changes
importlib.reload(scLightGAT.pipeline)

# Simulate command-line arguments for training
sys.argv = [
    "pipeline.py",
    "--mode", "train",
    "--train", "./data/train.h5ad",
    "--test", "./data/pbmc68k.h5ad",
    "--output", "/Group16T/common/lcy/dslab_lcy/GitRepo/scLightGAT/results",
    "--model_dir", "/Group16T/common/lcy/dslab_lcy/GitRepo/scLightGAT/models",
    "--train_dvae",
    "--use_gat",
    "--dvae_epochs", "5",
    "--gat_epochs", "300"
]

from scLightGAT.pipeline import main
main()