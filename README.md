# ML__ChEMBLForge
ML_ChEMBLForge is a SMILES > SELFIES-based LSTM generator that designs novel small molecules from ChEMBL seeds, focused on drug-likeness by enforcing Lipinski’s Rule of Five.
This Python script trains a recurrent neural network (LSTM) on SELFIES-encoded SMILES from a curated ChEMBL subset. It generates new molecules autonomously, checks that they meet key drug-like properties — LogP, TPSA, hydrogen bond donors and acceptors — consistent with Lipinski’s Rule of Five, and flags novel structures relative to the training set.
The final output is a CSV containing valid SMILES, chemical descriptors, a novelty label, and an overall novelty score, enabling further filtering or downstream docking studies.
# Key Features
 Learns molecular grammar from any ChEMBL subset (ChEMBLcsv2data.py)
 Generates valid, unique SELFIES/SMILES strings
 Filters molecules by key drug-likeness constraints (LogP, PSA, H-bond donors/acceptors)
 Checks for novelty vs. the training set (canonical SMILES)
 Outputs a detailed CSV with descriptors, novelty flags, and summary statistics

# Key points for how to use
Prepare your input molecules as a CSV with SMILES strings (e.g., carboxylic-lipinksi5.csv). You can use this script to train small molecules with different functional groups or core groups of drug molecules. Run the script to train the LSTM on this input. The model generates new SELFIES strings and decodes them to SMILES.  Each molecule’s LogP, QED, PSA, HBA, and HBD are computed.  Only molecules satisfying Lipinski’s Rule of Five filter are kept, and non-drug-like molecules are filtered out. Output: generated_molecules.csv → includes descriptors + a novelty flag + summary stats. 
# Requirements
We need Python 3.8+, RDKit, PyTorch, pandas, and selfies.
Install dependencies (example): 

pip install torch pandas selfies 

For RDKit, use conda:
conda install -c conda-forge rdkit

# Usage: 
python small-molecule.py
# Tuning Parameter
**n_to_generate** → controls how many molecules to generate. Here, I used 50; one can use 1000 to generate a large number of molecules.

**max_attempts** → allows retries to ensure enough valid structures.

**epochs** → more epochs = better training for complex SMILES.
**num_layers** → Number of hidden layers. Increasing the number of layers takes more time. 

Credit: Built using open-source libraries including PyTorch, RDKit, and SELFIES.
 
