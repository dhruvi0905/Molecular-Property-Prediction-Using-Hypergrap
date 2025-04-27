# Hypergraph Neural Networks for Molecular Property Prediction

## About
🧪 This project implements two hypergraph neural network (HGNN) models for predicting molecular properties: one based on **k-cliques** and another focused on **functional group-oriented** structures. These models capture complex, higher-order molecular interactions that traditional graph neural networks (GNNs) often miss. The models were evaluated on the **Tox21 NR-AR** dataset, with the k-clique model achieving an impressive **0.894 ROC AUC**. 🧬

## Features
- 🔗 **K-Clique Hypergraph Model:** Captures molecule structures via interconnected cliques.
- 🧩 **Functional Group Hypergraph Model:** Models interactions around key functional groups.
- ⚗️ **Higher-Order Interaction Modeling:** Moves beyond simple atom-to-atom connections.
- 📈 **Superior Performance:** Outperforms conventional GNNs in toxicity prediction tasks.
- 🧪 **Benchmarking:** Tested on the Tox21 NR-AR molecular dataset.

## Dataset
- **Tox21 NR-AR**
- Includes compounds labeled for toxicity related to androgen receptor activity.

## Requirements
```bash
python>=3.8
pytorch>=1.8
scikit-learn
rdkit
numpy
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run
1. **Train the k-clique hypergraph model:**
    ```bash
    python train_kclique.py
    ```

2. **Train the functional group hypergraph model:**
    ```bash
    python train_functional_group.py
    ```

3. **Evaluate the models:**
    ```bash
    python evaluate.py
    ```

## Project Structure
```
hypergraph-molecular-prediction/
├── models/
│   ├── kclique_model.py
│   ├── functional_group_model.py
├── data/
│   ├── tox21_nr_ar.csv
├── train_kclique.py
├── train_functional_group.py
├── evaluate.py
├── README.md
├── requirements.txt
```

## Results
- **K-Clique Model:** 0.894 ROC AUC
- **Functional Group Model:** Competitive performance with better interpretability

## Future Work
- Extend to multi-task prediction.
- Experiment with larger molecular datasets (e.g., ChEMBL).
- Incorporate 3D molecular structures.

## License
Open-source project under the MIT License.

---

*Advancing molecular property prediction with higher-order deep learning techniques.* 🚀

