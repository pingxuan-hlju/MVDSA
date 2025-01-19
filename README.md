# MVDSA

## Introduction  
The project  is an implementation of a multi-knowledge graph and multi-view entity feature learning method for predicting drug-related side effects (MVDSA). 

---

## Catalogs  
- **/data**: Contains the dataset used in our method.
- **dataloader.py**: Loading and processing drug and side effect data.
- **main.py**:  Contains the code implementation of MVDSA algorithm and training process.
- **tools.py**: Contains the early stopping function.

---

## Environment  
The MVDSA code has been implemented and tested in the following development environment: 

- Python == 3.8.10 
- Matplotlib == 3.5.2
- PyTorch == 1.12.1  
- NumPy == 1.22.4
- Scikit-learn == 1.2.2

---

## Dataset  
- **drugname.txt**: Contains the names of 708 drugs.  
- **se.txt**: Contains the names of 4192 side effects.
- **se_seSmilirity.zip**: Includes the side effcet similarities.
- **Similarity_Matrix_Drugs.txt**: Includes the drug  structural similarities.
- **drug_drug_sim_dis.txt**: Includes the drug functional similarities.
- **mat_drug_se.txt**: Includes the drug-side effect association.
- **Supplemenrary_Table_ST1.xls**: Lists the top 30 candidate side effects for each drug.

---

## How to Run the Code  
1. **Train and test the model**.  
    ```bash
    python main.py
    ```  
