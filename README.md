# Tema Invatare Automata - Sisteme de Recomandare si Ranking

Proiect pentru Machine Learning - implementare de algoritmi de clasificare si ranking pentru un dataset de restaurant.

## Structura Proiectului

```
├── 2_1.py                      # Task 2.1 - Regresie Logistica binara
├── 2_2.py                      # Task 2.2 - Sistem de recomandare sosuri
├── 3.py                        # Task 3 - Ranking pentru upsell cu ID3
├── ap_dataset.csv              # Dataset-ul cu bonuri de restaurant
├── requirements.txt            # Dependinte Python
├── raport.tex                  # Raportul in LaTeX
└── README.md                   # Acest fisier
```

## Cerinte

- Python 3.8+
- numpy
- pandas
- matplotlib

## Instalare

### Varianta 1: Cu pip
```bash
pip install -r requirements.txt
```

### Varianta 2: Cu conda
```bash
conda create -n ml_tema python=3.9
conda activate ml_tema
pip install -r requirements.txt
```

## Rulare

Fiecare task se ruleaza independent:

```bash
# Task 2.1 - Regresie Logistica
python 2_1.py

# Task 2.2 - Sistem de recomandare
python 2_2.py

# Task 3 - Ranking
python 3.py
```

## Output

Fiecare script genereaza grafice PNG in directorul curent:

- `2_1_*.png` - Grafice pentru Task 2.1 (ROC, matrice confuzie, loss, coeficienti)
- `2_2_*.png` - Grafice pentru Task 2.2 (metrici recomandare)
- `3_*.png` - Grafice pentru Task 3 (comparatie ranking, distributie scoruri)

## Dataset

`ap_dataset.csv` contine date de vanzari de la un restaurant cu urmatoarele coloane principale:
- `id_bon` - identificator unic pentru fiecare bon
- `data_bon` - data si ora tranzactiei
- `retail_product_name` - numele produsului
- `SalePriceWithVAT` - pretul cu TVA

## Descriere Task-uri

### Task 2.1
Regresie logistica implementata manual (gradient descent + metoda Newton) pentru a prezice daca un client care cumpara "Crazy Schnitzel" va cumpara si "Crazy Sauce".

### Task 2.2
Sistem de recomandare multi-sos care antreneaza cate un model de regresie logistica pentru fiecare sos si recomanda sosurile cu probabilitatea cea mai mare.

### Task 3
Sistem de ranking pentru upsell folosind arbori de decizie ID3. Scorul de ranking combina probabilitatea estimata cu pretul produsului.
