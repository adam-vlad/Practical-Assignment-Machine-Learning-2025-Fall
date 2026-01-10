# Task 2.2 - Un model de regresie logistica pentru fiecare sos + sistem de recomandare
# Pentru un cos dat, calculam P(sos | cos) si recomandam top-K sosuri

import numpy as np
import pandas as pd


SAUCES = [
    'Crazy Sauce', 
    'Cheddar Sauce', 
    'Extra Cheddar Sauce',
    'Garlic Sauce', 
    'Tomato Sauce', 
    'Blueberry Sauce',
    'Spicy Sauce', 
    'Pink Sauce'
]


def load_data(filepath='ap_dataset.csv'):
    """Incarca datele si analizeaza frecventa sosurilor"""
    print("="*60)
    print("INCARCARE DATE")
    print("="*60)
    
    df = pd.read_csv(filepath)
    print(f"Date incarcate: {len(df)} linii, {df['id_bon'].nunique()} bonuri")
    
    print("\nFrecventa sosurilor in dataset:")
    sauce_freq = {}
    for sauce in SAUCES:
        count = df[df['retail_product_name'] == sauce]['id_bon'].nunique()
        sauce_freq[sauce] = count
        print(f"  {sauce}: {count} bonuri")
    
    return df, sauce_freq


def analyze_sauces(df):
    """Analizeaza distributia sosurilor"""
    print("\n" + "="*60)
    print("ANALIZA SOSURI")
    print("="*60)
    
    all_products = df['retail_product_name'].unique()
    print(f"Total produse unice: {len(all_products)}")
    
    # sosuri existente
    existing_sauces = [s for s in SAUCES if s in all_products]
    missing_sauces = [s for s in SAUCES if s not in all_products]
    
    print(f"\nSosuri gasite in date: {len(existing_sauces)}")
    for s in existing_sauces:
        print(f"  - {s}")
    
    if missing_sauces:
        print(f"\nSosuri LIPSA din date: {len(missing_sauces)}")
        for s in missing_sauces:
            print(f"  - {s}")
    
    return existing_sauces


if __name__ == "__main__":
    df, sauce_freq = load_data('ap_dataset.csv')
    
    existing_sauces = analyze_sauces(df)
    
    # statistici per sos
    print("\n" + "="*60)
    print("STATISTICI SOSURI")
    print("="*60)
    
    total_receipts = df['id_bon'].nunique()
    print(f"\nProcent bonuri cu fiecare sos:")
    for sauce, count in sorted(sauce_freq.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = 100 * count / total_receipts
            print(f"  {sauce}: {pct:.2f}%")
    
    print("\n" + "="*60)
    print("TODO Ziua 2: Creare features si split train/test")
    print("="*60)
