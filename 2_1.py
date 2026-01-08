# Task 2.1 - Regresie Logistica: Crazy Sauce dat fiind Crazy Schnitzel
# Clasificare binara - prezice daca clientul ia sosul impreuna cu schnitzelul
# Ziua 1 - Structura de baza si incarcarea datelor

import numpy as np
import pandas as pd


def load_data(filepath='ap_dataset.csv'):
    """Incarca si exploreaza datele"""
    print("="*60)
    print("INCARCARE DATE")
    print("="*60)
    
    df = pd.read_csv(filepath)
    print(f"Date incarcate: {len(df)} linii")
    print(f"Bonuri unice: {df['id_bon'].nunique()}")
    print(f"Produse unice: {df['retail_product_name'].nunique()}")
    
    return df


def explore_target_products(df):
    """Exploreaza produsele target"""
    TARGET_SAUCE = 'Crazy Sauce'
    TARGET_PRODUCT = 'Crazy Schnitzel'
    
    receipts_with_cs = df[df['retail_product_name'] == TARGET_PRODUCT]['id_bon'].unique()
    print(f"\nBonuri cu {TARGET_PRODUCT}: {len(receipts_with_cs)}")
    
    # cate din aceste bonuri au si Crazy Sauce?
    df_filtered = df[df['id_bon'].isin(receipts_with_cs)]
    receipts_with_both = df_filtered[df_filtered['retail_product_name'] == TARGET_SAUCE]['id_bon'].nunique()
    
    print(f"Bonuri cu {TARGET_PRODUCT} si {TARGET_SAUCE}: {receipts_with_both}")
    print(f"Rata de cumparare impreuna: {100*receipts_with_both/len(receipts_with_cs):.1f}%")
    
    return receipts_with_cs


if __name__ == "__main__":
    df = load_data('ap_dataset.csv')
    
    print("\nColoane disponibile:")
    print(df.columns.tolist())
    
    print("\nPrimele 5 randuri:")
    print(df.head())
    
    print("\nStatistici produse:")
    print(df['retail_product_name'].value_counts().head(10))
    
    receipts_with_cs = explore_target_products(df)
    
    print("\n" + "="*60)
    print("TODO Ziua 2: Preprocesare date si creare features")
    print("="*60)
