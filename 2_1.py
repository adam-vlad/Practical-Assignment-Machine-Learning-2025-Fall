# Task 2.1 - Regresie Logistica: Crazy Sauce dat fiind Crazy Schnitzel
# Clasificare binara - prezice daca clientul ia sosul impreuna cu schnitzelul

import numpy as np
import pandas as pd


def load_and_preprocess(filepath='ap_dataset.csv'):
    """Incarca si preproceseaza datele"""
    print("="*60)
    print("INCARCARE SI PREPROCESARE DATE")
    print("="*60)
    
    df = pd.read_csv(filepath)
    print(f"Date incarcate: {len(df)} linii, {df['id_bon'].nunique()} bonuri")
    
    TARGET_SAUCE = 'Crazy Sauce'
    TARGET_PRODUCT = 'Crazy Schnitzel'
    
    # filtram doar bonurile care contin Crazy Schnitzel
    receipts_with_cs = df[df['retail_product_name'] == TARGET_PRODUCT]['id_bon'].unique()
    print(f"Bonuri cu {TARGET_PRODUCT}: {len(receipts_with_cs)}")
    
    df_filtered = df[df['id_bon'].isin(receipts_with_cs)].copy()
    
    all_products = df['retail_product_name'].unique()
    products_for_features = [p for p in all_products if p != TARGET_SAUCE]
    
    # categorii pentru features
    sides = [p for p in all_products if any(kw in p.lower() for kw in ['potato', 'fries', 'cartofi'])]
    drinks = [p for p in all_products if any(kw in p for kw in ['Pepsi', 'Cola', 'Dew', 'Aqua', 'Fanta'])]
    
    print(f"Garnituri gasite: {sides}")
    print(f"Bauturi gasite (primele 5): {drinks[:5]}")
    
    # construim features pentru fiecare bon
    data = []
    for id_bon in receipts_with_cs:
        receipt = df_filtered[df_filtered['id_bon'] == id_bon]
        row = {}
        
        # target
        row['y'] = 1 if TARGET_SAUCE in receipt['retail_product_name'].values else 0
        
        # features temporale
        dt = pd.to_datetime(receipt['data_bon'].iloc[0])
        row['day_of_week'] = dt.dayofweek + 1
        row['hour'] = dt.hour
        row['is_weekend'] = 1 if dt.dayofweek >= 5 else 0
        
        # features agregate
        row['cart_size'] = len(receipt)
        row['distinct_products'] = receipt['retail_product_name'].nunique()
        row['total_value'] = receipt['SalePriceWithVAT'].sum()
        
        # vectorul de produse
        product_counts = receipt['retail_product_name'].value_counts()
        products_in_cart = set(receipt['retail_product_name'].values)
        
        for p in products_for_features:
            row[f'count_{p}'] = product_counts.get(p, 0)
            row[f'has_{p}'] = 1 if p in products_in_cart else 0
        
        # features de interactiune
        row['has_any_side'] = 1 if any(s in products_in_cart for s in sides) else 0
        row['has_any_drink'] = 1 if any(d in products_in_cart for d in drinks) else 0
        
        data.append(row)
    
    df_final = pd.DataFrame(data)
    
    pos = df_final['y'].sum()
    neg = len(df_final) - pos
    print(f"\nDistributie target:")
    print(f"  y=1 (cu {TARGET_SAUCE}): {pos} ({100*pos/len(df_final):.1f}%)")
    print(f"  y=0 (fara {TARGET_SAUCE}): {neg} ({100*neg/len(df_final):.1f}%)")
    
    feature_cols = [c for c in df_final.columns if c != 'y']
    print(f"\nTotal features: {len(feature_cols)}")
    
    return df_final


def train_test_split(df, test_size=0.2, seed=42):
    """Imparte datele in train/test"""
    np.random.seed(seed)
    idx = np.random.permutation(len(df))
    split = int(len(df) * (1 - test_size))
    return df.iloc[idx[:split]].copy(), df.iloc[idx[split:]].copy()


if __name__ == "__main__":
    df = load_and_preprocess('ap_dataset.csv')
    
    df_train, df_test = train_test_split(df, test_size=0.2)
    print(f"\nTrain: {len(df_train)} bonuri, Test: {len(df_test)} bonuri")
    
    feature_cols = [c for c in df.columns if c != 'y']
    X_train, y_train = df_train[feature_cols].values, df_train['y'].values
    X_test, y_test = df_test[feature_cols].values, df_test['y'].values
    
    print(f"Dimensiune X_train: {X_train.shape}")
    print(f"Dimensiune X_test: {X_test.shape}")
    
    # normalizare z-score
    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
    sigma[sigma == 0] = 1
    X_train_n = (X_train - mu) / sigma
    X_test_n = (X_test - mu) / sigma
    
    print("\nDate pregatite pentru antrenare!")
    
    print("\n" + "="*60)
    print("TODO Ziua 3: Implementare Regresie Logistica cu Gradient Descent")
    print("="*60)
