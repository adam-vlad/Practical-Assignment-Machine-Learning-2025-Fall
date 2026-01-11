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


def split_receipts(df, test_size=0.2, seed=42):
    """Imparte bonurile in train/test"""
    rng = np.random.default_rng(seed)
    receipts = df['id_bon'].unique()
    rng.shuffle(receipts)
    split = int(len(receipts) * (1 - test_size))
    train_ids = set(receipts[:split])
    df_train = df[df['id_bon'].isin(train_ids)].copy()
    df_test = df[~df['id_bon'].isin(train_ids)].copy()
    return df_train, df_test


def identify_product_categories(df):
    """Identifica categoriile de produse"""
    all_products = df['retail_product_name'].unique()
    
    sides = [p for p in all_products 
             if any(kw in p.lower() for kw in ['potato', 'fries', 'cartofi'])]
    drinks = [p for p in all_products 
              if any(kw in p for kw in ['Pepsi', 'Cola', 'Dew', 'Aqua', 'Fanta', 'Mirinda', 'Limonada', 'Can', 'Doze'])]
    schnitzels = [p for p in all_products 
                  if any(kw in p.lower() for kw in ['schnitzel'])]
    
    return sides, drinks, schnitzels


def create_features_for_sauce(df, target_sauce, sides, drinks, schnitzels):
    """Creeaza features pentru un anumit sos"""
    all_products = df['retail_product_name'].unique()
    products_for_features = [p for p in all_products if p != target_sauce]
    
    receipts = df['id_bon'].unique()
    data = []
    
    for id_bon in receipts:
        receipt = df[df['id_bon'] == id_bon]
        row = {'id_bon': id_bon}
        
        products_in_cart = set(receipt['retail_product_name'].values)
        product_counts = receipt['retail_product_name'].value_counts()
        
        # target
        row['y'] = 1 if target_sauce in products_in_cart else 0
        row['actual_sauces'] = [s for s in SAUCES if s in products_in_cart]
        
        # features temporale
        dt = pd.to_datetime(receipt['data_bon'].iloc[0])
        row['day_of_week'] = dt.dayofweek + 1
        row['hour'] = dt.hour
        row['is_weekend'] = 1 if dt.dayofweek >= 5 else 0
        
        # features agregate (fara sosul target)
        receipt_filtered = receipt[receipt['retail_product_name'] != target_sauce]
        row['cart_size'] = len(receipt_filtered)
        row['distinct_products'] = receipt_filtered['retail_product_name'].nunique()
        row['total_value'] = receipt_filtered['SalePriceWithVAT'].sum()
        
        # vectorul de produse
        for p in products_for_features:
            row[f'count_{p}'] = product_counts.get(p, 0)
            row[f'has_{p}'] = 1 if p in products_in_cart else 0
        
        # features de categorie
        row['has_any_side'] = 1 if any(s in products_in_cart for s in sides) else 0
        row['has_any_drink'] = 1 if any(d in products_in_cart for d in drinks) else 0
        row['has_any_schnitzel'] = 1 if any(s in products_in_cart for s in schnitzels) else 0
        
        data.append(row)
    
    return pd.DataFrame(data), products_for_features


if __name__ == "__main__":
    df, sauce_freq = load_data('ap_dataset.csv')
    
    df_train, df_test = split_receipts(df, test_size=0.2, seed=42)
    print(f"\nTrain: {df_train['id_bon'].nunique()} bonuri")
    print(f"Test: {df_test['id_bon'].nunique()} bonuri")
    
    sides, drinks, schnitzels = identify_product_categories(df)
    print(f"\nCategorii gasite:")
    print(f"  - Garnituri: {len(sides)}")
    print(f"  - Bauturi: {len(drinks)}")
    print(f"  - Schnitzele: {len(schnitzels)}")
    
    # test pentru un sos
    print("\n" + "="*60)
    print("TEST CREARE FEATURES - Crazy Sauce")
    print("="*60)
    
    df_features, _ = create_features_for_sauce(df_train, 'Crazy Sauce', sides, drinks, schnitzels)
    print(f"Exemple create: {len(df_features)}")
    print(f"Features: {len([c for c in df_features.columns if c not in ['id_bon', 'y', 'actual_sauces']])}")
    print(f"Pozitive (cu Crazy Sauce): {df_features['y'].sum()}")
    
    print("\n" + "="*60)
    print("TODO Ziua 3: Implementare Regresie Logistica + antrenare pentru fiecare sos")
    print("="*60)
