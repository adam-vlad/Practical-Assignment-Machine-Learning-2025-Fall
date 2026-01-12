# Task 3 - Ranking pentru Upsell
# Score(produs | cos) = P(produs | cos) * pret(produs)

import numpy as np
import pandas as pd


SAUCES = [
    'Crazy Sauce', 'Cheddar Sauce', 'Extra Cheddar Sauce',
    'Garlic Sauce', 'Tomato Sauce', 'Blueberry Sauce',
    'Spicy Sauce', 'Pink Sauce'
]


def load_and_preprocess(filepath='ap_dataset.csv'):
    """Incarca si preproceseaza datele"""
    print("="*60)
    print("INCARCARE SI PREPROCESARE DATE")
    print("="*60)
    
    df = pd.read_csv(filepath)
    print(f"Date incarcate: {len(df)} linii, {df['id_bon'].nunique()} bonuri")
    
    # Construim liste de produse pentru fiecare bon
    receipts = []
    temporal_features = []
    receipt_ids = []
    
    for id_bon, group in df.groupby('id_bon'):
        products = group['retail_product_name'].tolist()
        receipts.append(products)
        receipt_ids.append(id_bon)
        
        dt = pd.to_datetime(group['data_bon'].iloc[0])
        temporal_features.append({
            'day_of_week': dt.dayofweek + 1,
            'hour': dt.hour,
            'is_weekend': 1 if dt.dayofweek >= 5 else 0
        })
    
    print(f"Bonuri procesate: {len(receipts)}")
    
    # Preturi medii
    all_products = df['retail_product_name'].unique()
    prices = df.groupby('retail_product_name')['SalePriceWithVAT'].mean().to_dict()
    
    # Identificam candidatii pentru upsell
    sauces = [s for s in SAUCES if s in all_products]
    
    drink_keywords = ['Pepsi', 'Cola', 'Dew', 'Fanta', 'Sprite', 'Aqua', 'Water', 
                      'Limonada', 'Mirinda', 'Twist', 'Can', 'Doze', '0.25L', '0.33L', '0.5L']
    drinks = [p for p in all_products if any(kw in p for kw in drink_keywords)]
    
    side_keywords = ['Fries', 'Potatoes', 'fries', 'potatoes', 'Cartofi']
    sides = [p for p in all_products if any(kw in p.lower() for kw in side_keywords)]
    
    upsell_candidates = list(set(sauces + drinks + sides))
    
    print(f"\nCandidati pentru upsell: {len(upsell_candidates)}")
    print(f"  - Sosuri: {len(sauces)}")
    print(f"  - Bauturi: {len(drinks)}")
    print(f"  - Garnituri: {len(sides)}")
    
    return receipts, temporal_features, receipt_ids, prices, upsell_candidates, all_products


def train_test_split_receipts(receipts, temporal_features, receipt_ids, test_size=0.2, seed=42):
    """Imparte datele in train/test"""
    np.random.seed(seed)
    n = len(receipts)
    idx = np.random.permutation(n)
    split = int(n * (1 - test_size))
    
    train_idx, test_idx = idx[:split], idx[split:]
    
    train_receipts = [receipts[i] for i in train_idx]
    train_temporal = [temporal_features[i] for i in train_idx]
    train_ids = [receipt_ids[i] for i in train_idx]
    
    test_receipts = [receipts[i] for i in test_idx]
    test_temporal = [temporal_features[i] for i in test_idx]
    test_ids = [receipt_ids[i] for i in test_idx]
    
    return (train_receipts, train_temporal, train_ids), (test_receipts, test_temporal, test_ids)


def create_features_for_product(receipt, temporal, all_products, target_product):
    """Creeaza features pentru un produs target"""
    features = []
    receipt_set = set(receipt)
    
    # Prezenta fiecarui produs (fara target)
    for p in all_products:
        if p != target_product:
            features.append(1 if p in receipt_set else 0)
    
    # Features temporale
    features.append(temporal['day_of_week'])
    features.append(temporal['is_weekend'])
    
    # Bucket pentru ora
    h = temporal['hour']
    if 6 <= h < 12:
        hour_bucket = 0  # dimineata
    elif 12 <= h < 15:
        hour_bucket = 1  # pranz
    elif 15 <= h < 21:
        hour_bucket = 2  # seara
    else:
        hour_bucket = 3  # noapte
    features.append(hour_bucket)
    
    return features


if __name__ == "__main__":
    data = load_and_preprocess('ap_dataset.csv')
    receipts, temporal_features, receipt_ids, prices, upsell_candidates, all_products = data
    
    train_data, test_data = train_test_split_receipts(
        receipts, temporal_features, receipt_ids, test_size=0.2
    )
    train_receipts, train_temporal, train_ids = train_data
    test_receipts, test_temporal, test_ids = test_data
    
    print(f"\nTrain: {len(train_receipts)} bonuri, Test: {len(test_receipts)} bonuri")
    
    # Test creare features
    print("\n" + "="*60)
    print("TEST CREARE FEATURES")
    print("="*60)
    
    sample_receipt = train_receipts[0]
    sample_temporal = train_temporal[0]
    target = upsell_candidates[0]
    
    print(f"Bon exemplu: {sample_receipt[:5]}...")
    print(f"Temporal: {sample_temporal}")
    print(f"Target: {target}")
    
    features = create_features_for_product(
        sample_receipt, sample_temporal, 
        [p for p in list(all_products) if p != target], target
    )
    print(f"Features create: {len(features)}")
    
    print("\n" + "="*60)
    print("TODO Ziua 6: Implementare arbore de decizie ID3")
    print("="*60)
