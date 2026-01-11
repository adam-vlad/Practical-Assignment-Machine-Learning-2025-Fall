# Task 3 - Ranking pentru Upsell
# Score(produs | cos) = P(produs | cos) * pret(produs)

import numpy as np
import pandas as pd


SAUCES = [
    'Crazy Sauce', 'Cheddar Sauce', 'Extra Cheddar Sauce',
    'Garlic Sauce', 'Tomato Sauce', 'Blueberry Sauce',
    'Spicy Sauce', 'Pink Sauce'
]


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


def identify_upsell_candidates(df):
    """Identifica produsele candidate pentru upsell"""
    print("\n" + "="*60)
    print("IDENTIFICARE CANDIDATI UPSELL")
    print("="*60)
    
    all_products = df['retail_product_name'].unique()
    prices = df.groupby('retail_product_name')['SalePriceWithVAT'].mean().to_dict()
    
    # Sosuri
    sauces = [s for s in SAUCES if s in all_products]
    print(f"\nSosuri gasite: {len(sauces)}")
    for s in sauces:
        print(f"  - {s}: {prices.get(s, 0):.2f} lei")
    
    # Bauturi
    drink_keywords = ['Pepsi', 'Cola', 'Dew', 'Fanta', 'Sprite', 'Aqua', 'Water', 
                      'Limonada', 'Mirinda', 'Twist', 'Can', 'Doze', '0.25L', '0.33L', '0.5L']
    drinks = [p for p in all_products if any(kw in p for kw in drink_keywords)]
    print(f"\nBauturi gasite: {len(drinks)}")
    for d in drinks[:5]:
        print(f"  - {d}: {prices.get(d, 0):.2f} lei")
    if len(drinks) > 5:
        print(f"  ... si inca {len(drinks)-5}")
    
    # Garnituri
    side_keywords = ['Fries', 'Potatoes', 'fries', 'potatoes', 'Cartofi']
    sides = [p for p in all_products if any(kw in p.lower() for kw in side_keywords)]
    print(f"\nGarnituri gasite: {len(sides)}")
    for s in sides:
        print(f"  - {s}: {prices.get(s, 0):.2f} lei")
    
    upsell_candidates = list(set(sauces + drinks + sides))
    print(f"\nTotal candidati upsell: {len(upsell_candidates)}")
    
    return upsell_candidates, prices


def analyze_basket_composition(df):
    """Analizeaza compozitia cosurilor"""
    print("\n" + "="*60)
    print("ANALIZA COSURI")
    print("="*60)
    
    basket_sizes = df.groupby('id_bon')['retail_product_name'].count()
    print(f"Dimensiune medie cos: {basket_sizes.mean():.2f} produse")
    print(f"Dimensiune mediana cos: {basket_sizes.median():.0f} produse")
    print(f"Dimensiune maxima cos: {basket_sizes.max()} produse")
    
    basket_values = df.groupby('id_bon')['SalePriceWithVAT'].sum()
    print(f"\nValoare medie bon: {basket_values.mean():.2f} lei")
    print(f"Valoare mediana bon: {basket_values.median():.2f} lei")
    
    return basket_sizes, basket_values


if __name__ == "__main__":
    df = load_data('ap_dataset.csv')
    
    upsell_candidates, prices = identify_upsell_candidates(df)
    basket_sizes, basket_values = analyze_basket_composition(df)
    
    # Distributia temporala
    print("\n" + "="*60)
    print("DISTRIBUTIE TEMPORALA")
    print("="*60)
    
    df['data_bon'] = pd.to_datetime(df['data_bon'])
    df['hour'] = df['data_bon'].dt.hour
    df['day_of_week'] = df['data_bon'].dt.dayofweek + 1
    
    hourly = df.groupby('hour')['id_bon'].nunique()
    print("\nBonuri pe ore:")
    for h in sorted(hourly.index):
        print(f"  Ora {h:02d}: {hourly[h]} bonuri")
    
    print("\n" + "="*60)
    print("TODO Ziua 2: Preprocesare si creare features")
    print("="*60)
