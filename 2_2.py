# Task 2.2 - Un model de regresie logistica pentru fiecare sos + sistem de recomandare
# Pentru un cos dat, calculam P(sos | cos) si recomandam top-K sosuri

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


class LogisticRegression:
    
    def __init__(self, lr=0.1, n_iter=1000, reg='l2', lambda_=0.01, tol=1e-6):
        self.lr = lr
        self.n_iter = n_iter
        self.reg = reg
        self.lambda_ = lambda_
        self.tol = tol
        self.w = None
        self.b = None
        self.loss_history = []
    
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _loss(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        if self.reg == 'l2':
            loss += (self.lambda_ / 2) * np.sum(self.w ** 2)
        return loss
    
    def fit(self, X, y, verbose=False):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        self.loss_history = []
        
        for i in range(self.n_iter):
            y_pred = self._sigmoid(X @ self.w + self.b)
            
            dw = (1/n_samples) * (X.T @ (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            if self.reg == 'l2':
                dw += self.lambda_ * self.w
            
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            loss = self._loss(y, y_pred)
            self.loss_history.append(loss)
            
            if verbose and i % 200 == 0:
                print(f"    Iter {i:4d}: loss = {loss:.6f}")
            
            if i > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol:
                break
        
        return self
    
    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def f1_score(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0


def hit_at_k(recommended, actual, k):
    top_k = recommended[:k]
    return 1 if any(s in actual for s in top_k) else 0


def precision_at_k(recommended, actual, k):
    top_k = recommended[:k]
    if len(top_k) == 0:
        return 0
    hits = sum(1 for s in top_k if s in actual)
    return hits / len(top_k)


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
    rng = np.random.default_rng(seed)
    receipts = df['id_bon'].unique()
    rng.shuffle(receipts)
    split = int(len(receipts) * (1 - test_size))
    train_ids = set(receipts[:split])
    df_train = df[df['id_bon'].isin(train_ids)].copy()
    df_test = df[~df['id_bon'].isin(train_ids)].copy()
    return df_train, df_test


def identify_product_categories(df):
    all_products = df['retail_product_name'].unique()
    
    sides = [p for p in all_products 
             if any(kw in p.lower() for kw in ['potato', 'fries', 'cartofi'])]
    drinks = [p for p in all_products 
              if any(kw in p for kw in ['Pepsi', 'Cola', 'Dew', 'Aqua', 'Fanta', 'Mirinda', 'Limonada', 'Can', 'Doze'])]
    schnitzels = [p for p in all_products 
                  if any(kw in p.lower() for kw in ['schnitzel'])]
    
    return sides, drinks, schnitzels


def create_features_for_sauce(df, target_sauce, sides, drinks, schnitzels):
    all_products = df['retail_product_name'].unique()
    products_for_features = [p for p in all_products if p != target_sauce]
    
    receipts = df['id_bon'].unique()
    data = []
    
    for id_bon in receipts:
        receipt = df[df['id_bon'] == id_bon]
        row = {'id_bon': id_bon}
        
        products_in_cart = set(receipt['retail_product_name'].values)
        product_counts = receipt['retail_product_name'].value_counts()
        
        row['y'] = 1 if target_sauce in products_in_cart else 0
        row['actual_sauces'] = [s for s in SAUCES if s in products_in_cart]
        
        dt = pd.to_datetime(receipt['data_bon'].iloc[0])
        row['day_of_week'] = dt.dayofweek + 1
        row['hour'] = dt.hour
        row['is_weekend'] = 1 if dt.dayofweek >= 5 else 0
        
        receipt_filtered = receipt[receipt['retail_product_name'] != target_sauce]
        row['cart_size'] = len(receipt_filtered)
        row['distinct_products'] = receipt_filtered['retail_product_name'].nunique()
        row['total_value'] = receipt_filtered['SalePriceWithVAT'].sum()
        
        for p in products_for_features:
            row[f'count_{p}'] = product_counts.get(p, 0)
            row[f'has_{p}'] = 1 if p in products_in_cart else 0
        
        row['has_any_side'] = 1 if any(s in products_in_cart for s in sides) else 0
        row['has_any_drink'] = 1 if any(d in products_in_cart for d in drinks) else 0
        row['has_any_schnitzel'] = 1 if any(s in products_in_cart for s in schnitzels) else 0
        
        row['interact_schnitzel_x_side'] = row['has_any_schnitzel'] * row['has_any_side']
        row['interact_schnitzel_x_drink'] = row['has_any_schnitzel'] * row['has_any_drink']
        row['interact_side_x_drink'] = row['has_any_side'] * row['has_any_drink']
        
        data.append(row)
    
    return pd.DataFrame(data), products_for_features


def train_test_split(df, test_size=0.2, seed=42):
    np.random.seed(seed)
    idx = np.random.permutation(len(df))
    split = int(len(df) * (1 - test_size))
    return df.iloc[idx[:split]].copy(), df.iloc[idx[split:]].copy()


class SauceRecommender:
    """Antrenam cate un model LR pentru fiecare sos si le folosim pentru recomandari."""
    
    def __init__(self):
        self.models = {}
        self.norm_params = {}
        self.feature_cols = {}
        self.sauce_popularity = {}
        self.sides = []
        self.drinks = []
        self.schnitzels = []
        self.all_products = []
    
    def fit(self, df, sauce_freq, verbose=True):
        self.sauce_popularity = sauce_freq
        self.all_products = df['retail_product_name'].unique()
        
        self.sides, self.drinks, self.schnitzels = identify_product_categories(df)
        
        print("\n" + "="*60)
        print("ANTRENARE MODELE PENTRU FIECARE SOS")
        print("="*60)
        print(f"Categorii gasite:")
        print(f"  - Garnituri: {len(self.sides)}")
        print(f"  - Bauturi: {len(self.drinks)}")
        print(f"  - Schnitzele: {len(self.schnitzels)}")
        
        for sauce in SAUCES:
            if sauce_freq.get(sauce, 0) == 0:
                print(f"\n[SKIP] {sauce} - nu exista in date")
                continue
            
            if verbose:
                print(f"\n>>> Antrenare: {sauce}")
            
            df_features, _ = create_features_for_sauce(
                df, sauce, self.sides, self.drinks, self.schnitzels
            )
            
            df_train, df_val = train_test_split(df_features, test_size=0.2)
            
            feature_cols = [c for c in df_features.columns 
                           if c not in ['id_bon', 'y', 'actual_sauces']]
            
            X_train = df_train[feature_cols].values.astype(float)
            y_train = df_train['y'].values
            
            n_pos = y_train.sum()
            if verbose:
                print(f"    Exemple pozitive: {n_pos}/{len(y_train)} ({100*n_pos/len(y_train):.1f}%)")
            
            if n_pos == 0:
                print(f"    [SKIP] 0 exemple pozitive")
                continue
            
            mu = X_train.mean(axis=0)
            sigma = X_train.std(axis=0)
            sigma[sigma == 0] = 1
            X_train_n = (X_train - mu) / sigma
            
            model = LogisticRegression(lr=0.1, n_iter=500, reg='l2', lambda_=0.01)
            model.fit(X_train_n, y_train, verbose=False)
            
            X_val = df_val[feature_cols].values.astype(float)
            y_val = df_val['y'].values
            X_val_n = (X_val - mu) / sigma
            y_pred = model.predict(X_val_n)
            acc = accuracy(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            
            if verbose:
                print(f"    Validare - Accuracy: {acc:.4f}, F1: {f1:.4f}")
            
            self.models[sauce] = model
            self.norm_params[sauce] = {'mu': mu, 'sigma': sigma}
            self.feature_cols[sauce] = feature_cols
        
        print(f"\nModele antrenate: {len(self.models)}/{len(SAUCES)} sosuri")
    
    def _create_features_for_receipt(self, df_receipt, target_sauce, all_products):
        products_in_cart = set(df_receipt['retail_product_name'].values)
        product_counts = df_receipt['retail_product_name'].value_counts()
        
        products_for_features = [p for p in all_products if p != target_sauce]
        
        features = {}
        
        dt = pd.to_datetime(df_receipt['data_bon'].iloc[0])
        features['day_of_week'] = dt.dayofweek + 1
        features['hour'] = dt.hour
        features['is_weekend'] = 1 if dt.dayofweek >= 5 else 0
        
        receipt_filtered = df_receipt[df_receipt['retail_product_name'] != target_sauce]
        features['cart_size'] = len(receipt_filtered)
        features['distinct_products'] = receipt_filtered['retail_product_name'].nunique()
        features['total_value'] = receipt_filtered['SalePriceWithVAT'].sum()
        
        for p in products_for_features:
            features[f'count_{p}'] = product_counts.get(p, 0)
            features[f'has_{p}'] = 1 if p in products_in_cart else 0
        
        features['has_any_side'] = 1 if any(s in products_in_cart for s in self.sides) else 0
        features['has_any_drink'] = 1 if any(d in products_in_cart for d in self.drinks) else 0
        features['has_any_schnitzel'] = 1 if any(s in products_in_cart for s in self.schnitzels) else 0
        features['interact_schnitzel_x_side'] = features['has_any_schnitzel'] * features['has_any_side']
        features['interact_schnitzel_x_drink'] = features['has_any_schnitzel'] * features['has_any_drink']
        features['interact_side_x_drink'] = features['has_any_side'] * features['has_any_drink']
        
        return features
    
    def predict_proba_for_sauce(self, df_receipt, sauce, all_products=None):
        if sauce not in self.models:
            return 0.0
        if all_products is None:
            all_products = self.all_products
        
        features = self._create_features_for_receipt(df_receipt, sauce, all_products)
        X = np.array([features.get(col, 0) for col in self.feature_cols[sauce]]).reshape(1, -1)
        
        mu = self.norm_params[sauce]['mu']
        sigma = self.norm_params[sauce]['sigma']
        X_n = (X - mu) / sigma
        
        return self.models[sauce].predict_proba(X_n)[0]
    
    def recommend(self, df_receipt, exclude_sauces=None, top_k=3):
        if exclude_sauces is None:
            products_in_cart = set(df_receipt['retail_product_name'].values)
            exclude_sauces = [s for s in SAUCES if s in products_in_cart]
        
        probs = {}
        for sauce in self.models.keys():
            if sauce not in exclude_sauces:
                probs[sauce] = self.predict_proba_for_sauce(df_receipt, sauce)
        
        ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return [s for s, p in ranked[:top_k]], ranked[:top_k]
    
    def recommend_baseline(self, exclude_sauces=None, top_k=3):
        if exclude_sauces is None:
            exclude_sauces = []
        
        ranking = sorted(self.sauce_popularity.items(), key=lambda x: x[1], reverse=True)
        return [s for s, c in ranking if s not in exclude_sauces][:top_k]


def evaluate_recommender(recommender, df, k_values=[1, 3, 5]):
    print("\n" + "="*60)
    print("EVALUARE SISTEM DE RECOMANDARE")
    print("="*60)
    
    results = {k: {'hit': [], 'precision': [], 
                   'hit_baseline': [], 'precision_baseline': []} 
               for k in k_values}
    
    receipts = df['id_bon'].unique()
    n_with_sauce = 0
    
    for id_bon in receipts:
        df_receipt = df[df['id_bon'] == id_bon]
        products_in_cart = list(df_receipt['retail_product_name'].values)
        actual_sauces = [s for s in SAUCES if s in products_in_cart]
        
        if len(actual_sauces) == 0:
            continue
        
        for target_sauce in actual_sauces:
            df_partial = df_receipt[df_receipt['retail_product_name'] != target_sauce]
            if df_partial.empty:
                continue
            
            exclude = [s for s in SAUCES if s in df_partial['retail_product_name'].values]
            
            rec_sauces, _ = recommender.recommend(df_partial, exclude_sauces=exclude, top_k=max(k_values))
            rec_baseline = recommender.recommend_baseline(exclude_sauces=exclude, top_k=max(k_values))
            
            for k in k_values:
                results[k]['hit'].append(hit_at_k(rec_sauces, [target_sauce], k))
                results[k]['precision'].append(precision_at_k(rec_sauces, [target_sauce], k))
                results[k]['hit_baseline'].append(hit_at_k(rec_baseline, [target_sauce], k))
                results[k]['precision_baseline'].append(precision_at_k(rec_baseline, [target_sauce], k))
            n_with_sauce += 1
    
    print(f"Bonuri cu cel putin un sos in test: {n_with_sauce}")
    
    print("\n" + "-"*60)
    print(f"{'Metrica':<25} {'K=1':>10} {'K=3':>10} {'K=5':>10}")
    print("-"*60)
    
    for metric in ['hit', 'precision']:
        vals_model = [np.mean(results[k][metric]) for k in k_values]
        vals_baseline = [np.mean(results[k][f'{metric}_baseline']) for k in k_values]
        
        print(f"{metric.capitalize()}@K (Model)      {vals_model[0]:>10.4f} {vals_model[1]:>10.4f} {vals_model[2]:>10.4f}")
        print(f"{metric.capitalize()}@K (Baseline)   {vals_baseline[0]:>10.4f} {vals_baseline[1]:>10.4f} {vals_baseline[2]:>10.4f}")
        print()
    
    return results


def plot_recommendation_metrics(results, k_values=[1, 3, 5]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(k_values))
    width = 0.35
    
    for ax, metric in zip(axes, ['hit', 'precision']):
        vals_model = [np.mean(results[k][metric]) for k in k_values]
        vals_baseline = [np.mean(results[k][f'{metric}_baseline']) for k in k_values]
        
        bars1 = ax.bar(x - width/2, vals_model, width, label='Model LR', color='steelblue')
        bars2 = ax.bar(x + width/2, vals_baseline, width, label='Baseline (Popularitate)', color='coral')
        
        ax.set_xlabel('K')
        ax.set_ylabel(f'{metric.capitalize()}@K')
        ax.set_title(f'{metric.capitalize()}@K: Model vs Baseline')
        ax.set_xticks(x)
        ax.set_xticklabels([f'K={k}' for k in k_values])
        ax.legend()
        ax.set_ylim(0, 1)
        
        for bar in bars1:
            h = bar.get_height()
            ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        for bar in bars2:
            h = bar.get_height()
            ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('2_2_recommendation_metrics.png', dpi=150)
    plt.close()
    print("Salvat: 2_2_recommendation_metrics.png")


def show_example_recommendations(recommender, df, n_examples=5):
    print("\n" + "="*60)
    print("EXEMPLE DE RECOMANDARI")
    print("="*60)

    receipts = df['id_bon'].unique()
    np.random.seed(789)
    
    count = 0
    for id_bon in np.random.permutation(receipts):
        df_receipt = df[df['id_bon'] == id_bon]
        products = list(df_receipt['retail_product_name'].values)
        actual_sauces = [s for s in SAUCES if s in products]
        
        if len(actual_sauces) == 0:
            continue
        
        target = actual_sauces[0]
        df_partial = df_receipt[df_receipt['retail_product_name'] != target]
        if df_partial.empty:
            continue
        
        exclude = [s for s in SAUCES if s in df_partial['retail_product_name'].values]
        rec_sauces, rec_with_probs = recommender.recommend(df_partial, exclude_sauces=exclude, top_k=5)
        baseline = recommender.recommend_baseline(exclude_sauces=exclude, top_k=5)
        
        count += 1
        print(f"\n--- Exemplu {count} ---")
        print(f"Produse in bon: {products[:5]}{'...' if len(products) > 5 else ''}")
        print(f"Sos real (ascuns): {target}")
        print(f"Top-5 recomandari (Model):")
        for j, (sauce, prob) in enumerate(rec_with_probs, 1):
            hit = "CORECT!" if sauce == target else ""
            print(f"  {j}. {sauce}: {prob:.4f} {hit}")
        print(f"Top-5 baseline: {baseline}")
        
        if count >= n_examples:
            break


if __name__ == "__main__":
    df, sauce_freq = load_data('ap_dataset.csv')
    df_train, df_test = split_receipts(df, test_size=0.2, seed=42)
    print(f"Train: {df_train['id_bon'].nunique()} bonuri, Test: {df_test['id_bon'].nunique()} bonuri")
    
    sauce_freq_train = {s: df_train[df_train['retail_product_name'] == s]['id_bon'].nunique()
                        for s in SAUCES}
    
    recommender = SauceRecommender()
    recommender.fit(df_train, sauce_freq_train, verbose=True)
    
    k_values = [1, 3, 5]
    results = evaluate_recommender(recommender, df_test, k_values)
    
    print("\n--- GENERARE GRAFICE ---")
    plot_recommendation_metrics(results, k_values)
    
    show_example_recommendations(recommender, df_test, n_examples=5)
    
    print("\n" + "="*60)
    print("TASK 2.2 COMPLET!")
    print("="*60)
    print("Fisiere generate:")
    print("  - 2_2_recommendation_metrics.png")
    
    print("\nRezumat (pe setul de TEST):")
    print(f"  - Modele antrenate: {len(recommender.models)} sosuri")
    for k in k_values:
        h = np.mean(results[k]['hit'])
        hb = np.mean(results[k]['hit_baseline'])
        p = np.mean(results[k]['precision'])
        pb = np.mean(results[k]['precision_baseline'])
        print(f"  - Hit@{k}: {h:.4f} (vs baseline {hb:.4f}) | Precision@{k}: {p:.4f} (vs {pb:.4f})")
