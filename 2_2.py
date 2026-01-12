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


def f1_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


def load_data(filepath='ap_dataset.csv'):
    df = pd.read_csv(filepath)
    sauce_freq = {}
    for sauce in SAUCES:
        count = df[df['retail_product_name'] == sauce]['id_bon'].nunique()
        sauce_freq[sauce] = count
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
    sides = [p for p in all_products if any(kw in p.lower() for kw in ['potato', 'fries', 'cartofi'])]
    drinks = [p for p in all_products if any(kw in p for kw in ['Pepsi', 'Cola', 'Dew', 'Aqua', 'Fanta', 'Mirinda', 'Limonada', 'Can', 'Doze'])]
    schnitzels = [p for p in all_products if any(kw in p.lower() for kw in ['schnitzel'])]
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
        
        data.append(row)
    
    return pd.DataFrame(data), products_for_features


def train_test_split(df, test_size=0.2, seed=42):
    np.random.seed(seed)
    idx = np.random.permutation(len(df))
    split = int(len(df) * (1 - test_size))
    return df.iloc[idx[:split]].copy(), df.iloc[idx[split:]].copy()


if __name__ == "__main__":
    print("="*60)
    print("INCARCARE DATE")
    print("="*60)
    
    df, sauce_freq = load_data('ap_dataset.csv')
    print(f"Date incarcate: {len(df)} linii, {df['id_bon'].nunique()} bonuri")
    
    df_train, df_test = split_receipts(df, test_size=0.2, seed=42)
    print(f"Train: {df_train['id_bon'].nunique()} bonuri, Test: {df_test['id_bon'].nunique()} bonuri")
    
    sides, drinks, schnitzels = identify_product_categories(df)
    
    print("\n" + "="*60)
    print("ANTRENARE MODELE PENTRU FIECARE SOS")
    print("="*60)
    
    models = {}
    
    for sauce in SAUCES:
        if sauce_freq.get(sauce, 0) == 0:
            print(f"\n[SKIP] {sauce} - nu exista in date")
            continue
        
        print(f"\n>>> Antrenare: {sauce}")
        
        df_features, _ = create_features_for_sauce(df_train, sauce, sides, drinks, schnitzels)
        df_tr, df_val = train_test_split(df_features, test_size=0.2)
        
        feature_cols = [c for c in df_features.columns if c not in ['id_bon', 'y', 'actual_sauces']]
        
        X_train = df_tr[feature_cols].values.astype(float)
        y_train = df_tr['y'].values
        
        n_pos = y_train.sum()
        print(f"    Exemple pozitive: {n_pos}/{len(y_train)} ({100*n_pos/len(y_train):.1f}%)")
        
        if n_pos == 0:
            print(f"    [SKIP] 0 exemple pozitive")
            continue
        
        # normalizare
        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0)
        sigma[sigma == 0] = 1
        X_train_n = (X_train - mu) / sigma
        
        model = LogisticRegression(lr=0.1, n_iter=500, reg='l2', lambda_=0.01)
        model.fit(X_train_n, y_train, verbose=False)
        
        # validare
        X_val = df_val[feature_cols].values.astype(float)
        y_val = df_val['y'].values
        X_val_n = (X_val - mu) / sigma
        y_pred = model.predict(X_val_n)
        acc = accuracy(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        print(f"    Validare - Accuracy: {acc:.4f}, F1: {f1:.4f}")
        
        models[sauce] = {'model': model, 'mu': mu, 'sigma': sigma, 'feature_cols': feature_cols}
    
    print(f"\nModele antrenate: {len(models)}/{len(SAUCES)} sosuri")
    
    print("\n" + "="*60)
    print("TODO Ziua 6: Implementare sistem de recomandare + evaluare")
    print("="*60)
