# Task 2.1 - Regresie Logistica: Crazy Sauce dat fiind Crazy Schnitzel
# Clasificare binara - prezice daca clientul ia sosul impreuna cu schnitzelul

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LogisticRegression:
    """Regresie logistica implementata manual cu gradient descent si metoda Newton."""
    
    def __init__(self, lr=0.1, n_iter=1000, method='gd', reg='l2', lambda_=0.01, tol=1e-6):
        self.lr = lr
        self.n_iter = n_iter
        self.method = method
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
    
    def _gradient_descent_step(self, X, y):
        n = X.shape[0]
        y_pred = self._sigmoid(X @ self.w + self.b)
        
        dw = (1/n) * (X.T @ (y_pred - y))
        db = (1/n) * np.sum(y_pred - y)
        
        if self.reg == 'l2':
            dw += self.lambda_ * self.w
        
        self.w -= self.lr * dw
        self.b -= self.lr * db
        return y_pred
    
    def _newton_step(self, X, y):
        n = X.shape[0]
        y_pred = self._sigmoid(X @ self.w + self.b)
        
        gradient = (1/n) * (X.T @ (y_pred - y))
        if self.reg == 'l2':
            gradient += self.lambda_ * self.w
        
        # matricea Hessian
        d = y_pred * (1 - y_pred)
        d = np.clip(d, 1e-10, None)
        H = (1/n) * (X.T @ np.diag(d) @ X)
        H += (self.lambda_ + 1e-5) * np.eye(H.shape[0])
        
        try:
            self.w -= np.linalg.solve(H, gradient)
        except:
            self.w -= self.lr * gradient
        
        self.b -= self.lr * np.mean(y_pred - y)
        return y_pred
    
    def fit(self, X, y, verbose=True):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        self.loss_history = []
        
        for i in range(self.n_iter):
            if self.method == 'newton':
                y_pred = self._newton_step(X, y)
            else:
                y_pred = self._gradient_descent_step(X, y)
            
            loss = self._loss(y, y_pred)
            self.loss_history.append(loss)
            
            if verbose and i % 100 == 0:
                print(f"  Iter {i:4d}: loss = {loss:.6f}")
            
            if i > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol:
                if verbose:
                    print(f"  Convergenta la iteratia {i}")
                break
        
        return self
    
    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# Metrici

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])


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


def roc_auc(y_true, y_proba):
    sorted_idx = np.argsort(y_proba)[::-1]
    y_sorted = y_true[sorted_idx]
    
    tpr_list, fpr_list = [0], [0]
    tp, fp = 0, 0
    pos, neg = np.sum(y_true == 1), np.sum(y_true == 0)
    
    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / pos if pos > 0 else 0)
        fpr_list.append(fp / neg if neg > 0 else 0)
    
    auc = sum((fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2 
              for i in range(1, len(fpr_list)))
    return auc, fpr_list, tpr_list


def evaluate(y_true, y_pred, y_proba=None):
    results = {
        'accuracy': accuracy(y_true, y_pred),
        'precision': precision(y_true, y_pred),
        'recall': recall(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'cm': confusion_matrix(y_true, y_pred)
    }
    if y_proba is not None:
        results['auc'], results['fpr'], results['tpr'] = roc_auc(y_true, y_proba)
    return results


def print_metrics(results, title="Rezultate"):
    print(f"\n{'='*50}")
    print(title)
    print('='*50)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1']:.4f}")
    if 'auc' in results:
        print(f"ROC-AUC:   {results['auc']:.4f}")
    print(f"\nMatrice de confuzie:")
    print(f"              Pred 0  Pred 1")
    print(f"  Actual 0     {results['cm'][0,0]:4d}    {results['cm'][0,1]:4d}")
    print(f"  Actual 1     {results['cm'][1,0]:4d}    {results['cm'][1,1]:4d}")


# Ploturi

def plot_loss(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, 'b-', lw=2)
    plt.xlabel('Iteratie')
    plt.ylabel('Loss')
    plt.title('Curba de loss - Gradient Descent')
    plt.grid(True, alpha=0.3)
    plt.savefig('2_1_loss_curve.png', dpi=150)
    plt.close()
    print("Salvat: 2_1_loss_curve.png")


def plot_roc(fpr, tpr, auc_val):
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC = {auc_val:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', lw=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curba ROC')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig('2_1_roc_curve.png', dpi=150)
    plt.close()
    print("Salvat: 2_1_roc_curve.png")


def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xticks([0, 1], ['Fara sos', 'Cu sos'])
    plt.yticks([0, 1], ['Fara sos', 'Cu sos'])
    plt.xlabel('Predictie')
    plt.ylabel('Real')
    plt.title('Matrice de confuzie')
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i,j] > cm.max()/2 else 'black'
            plt.text(j, i, cm[i,j], ha='center', va='center', color=color, fontsize=16)
    plt.savefig('2_1_confusion_matrix.png', dpi=150)
    plt.close()
    print("Salvat: 2_1_confusion_matrix.png")


# Preprocesare date

def load_and_preprocess(filepath='ap_dataset.csv'):
    print("="*60)
    print("INCARCARE SI PREPROCESARE DATE")
    print("="*60)
    
    df = pd.read_csv(filepath)
    print(f"Date incarcate: {len(df)} linii, {df['id_bon'].nunique()} bonuri")
    
    TARGET_SAUCE = 'Crazy Sauce'
    TARGET_PRODUCT = 'Crazy Schnitzel'
    
    receipts_with_cs = df[df['retail_product_name'] == TARGET_PRODUCT]['id_bon'].unique()
    print(f"Bonuri cu {TARGET_PRODUCT}: {len(receipts_with_cs)}")
    
    df_filtered = df[df['id_bon'].isin(receipts_with_cs)].copy()
    
    all_products = df['retail_product_name'].unique()
    products_for_features = [p for p in all_products if p != TARGET_SAUCE]
    
    sides = [p for p in all_products if any(kw in p.lower() for kw in ['potato', 'fries', 'cartofi'])]
    drinks = [p for p in all_products if any(kw in p for kw in ['Pepsi', 'Cola', 'Dew', 'Aqua', 'Fanta', 'Mirinda', 'Limonada'])]
    
    data = []
    for id_bon in receipts_with_cs:
        receipt = df_filtered[df_filtered['id_bon'] == id_bon]
        row = {}
        
        row['y'] = 1 if TARGET_SAUCE in receipt['retail_product_name'].values else 0
        
        dt = pd.to_datetime(receipt['data_bon'].iloc[0])
        row['day_of_week'] = dt.dayofweek + 1
        row['hour'] = dt.hour
        row['is_weekend'] = 1 if dt.dayofweek >= 5 else 0
        
        row['cart_size'] = len(receipt)
        row['distinct_products'] = receipt['retail_product_name'].nunique()
        row['total_value'] = receipt['SalePriceWithVAT'].sum()
        
        product_counts = receipt['retail_product_name'].value_counts()
        products_in_cart = set(receipt['retail_product_name'].values)
        
        for p in products_for_features:
            row[f'count_{p}'] = product_counts.get(p, 0)
            row[f'has_{p}'] = 1 if p in products_in_cart else 0
        
        row['has_any_side'] = 1 if any(s in products_in_cart for s in sides) else 0
        row['has_any_drink'] = 1 if any(d in products_in_cart for d in drinks) else 0
        
        data.append(row)
    
    df_final = pd.DataFrame(data)
    
    pos = df_final['y'].sum()
    neg = len(df_final) - pos
    print(f"\nDistributie target:")
    print(f"  y=1 (cu {TARGET_SAUCE}): {pos} ({100*pos/len(df_final):.1f}%)")
    print(f"  y=0 (fara {TARGET_SAUCE}): {neg} ({100*neg/len(df_final):.1f}%)")
    
    return df_final


def train_test_split(df, test_size=0.2, seed=42):
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
    
    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
    sigma[sigma == 0] = 1
    X_train_n = (X_train - mu) / sigma
    X_test_n = (X_test - mu) / sigma
    
    # baseline
    majority = np.bincount(y_train).argmax()
    baseline_pred = np.full_like(y_test, majority)
    baseline_acc = np.mean(y_test == baseline_pred)
    print(f"\nBASELINE (clasa majoritara = {majority}): Accuracy = {baseline_acc:.4f}")
    
    # gradient descent
    print(f"\n{'='*50}")
    print("REGRESIE LOGISTICA (Gradient Descent)")
    print('='*50)
    model_gd = LogisticRegression(lr=0.1, n_iter=1000, method='gd', reg='l2', lambda_=0.01)
    model_gd.fit(X_train_n, y_train, verbose=True)
    
    y_pred_gd = model_gd.predict(X_test_n)
    y_proba_gd = model_gd.predict_proba(X_test_n)
    
    results_gd = evaluate(y_test, y_pred_gd, y_proba_gd)
    print_metrics(results_gd, "Rezultate - Gradient Descent")
    
    # metoda Newton
    print(f"\n{'='*50}")
    print("REGRESIE LOGISTICA (Metoda Newton)")
    print('='*50)
    model_newton = LogisticRegression(lr=1.0, n_iter=50, method='newton', reg='l2', lambda_=0.1)
    model_newton.fit(X_train_n, y_train, verbose=True)
    
    y_pred_newton = model_newton.predict(X_test_n)
    y_proba_newton = model_newton.predict_proba(X_test_n)
    
    results_newton = evaluate(y_test, y_pred_newton, y_proba_newton)
    print_metrics(results_newton, "Rezultate - Metoda Newton")
    
    # grafice
    print(f"\n{'='*50}")
    print("GENERARE GRAFICE")
    print('='*50)
    plot_loss(model_gd.loss_history)
    plot_roc(results_gd['fpr'], results_gd['tpr'], results_gd['auc'])
    plot_confusion_matrix(results_gd['cm'])
    
    print("\n" + "="*60)
    print("TODO Ziua 5: Adaugare interpretare coeficienti + finalizare")
    print("="*60)
