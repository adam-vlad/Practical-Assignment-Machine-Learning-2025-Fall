# Task 3 - Ranking pentru Upsell
# Score(produs | cos) = P(produs | cos) * pret(produs)

import numpy as np
import pandas as pd
from collections import Counter


SAUCES = [
    'Crazy Sauce', 'Cheddar Sauce', 'Extra Cheddar Sauce',
    'Garlic Sauce', 'Tomato Sauce', 'Blueberry Sauce',
    'Spicy Sauce', 'Pink Sauce'
]


class DecisionTreeID3:
    """Arbore de decizie ID3 pentru clasificare binara."""
    
    def __init__(self, max_depth=5, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.feature_names = None
    
    def _entropy(self, y):
        if len(y) == 0:
            return 0
        counts = Counter(y)
        probs = [c / len(y) for c in counts.values()]
        return -sum(p * np.log2(p + 1e-10) for p in probs)
    
    def _information_gain(self, X, y, feature_idx):
        parent_entropy = self._entropy(y)
        
        values = np.unique(X[:, feature_idx])
        
        child_entropy = 0
        for v in values:
            mask = X[:, feature_idx] == v
            if np.sum(mask) > 0:
                weight = np.sum(mask) / len(y)
                child_entropy += weight * self._entropy(y[mask])
        
        return parent_entropy - child_entropy
    
    def _build_tree(self, X, y, depth, used_features):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return {'leaf': True, 'prob': np.mean(y) if len(y) > 0 else 0.5}
        
        if len(np.unique(y)) == 1:
            return {'leaf': True, 'prob': y[0]}
        
        best_gain = -1
        best_feature = None
        
        for f in range(X.shape[1]):
            if f in used_features:
                continue
            if len(np.unique(X[:, f])) < 2:
                continue
            
            gain = self._information_gain(X, y, f)
            if gain > best_gain:
                best_gain = gain
                best_feature = f
        
        if best_feature is None or best_gain <= 0:
            return {'leaf': True, 'prob': np.mean(y)}
        
        node = {
            'leaf': False,
            'feature': best_feature,
            'feature_name': self.feature_names[best_feature] if self.feature_names else f'f{best_feature}',
            'children': {}
        }
        
        new_used = used_features | {best_feature}
        
        for v in np.unique(X[:, best_feature]):
            mask = X[:, best_feature] == v
            if np.sum(mask) > 0:
                node['children'][v] = self._build_tree(X[mask], y[mask], depth + 1, new_used)
            else:
                node['children'][v] = {'leaf': True, 'prob': np.mean(y)}
        
        node['default_prob'] = np.mean(y)
        
        return node
    
    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        X = np.array(X)
        y = np.array(y)
        self.tree = self._build_tree(X, y, depth=0, used_features=set())
        return self
    
    def _predict_one(self, x, node):
        if node['leaf']:
            return node['prob']
        
        f = node['feature']
        v = x[f]
        
        if v in node['children']:
            return self._predict_one(x, node['children'][v])
        else:
            return node.get('default_prob', 0.5)
    
    def predict_proba(self, X):
        X = np.array(X)
        if X.ndim == 1:
            return self._predict_one(X, self.tree)
        return np.array([self._predict_one(x, self.tree) for x in X])


class ID3Ranker:
    """Antrenam cate un arbore pentru fiecare produs candidat."""
    
    def __init__(self, max_depth=5, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = {}
        self.feature_names = None
        self.all_products = []
    
    def _create_features(self, receipt, temporal, all_products):
        features = []
        receipt_set = set(receipt)
        
        for p in all_products:
            features.append(1 if p in receipt_set else 0)
        
        features.append(temporal['day_of_week'])
        features.append(temporal['is_weekend'])
        
        h = temporal['hour']
        if 6 <= h < 12:
            hour_bucket = 0
        elif 12 <= h < 15:
            hour_bucket = 1
        elif 15 <= h < 21:
            hour_bucket = 2
        else:
            hour_bucket = 3
        features.append(hour_bucket)
        
        return features
    
    def fit(self, receipts, temporal_features, candidates):
        all_products = set()
        for r in receipts:
            all_products.update(r)
        self.all_products = sorted(all_products)
        
        self.feature_names = [f'has_{p}' for p in self.all_products]
        self.feature_names += ['day_of_week', 'is_weekend', 'hour_bucket']
        
        print(f"  Antrenare ID3 pentru {len(candidates)} produse candidate...")
        
        for i, candidate in enumerate(candidates):
            if candidate not in all_products:
                continue
            
            X = []
            y = []
            
            for receipt, temp in zip(receipts, temporal_features):
                receipt_without_candidate = [p for p in receipt if p != candidate]
                features = self._create_features(
                    receipt_without_candidate, 
                    temp, 
                    [p for p in self.all_products if p != candidate]
                )
                X.append(features)
                y.append(1 if candidate in receipt else 0)
            
            X = np.array(X)
            y = np.array(y)
            
            tree = DecisionTreeID3(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split
            )
            feature_names_for_candidate = [f'has_{p}' for p in self.all_products if p != candidate]
            feature_names_for_candidate += ['day_of_week', 'is_weekend', 'hour_bucket']
            tree.fit(X, y, feature_names=feature_names_for_candidate)
            
            self.trees[candidate] = tree
            
            if (i + 1) % 10 == 0:
                print(f"    Progres: {i+1}/{len(candidates)} arbori")
        
        print(f"  Gata! {len(self.trees)} arbori antrenati")
        return self
    
    def predict_proba(self, cart, temporal_context, candidate):
        if candidate not in self.trees:
            return 0.0
        
        products_for_features = [p for p in self.all_products if p != candidate]
        features = self._create_features(cart, temporal_context, products_for_features)
        
        return self.trees[candidate].predict_proba(np.array(features))
    
    def rank_candidates(self, cart, temporal_context, candidates, prices=None):
        scores = []
        for c in candidates:
            if c in cart:
                continue
            
            prob = self.predict_proba(cart, temporal_context, c)
            
            if prices and c in prices:
                score = prob * prices[c]
            else:
                score = prob
            
            scores.append((c, score, prob))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class PopularityBaseline:
    """Baseline simplu - ordonam dupa popularitate."""
    
    def __init__(self):
        self.popularity = {}
        self.revenue = {}
    
    def fit(self, receipts, prices=None):
        product_count = {}
        for receipt in receipts:
            for p in set(receipt):
                product_count[p] = product_count.get(p, 0) + 1
        
        total_receipts = len(receipts)
        for p, count in product_count.items():
            self.popularity[p] = count / total_receipts
            if prices and p in prices:
                self.revenue[p] = count * prices[p]
            else:
                self.revenue[p] = count
        
        return self
    
    def rank_by_popularity(self, candidates, exclude=None):
        if exclude is None:
            exclude = set()
        else:
            exclude = set(exclude)
        
        scores = [(c, self.popularity.get(c, 0)) for c in candidates if c not in exclude]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def rank_by_revenue(self, candidates, exclude=None):
        if exclude is None:
            exclude = set()
        else:
            exclude = set(exclude)
        
        scores = [(c, self.revenue.get(c, 0)) for c in candidates if c not in exclude]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


def hit_at_k(ranked_list, target, k):
    top_k = [item[0] for item in ranked_list[:k]]
    return 1 if target in top_k else 0


def load_and_preprocess(filepath='ap_dataset.csv'):
    print("="*60)
    print("INCARCARE SI PREPROCESARE DATE")
    print("="*60)
    
    df = pd.read_csv(filepath)
    print(f"Date incarcate: {len(df)} linii, {df['id_bon'].nunique()} bonuri")
    
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
    
    all_products = df['retail_product_name'].unique()
    prices = df.groupby('retail_product_name')['SalePriceWithVAT'].mean().to_dict()
    
    sauces = [s for s in SAUCES if s in all_products]
    
    drink_keywords = ['Pepsi', 'Cola', 'Dew', 'Fanta', 'Sprite', 'Aqua', 'Water', 
                      'Limonada', 'Mirinda', 'Twist', 'Can', 'Doze', '0.25L', '0.33L', '0.5L']
    drinks = [p for p in all_products if any(kw in p for kw in drink_keywords)]
    
    side_keywords = ['Fries', 'Potatoes', 'fries', 'potatoes', 'Cartofi']
    sides = [p for p in all_products if any(kw in p.lower() for kw in side_keywords)]
    
    upsell_candidates = list(set(sauces + drinks + sides))
    
    print(f"\nCandidati pentru upsell: {len(upsell_candidates)}")
    
    return receipts, temporal_features, receipt_ids, prices, upsell_candidates, all_products


def train_test_split_receipts(receipts, temporal_features, receipt_ids, test_size=0.2, seed=42):
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


if __name__ == "__main__":
    data = load_and_preprocess('ap_dataset.csv')
    receipts, temporal_features, receipt_ids, prices, upsell_candidates, all_products = data
    
    train_data, test_data = train_test_split_receipts(
        receipts, temporal_features, receipt_ids, test_size=0.2
    )
    train_receipts, train_temporal, train_ids = train_data
    test_receipts, test_temporal, test_ids = test_data
    
    print(f"\nTrain: {len(train_receipts)} bonuri, Test: {len(test_receipts)} bonuri")
    
    print("\n" + "="*60)
    print("ANTRENARE BASELINE")
    print("="*60)
    baseline = PopularityBaseline()
    baseline.fit(train_receipts, prices)
    print("Gata!")
    
    print("\n" + "="*60)
    print("ANTRENARE ID3 (Decision Tree)")
    print("="*60)
    id3_ranker = ID3Ranker(max_depth=5, min_samples_split=10)
    id3_ranker.fit(train_receipts, train_temporal, upsell_candidates)
    
    # Test rapid
    print("\n" + "="*60)
    print("TEST RANKING")
    print("="*60)
    
    sample_receipt = test_receipts[0]
    sample_temporal = test_temporal[0]
    
    print(f"Bon test: {sample_receipt[:5]}...")
    
    ranking = id3_ranker.rank_candidates(sample_receipt, sample_temporal, upsell_candidates, prices)
    print(f"\nTop 5 ID3:")
    for i, (prod, score, prob) in enumerate(ranking[:5], 1):
        print(f"  {i}. {prod}: score={score:.4f} (prob={prob:.4f})")
    
    baseline_ranking = baseline.rank_by_popularity(upsell_candidates, exclude=sample_receipt)
    print(f"\nTop 5 Baseline:")
    for i, (prod, pop) in enumerate(baseline_ranking[:5], 1):
        print(f"  {i}. {prod}: pop={pop:.4f}")
    
    print("\n" + "="*60)
    print("TODO Ziua 8: Evaluare completa + grafice + finalizare")
    print("="*60)
