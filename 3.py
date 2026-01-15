# Task 3 - Ranking pentru Upsell
# Score(produs | cos) = P(produs | cos) * pret(produs)
# Folosim ID3 (arbore de decizie) pentru a estima probabilitatea

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        # conditii de oprire
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return {'leaf': True, 'prob': np.mean(y) if len(y) > 0 else 0.5}
        
        if len(np.unique(y)) == 1:
            return {'leaf': True, 'prob': y[0]}
        
        # gasim cel mai bun feature
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
        
        # prezenta fiecarui produs
        for p in all_products:
            features.append(1 if p in receipt_set else 0)
        
        # features temporale
        features.append(temporal['day_of_week'])
        features.append(temporal['is_weekend'])
        
        # bucket pentru ora
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


def mrr(ranked_list, target):
    for i, (item, _, _) in enumerate(ranked_list, 1):
        if item == target:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_list, target, k):
    for i, (item, _, _) in enumerate(ranked_list[:k], 1):
        if item == target:
            return 1.0 / np.log2(i + 1)
    return 0.0


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
    
    # identificam categoriile pentru upsell
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


def evaluate_ranker(ranker, baseline, test_data, candidates, prices, k_values=[1, 3, 5]):
    """Evaluare leave-one-out: scoatem un produs si vedem daca il recomandam inapoi."""
    test_receipts, test_temporal, test_ids = test_data
    
    results = {
        'model': {k: {'hit': [], 'mrr': [], 'ndcg': []} for k in k_values},
        'baseline_pop': {k: {'hit': [], 'mrr': [], 'ndcg': []} for k in k_values},
        'baseline_rev': {k: {'hit': [], 'mrr': [], 'ndcg': []} for k in k_values}
    }
    
    n_evaluated = 0
    
    for receipt, temporal in zip(test_receipts, test_temporal):
        candidates_in_receipt = [p for p in receipt if p in candidates]
        
        if len(candidates_in_receipt) == 0:
            continue
        
        for target in candidates_in_receipt:
            partial_cart = [p for p in receipt if p != target]
            
            if len(partial_cart) == 0:
                continue
            
            recommendation_candidates = [c for c in candidates if c not in partial_cart]
            
            model_ranking = ranker.rank_candidates(
                partial_cart, temporal, recommendation_candidates, prices
            )
            
            baseline_pop = [(c, p) for c, p in baseline.rank_by_popularity(
                recommendation_candidates, exclude=partial_cart
            )]
            baseline_pop = [(c, s, s) for c, s in baseline_pop]
            
            baseline_rev = [(c, r) for c, r in baseline.rank_by_revenue(
                recommendation_candidates, exclude=partial_cart
            )]
            baseline_rev = [(c, s, s) for c, s in baseline_rev]
            
            for k in k_values:
                results['model'][k]['hit'].append(hit_at_k(model_ranking, target, k))
                results['model'][k]['mrr'].append(mrr(model_ranking, target))
                results['model'][k]['ndcg'].append(ndcg_at_k(model_ranking, target, k))
                
                results['baseline_pop'][k]['hit'].append(hit_at_k(baseline_pop, target, k))
                results['baseline_pop'][k]['mrr'].append(mrr(baseline_pop, target))
                results['baseline_pop'][k]['ndcg'].append(ndcg_at_k(baseline_pop, target, k))
                
                results['baseline_rev'][k]['hit'].append(hit_at_k(baseline_rev, target, k))
                results['baseline_rev'][k]['mrr'].append(mrr(baseline_rev, target))
                results['baseline_rev'][k]['ndcg'].append(ndcg_at_k(baseline_rev, target, k))
            
            n_evaluated += 1
    
    print(f"\n  Evaluari leave-one-out: {n_evaluated}")
    
    return results


def print_results(results, k_values, title):
    print(f"\n{'-'*70}")
    print(f"{title}")
    print(f"{'-'*70}")
    print(f"{'Metoda':<25} {'Metrica':<12} {'K=1':>10} {'K=3':>10} {'K=5':>10}")
    print("-"*70)
    
    methods = [
        ('model', 'ID3'),
        ('baseline_pop', 'Baseline (Popularitate)'),
        ('baseline_rev', 'Baseline (Venit)')
    ]
    
    for method_key, method_name in methods:
        for metric in ['hit', 'mrr', 'ndcg']:
            vals = [np.mean(results[method_key][k][metric]) for k in k_values]
            metric_name = f"{metric.upper()}@K" if metric != 'mrr' else "MRR"
            print(f"{method_name:<25} {metric_name:<12} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f}")
        print()


def plot_comparison(results, k_values=[1, 3, 5]):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['hit', 'mrr', 'ndcg']
    titles = ['Hit@K', 'MRR', 'NDCG@K']
    
    colors = ['steelblue', 'coral', 'seagreen']
    
    for ax, metric, title in zip(axes, metrics, titles):
        x = np.arange(len(k_values))
        width = 0.25
        
        vals_model = [np.mean(results['model'][k][metric]) for k in k_values]
        vals_pop = [np.mean(results['baseline_pop'][k][metric]) for k in k_values]
        vals_rev = [np.mean(results['baseline_rev'][k][metric]) for k in k_values]
        
        ax.bar(x - width, vals_model, width, label='ID3', color=colors[0])
        ax.bar(x, vals_pop, width, label='Baseline (Pop)', color=colors[1])
        ax.bar(x + width, vals_rev, width, label='Baseline (Rev)', color=colors[2])
        
        ax.set_xlabel('K')
        ax.set_ylabel(title)
        ax.set_title(f'{title} - Comparatie')
        ax.set_xticks(x)
        ax.set_xticklabels([f'K={k}' for k in k_values])
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('3_ranking_comparison.png', dpi=150)
    plt.close()
    print("Salvat: 3_ranking_comparison.png")


def plot_score_distribution(ranker, test_data, candidates, prices):
    test_receipts, test_temporal, _ = test_data
    
    probs = []
    scores = []
    
    np.random.seed(42)
    n_samples = min(100, len(test_receipts))
    sampled_idx = np.random.choice(len(test_receipts), n_samples, replace=False)
    
    for i in sampled_idx:
        receipt = test_receipts[i]
        temporal = test_temporal[i]
        ranking = ranker.rank_candidates(receipt, temporal, candidates, prices)
        for prod, score, prob in ranking[:5]:
            probs.append(prob)
            scores.append(score)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(probs, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('P(produs | cos, timp)')
    ax1.set_ylabel('Frecventa')
    ax1.set_title('ID3 - Distributia probabilitatilor')
    
    ax2.hist(scores, bins=30, color='forestgreen', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Score = P x Pret')
    ax2.set_ylabel('Frecventa')
    ax2.set_title('ID3 - Distributia scorurilor')
    
    plt.tight_layout()
    plt.savefig('3_score_distribution_id3.png', dpi=150)
    plt.close()
    print("Salvat: 3_score_distribution_id3.png")


def show_examples(ranker, baseline, test_data, candidates, prices, n_examples=5):
    test_receipts, test_temporal, _ = test_data
    
    print(f"\n{'='*60}")
    print("EXEMPLE DE RANKING - ID3")
    print("="*60)
    
    np.random.seed(456)
    valid_indices = [i for i, r in enumerate(test_receipts) 
                     if any(p in candidates for p in r)]
    sampled = np.random.choice(valid_indices, min(n_examples, len(valid_indices)), replace=False)
    
    for idx, i in enumerate(sampled, 1):
        receipt = test_receipts[i]
        temporal = test_temporal[i]
        
        candidates_in = [p for p in receipt if p in candidates]
        if len(candidates_in) == 0:
            continue
        
        target = candidates_in[0]
        partial_cart = [p for p in receipt if p != target]
        rec_candidates = [c for c in candidates if c not in partial_cart]
        
        model_ranking = ranker.rank_candidates(partial_cart, temporal, rec_candidates, prices)
        baseline_pop = baseline.rank_by_popularity(rec_candidates, exclude=partial_cart)
        
        print(f"\n--- Exemplu {idx} ---")
        print(f"Context temporal: zi={temporal['day_of_week']}, ora={temporal['hour']}, weekend={temporal['is_weekend']}")
        print(f"Cos partial: {partial_cart[:4]}{'...' if len(partial_cart) > 4 else ''}")
        print(f"Target (scos): {target}")
        
        print(f"\nTop-5 ID3 (Score = P x Pret):")
        for j, (prod, score, prob) in enumerate(model_ranking[:5], 1):
            hit = "CORECT!" if prod == target else ""
            print(f"  {j}. {prod}: score={score:.4f} (prob={prob:.4f}, pret={prices.get(prod, 0):.2f}) {hit}")
        
        print(f"\nTop-5 Baseline (Popularitate):")
        for j, (prod, pop) in enumerate(baseline_pop[:5], 1):
            hit = "CORECT!" if prod == target else ""
            print(f"  {j}. {prod}: pop={pop:.4f} {hit}")


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
    
    k_values = [1, 3, 5]
    
    print("\n" + "="*60)
    print("EVALUARE ID3")
    print("="*60)
    results = evaluate_ranker(id3_ranker, baseline, test_data, upsell_candidates, prices, k_values)
    print_results(results, k_values, "Rezultate - ID3 vs Baseline-uri")
    
    print("\n" + "="*60)
    print("GENERARE GRAFICE")
    print("="*60)
    plot_comparison(results, k_values)
    plot_score_distribution(id3_ranker, test_data, upsell_candidates, prices)
    
    show_examples(id3_ranker, baseline, test_data, upsell_candidates, prices, n_examples=3)
    
    print("\n" + "="*60)
    print("TASK 3 COMPLET!")
    print("="*60)
    print("Fisiere generate:")
    print("  - 3_ranking_comparison.png")
    print("  - 3_score_distribution_id3.png")
    
    print("\nSUMAR PERFORMANTA (Hit@K):")
    print("-"*60)
    print(f"{'Model':<25} {'Hit@1':>10} {'Hit@3':>10} {'Hit@5':>10}")
    print("-"*60)
    
    h1 = np.mean(results['model'][1]['hit'])
    h3 = np.mean(results['model'][3]['hit'])
    h5 = np.mean(results['model'][5]['hit'])
    print(f"{'ID3':<25} {h1:>10.4f} {h3:>10.4f} {h5:>10.4f}")
    
    h1 = np.mean(results['baseline_pop'][1]['hit'])
    h3 = np.mean(results['baseline_pop'][3]['hit'])
    h5 = np.mean(results['baseline_pop'][5]['hit'])
    print(f"{'Baseline (Popularitate)':<25} {h1:>10.4f} {h3:>10.4f} {h5:>10.4f}")
    
    h1 = np.mean(results['baseline_rev'][1]['hit'])
    h3 = np.mean(results['baseline_rev'][3]['hit'])
    h5 = np.mean(results['baseline_rev'][5]['hit'])
    print(f"{'Baseline (Venit)':<25} {h1:>10.4f} {h3:>10.4f} {h5:>10.4f}")
