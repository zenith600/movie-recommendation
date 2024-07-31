import numpy as np
import pandas as pd
from sklearn.metrics import precision_score

def compute_precision_at_k(recommendations, relevant_items, k):

    top_k_recommendations = recommendations[:k]
    relevant_set = set(relevant_items)
    recommended_set = set(top_k_recommendations)

    # Intersection of recommended and relevant items
    intersection = len(relevant_set.intersection(recommended_set))

    return intersection / k


# Example usage

# Assume these are your lists of recommendations and relevant items
user_recommendations = {
    'user1': {'user_based': ['item1', 'item2', 'item3', 'item4', 'item5'],
              'item_based': ['item1', 'item3', 'item2', 'item4', 'item6'],
              'content_based': ['item2', 'item3', 'item1', 'item5', 'item4'],
              'hybrid': ['item3', 'item2', 'item5', 'item1', 'item4']},
}

relevant_items = {
    'user1': ['item1', 'item2', 'item3'],
}

# k values for evaluation
k_values = [1, 2, 3, 4, 5]

# Dictionary to store precision results
precision_results = {'user_based': [], 'item_based': [], 'content_based': [], 'hybrid': []}

for user, recs in user_recommendations.items():
    relevant = relevant_items.get(user, [])

    for k in k_values:
        for method in precision_results.keys():
            precision = compute_precision_at_k(recs[method], relevant, k)
            precision_results[method].append(precision)

for method in precision_results.keys():
    print(f'{method.replace("_", " ").title()}:')
    for k, precision in zip(k_values, precision_results[method]):
        print(f'  Precision@{k}: {precision:.2f}')
