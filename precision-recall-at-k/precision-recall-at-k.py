def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    hit = 0

    for rec in recommended[:k]:
            if rec in relevant:
                hit += 1
    
    p = hit / k
    c = hit / len(relevant)
    
    return [p, c]

    