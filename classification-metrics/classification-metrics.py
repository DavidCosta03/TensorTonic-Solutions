import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    if len(y_true) != len(y_pred):
        raise ValueError("Inputs must be equal length")
    n = len(y_true)
    if n == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    classes = sorted(set(y_true) | set(y_pred))
    tp = {c: 0 for c in classes}
    fp = {c: 0 for c in classes}
    fn = {c: 0 for c in classes}
    support = {c: 0 for c in classes}

    correct = 0
    for t, p in zip(y_true, y_pred):
        support[t] += 1
        if t == p:
            tp[t] += 1
            correct += 1
        else:
            fp[p] += 1
            fn[t] += 1

    accuracy = correct / n

    def prf(tps, fps, fns):
        prec = tps / (tps + fps) if (tps + fps) else 0.0
        rec = tps / (tps + fns) if (tps + fns) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f1

    if average == "micro":
        TP = sum(tp.values()); FP = sum(fp.values()); FN = sum(fn.values())
        precision, recall, f1 = prf(TP, FP, FN)

    elif average == "macro":
        ps, rs, fs = [], [], []
        for c in classes:
            p, r, f = prf(tp[c], fp[c], fn[c])
            ps.append(p); rs.append(r); fs.append(f)
        k = len(classes)
        precision = sum(ps) / k; recall = sum(rs) / k; f1 = sum(fs) / k

    elif average == "weighted":
        precision = recall = f1 = 0.0
        for c in classes:
            p, r, f = prf(tp[c], fp[c], fn[c])
            w = support[c] / n
            precision += p * w; recall += r * w; f1 += f * w

    elif average == "binary":
        precision, recall, f1 = prf(
            tp.get(pos_label, 0), fp.get(pos_label, 0), fn.get(pos_label, 0)
        )
    else:
        raise ValueError(f"Unknown average mode: {average}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}