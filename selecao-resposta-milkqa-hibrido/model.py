"""
Modelo HIBRIDO de selecao de resposta para o MilkQA: funde um sinal
lexical classico (BM25, via rank_bm25) com o vetor denso de pares do
BERTimbau (embeddings congelados, cacheados pelo projeto base), e treina
uma Rede Neural (MLP) em PyTorch, com camada de saida Sigmoid explicita,
sobre esse vetor enriquecido.

Arquitetura (4 camadas, ver especificacao do projeto):
  1. PLN classico       -> classical_features.BM25Layer   (score BM25)
  2. Matematica do BERT -> bert_math.{cosine_similarity, absolute_difference,
                                       elementwise_product, pair_features}
  3. Feature Fusion     -> feature_fusion() (neste arquivo)
  4. Rede Neural (MLP)  -> HybridMLP (neste arquivo), PyTorch puro

Vetor enriquecido = [q, a, |q-a|, q*a] (3072d, BERTimbau) ++
                     [cos(q,a)] (1d, BERTimbau) ++
                     [BM25 normalizado] (1d, PLN classico)
                   = 3074 dimensoes.

Metricas: Accuracy@1 e MRR sobre os splits oficiais do MilkQA (identico as
outras variantes do projeto, para comparacao direta).

Uso: py model.py
"""
import itertools
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset

from bert_math import cosine_similarity, pair_features
from classical_features import BM25Layer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_NEG_TRAIN = 8
GRID = {
    "hidden_size": [128, 256],
    "dropout": [0.2, 0.4],
    "lr": [1e-3, 1e-4],
}
EPOCHS = 25
PATIENCE = 4
HARD_NEG_FRAC = 0.5  # mesma estrategia de hard negative mining do projeto base


# ---------------------------------------------------------------------
# 3. Feature Fusion -- funde o vetor denso do BERT com sinais escalares
#    (cosseno do BERT + score BM25 do PLN classico)
# ---------------------------------------------------------------------

def feature_fusion(q_emb, a_emb, bm25_score):
    """Enriquece o vetor de pares do BERT (3072d) concatenando dois scores
    escalares: a similaridade de cosseno explicita entre os embeddings
    (sinal semantico) e o score BM25 normalizado da camada classica (sinal
    lexical). Resultado: vetor unico de 3074 dimensoes.

    Funciona tanto para um unico par (q_emb, a_emb com forma (768,),
    bm25_score escalar) quanto para um lote (forma (N, 768), bm25_score
    com forma (N,)), usado na avaliacao por ranqueamento.
    """
    dense = pair_features(q_emb, a_emb)                       # (..., 3072)
    cos = cosine_similarity(q_emb, a_emb)                     # (...,)
    extra = torch.stack([cos, bm25_score], dim=-1)            # (..., 2)
    return torch.cat([dense, extra], dim=-1)                  # (..., 3074)


# ---------------------------------------------------------------------
# 4. Rede Neural (MLP) em PyTorch -- camadas explicitas + Sigmoid final
# ---------------------------------------------------------------------

class HybridMLP(nn.Module):
    """Perceptron multicamadas para classificacao binaria (match / nao-match)
    do par (pergunta, candidata), a partir do vetor enriquecido (BERT + PLN
    classico). Camadas lineares (nn.Linear) intercaladas com ReLU e Dropout;
    saida final passa por nn.Sigmoid explicito, produzindo diretamente uma
    probabilidade em [0, 1] (por isso o treino usa nn.BCELoss, nao
    BCEWithLogitsLoss -- o logit ja foi convertido a probabilidade dentro do
    modelo)."""

    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------
# Montagem de pares (treino) e avaliacao (ranqueamento), com fusao de
# features aplicada em cada par -- espelha model.py do projeto base, mas
# usando feature_fusion() em vez de pair_features() puro.
# ---------------------------------------------------------------------

def build_pairs(split, query_texts, query_emb, corpus_emb, bm25, n_neg,
                 hard_frac=0.0):
    rows = list(split)
    n_hard = int(round(n_neg * hard_frac))
    n_random = n_neg - n_hard

    X, y = [], []
    for row in rows:
        qid = row["query-id"]
        pos_id = row["positive-doc-id"]
        pool = row["candidates-ids"]
        candidates = [c for c in pool if c != pos_id]

        if n_hard > 0 and len(candidates) > n_hard:
            q = torch.nn.functional.normalize(query_emb[qid].unsqueeze(0), dim=-1)
            cand_embs = torch.nn.functional.normalize(
                torch.stack([corpus_emb[c] for c in candidates]), dim=-1)
            sims = (cand_embs @ q.T).squeeze(-1)
            hard_idx = torch.argsort(-sims)[:n_hard].tolist()
            hard_negs = [candidates[i] for i in hard_idx]
            remaining = [c for c in candidates if c not in hard_negs]
            random_negs = random.sample(remaining, min(n_random, len(remaining)))
            negs = hard_negs + random_negs
        else:
            negs = random.sample(candidates, min(n_neg, len(candidates)))

        q_text = query_texts[qid]
        bm25_scores = bm25.normalized_scores_for_pool(qid, q_text, pool)
        q = query_emb[qid]

        X.append(feature_fusion(q, corpus_emb[pos_id],
                                 torch.tensor(bm25_scores[pos_id])))
        y.append(1)
        for neg_id in negs:
            X.append(feature_fusion(q, corpus_emb[neg_id],
                                     torch.tensor(bm25_scores[neg_id])))
            y.append(0)

    return torch.stack(X), torch.tensor(y, dtype=torch.float32)


def rank_eval(model, split, query_texts, query_emb, corpus_emb, bm25):
    model.eval()
    acc1, mrr = [], []
    with torch.no_grad():
        for row in split:
            qid, pos_id, cands = row["query-id"], row["positive-doc-id"], row["candidates-ids"]
            q_text = query_texts[qid]
            bm25_scores = bm25.normalized_scores_for_pool(qid, q_text, cands)

            q = query_emb[qid].unsqueeze(0).expand(len(cands), -1)
            a = torch.stack([corpus_emb[c] for c in cands])
            bm25_vec = torch.tensor([bm25_scores[c] for c in cands], dtype=torch.float32)

            feats = feature_fusion(q, a, bm25_vec)
            scores = model(feats).numpy()  # ja e probabilidade (Sigmoid interno)

            order = np.argsort(-scores)
            ranked_ids = [cands[i] for i in order]
            rank = ranked_ids.index(pos_id) + 1

            acc1.append(1.0 if rank == 1 else 0.0)
            mrr.append(1.0 / rank)
    return float(np.mean(acc1)), float(np.mean(mrr))


def train_one(Xtr, ytr, dev_split, query_texts, query_emb, corpus_emb, bm25,
              hidden, dropout, lr):
    torch.manual_seed(SEED)
    model = HybridMLP(Xtr.shape[1], hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()  # modelo ja retorna probabilidade (Sigmoid interno)

    best_mrr, best_state, no_improve = -1.0, None, 0
    n = Xtr.shape[0]
    bs = 64
    g = torch.Generator().manual_seed(SEED)

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            probs = model(Xtr[idx])
            loss = criterion(probs, ytr[idx])
            loss.backward()
            opt.step()

        acc1, mrr = rank_eval(model, dev_split, query_texts, query_emb, corpus_emb, bm25)
        if mrr > best_mrr:
            best_mrr, best_state, no_improve = mrr, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_mrr


def main():
    print("Carregando embeddings BERTimbau cacheados (congelados)...")
    query_emb = torch.load("../selecao-resposta-milkqa/cache/queries_embeddings.pt", weights_only=False)
    corpus_emb = torch.load("../selecao-resposta-milkqa/cache/corpus_embeddings.pt", weights_only=False)
    print(f"  {len(query_emb)} perguntas | {len(corpus_emb)} respostas | dim={next(iter(query_emb.values())).shape[0]}")

    print("Carregando textos (para tokenizacao/BM25)...")
    import csv

    def load_csv(path):
        rows = {}
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[row["id"]] = row["text"]
        return rows

    corpus_texts = load_csv("../selecao-resposta-milkqa/datasets/corpus.csv")
    query_texts = load_csv("../selecao-resposta-milkqa/datasets/queries.csv")

    print("Construindo indice BM25 (rank_bm25.BM25Okapi) sobre o corpus...")
    bm25 = BM25Layer(list(corpus_texts.keys()), list(corpus_texts.values()))

    print("Carregando splits oficiais do MilkQA (train/dev/test)...")
    ds = load_dataset("eduagarcia/MilkQA")
    train_split, dev_split, test_split = ds["train"], ds["dev"], ds["test"]
    print(f"  treino={len(train_split)} | dev={len(dev_split)} | teste={len(test_split)}")

    print(f"\nMontando pares de treino (todas as {len(train_split)} perguntas, "
          f"{N_NEG_TRAIN} negativos cada, {HARD_NEG_FRAC:.0%} deles 'dificeis', "
          f"vetor enriquecido BERT+BM25)...")
    Xtr, ytr = build_pairs(train_split, query_texts, query_emb, corpus_emb, bm25,
                            N_NEG_TRAIN, hard_frac=HARD_NEG_FRAC)
    print(f"  {Xtr.shape[0]} pares ({int(ytr.sum())} positivos, {int((1 - ytr).sum())} negativos) "
          f"| dim do vetor enriquecido = {Xtr.shape[1]}")

    print("\n===== Grid Search (selecao pelo MRR no conjunto de dev) =====")
    combos = list(itertools.product(GRID["hidden_size"], GRID["dropout"], GRID["lr"]))
    results = []
    best_overall = {"mrr": -1.0, "model": None, "cfg": None}
    for hidden, dropout, lr in combos:
        model, dev_mrr = train_one(Xtr, ytr, dev_split, query_texts, query_emb,
                                    corpus_emb, bm25, hidden, dropout, lr)
        dev_acc1, _ = rank_eval(model, dev_split, query_texts, query_emb, corpus_emb, bm25)
        results.append({"hidden_size": hidden, "dropout": dropout, "lr": lr,
                         "dev_mrr": dev_mrr, "dev_acc1": dev_acc1})
        print(f"  hidden={hidden:4d} dropout={dropout:.1f} lr={lr:.0e} "
              f"-> dev MRR={dev_mrr:.4f} Acc@1={dev_acc1:.4f}")
        if dev_mrr > best_overall["mrr"]:
            best_overall = {"mrr": dev_mrr, "model": model, "cfg": (hidden, dropout, lr)}

    hidden, dropout, lr = best_overall["cfg"]
    print(f"\nMelhor configuracao: hidden={hidden}, dropout={dropout}, lr={lr} "
          f"(dev MRR={best_overall['mrr']:.4f})")

    print("\n===== Avaliacao final no conjunto de TESTE (300 perguntas, 50 candidatas cada) =====")
    model = best_overall["model"]
    test_acc1, test_mrr = rank_eval(model, test_split, query_texts, query_emb, corpus_emb, bm25)
    print(f"Accuracy@1 (teste, hibrido BERT+BM25+MLP) = {test_acc1:.4f}")
    print(f"MRR (teste, hibrido BERT+BM25+MLP)        = {test_mrr:.4f}")
    print(f"Baseline aleatorio esperado: Accuracy@1 ~= {1/50:.4f} | MRR ~= {sum(1/k for k in range(1,51))/50:.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "hidden_size": hidden, "dropout": dropout, "lr": lr,
        "in_dim": Xtr.shape[1], "test_acc1": test_acc1, "test_mrr": test_mrr,
    }, "checkpoints/best_model_hibrido.pt")
    pd.DataFrame(results).sort_values("dev_mrr", ascending=False).to_csv(
        "checkpoints/grid_search_results.csv", index=False)
    print("\nCheckpoint e grid search salvos em checkpoints/")


if __name__ == "__main__":
    main()
