"""
Artefato 3 (adaptado): treina um classificador de pares (pergunta,
resposta-candidata) sobre os embeddings BERTimbau cacheados, para a tarefa
de SELECAO DE RESPOSTA do MilkQA.

Por que classificacao de pares em vez de classificacao direta: nao ha um
conjunto fixo de "classes" aqui -- o alvo e decidir, para uma pergunta e um
pool de ate 50 respostas candidatas, qual delas e a correta. A abordagem
classica (usada em varios trabalhos de answer selection) e transformar isso
num problema de classificacao binaria por par: (pergunta, candidata) ->
{match, nao-match}. Cada par gera um vetor de features combinando os dois
embeddings no estilo InferSent/SNLI: [q, a, |q-a|, q*a] (3072 dimensoes),
que o MLP consome. Na inferencia, TODAS as candidatas de uma pergunta sao
pontuadas e ranqueadas pela probabilidade de "match"; a de maior pontuacao
e a resposta selecionada.

Metricas: como e um problema de RANQUEAMENTO (nao classes fixas), F1-macro
nao se aplica diretamente -- usamos Accuracy@1 (a resposta certa ficou em
1o lugar?) e MRR -- Mean Reciprocal Rank (1/posicao da resposta certa no
ranking; MRR=1 se sempre acerta em 1o, cai conforme a resposta certa fica
mais atras).

Uso: py model.py
"""
import itertools
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_NEG_TRAIN = 8       # negativos amostrados por positivo, no treino (era 4)
GRID = {
    "hidden_size": [128, 256],
    "dropout": [0.2, 0.4],
    "lr": [1e-3, 1e-4],
}
EPOCHS = 25
PATIENCE = 4
TRAIN_SUBSAMPLE = None  # None = usa as 2307 perguntas de treino inteiras (era 800)
HARD_NEG_FRAC = 0.5     # fracao dos negativos escolhida por hard negative mining (0 = so aleatorio)


class PairMLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def pair_features(q_emb, a_emb):
    return torch.cat([q_emb, a_emb, torch.abs(q_emb - a_emb), q_emb * a_emb], dim=-1)


def build_pairs(split, query_emb, corpus_emb, n_neg, subsample=None,
                 hard_frac=0.0):
    """Monta pares (query, candidata, label) -- 1 positivo + n_neg negativos
    por query.

    hard_frac > 0 ativa HARD NEGATIVE MINING: em vez de amostrar os n_neg
    negativos totalmente ao acaso, uma fracao deles (hard_frac) e escolhida
    entre as candidatas ERRADAS mais parecidas com a pergunta (maior
    similaridade de cosseno no embedding do BERTimbau cru). A ideia: um
    negativo aleatorio entre 49 candidatas costuma ser obviamente errado
    (o MLP aprende pouco com ele); um negativo "dificil" -- uma resposta
    sobre outro assunto mas que compartilha vocabulario veterinario/
    agropecuario com a pergunta -- forca o classificador a aprender a
    distincao fina que realmente importa na hora de rankear as 50
    candidatas do teste. O restante (1 - hard_frac) continua aleatorio,
    para nao perder diversidade e o modelo nao "decorar" so os casos
    dificeis.
    """
    rows = list(split)
    if subsample:
        rng = random.Random(SEED)
        rows = rng.sample(rows, min(subsample, len(rows)))

    n_hard = int(round(n_neg * hard_frac))
    n_random = n_neg - n_hard

    X, y = [], []
    for row in rows:
        qid = row["query-id"]
        pos_id = row["positive-doc-id"]
        candidates = [c for c in row["candidates-ids"] if c != pos_id]

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

        q = query_emb[qid]
        X.append(pair_features(q, corpus_emb[pos_id])); y.append(1)
        for neg_id in negs:
            X.append(pair_features(q, corpus_emb[neg_id])); y.append(0)

    return torch.stack(X), torch.tensor(y, dtype=torch.float32)


def cosine_baseline_eval(split, query_emb, corpus_emb):
    """Baseline sem treino: ranqueia as candidatas so pela similaridade de
    cosseno entre o embedding da pergunta e o da candidata (sem MLP)."""
    acc1, mrr = [], []
    for row in split:
        qid, pos_id, cands = row["query-id"], row["positive-doc-id"], row["candidates-ids"]
        q = torch.nn.functional.normalize(query_emb[qid].unsqueeze(0), dim=-1)
        a = torch.nn.functional.normalize(torch.stack([corpus_emb[c] for c in cands]), dim=-1)
        scores = (a @ q.T).squeeze(-1).numpy()

        order = np.argsort(-scores)
        ranked_ids = [cands[i] for i in order]
        rank = ranked_ids.index(pos_id) + 1

        acc1.append(1.0 if rank == 1 else 0.0)
        mrr.append(1.0 / rank)
    return float(np.mean(acc1)), float(np.mean(mrr))


def rank_eval(model, split, query_emb, corpus_emb):
    """Para cada query, pontua as 50 candidatas e mede Accuracy@1 e MRR."""
    model.eval()
    acc1, mrr = [], []
    with torch.no_grad():
        for row in split:
            qid, pos_id, cands = row["query-id"], row["positive-doc-id"], row["candidates-ids"]
            q = query_emb[qid].unsqueeze(0).expand(len(cands), -1)
            a = torch.stack([corpus_emb[c] for c in cands])
            feats = pair_features(q, a)
            scores = torch.sigmoid(model(feats)).numpy()

            order = np.argsort(-scores)
            ranked_ids = [cands[i] for i in order]
            rank = ranked_ids.index(pos_id) + 1  # 1-indexado

            acc1.append(1.0 if rank == 1 else 0.0)
            mrr.append(1.0 / rank)
    return float(np.mean(acc1)), float(np.mean(mrr))


def train_one(Xtr, ytr, dev_split, query_emb, corpus_emb, hidden, dropout, lr):
    torch.manual_seed(SEED)
    model = PairMLP(Xtr.shape[1], hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

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
            logits = model(Xtr[idx])
            loss = criterion(logits, ytr[idx])
            loss.backward()
            opt.step()

        acc1, mrr = rank_eval(model, dev_split, query_emb, corpus_emb)
        if mrr > best_mrr:
            best_mrr, best_state, no_improve = mrr, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_mrr


def main():
    print("Carregando embeddings cacheados...")
    query_emb = torch.load("cache/queries_embeddings.pt", weights_only=False)
    corpus_emb = torch.load("cache/corpus_embeddings.pt", weights_only=False)
    print(f"  {len(query_emb)} perguntas | {len(corpus_emb)} respostas | dim={next(iter(query_emb.values())).shape[0]}")

    print("Carregando splits oficiais do MilkQA (train/dev/test)...")
    ds = load_dataset("eduagarcia/MilkQA")
    train_split, dev_split, test_split = ds["train"], ds["dev"], ds["test"]
    print(f"  treino={len(train_split)} | dev={len(dev_split)} | teste={len(test_split)}")

    n_desc = TRAIN_SUBSAMPLE if TRAIN_SUBSAMPLE else f"todas ({len(train_split)})"
    print(f"\nMontando pares de treino ({n_desc} perguntas, {N_NEG_TRAIN} negativos cada, "
          f"{HARD_NEG_FRAC:.0%} deles 'dificeis' por similaridade de cosseno)...")
    Xtr, ytr = build_pairs(train_split, query_emb, corpus_emb, N_NEG_TRAIN,
                            subsample=TRAIN_SUBSAMPLE, hard_frac=HARD_NEG_FRAC)
    print(f"  {Xtr.shape[0]} pares ({int(ytr.sum())} positivos, {int((1 - ytr).sum())} negativos)")

    print("\n===== Baseline sem treino (similaridade de cosseno pura) =====")
    base_acc1, base_mrr = cosine_baseline_eval(test_split, query_emb, corpus_emb)
    print(f"  Accuracy@1 = {base_acc1:.4f} | MRR = {base_mrr:.4f}  (embedding BERTimbau cru, sem MLP)")

    print("\n===== Grid Search (selecao pelo MRR no conjunto de dev) =====")
    combos = list(itertools.product(GRID["hidden_size"], GRID["dropout"], GRID["lr"]))
    results = []
    best_overall = {"mrr": -1.0, "model": None, "cfg": None}
    for hidden, dropout, lr in combos:
        model, dev_mrr = train_one(Xtr, ytr, dev_split, query_emb, corpus_emb, hidden, dropout, lr)
        dev_acc1, _ = rank_eval(model, dev_split, query_emb, corpus_emb)
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
    test_acc1, test_mrr = rank_eval(model, test_split, query_emb, corpus_emb)
    print(f"Accuracy@1 (teste, MLP treinado)   = {test_acc1:.4f}")
    print(f"MRR (teste, MLP treinado)          = {test_mrr:.4f}")
    print(f"Accuracy@1 (teste, cosseno cru)    = {base_acc1:.4f}")
    print(f"MRR (teste, cosseno cru)           = {base_mrr:.4f}")
    print(f"Baseline aleatorio esperado: Accuracy@1 ~= {1/50:.4f} | MRR ~= {sum(1/k for k in range(1,51))/50:.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "hidden_size": hidden, "dropout": dropout, "lr": lr,
        "in_dim": Xtr.shape[1], "test_acc1": test_acc1, "test_mrr": test_mrr,
        "cosine_baseline_acc1": base_acc1, "cosine_baseline_mrr": base_mrr,
    }, "checkpoints/best_model.pt")
    pd.DataFrame(results).sort_values("dev_mrr", ascending=False).to_csv(
        "checkpoints/grid_search_results.csv", index=False)
    print("\nCheckpoint e grid search salvos em checkpoints/")


if __name__ == "__main__":
    main()
