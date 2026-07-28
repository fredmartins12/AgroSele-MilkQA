"""
Treino e avaliacao do Cross-Encoder, em cima do cache de features extraido
por extrair_features_pares.py (vetor pooler_output de 768d para cada par
(pergunta, candidata), com a pergunta e a candidata processadas JUNTAS pelo
BERT -- ao contrario do bi-encoder, aqui nao existe "embedding da pergunta"
e "embedding da resposta" separados, so um vetor que ja representa o PAR
inteiro, calculado com atencao cruzada entre os dois textos.

Como o BERT ficou congelado na extracao, essa etapa e rapida: e só treinar
uma cabeca de classificacao pequena sobre vetores ja prontos (768d, nao
3072d/3074d como nas variantes bi-encoder, porque aqui nao ha q_emb/a_emb
separados pra concatenar/subtrair/multiplicar -- o proprio pooler_output ja
"e" a representacao do par).

Uso: py cross_encoder_model.py (depois que extrair_features_pares.py tiver
terminado e cache/pair_features_crossencoder.pt estiver completo)
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

CAMINHO_CACHE = "cache/pair_features_crossencoder.pt"
N_PERGUNTAS_TREINO = 2307
N_NEGATIVOS_TREINO = 8
GRADE = {
    "ocultas": [64, 128],
    "dropout": [0.2, 0.4],
    "lr": [1e-3, 1e-4],
}
EPOCAS = 30
PACIENCIA = 5


class CabecaCrossEncoder(nn.Module):
    """Cabeca de classificacao sobre o vetor [CLS]/pooler do par ja
    codificado em conjunto pelo BERT (768d -- nao ha q/a separados aqui)."""

    def __init__(self, dim_entrada, ocultas, dropout):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(dim_entrada, ocultas), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ocultas, 1),
        )

    def forward(self, x):
        return self.rede(x).squeeze(-1)


def montar_pares_treino(conjunto, n_perguntas, n_negativos):
    linhas = list(conjunto)
    rng = random.Random(SEED)
    linhas = rng.sample(linhas, min(n_perguntas, len(linhas)))
    pares = []
    for linha in linhas:
        id_pergunta, id_certa = linha["query-id"], linha["positive-doc-id"]
        candidatas = [c for c in linha["candidates-ids"] if c != id_certa]
        negativos = rng.sample(candidatas, min(n_negativos, len(candidatas)))
        pares.append((id_pergunta, id_certa, 1))
        for id_neg in negativos:
            pares.append((id_pergunta, id_neg, 0))
    return pares


def montar_X_y(pares, cache):
    X = torch.stack([cache[f"{q}||{c}"] for q, c, _ in pares])
    y = torch.tensor([r for _, _, r in pares], dtype=torch.float32)
    return X, y


def avaliar_ranking(modelo, conjunto, cache):
    modelo.eval()
    lista_acuracia1, lista_mrr = [], []
    with torch.no_grad():
        for linha in conjunto:
            id_pergunta, id_certa, candidatas = linha["query-id"], linha["positive-doc-id"], linha["candidates-ids"]
            features = torch.stack([cache[f"{id_pergunta}||{c}"] for c in candidatas])
            pontuacoes = torch.sigmoid(modelo(features)).numpy()

            ordem = np.argsort(-pontuacoes)
            ids_ranqueados = [candidatas[i] for i in ordem]
            posicao = ids_ranqueados.index(id_certa) + 1
            lista_acuracia1.append(1.0 if posicao == 1 else 0.0)
            lista_mrr.append(1.0 / posicao)
    return float(np.mean(lista_acuracia1)), float(np.mean(lista_mrr))


def treinar_um_modelo(X_treino, y_treino, conjunto_dev, cache, ocultas, dropout, lr):
    torch.manual_seed(SEED)
    modelo = CabecaCrossEncoder(X_treino.shape[1], ocultas, dropout)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    funcao_perda = nn.BCEWithLogitsLoss()

    melhor_mrr, melhor_estado, sem_melhora = -1.0, None, 0
    n = X_treino.shape[0]
    tamanho_lote = 64
    gerador = torch.Generator().manual_seed(SEED)

    for epoca in range(EPOCAS):
        modelo.train()
        permutacao = torch.randperm(n, generator=gerador)
        for i in range(0, n, tamanho_lote):
            indices = permutacao[i:i + tamanho_lote]
            otimizador.zero_grad()
            logits = modelo(X_treino[indices])
            perda = funcao_perda(logits, y_treino[indices])
            perda.backward()
            otimizador.step()

        _, mrr_dev = avaliar_ranking(modelo, conjunto_dev, cache)
        if mrr_dev > melhor_mrr:
            melhor_mrr, melhor_estado, sem_melhora = mrr_dev, {k: v.clone() for k, v in modelo.state_dict().items()}, 0
        else:
            sem_melhora += 1
            if sem_melhora >= PACIENCIA:
                break

    modelo.load_state_dict(melhor_estado)
    return modelo, melhor_mrr


def main():
    print("Carregando cache de features do cross-encoder...")
    cache = torch.load(CAMINHO_CACHE, weights_only=False)
    print(f"  {len(cache)} pares no cache | dim={next(iter(cache.values())).shape[0]}")

    print("Carregando splits oficiais do MilkQA...")
    ds = load_dataset("eduagarcia/MilkQA")
    conjunto_treino, conjunto_dev, conjunto_teste = ds["train"], ds["dev"], ds["test"]

    print(f"\nMontando pares de treino ({N_PERGUNTAS_TREINO} perguntas, {N_NEGATIVOS_TREINO} negativos)...")
    pares_treino = montar_pares_treino(conjunto_treino, N_PERGUNTAS_TREINO, N_NEGATIVOS_TREINO)
    X_treino, y_treino = montar_X_y(pares_treino, cache)
    print(f"  {X_treino.shape[0]} pares ({int(y_treino.sum())} positivos, {int((1 - y_treino).sum())} negativos)")

    print("\n===== Grid Search (selecao pelo MRR no dev) =====")
    combinacoes = list(itertools.product(GRADE["ocultas"], GRADE["dropout"], GRADE["lr"]))
    resultados = []
    melhor_geral = {"mrr": -1.0, "modelo": None, "config": None}
    for ocultas, dropout, lr in combinacoes:
        modelo, mrr_dev = treinar_um_modelo(X_treino, y_treino, conjunto_dev, cache, ocultas, dropout, lr)
        acuracia1_dev, _ = avaliar_ranking(modelo, conjunto_dev, cache)
        resultados.append({"ocultas": ocultas, "dropout": dropout, "lr": lr,
                            "mrr_dev": mrr_dev, "acuracia1_dev": acuracia1_dev})
        print(f"  ocultas={ocultas:4d} dropout={dropout:.1f} lr={lr:.0e} "
              f"-> dev MRR={mrr_dev:.4f} Acc@1={acuracia1_dev:.4f}")
        if mrr_dev > melhor_geral["mrr"]:
            melhor_geral = {"mrr": mrr_dev, "modelo": modelo, "config": (ocultas, dropout, lr)}

    ocultas, dropout, lr = melhor_geral["config"]
    print(f"\nMelhor configuracao: ocultas={ocultas}, dropout={dropout}, lr={lr} "
          f"(dev MRR={melhor_geral['mrr']:.4f})")

    print("\n===== Avaliacao final no TESTE (300 perguntas, 50 candidatas cada) =====")
    modelo = melhor_geral["modelo"]
    acuracia1_teste, mrr_teste = avaliar_ranking(modelo, conjunto_teste, cache)
    print(f"Accuracy@1 (teste, cross-encoder) = {acuracia1_teste:.4f}")
    print(f"MRR (teste, cross-encoder)        = {mrr_teste:.4f}")
    print("\nComparar com: congelado bi-encoder = 0.570/0.679 | hibrido BERT+BM25 = 0.663/0.753 "
          "| fine-tuning completo = 0.690/0.782")

    torch.save({
        "model_state": modelo.state_dict(),
        "ocultas": ocultas, "dropout": dropout, "lr": lr,
        "dim_entrada": X_treino.shape[1],
        "acuracia1_teste": acuracia1_teste, "mrr_teste": mrr_teste,
    }, "checkpoints/best_model_crossencoder.pt")
    pd.DataFrame(resultados).sort_values("mrr_dev", ascending=False).to_csv(
        "checkpoints/grid_search_results.csv", index=False)
    print("\nCheckpoint e grid search salvos em checkpoints/")


if __name__ == "__main__":
    main()
