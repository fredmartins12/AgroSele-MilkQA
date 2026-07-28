"""
Extracao (com CACHE incremental e retomavel) do vetor [CLS]/pooler do
Cross-Encoder para cada par (pergunta, candidata) necessario nos splits de
treino/dev/teste do MilkQA.

Por que isso e diferente do bi-encoder (../selecao-resposta-milkqa/ e
../selecao-resposta-milkqa-finetune/): la, cada TEXTO (pergunta ou resposta)
e codificado uma unica vez e reaproveitado em varios pares. Aqui, o BERT
recebe a pergunta e a candidata JUNTAS na mesma sequencia
([CLS] pergunta [SEP] candidata [SEP]), entao a codificacao e especifica de
cada PAR -- nao ha como reaproveitar entre pares diferentes, mesmo que
compartilhem a mesma pergunta ou a mesma candidata.

Perfilamento real (profile_crossencoder.py, texto de verdade do MilkQA):
~362ms por par em CPU. Para o conjunto de teste sozinho (300 perguntas x 50
candidatas = 15.000 pares) isso ja da ~1.5h; somando dev (2.500 pares) e
treino, o total fica na casa de ~4h -- por isso a extracao roda uma unica
vez, com CACHE em disco e checkpoint incremental (salva a cada N pares),
para que uma interrupcao no meio nao jogue fora o trabalho ja feito (mesma
licao do fine-tuning: ../selecao-resposta-milkqa-finetune, que sobreviveu a
uma queda do PC via checkpoint por epoca).

O BERT fica CONGELADO aqui (sem fine-tuning) -- o objetivo deste
experimento e isolar o efeito da ARQUITETURA cross-encoder (atencao
conjunta entre pergunta e candidata) em relacao ao bi-encoder, sem misturar
com o efeito de ajustar pesos. Depois de cacheado, o vetor pooler_output de
cada par e usado para treinar uma cabeca de classificacao pequena (rapido,
sobre features fixas) em cross_encoder_model.py.

Uso: py extrair_features_pares.py
"""
import os
import random
import time

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

SEMENTE = 42
random.seed(SEMENTE)

NOME_MODELO = "neuralmind/bert-base-portuguese-cased"
TAMANHO_MAX_TOKENS = 256
TAMANHO_LOTE = 8
N_PERGUNTAS_TREINO = 2307   # mesma escala das outras variantes, pra comparacao justa
N_NEGATIVOS_TREINO = 8

CAMINHO_CACHE = "cache/pair_features_crossencoder.pt"
SALVAR_A_CADA = 500  # pares -- checkpoint incremental


def montar_pares_treino(conjunto, n_perguntas, n_negativos):
    linhas = list(conjunto)
    rng = random.Random(SEMENTE)
    linhas = rng.sample(linhas, min(n_perguntas, len(linhas)))

    pares = []  # (id_pergunta, id_candidata, rotulo)
    for linha in linhas:
        id_pergunta, id_certa = linha["query-id"], linha["positive-doc-id"]
        candidatas = [c for c in linha["candidates-ids"] if c != id_certa]
        negativos = rng.sample(candidatas, min(n_negativos, len(candidatas)))
        pares.append((id_pergunta, id_certa, 1))
        for id_neg in negativos:
            pares.append((id_pergunta, id_neg, 0))
    return pares


def montar_pares_avaliacao(conjunto):
    """Para dev/teste precisamos de TODAS as 50 candidatas de cada pergunta
    (e assim que a avaliacao por ranking funciona)."""
    pares = []
    for linha in conjunto:
        id_pergunta, id_certa = linha["query-id"], linha["positive-doc-id"]
        for id_cand in linha["candidates-ids"]:
            pares.append((id_pergunta, id_cand, 1 if id_cand == id_certa else 0))
    return pares


def main():
    print("Carregando textos (corpus.csv / queries.csv)...")
    corpus_df = pd.read_csv("../selecao-resposta-milkqa-finetune/datasets/corpus.csv")
    queries_df = pd.read_csv("../selecao-resposta-milkqa-finetune/datasets/queries.csv")
    texto_resposta = dict(zip(corpus_df["id"].astype(str), corpus_df["text"]))
    texto_pergunta = dict(zip(queries_df["id"].astype(str), queries_df["text"]))

    print("Carregando splits oficiais do MilkQA...")
    ds = load_dataset("eduagarcia/MilkQA")
    conjunto_treino, conjunto_dev, conjunto_teste = ds["train"], ds["dev"], ds["test"]

    print(f"\nMontando pares necessarios (treino={N_PERGUNTAS_TREINO} perguntas x "
          f"{N_NEGATIVOS_TREINO} negativos, dev e teste completos)...")
    pares_treino = montar_pares_treino(conjunto_treino, N_PERGUNTAS_TREINO, N_NEGATIVOS_TREINO)
    pares_dev = montar_pares_avaliacao(conjunto_dev)
    pares_teste = montar_pares_avaliacao(conjunto_teste)
    todos_os_pares = pares_treino + pares_dev + pares_teste
    print(f"  treino={len(pares_treino)} | dev={len(pares_dev)} | teste={len(pares_teste)} "
          f"| total={len(todos_os_pares)} pares")

    os.makedirs("cache", exist_ok=True)
    if os.path.exists(CAMINHO_CACHE):
        print(f"\nCache parcial encontrado em {CAMINHO_CACHE}, retomando de onde parou...")
        cache = torch.load(CAMINHO_CACHE, weights_only=False)
    else:
        cache = {}
    print(f"  {len(cache)} pares ja no cache")

    pendentes = [(q, c) for q, c, _ in todos_os_pares if f"{q}||{c}" not in cache]
    print(f"  {len(pendentes)} pares pendentes de extracao")

    if not pendentes:
        print("Nada a fazer -- cache ja completo.")
        return

    print(f"\nCarregando {NOME_MODELO} (congelado, so forward)...")
    tokenizador = AutoTokenizer.from_pretrained(NOME_MODELO)
    bert = AutoModel.from_pretrained(NOME_MODELO)
    bert.eval()
    for p in bert.parameters():
        p.requires_grad = False

    t0 = time.time()
    processados_nesta_rodada = 0
    with torch.no_grad():
        for i in range(0, len(pendentes), TAMANHO_LOTE):
            lote = pendentes[i:i + TAMANHO_LOTE]
            lote_perguntas = [texto_pergunta[q] for q, c in lote]
            lote_respostas = [texto_resposta[c] for q, c in lote]

            entrada = tokenizador(
                lote_perguntas, lote_respostas,
                return_tensors="pt", truncation=True,
                max_length=TAMANHO_MAX_TOKENS, padding=True,
            )
            saida = bert(**entrada)
            vetores = saida.pooler_output  # (lote, 768)

            for (q, c), vetor in zip(lote, vetores):
                cache[f"{q}||{c}"] = vetor.clone()
            processados_nesta_rodada += len(lote)

            if processados_nesta_rodada % SALVAR_A_CADA < TAMANHO_LOTE:
                torch.save(cache, CAMINHO_CACHE)
                decorrido = time.time() - t0
                taxa = processados_nesta_rodada / decorrido
                restantes = len(pendentes) - processados_nesta_rodada
                eta_h = (restantes / taxa) / 3600 if taxa > 0 else float("nan")
                print(f"  {len(cache)}/{len(todos_os_pares)} pares no cache "
                      f"({processados_nesta_rodada} nesta rodada, {decorrido/60:.1f}min, "
                      f"ETA restante ~{eta_h:.2f}h)", flush=True)

    torch.save(cache, CAMINHO_CACHE)
    print(f"\nExtracao completa: {len(cache)} pares no cache. "
          f"Tempo total desta rodada: {(time.time() - t0)/3600:.2f}h")


if __name__ == "__main__":
    main()
