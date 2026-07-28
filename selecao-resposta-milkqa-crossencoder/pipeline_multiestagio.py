"""
Pipeline hibrido MULTI-ESTAGIO (retrieval leve + reranking profundo):

  Estagio 1 (recuperacao rapida): BM25 pontua as 50 candidatas de cada
  pergunta e filtra so as top-K mais promissoras. Custo desprezivel
  (contagem de termos, sem rede neural).

  Estagio 2 (reranking caro): o Cross-Encoder -- que precisa rodar o BERT
  inteiro PARA CADA PAR pergunta-candidata, sem cache possivel entre pares
  diferentes -- so e aplicado as K candidatas que sobraram do estagio 1,
  em vez das 50 originais.

Essa e a arquitetura padrao usada em sistemas de busca/recuperacao de
informacao em producao (ex.: um primeiro estagio BM25/embedding barato
filtra milhares de documentos para uma centena, e so essa centena passa por
um reranqueador caro) -- aqui adaptada em escala pequena (50 candidatas por
pergunta) para o MilkQA, mas com o mesmo principio: **nunca rodar o modelo
caro em candidatas obviamente irrelevantes**.

Usa o cache de pares ja extraido por extrair_features_pares.py (que cobre
as 50 candidatas de cada pergunta de dev/teste) -- entao aqui nao rodamos o
BERT de novo, so filtramos quais dessas 50 pontuacoes ja calculadas entram
no reranking (simulando, em termos de ACURACIA, o que aconteceria numa
implantacao real onde o estagio 2 so seria executado para as K candidatas
filtradas, economizando (50-K)/50 das chamadas ao BERT).

Uso: py pipeline_multiestagio.py (depois de cross_encoder_model.py ja ter
treinado e salvo checkpoints/best_model_crossencoder.pt)
"""
import re

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from rank_bm25 import BM25Okapi

from cross_encoder_model import CabecaCrossEncoder

LISTA_STOPWORDS_PT = set("""
a ao aos aquela aquelas aquele aqueles aquilo as até com como da das de dele
deles depois do dos e ela elas ele eles em entre era eram essa essas esse
esses esta estamos estas estava estavam este esteja estejam estejamos estes
esteve estive estivemos estiveram estivesse estivessem estivéramos
estivéssemos estou está estávamos estão eu foi fomos for fora foram forem
formos fosse fossem fui fôramos fôssemos haja hajam hajamos havemos hei
houve houvemos houver houvera houveram houverei houverem houveremos
houveria houveriam houvermos houverá houverão houveríamos houvesse
houvessem houvéramos houvéssemos há hão isso isto já lhe lhes mais mas me
mesmo meu meus minha minhas muito na nas nem no nos nossa nossas nosso
nossos num numa não nós o os ou para pela pelas pelo pelos por qual quando
que quem se seja sejam sejamos sem serei seremos seria seriam será serão
seríamos seu seus somos sou sua suas são só também te tem temos tenha
tenham tenhamos tenho terei teremos teria teriam terá terão teríamos teu
teus teve tinha tinham tive tivemos tiver tivera tiveram tiverem tivermos
tivesse tivessem tivéramos tivéssemos tu tua tuas tá um uma você vocês
vos à às éramos é
""".split())
_padrao_token = re.compile(r"[a-zà-öø-ÿ0-9]+")


def tokenizar(texto):
    palavras = _padrao_token.findall(texto.lower())
    return [p for p in palavras if p not in LISTA_STOPWORDS_PT and len(p) > 1]


def carregar_csv(caminho):
    linhas = {}
    import csv
    with open(caminho, encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            linhas[linha["id"]] = linha["text"]
    return linhas


def avaliar_pipeline(conjunto, bm25, indice_do_id, cache, modelo_crossencoder, k):
    """Estagio 1: BM25 escolhe as top-k candidatas. Estagio 2: cross-encoder
    reranqueia so essas k (as demais ficam nas posicoes finais do ranking,
    ordenadas pelo proprio BM25, ja que nunca foram avaliadas pelo estagio 2)."""
    modelo_crossencoder.eval()
    lista_acuracia1, lista_mrr = [], []
    chamadas_crossencoder = 0

    with torch.no_grad():
        for linha in conjunto:
            id_pergunta, id_certa, candidatas = linha["query-id"], linha["positive-doc-id"], linha["candidates-ids"]

            escores_bm25 = bm25.get_scores(tokenizar(_texto_pergunta[id_pergunta]))
            candidatas_ordenadas_bm25 = sorted(
                candidatas, key=lambda c: -escores_bm25[indice_do_id[c]])

            top_k = candidatas_ordenadas_bm25[:k]
            resto = candidatas_ordenadas_bm25[k:]  # nao passam pelo cross-encoder

            features_top_k = torch.stack([cache[f"{id_pergunta}||{c}"] for c in top_k])
            pontuacoes_top_k = torch.sigmoid(modelo_crossencoder(features_top_k)).numpy()
            chamadas_crossencoder += len(top_k)

            ordem_top_k = np.argsort(-pontuacoes_top_k)
            ranking_final = [top_k[i] for i in ordem_top_k] + resto

            posicao = ranking_final.index(id_certa) + 1
            lista_acuracia1.append(1.0 if posicao == 1 else 0.0)
            lista_mrr.append(1.0 / posicao)

    return float(np.mean(lista_acuracia1)), float(np.mean(lista_mrr)), chamadas_crossencoder


def main():
    global _texto_pergunta
    print("Carregando textos e construindo indice BM25...")
    textos_respostas = carregar_csv("../selecao-resposta-milkqa-finetune/datasets/corpus.csv")
    _texto_pergunta = carregar_csv("../selecao-resposta-milkqa-finetune/datasets/queries.csv")

    ids_documentos = list(textos_respostas.keys())
    indice_do_id = {id_doc: i for i, id_doc in enumerate(ids_documentos)}
    corpus_tokenizado = [tokenizar(t) for t in textos_respostas.values()]
    bm25 = BM25Okapi(corpus_tokenizado)

    print("Carregando cache de pares do cross-encoder...")
    cache = torch.load("cache/pair_features_crossencoder.pt", weights_only=False)

    print("Carregando cross-encoder treinado...")
    checkpoint = torch.load("checkpoints/best_model_crossencoder.pt", weights_only=False)
    modelo = CabecaCrossEncoder(checkpoint["dim_entrada"], checkpoint["ocultas"], checkpoint["dropout"])
    modelo.load_state_dict(checkpoint["model_state"])

    print("Carregando splits oficiais do MilkQA...")
    ds = load_dataset("eduagarcia/MilkQA")
    conjunto_teste = ds["test"]

    print(f"\n===== Pipeline multi-estagio (BM25 filtra top-K -> cross-encoder reranqueia) =====")
    print(f"Baseline: cross-encoder puro sobre as 50 candidatas -> "
          f"Acc@1={checkpoint['acuracia1_teste']:.4f} MRR={checkpoint['mrr_teste']:.4f} "
          f"(50 chamadas ao BERT por pergunta)\n")

    resultados = []
    for k in [3, 5, 10, 20, 50]:
        acuracia1, mrr, chamadas = avaliar_pipeline(
            conjunto_teste, bm25, indice_do_id, cache, modelo, k)
        economia = 1 - (k / 50)
        print(f"  K={k:2d} -> Acc@1={acuracia1:.4f} MRR={mrr:.4f} "
              f"| chamadas ao cross-encoder: {chamadas} (media {chamadas/len(conjunto_teste):.0f}/pergunta) "
              f"| economia de {economia:.0%} sobre rodar em todas as 50")
        resultados.append({"k": k, "acuracia1": acuracia1, "mrr": mrr, "economia_chamadas": economia})

    pd.DataFrame(resultados).to_csv("checkpoints/resultados_pipeline_multiestagio.csv", index=False)
    print("\nResultados salvos em checkpoints/resultados_pipeline_multiestagio.csv")


if __name__ == "__main__":
    main()
