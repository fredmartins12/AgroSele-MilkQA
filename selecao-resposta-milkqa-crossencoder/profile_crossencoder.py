"""
Perfilamento de custo real do Cross-Encoder ANTES de decidir a escala do
treino -- mesma licao aprendida no experimento de fine-tuning do projeto
irmao (selecao-resposta-milkqa-finetune): perfilar com texto curto/repetido
subestimou o custo real em ~3x. Aqui perfilamos direto com pares reais do
MilkQA (pergunta, resposta), do jeito que vao ser processados de verdade.

Diferenca de custo em relacao ao bi-encoder (congelado ou fine-tuning): no
cross-encoder NAO HA CACHE possivel -- cada par (pergunta, candidata) e uma
sequencia conjunta unica ([CLS] pergunta [SEP] candidata [SEP]), entao o
BERT roda uma vez por PAR, nao uma vez por TEXTO. Para 2307 perguntas de
treino com N negativos, isso significa 2307*(1+N) passadas pelo BERT so no
treino -- bem mais caro que o bi-encoder, que reaproveita a mesma
codificacao de uma resposta em varios pares.

Uso: py profile_crossencoder.py
"""
import time

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

NOME_MODELO = "neuralmind/bert-base-portuguese-cased"
TAMANHO_MAX_TOKENS = 256
N_PARES_TESTE = 32
TAMANHO_LOTE = 8


def main():
    print(f"Carregando {NOME_MODELO}...")
    tokenizador = AutoTokenizer.from_pretrained(NOME_MODELO)
    bert = AutoModel.from_pretrained(NOME_MODELO)
    bert.eval()
    for p in bert.parameters():
        p.requires_grad = False

    print("Carregando textos reais do MilkQA...")
    corpus_df = pd.read_csv("../selecao-resposta-milkqa-finetune/datasets/corpus.csv")
    queries_df = pd.read_csv("../selecao-resposta-milkqa-finetune/datasets/queries.csv")

    perguntas = queries_df["text"].tolist()[:N_PARES_TESTE]
    respostas = corpus_df["text"].tolist()[:N_PARES_TESTE]

    print(f"\nPerfilando {N_PARES_TESTE} pares reais (pergunta, resposta), "
          f"lote={TAMANHO_LOTE}, max_length={TAMANHO_MAX_TOKENS}...")

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N_PARES_TESTE, TAMANHO_LOTE):
            lote_perguntas = perguntas[i:i + TAMANHO_LOTE]
            lote_respostas = respostas[i:i + TAMANHO_LOTE]
            entrada = tokenizador(
                lote_perguntas, lote_respostas,
                return_tensors="pt", truncation=True,
                max_length=TAMANHO_MAX_TOKENS, padding=True,
            )
            saida = bert(**entrada)
            _ = saida.pooler_output  # [CLS] processado, o que a cabeca vai consumir
    tempo_total = time.time() - t0

    tempo_por_par = tempo_total / N_PARES_TESTE
    print(f"\nTempo total: {tempo_total:.1f}s para {N_PARES_TESTE} pares")
    print(f"Tempo por par: {tempo_por_par * 1000:.0f}ms")

    # projecoes de escala pra decidir o tamanho do treino
    print("\n===== Projecoes de tempo (so forward, sem grad -- BERT congelado) =====")
    for n_perguntas, n_neg in [(300, 4), (800, 4), (2307, 4), (2307, 8)]:
        n_pares_treino = n_perguntas * (1 + n_neg)
        n_pares_avaliacao = 50 * 50 + 300 * 50  # dev + teste, sem dedup possivel no cross-encoder
        total_pares = n_pares_treino + n_pares_avaliacao
        horas = (total_pares * tempo_por_par) / 3600
        print(f"  {n_perguntas:5d} perguntas, {n_neg} negativos -> "
              f"{n_pares_treino:6d} pares treino + {n_pares_avaliacao:6d} avaliacao "
              f"= {total_pares:6d} pares -> ~{horas:.2f}h (so extracao de features, sem contar epocas)")


if __name__ == "__main__":
    main()
