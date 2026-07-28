# Seleção de Resposta em PLN Agropecuário — MilkQA (variante: Cross-Encoder + Pipeline Multi-Estágio)

Quinta e sexta variantes do projeto [`AgroSele`](../selecao-resposta-milkqa/):
até aqui, todas as arquiteturas (congelada, fine-tuning, TF-IDF, híbrida)
eram **bi-encoder** — pergunta e resposta viram embeddings separados,
combinados depois. Esta pasta implementa um **Cross-Encoder** de verdade
(pergunta e candidata processadas *juntas* pelo BERT, com atenção cruzada
entre as duas) e um **pipeline multi-estágio** (BM25 filtra rápido, o
Cross-Encoder reranqueia só o que sobrou) — as duas lacunas identificadas
numa revisão externa do projeto (Gemini) que o bi-encoder, por construção,
não cobre.

## Por que Cross-Encoder é diferente

No bi-encoder, `emb(pergunta)` e `emb(candidata)` são calculados de forma
totalmente independente — o BERT nunca "vê" os dois textos ao mesmo tempo.
No Cross-Encoder, a entrada do BERT é a sequência conjunta
`[CLS] pergunta [SEP] candidata [SEP]`, processada com **atenção cruzada**:
cada palavra da pergunta pode atender diretamente a cada palavra da
candidata (e vice-versa) em todas as camadas do Transformer, antes de
qualquer decisão de classificação. Isso costuma dar mais qualidade, ao
custo de não poder cachear/reaproveitar embeddings entre pares diferentes
— cada par exige uma passada nova pelo BERT.

## Arquitetura

```text
CROSS-ENCODER (Nível 4)
  pergunta + candidata -> [CLS] pergunta [SEP] candidata [SEP]
    -> BERTimbau (congelado) -> pooler_output (768d, já representa o PAR)
    -> cabeça de classificação (Linear -> ReLU -> Dropout -> Linear)
    -> match / não-match

PIPELINE MULTI-ESTÁGIO (Nível 5)
  pergunta + 50 candidatas
    -> Estágio 1 (BM25, barato): filtra as top-K candidatas
    -> Estágio 2 (Cross-Encoder, caro): reranqueia só as K filtradas
    -> ranking final (as 50-K restantes ficam atrás, na ordem do BM25)
```

## Por que o BERT ficou congelado aqui

Fine-tunar um Cross-Encoder de verdade (backprop através do BERT, pra cada
par) seria proibitivo em CPU: perfilamento real (`profile_crossencoder.py`)
mediu ~362ms por par só no forward. Congelando o BERT, isolamos o efeito da
**arquitetura** (atenção cruzada) do efeito de ajustar pesos — comparável,
em espírito, à variante congelada do bi-encoder. Mesmo assim, como não há
cache possível entre pares diferentes, a extração de features (uma vez,
depois reaproveitada) ainda levou ~2,25h em CPU para os 38.263 pares
necessários (treino + dev + teste completos) — ver `extrair_features_pares.py`,
com checkpoint incremental a cada 500 pares (mesma lição de resiliência do
projeto de fine-tuning).

## Resultado — Cross-Encoder

| Método | Accuracy@1 (teste, 300) | MRR (teste) |
|---|---:|---:|
| Aleatório | 0,020 | 0,090 |
| Cosseno BERTimbau puro (bi-encoder, sem treino) | 0,277 | 0,392 |
| TF-IDF à mão (sem BERT) | 0,503 | 0,610 |
| Bi-encoder congelado + MLP | 0,570 | 0,679 |
| **Cross-Encoder congelado (atenção cruzada)** | **0,617** | **0,715** |
| Híbrido: bi-encoder congelado + BM25 + MLP | 0,663 | 0,753 |
| Fine-tuning parcial do bi-encoder | 0,690 | 0,782 |

O Cross-Encoder **supera o bi-encoder congelado equivalente** (0,617 vs.
0,570 de Accuracy@1) sem nenhum treinamento do BERT — só trocando a
arquitetura de "dois embeddings separados" para "atenção conjunta". Isso
confirma a expectativa da literatura: cross-encoders tendem a superar
bi-encoders na mesma configuração, ao custo de não permitirem indexação/
busca eficiente (não dá para pré-computar o embedding de uma candidata
independente da pergunta). Ainda assim, fica atrás do híbrido e do
fine-tuning — sugerindo que o sinal lexical explícito (BM25) e o ajuste de
pesos continuam sendo mais valiosos que a arquitetura de atenção cruzada
isolada, ao menos nesta escala de dado e sem fine-tuning do cross-encoder.

## Resultado — Pipeline Multi-Estágio (BM25 → Cross-Encoder)

| K (candidatas reranqueadas) | Accuracy@1 | MRR | Chamadas ao BERT/pergunta | Economia |
|---:|---:|---:|---:|---:|
| 3 | 0,530 | 0,617 | 3 | 94% |
| 5 | 0,553 | 0,636 | 5 | 90% |
| 10 | 0,590 | 0,675 | 10 | 80% |
| 20 | 0,600 | 0,696 | 20 | 60% |
| 50 (= Cross-Encoder puro) | 0,617 | 0,716 | 50 | 0% |

Com **K=10**, o pipeline perde só 0,027 de Accuracy@1 (0,617 → 0,590) em
troca de **80% menos chamadas ao BERT** por pergunta — uma economia
computacional grande por uma perda de qualidade pequena. Esse é o
princípio central de arquiteturas de recuperação em produção (um estágio
barato filtra o grosso, um estágio caro refina só o que sobrou), aqui
demonstrado em miniatura sobre o pool de 50 candidatas do MilkQA.

## Estrutura do Projeto

```text
.
├── profile_crossencoder.py       # perfilamento de custo real antes de escolher a escala
├── extrair_features_pares.py     # extracao com cache/checkpoint incremental (~2,25h)
├── cross_encoder_model.py         # cabeca de classificacao + grid search + avaliacao
├── pipeline_multiestagio.py       # BM25 filtra top-K -> cross-encoder reranqueia
├── cache/
│   └── pair_features_crossencoder.pt   # {"qid||cid": Tensor[768]} para os 38.263 pares
└── checkpoints/
    ├── best_model_crossencoder.pt
    ├── grid_search_results.csv
    └── resultados_pipeline_multiestagio.csv
```

## Como Executar

```bash
pip install torch transformers datasets pandas numpy rank_bm25
py profile_crossencoder.py         # opcional: confirma o custo por par nesta maquina
py extrair_features_pares.py       # ~2-4h, retomavel (cache incremental)
py cross_encoder_model.py          # ~1 min (so treina a cabeca sobre features ja prontas)
py pipeline_multiestagio.py        # ~1 min (reaproveita cache + modelo treinado)
```

## Referências

- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional
  Transformers for Language Understanding*. (arquitetura base do
  Cross-Encoder: um único Transformer processando os dois textos)
- Reimers, N.; Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings
  using Siamese BERT-Networks*. (contraste formal entre bi-encoder e
  cross-encoder, e o trade-off custo/qualidade entre os dois)
- Nogueira, R.; Cho, K. (2019). *Passage Re-ranking with BERT*.
  (cross-encoder aplicado a reranking em recuperação de informação —
  inspiração direta do pipeline multi-estágio desta pasta)
- Robertson, S.; Zaragoza, H. (2009). *The Probabilistic Relevance
  Framework: BM25 and Beyond*.
- Criscuolo, M. et al. (2018). *MilkQA: a Dataset of Consumer Questions for
  the Task of Answer Selection*. [arXiv:1801.03460](https://arxiv.org/abs/1801.03460)
