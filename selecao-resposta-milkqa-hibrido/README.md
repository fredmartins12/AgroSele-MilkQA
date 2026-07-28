# Seleção de Resposta em PLN Agropecuário — MilkQA (variante: híbrida, BERT + BM25)

Quarta variante do projeto [`AgroSele`](../selecao-resposta-milkqa/): em vez de
comparar PLN clássico e BERTimbau como alternativas separadas (como nas
pastas [`-classico`](../selecao-resposta-milkqa-classico/) e
[`-finetune`](../selecao-resposta-milkqa-finetune/)), esta versão **funde os
dois sinais em um único vetor de features**, enriquecendo a representação
densa do BERTimbau com um score lexical clássico (BM25) antes de entregá-la
a um MLP.

## Arquitetura

```text
1. PLN clássico (classical_features.py)
   texto -> tokenização + stopwords -> índice BM25 (rank_bm25.BM25Okapi)
   -> score BM25(pergunta, candidata), normalizado por pool de 50 candidatas

2. Matemática do BERT (bert_math.py) -- funções explícitas
   cosine_similarity(u, v)       = (u·v) / (||u|| ||v||)
   absolute_difference(u, v)     = |u - v|
   elementwise_product(u, v)     = u * v
   pair_features(q, a)           = [q, a, |q-a|, q*a]           -> 3072d

3. Feature Fusion (model.py: feature_fusion())
   vetor enriquecido = pair_features(q,a) ++ [cos(q,a)] ++ [BM25 normalizado]
                      = 3072 + 1 + 1 = 3074 dimensões

4. Rede Neural -- HybridMLP (model.py), PyTorch puro
   Linear(3074 -> hidden) -> ReLU -> Dropout
   Linear(hidden -> hidden/2) -> ReLU -> Dropout
   Linear(hidden/2 -> 1) -> Sigmoid   (saída final explícita, treino com BCELoss)
```

Os embeddings do BERTimbau são os mesmos já cacheados em
`../selecao-resposta-milkqa/cache/` (**congelados**, sem fine-tuning) — o
ganho desta variante vem inteiramente da fusão com o sinal lexical, não de
ajustar os pesos do BERT.

## Por que este experimento

As duas variantes anteriores mostraram que PLN clássico (TF-IDF, 0,503 de
Acc@1) e BERTimbau congelado (0,570) capturam sinais complementares:
vocabulário técnico exato de um lado, proximidade semântica geral de outro.
A pergunta natural é se **combinar os dois sinais no mesmo vetor de entrada**
do classificador — em vez de escolher um ou outro — aproveita o melhor dos
dois sem precisar do custo computacional do fine-tuning (~11h em CPU).

## Resultado

| Método | Accuracy@1 (teste, 300) | MRR (teste) | Custo de treino |
|---|---:|---:|---:|
| Aleatório | 0,020 | 0,090 | — |
| Cosseno BERTimbau puro (sem treino) | 0,277 | 0,392 | — |
| TF-IDF + cosseno à mão (sem treino) | 0,503 | 0,610 | ~4 s |
| BERTimbau congelado + MLP | 0,570 | 0,679 | ~1-2 min |
| **Híbrido: BERTimbau congelado + BM25 + MLP** | **0,663** | **0,753** | **~2 min** |
| Fine-tuning parcial do BERTimbau | 0,690 | 0,782 | ~11 h |

## O achado principal

A fusão de features recupera **quase todo o ganho do fine-tuning** (0,663
vs. 0,690 de Accuracy@1; 0,753 vs. 0,782 de MRR) **sem tocar em nenhum peso
do BERT** — apenas concatenando um score BM25 barato de calcular ao vetor
denso já existente. Contra a versão só-congelada (sem BM25), o ganho é
grande: +16% de Accuracy@1 (0,570 → 0,663) e +11% de MRR (0,679 → 0,753),
pelo custo de uma dependência extra (`rank_bm25`) e alguns milissegundos de
indexação.

Isso confirma, de um ângulo diferente, o achado da variante clássica: o
BERTimbau congelado "borra" o sinal de correspondência lexical exata que é
importante neste domínio técnico. Em vez de esperar que o MLP reconstrua
esse sinal a partir de um embedding genérico, entregar o sinal lexical já
pronto (BM25) como *feature* explícita é uma forma muito mais barata de
recuperar a mesma informação que o fine-tuning aprende implicitamente ao
reajustar os pesos do BERT para o vocabulário do domínio.

## Estrutura do Projeto

```text
.
├── classical_features.py   # tokenização, stopwords, índice BM25 (camada 1)
├── bert_math.py             # cosine_similarity, |u-v|, u*v, pair_features (camada 2)
├── model.py                  # feature_fusion, HybridMLP, treino, avaliação (camadas 3-4)
├── checkpoints/
│   ├── best_model_hibrido.pt
│   └── grid_search_results.csv
└── README.md
```

Reaproveita, sem duplicar, os artefatos das pastas irmãs:
`../selecao-resposta-milkqa/cache/*.pt` (embeddings BERTimbau) e
`../selecao-resposta-milkqa/datasets/*.csv` (textos de corpus e queries).

## Como Executar

```bash
pip install torch transformers datasets pandas numpy rank_bm25
python model.py   # ~2 min (indexação BM25 + grid search + avaliação)
```

Pré-requisito: os embeddings cacheados em `../selecao-resposta-milkqa/cache/`
precisam existir (gerados por `../selecao-resposta-milkqa/pre_processing.py`).

## Referências

- Criscuolo, M. et al. (2018). *MilkQA: a Dataset of Consumer Questions for
  the Task of Answer Selection*. [arXiv:1801.03460](https://arxiv.org/abs/1801.03460)
- Robertson, S.; Zaragoza, H. (2009). *The Probabilistic Relevance
  Framework: BM25 and Beyond*. Foundations and Trends in Information
  Retrieval. (formulação do BM25 implementada por `rank_bm25`)
- Conneau, A. et al. (2017). *Supervised Learning of Universal Sentence
  Representations from Natural Language Inference Data*. (esquema de
  features de pares [u, v, |u-v|, u*v])
- Souza, F., Nogueira, R., Lotufo, R. (2020). *BERTimbau: Pretrained BERT
  Models for Brazilian Portuguese*.
