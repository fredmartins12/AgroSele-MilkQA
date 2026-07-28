# Seleção de Resposta em PLN Agropecuário — MilkQA

> **⚠️ Atualização**: a versão com fine-tuning parcial do BERTimbau, em
> [`../selecao-resposta-milkqa-finetune/`](../selecao-resposta-milkqa-finetune/),
> **supera esta versão congelada** (Accuracy@1 = 0,690 vs 0,570; MRR =
> 0,782 vs 0,679 — ganho de ~21% e ~15%). **Essa pasta passou a ser a
> entrega recomendada.** Esta versão (congelada) permanece como baseline
> de comparação e como a versão mais rápida/leve de rodar (~7min vs ~11h
> em CPU), útil se recursos computacionais forem uma restrição.

Sistema de **seleção de resposta** (*answer selection*) sobre o
[MilkQA](https://arxiv.org/abs/1801.03460), dataset real de perguntas e
respostas do atendimento ao produtor rural da Embrapa Gado de Leite
(2003–2012): dado uma pergunta real de um produtor (ex.: *"Meu tio Paulo
mora em uma chácara... o leite azeda muito rápido... o que devemos
fazer?"*) e um conjunto de 50 respostas candidatas, o sistema aponta qual
delas é a correta.

Projeto desenvolvido para a disciplina de Processamento de Linguagem
Natural (PLN), seguindo o mesmo rigor metodológico (desacoplamento entre
extração de *features* e treino, *cache* de *embeddings*, seleção de
hiperparâmetros por *grid search*) do projeto de referência
[`MultimodalEmotionReconizer`](../../MultimodalEmotionReconizer/).

> **Nota**: este projeto substitui uma versão anterior baseada em dataset
> sintético de classificação financeira
> (`../classificacao-financeira-pecuaria/`), abandonada em favor do MilkQA
> por ser um dataset **real, publicado e citável**, em vez de gerado
> artificialmente. A pasta anterior foi mantida no repositório para
> referência, mas não é mais o projeto de entrega.

## Por que o MilkQA

Não existe dataset público de lançamentos financeiros de fazendas de
pecuária em português (verificado por pesquisa). O MilkQA, por outro lado,
é um recurso real e citável: 2.657 pares de pergunta-resposta escritos por
produtores rurais de níveis de literacia muito diferentes, com respostas
elaboradas por especialistas da Embrapa, publicado pelo NILC/USP com *splits*
oficiais de treino (2307), validação (50) e teste (300) — o que resolve de
saída o problema de conjunto de teste pequeno que auditamos no projeto de
referência.

## A Tarefa

Diferente de classificação em categorias fixas, a tarefa aqui é **ranquear**:
dado uma pergunta e um conjunto de 50 respostas candidatas (uma correta, 49
distratoras), o sistema deve pontuar cada candidata e ordenar pela
probabilidade de ser a resposta certa.

```text
pergunta + 50 candidatas
      ↓
BERTimbau (congelado) → embeddings 768d (cache)
      ↓
para cada par (pergunta, candidata):
  features = [emb_pergunta, emb_candidata, |diferença|, produto]  (3072d)
      ↓
MLP binário: match / não-match
      ↓
ranking das 50 candidatas pela pontuação → resposta selecionada
```

## Estrutura do Projeto

```text
.
├── pre_processing.py        # extrai embeddings BERTimbau de corpus + queries
├── model.py                  # MLP pareado + grid search + avaliação por ranking
├── datasets/
│   ├── corpus.csv             # 2657 respostas (exportado para reprodutibilidade offline)
│   └── queries.csv            # 2657 perguntas
├── cache/
│   ├── corpus_embeddings.pt    # dict {id: Tensor[768]}
│   └── queries_embeddings.pt   # dict {id: Tensor[768]}
├── checkpoints/
│   ├── best_model.pt
│   └── grid_search_results.csv
└── docs/
    └── artigo_sbc.docx
```

## Como Executar

```bash
pip install torch transformers datasets pandas scikit-learn numpy
python pre_processing.py   # ~20-25 min em CPU (2657+2657 textos, alguns longos)
python model.py            # ~1-2 min (grid search + avaliação)
```

## Resultados

Versão final: treino com as **2.307 perguntas completas** (não mais
subamostra) e **8 negativos por positivo** (não mais 4). Grid search sobre
`hidden_size ∈ {128, 256}`, `dropout ∈ {0.2, 0.4}`, `lr ∈ {1e-3, 1e-4}`
(8 combinações), seleção pelo MRR no conjunto de validação oficial (50
perguntas).

| hidden | dropout | lr | MRR (val.) | Acc@1 (val.) |
|---:|---:|---:|---:|---:|
| 128 | 0,4 | 1e-4 | **0,735** | 0,64 |
| 256 | 0,2 | 1e-4 | 0,734 | 0,64 |
| 256 | 0,4 | 1e-4 | 0,731 | 0,64 |
| 128 | 0,4 | 1e-3 | 0,731 | 0,64 |
| 128 | 0,2 | 1e-4 | 0,728 | 0,64 |
| 256 | 0,2 | 1e-3 | 0,726 | 0,62 |
| 128 | 0,2 | 1e-3 | 0,724 | 0,62 |
| 256 | 0,4 | 1e-3 | 0,708 | 0,60 |

**Resultado final no conjunto de teste** (300 perguntas, isolado da seleção
de hiperparâmetros), comparado contra dois baselines:

| Método | Accuracy@1 | MRR |
|---|---:|---:|
| **MLP treinado (final)** | **0,570** | **0,679** |
| Similaridade de cosseno pura (sem treino) | 0,277 | 0,392 |
| Aleatório | 0,020 | 0,090 |

O MLP treinado **dobra** o Accuracy@1 em relação a usar o embedding do
BERTimbau puro por similaridade de cosseno (0,277 → 0,570), mostrando que
o classificador de pares aprende uma noção de correspondência
pergunta-resposta que vai além da proximidade semântica bruta capturada
pelo modelo congelado. Contra o acaso, o resultado final é **28× melhor**
em Accuracy@1.

**Evolução do experimento** (mesma arquitetura, mais dado de treino):

| Versão | Perguntas de treino | Negativos/positivo | Acc@1 (teste) | MRR (teste) |
|---|---:|---:|---:|---:|
| Inicial | 800 (subamostra) | 4 | 0,500 | 0,622 |
| Final | 2.307 (completo) | 8 | **0,570** | **0,679** |

Ver `docs/artigo_sbc.docx` para discussão completa.

## Duas melhorias testadas depois — e que não venceram (documentado por rigor)

Após o resultado acima, dois refinamentos adicionais foram tentados. Nenhum
dos dois trouxe ganho estatisticamente relevante — registrados aqui porque
"o que não funcionou" também é resultado científico válido, e porque
mostra que o sistema já está perto do teto alcançável com esta arquitetura
(BERTimbau congelado + MLP) sem uma mudança estrutural maior.

1. **Ensemble MLP + similaridade de cosseno** (combinar as duas pontuações
   por média ponderada, sem re-treinar nada): piorou monotonicamente
   conforme o peso do cosseno aumentava (Acc@1 caiu de 0,570 com peso 0 do
   cosseno até 0,503 com peso igual). O MLP já usa a informação do
   embedding de forma mais refinada do que uma média simples com o
   cosseno bruto consegue agregar.
2. **Hard negative mining** (metade dos negativos de treino escolhidos
   entre as candidatas erradas mais parecidas com a pergunta, em vez de
   sorteio uniforme): Acc@1 = 0,573 e MRR = 0,677 — estatisticamente
   empatado com a versão de negativos 100% aleatórios (0,570 / 0,679). O
   checkpoint salvo em `checkpoints/best_model.pt` corresponde a esta
   versão (`HARD_NEG_FRAC = 0.5` em `model.py`).

## Limitações e Trabalho Futuro

- Textos truncados em 256 tokens (BERTimbau); respostas mais longas
  (média de 198 palavras, até 682) perdem informação de cauda — testar
  max_length maior é o próximo experimento natural, com custo de mais
  tempo de pré-processamento.
- `TRAIN_SUBSAMPLE`, `N_NEG_TRAIN` e `HARD_NEG_FRAC` em `model.py` estão
  documentados e são fáceis de re-experimentar por quem for dar
  continuidade ao projeto.

**Atualização — fine-tuning parcial testado e bem-sucedido**: o
fine-tuning das últimas camadas do BERTimbau foi implementado e testado
em [`../selecao-resposta-milkqa-finetune/`](../selecao-resposta-milkqa-finetune/)
(cópia isolada deste projeto, para não arriscar este resultado). Uma
primeira tentativa, limitada a 300 das 2.307 perguntas de treino por
restrição de tempo de CPU, tinha ficado pior (Acc@1=0,487). Repetindo com
o conjunto de treino completo (mesma escala de dado desta versão
congelada) — rodada overnight de ~11h em CPU — o fine-tuning **superou
esta versão congelada**: Acc@1=0,690 e MRR=0,782 (ganhos de ~21% e ~15%).
**A pasta de fine-tuning passou a ser a recomendação final do projeto**;
esta versão congelada permanece documentada como baseline mais rápida de
reproduzir.

## Família de experimentos (todas as variantes do projeto)

Esta pasta é o ponto de partida (bi-encoder congelado); as demais reaproveitam
seu cache de embeddings ou seus dados. Accuracy@1 / MRR no teste (300 perguntas):

| Pasta | Ideia | Acc@1 | MRR |
|---|---|---:|---:|
| `selecao-resposta-milkqa-classico/` | TF-IDF + cosseno, à mão, sem BERT | 0,503 | 0,610 |
| **`selecao-resposta-milkqa/`** (esta) | Bi-encoder congelado + MLP | 0,570 | 0,679 |
| `selecao-resposta-milkqa-crossencoder/` | Cross-Encoder (atenção cruzada), congelado | 0,617 | 0,715 |
| `selecao-resposta-milkqa-hibrido/` | Bi-encoder congelado + BM25 fundido + MLP | 0,663 | 0,753 |
| `selecao-resposta-milkqa-finetune/` | Fine-tuning parcial do BERTimbau (recomendada) | 0,690 | 0,782 |
| `selecao-resposta-milkqa-crossencoder/` (pipeline) | BM25 filtra top-10 → Cross-Encoder reranqueia | 0,590 | 0,675 |

## Referências

- Criscuolo, M. et al. (2018). *MilkQA: a Dataset of Consumer Questions for
  the Task of Answer Selection*. [arXiv:1801.03460](https://arxiv.org/abs/1801.03460)
- Souza, F., Nogueira, R., Lotufo, R. (2020). *BERTimbau: Pretrained BERT
  Models for Brazilian Portuguese*.
