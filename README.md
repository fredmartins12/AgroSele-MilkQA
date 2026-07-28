# AgroSele — Seleção Automática de Respostas Técnicas para Produtores Rurais de Pecuária Leiteira em Português

Projeto individual de Processamento de Linguagem Natural (PLN) — UFPB, Prof.
Dr. Yuri Malheiros, período 2026.1. Discente: Frederico Botelho Martins —
matrícula 20230090098.

Sistema de **seleção de resposta** (*answer selection*) sobre o
[MilkQA](https://arxiv.org/abs/1801.03460) (Criscuolo et al., 2018), dataset
real de perguntas e respostas do atendimento ao produtor rural da Embrapa
Gado de Leite (2003–2012): dado uma pergunta real de um produtor e um
conjunto de 50 respostas candidatas, o sistema aponta qual delas é a
correta. O tema conecta-se a um software de gestão para fazendas leiteiras
já em produção, desenvolvido de forma independente pelo autor, que motivou
a escolha do domínio e do dataset (ver `docs/Fase1_Proposta_AgroSele.docx`).

**Repositório:** https://github.com/fredmartins12/AgroSele-MilkQA
**Slides da apresentação:** `docs/Apresentacao_AgroSele.pptx`

## Documentação (entregas da disciplina)

Toda a documentação formal está em [`docs/`](docs/) (cópia de conveniência
dos mesmos arquivos que também estão em
`selecao-resposta-milkqa/docs/`, a pasta "canônica" onde o artigo é
referenciado internamente pelos READMEs de cada variante):

| Arquivo | Conteúdo |
|---|---|
| `Fase1_Proposta_AgroSele.docx` | Motivação, justificativa metodológica (por que não uma LLM comercial), pergunta de pesquisa |
| `Fase2_ReferencialTeorico_AgroSele.docx` | Tabela comparativa com 10 trabalhos relacionados verificados |
| `Fase3_Metodologia_Conclusao_AgroSele.docx` | Metodologia completa das 6 variantes + conclusão |
| `artigo_sbc.docx` | Artigo final em formato SBC, com todas as seções preenchidas (Resumo, Abstract, Trabalhos Relacionados, Metodologia, Resultados 4.1–4.10, Conclusão, Referências) |

## As Seis Variantes (da mais simples à mais sofisticada)

Cada pasta é **auto-contida e executável**: código-fonte, um notebook
Jupyter já executado (com outputs reais, variáveis em português, comentado),
README próprio com discussão detalhada, e os checkpoints/caches necessários
para reproduzir os resultados sem precisar re-treinar do zero.

| # | Pasta | Ideia central | Accuracy@1 | MRR |
|---:|---|---|---:|---:|
| — | *(baseline)* | Aleatório | 0,020 | 0,090 |
| — | *(baseline)* | Similaridade de cosseno, BERTimbau sem treino | 0,277 | 0,392 |
| 1 | [`selecao-resposta-milkqa-classico/`](selecao-resposta-milkqa-classico/) | TF-IDF + cosseno, implementado à mão, sem BERT | 0,503 | 0,610 |
| 2 | [`selecao-resposta-milkqa/`](selecao-resposta-milkqa/) | Bi-encoder: BERTimbau congelado + MLP | 0,570 | 0,679 |
| 3 | [`selecao-resposta-milkqa-crossencoder/`](selecao-resposta-milkqa-crossencoder/) | Cross-Encoder: atenção cruzada pergunta↔candidata, congelado | 0,617 | 0,715 |
| 3b | `selecao-resposta-milkqa-crossencoder/` (pipeline) | BM25 filtra top-10 → Cross-Encoder reranqueia | 0,590 | 0,675 |
| 4 | [`selecao-resposta-milkqa-hibrido/`](selecao-resposta-milkqa-hibrido/) | Bi-encoder congelado + BM25 fundido no vetor de entrada | 0,663 | 0,753 |
| 5 | [`selecao-resposta-milkqa-finetune/`](selecao-resposta-milkqa-finetune/) | Fine-tuning parcial do BERTimbau (**melhor resultado isolado**) | **0,690** | **0,782** |

## O Fio Condutor dos Experimentos

1. **TF-IDF quase dobra o BERTimbau cru** (0,503 vs 0,277): num domínio de
   vocabulário técnico compartilhado (nomes de doença, insumo,
   procedimento), correspondência lexical exata vale muito, e um embedding
   genérico e não-ajustado não captura isso sozinho.
2. **Um classificador treinado sobre embeddings congelados já ajuda bastante**
   (0,570): o MLP aprende uma noção de correspondência que vai além da
   similaridade bruta.
3. **Trocar a arquitetura (Cross-Encoder) ajuda mais que esperado, sem
   nenhum treino do BERT** (0,617): atenção cruzada entre pergunta e
   candidata captura sinal que dois embeddings separados não expõem.
4. **Fundir o sinal lexical explicitamente (híbrido BM25) ajuda mais ainda,
   e mais barato** (0,663, ~2 minutos de treino): quase alcança o
   fine-tuning completo sem tocar nos pesos do BERT.
5. **Fine-tuning de verdade continua sendo o melhor resultado isolado**
   (0,690), mas custou ~11 horas em CPU — e uma primeira tentativa com
   dado insuficiente (300 de 2.307 perguntas) tinha indicado, erroneamente,
   que fine-tuning não compensava o custo.
6. **Um pipeline de dois estágios recupera quase toda a qualidade do
   Cross-Encoder a uma fração do custo** (K=10 → só 0,027 de perda de
   Accuracy@1 por 80% menos chamadas ao BERT) — o princípio central de
   sistemas de busca em produção, aqui demonstrado em miniatura.

Ver `docs/artigo_sbc.docx`, Seção 4 e 5, para a discussão completa.

## Como Reproduzir

Cada pasta tem seu próprio README com instruções detalhadas. Em resumo,
a ordem de dependência entre pastas (por causa de caches/dados
compartilhados via caminho relativo) é:

```
selecao-resposta-milkqa/          -- gera cache/ (embeddings) e datasets/ (CSVs), usados por classico e hibrido
selecao-resposta-milkqa-finetune/ -- gera datasets/ (CSVs), usados por hibrido e crossencoder
        |
        ├── selecao-resposta-milkqa-classico/      (reusa datasets/ do bi-encoder)
        ├── selecao-resposta-milkqa-hibrido/        (reusa cache/ do bi-encoder + datasets/ do finetune)
        └── selecao-resposta-milkqa-crossencoder/   (reusa datasets/ do finetune)
```

Os caches e checkpoints necessários já estão incluídos nesta pasta — não é
preciso rodar a extração de embeddings (~20-25 min) nem a extração de pares
do Cross-Encoder (~2h15min) de novo, exceto se quiser reproduzir do zero.
Cada notebook (`notebook_*.ipynb`) já foi executado e contém os outputs
reais, incluindo uma célula de verificação que recalcula alguns exemplos ao
vivo pelo BERT e confirma que batem com o cache salvo.

```bash
pip install torch transformers datasets pandas numpy rank_bm25 jupyter
```

## Estrutura desta Pasta

```text
AgroSele-Entrega-Final/
├── README.md                              (este arquivo)
├── docs/                                    4 documentos da disciplina (cópia de conveniência)
├── selecao-resposta-milkqa/                 variante 2: bi-encoder congelado
├── selecao-resposta-milkqa-classico/        variante 1: TF-IDF à mão
├── selecao-resposta-milkqa-hibrido/         variante 4: híbrido BERT+BM25
├── selecao-resposta-milkqa-crossencoder/    variantes 3 e 3b: Cross-Encoder + pipeline
└── selecao-resposta-milkqa-finetune/        variante 5: fine-tuning parcial (recomendada)
```

## Referências Principais

- Criscuolo, M.; Fonseca, E. R.; Aluísio, S. M.; Sperança-Criscuolo, A. C.
  (2018). *MilkQA: a Dataset of Consumer Questions for the Task of Answer
  Selection*. [arXiv:1801.03460](https://arxiv.org/abs/1801.03460)
- Souza, F.; Nogueira, R.; Lotufo, R. (2020). *BERTimbau: Pretrained BERT
  Models for Brazilian Portuguese*.
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional
  Transformers for Language Understanding*.
- Conneau, A. et al. (2017). *Supervised Learning of Universal Sentence
  Representations from Natural Language Inference Data* (InferSent).
- Reimers, N.; Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings
  using Siamese BERT-Networks*.
- Robertson, S.; Zaragoza, H. (2009). *The Probabilistic Relevance
  Framework: BM25 and Beyond*.
- Nogueira, R.; Cho, K. (2019). *Passage Re-ranking with BERT*.

Ver `docs/artigo_sbc.docx` para a lista completa de referências.
