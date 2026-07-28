# Seleção de Resposta em PLN Agropecuário — MilkQA (variante: fine-tuning parcial)

> **Esta é uma cópia experimental** de
> [`../selecao-resposta-milkqa/`](../selecao-resposta-milkqa/), criada para
> testar **fine-tuning parcial do BERTimbau** sem arriscar o resultado já
> validado da pasta original.
>
> **Resultado final: o fine-tuning parcial VENCE a versão congelada**
> (Acc@1 = 0,690 vs 0,570; MRR = 0,782 vs 0,679) — ganho real de ~21% em
> Accuracy@1 e ~15% em MRR. **Esta pasta passa a ser a entrega
> recomendada.**
>
> O caminho até aqui teve uma volta importante, documentada com
> honestidade científica na seção "Experimento de Fine-Tuning" abaixo: uma
> primeira tentativa, com apenas 300 das 2.307 perguntas de treino
> disponíveis (limitação de tempo de CPU), tinha ficado **pior** que a
> versão congelada (Acc@1=0,487). A hipótese — de que o problema era
> volume de dado, não o método — foi testada retreinando com o conjunto
> completo (2.307 perguntas, 6 negativos por positivo, 5 épocas, ~11h em
> CPU, incluindo uma retomada de checkpoint após o computador reiniciar no
> meio do treino), confirmando a hipótese.

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

## Resultados (herdados da versão congelada — ver "Experimento de Fine-Tuning" abaixo para o resultado final desta pasta)

> Esta seção documenta os resultados da versão **congelada** (herdada de
> `../selecao-resposta-milkqa/`, já que esta pasta começou como cópia
> dela). O resultado final desta pasta — fine-tuning parcial, que supera
> os números abaixo — está na seção
> ["Experimento de Fine-Tuning"](#experimento-de-fine-tuning-esta-pasta).

Versão final (congelada): treino com as **2.307 perguntas completas** (não mais
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

## Experimento de Fine-Tuning (esta pasta)

Terceira tentativa de melhoria: em vez de manter o BERTimbau totalmente
congelado, a **última camada do encoder** (1 de 12, ~7% dos parâmetros do
modelo, 7,7M de 108,9M) e o *pooler* foram descongelados e treinados
junto com a cabeça MLP, com taxa de aprendizado menor para o BERT
(2×10⁻⁵) do que para a cabeça (1×10⁻³) — prática padrão em fine-tuning.
Implementado em `finetune_model.py`.

### Tentativa 1 (falhou): amostra pequena por restrição de tempo

Sem *cache* de embeddings (o texto passa pelo BERT a cada passo de
treino, não é mais um lookup instantâneo), o custo computacional em CPU
é ordens de magnitude maior. Um primeiro teste de perfilamento (com texto
curto e repetido) subestimou esse custo em ~3×; o valor real, medido com
texto do MilkQA, foi de **~505ms por exemplo de treino**. Por essa
restrição, a primeira rodada usou apenas **300 perguntas** (de 2.307
disponíveis) e **3 negativos por positivo** — 17× menos exemplos que a
versão congelada. O experimento completo (6 épocas + avaliação final)
levou 102 minutos em CPU e **ficou pior que a versão congelada**
(Acc@1=0,487 vs 0,570). O log de treino mostrou instabilidade entre
épocas no acompanhamento de validação (Acc@1 variando 0,27→0,20→0,40→
0,40→0,47→0,33 sem convergência monotônica) — sintoma de volume de dados
insuficiente para o número de parâmetros sendo ajustados. Hipótese
registrada: o problema era volume de dado, não o método.

### Tentativa 2 (venceu): dataset completo, rodada overnight

Para testar a hipótese, o treino foi refeito com o **conjunto de treino
completo** (2.307 perguntas, 6 negativos por positivo, 16.149 exemplos
por época, 5 épocas) — a mesma escala de dado da versão congelada. Rodada
em background por ~11h em CPU, com prioridade de processo elevada e
outros programas fechados para dedicar a máquina inteira ao treino.
Sobreviveu a uma queda do computador no meio da época 3 (reiniciado por
travamento do sistema, não relacionado ao treino): um checkpoint de
segurança salvo ao fim de cada época permitiu retomar exatamente de onde
parou, sem perder as duas primeiras épocas já concluídas.

| Métrica | Congelada (2.307 perguntas) | Fine-tuning tentativa 1 (300) | Fine-tuning tentativa 2 (2.307) |
|---|---:|---:|---:|
| Accuracy@1 (teste, 300) | 0,570 | 0,487 | **0,690** |
| MRR (teste, 300) | 0,679 | 0,606 | **0,782** |

### Interpretação

A hipótese se confirmou: o fine-tuning parcial **não era pior em
princípio** — só precisava do mesmo volume de dado que a versão
congelada já tinha. Com dado equivalente, especializar a última camada
do BERTimbau no vocabulário técnico do domínio (nomes de doença,
procedimento, insumo) supera manter o modelo totalmente congelado, com
ganho de ~21% em Accuracy@1 e ~15% em MRR. **Esta é agora a versão
recomendada do projeto.**

## Limitações e Trabalho Futuro

- Textos truncados em 256 tokens (BERTimbau); respostas mais longas
  (média de 198 palavras, até 682) perdem informação de cauda — testar
  max_length maior é o próximo experimento natural, com custo de mais
  tempo de pré-processamento.
- Testar descongelar mais de 1 camada (2-3 camadas do encoder), agora que
  ficou claro que o gargalo era dado, não capacidade do backbone — com o
  dataset completo já disponível, vale ver se descongelar mais camadas
  melhora ainda mais o resultado, ou se 1 camada já captura a maior parte
  do ganho disponível.
- `TRAIN_QUERIES`, `N_NEG_TRAIN`, `N_UNFROZEN_LAYERS` em `finetune_model.py`
  estão documentados e são fáceis de re-experimentar por quem for dar
  continuidade ao projeto. `RESUME_CHECKPOINT`/`RESUME_FROM_EPOCH` permitem
  retomar de um checkpoint salvo, caso o treino seja interrompido de novo.

## Referências

- Criscuolo, M. et al. (2018). *MilkQA: a Dataset of Consumer Questions for
  the Task of Answer Selection*. [arXiv:1801.03460](https://arxiv.org/abs/1801.03460)
- Souza, F., Nogueira, R., Lotufo, R. (2020). *BERTimbau: Pretrained BERT
  Models for Brazilian Portuguese*.
