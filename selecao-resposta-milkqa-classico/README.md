# Seleção de Resposta em PLN Agropecuário — MilkQA (variante: PLN clássico, sem deep learning)

Terceira variante do projeto [`AgroSele`](../selecao-resposta-milkqa/), agora
**sem BERTimbau, sem embeddings neurais, sem nenhuma biblioteca de PLN de
alto nível** (não usa `sklearn.feature_extraction`, não usa
`sentence-transformers`). Todo o pipeline — tokenização, remoção de
stopwords, TF-IDF e similaridade de cosseno — foi implementado **à mão**,
em Python puro, sobre o modelo clássico de espaço vetorial (*Vector Space
Model*, Salton et al., 1975), o fundamento estatístico/lexical sobre o
qual técnicas modernas de PLN se apoiam.

## Por que este experimento

Os outros dois projetos da série ([congelado](../selecao-resposta-milkqa/)
e [fine-tuning](../selecao-resposta-milkqa-finetune/)) demonstram uso de
modelos de linguagem pré-treinados. Este aqui demonstra o oposto: que
técnicas de PLN anteriores a modelos de linguagem profundos — pura
contagem de palavras e álgebra linear simples — já resolvem boa parte do
problema, e serve como baseline honesto para medir exatamente quanto valor
o BERTimbau (congelado ou ajustado) adiciona sobre o fundamento clássico.

## Pipeline (implementado do zero em `tfidf_model.py`)

```text
Texto bruto
      |
Tokenização (regex + minúsculas)
      |
Remoção de stopwords (lista PT-BR de ~200 palavras, embutida no código)
      |
TF (frequência do termo no documento, normalizada)
      |
IDF (log(N / (1 + df(termo))) + 1, calculado sobre o corpus de 2.657 respostas)
      |
TF-IDF = TF × IDF, vetor esparso normalizado (norma L2 = 1)
      |
Similaridade de cosseno entre vetor da pergunta e de cada candidata
      |
Ranking das 50 candidatas pela similaridade → resposta selecionada
```

Nenhuma etapa envolve treinamento — é um método puramente estatístico,
sem parâmetros aprendidos por gradiente.

## Resultado

| Método | Accuracy@1 (teste, 300) | MRR (teste) | Tempo total |
|---|---:|---:|---:|
| Aleatório | 0,020 | 0,090 | — |
| Similaridade de cosseno com BERTimbau puro (sem treino) | 0,277 | 0,392 | ~4 min (embeddings) |
| **TF-IDF + cosseno, implementado à mão (sem treino)** | **0,503** | **0,610** | **4,4 segundos** |
| Embeddings BERTimbau congelados + MLP treinado | 0,570 | 0,679 | ~7 min |
| Fine-tuning parcial do BERTimbau (dataset completo) | 0,690 | 0,782 | ~11 h |

## O achado principal

O TF-IDF implementado à mão **quase dobra** o resultado do BERTimbau cru
usado sem nenhum treinamento (0,503 vs 0,277 de Accuracy@1) — e o faz em
4,4 segundos contra minutos de extração de embeddings. Isso não é um erro
nem uma coincidência: perguntas e respostas do MilkQA compartilham
vocabulário técnico específico (nomes de doença, insumo, procedimento) que
o TF-IDF captura por correspondência lexical **exata**, enquanto o
embedding do BERTimbau, sendo uma média de representações contextuais
treinadas para um objetivo genérico de linguagem, "borra" esse sinal em um
espaço semântico contínuo que não foi calibrado para esta tarefa
específica sem alguma forma de treinamento supervisionado.

Esse resultado é consistente com um fenômeno bem documentado na literatura
de recuperação de informação: baselines lexicais clássicos (TF-IDF, BM25)
são notoriamente difíceis de superar por métodos neurais não treinados
especificamente para a tarefa — e só voltam a perder quando o modelo
neural recebe supervisão direta sobre o problema (como o classificador de
pares treinado na versão congelada, ou o fine-tuning parcial na versão
final). O TF-IDF só é superado depois que **algum treinamento** entra em
cena — evidenciando que o ganho do BERTimbau não vem do embedding em si,
mas da capacidade de ajustá-lo à tarefa.

## Limitações desta abordagem

- TF-IDF não captura sinônimos, paráfrases ou relações semânticas entre
  palavras diferentes com significado semelhante — dois textos que dizem a
  mesma coisa com vocabulário distinto teriam similaridade baixa.
- Sensível a variações morfológicas não normalizadas (o tokenizador não
  aplica stemming/lematização); "vacina" e "vacinação" são termos
  distintos no vocabulário, por exemplo.
- Não há nenhuma forma de generalização além do vocabulário observado no
  corpus — termos novos, fora do vocabulário de treino, são ignorados.

## Como executar

```bash
pip install datasets  # única dependência externa, só para carregar os splits oficiais
python tfidf_model.py
```

Reaproveita `corpus.csv` e `queries.csv` já exportados em
`../selecao-resposta-milkqa/datasets/` — não precisa baixar nada além dos
splits oficiais do MilkQA (metadados leves, sem texto).

## Referências

- Salton, G.; Wong, A.; Yang, C. S. (1975). *A Vector Space Model for
  Automatic Indexing*. Communications of the ACM.
- Criscuolo, M. et al. (2018). *MilkQA: a Dataset of Consumer Questions for
  the Task of Answer Selection*. [arXiv:1801.03460](https://arxiv.org/abs/1801.03460)
