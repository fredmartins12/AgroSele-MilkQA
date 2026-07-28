"""
Baseline classico de PLN para selecao de resposta no MilkQA -- SEM
BERTimbau, SEM embeddings neurais, SEM bibliotecas de alto nivel (nao usa
sklearn.feature_extraction nem sklearn.metrics.pairwise). Tokenizacao,
remocao de stopwords, TF-IDF e similaridade de cosseno sao implementados
a mao, em Python puro, para demonstrar o fundamento estatistico/lexical
sobre o qual tecnicas modernas de PLN se apoiam.

Modelo de espaco vetorial (Vector Space Model, Salton et al. 1975):
  1. Tokenizacao: minusculizacao + remocao de pontuacao + split por espaco
  2. Remocao de stopwords (lista fixa em portugues)
  3. TF (term frequency): contagem normalizada de cada termo no documento
  4. IDF (inverse document frequency): log(N / (1 + df(termo)))
  5. TF-IDF = TF * IDF, vetor esparso por documento
  6. Similaridade de cosseno entre o vetor da pergunta e de cada candidata
  7. Ranking das candidatas pela similaridade -- sem nenhum treinamento

Uso: py tfidf_model.py
"""
import csv
import math
import re
import time
from collections import Counter, defaultdict

from datasets import load_dataset

# ---------------------------------------------------------------------
# 1. Tokenizacao e stopwords (implementadas a mao)
# ---------------------------------------------------------------------

STOPWORDS_PT = set("""
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

_token_re = re.compile(r"[a-zà-öø-ÿ0-9]+")


def tokenize(text):
    """Minusculiza, extrai tokens alfanumericos (regex), remove stopwords."""
    words = _token_re.findall(text.lower())
    return [w for w in words if w not in STOPWORDS_PT and len(w) > 1]


# ---------------------------------------------------------------------
# 2. TF-IDF a mao (sem sklearn)
# ---------------------------------------------------------------------

class TfidfIndex:
    """Indice TF-IDF construido a mao sobre uma colecao de documentos."""

    def __init__(self, doc_ids, doc_texts):
        self.doc_ids = list(doc_ids)
        self.tokens_by_doc = {did: tokenize(t) for did, t in zip(doc_ids, doc_texts)}

        # document frequency: em quantos documentos cada termo aparece
        df = defaultdict(int)
        for toks in self.tokens_by_doc.values():
            for term in set(toks):
                df[term] += 1

        n_docs = len(self.doc_ids)
        # IDF suavizado: log(N / (1 + df)) + 1  (evita idf=0 para termos universais)
        self.idf = {term: math.log(n_docs / (1 + d)) + 1.0 for term, d in df.items()}

        # TF-IDF vector (esparso, como dict termo->peso) por documento, ja normalizado
        self.vectors = {}
        for did, toks in self.tokens_by_doc.items():
            self.vectors[did] = self._tfidf_vector(toks)

    def _tfidf_vector(self, tokens):
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        vec = {}
        for term, c in counts.items():
            tf = c / total
            idf = self.idf.get(term, 0.0)
            if idf > 0:
                vec[term] = tf * idf
        # normaliza p/ norma L2 = 1 (necessario para cosseno = produto escalar puro)
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        return {t: w / norm for t, w in vec.items()}

    def vectorize_query(self, text):
        """TF-IDF de um texto novo, usando o IDF ja aprendido do corpus."""
        return self._tfidf_vector(tokenize(text))

    @staticmethod
    def cosine(vec_a, vec_b):
        """Similaridade de cosseno entre dois vetores esparsos (dict). Como
        ambos ja estao normalizados (norma 1), cosseno = produto escalar."""
        # itera sobre o menor dict por eficiencia
        if len(vec_a) > len(vec_b):
            vec_a, vec_b = vec_b, vec_a
        return sum(w * vec_b.get(t, 0.0) for t, w in vec_a.items())


# ---------------------------------------------------------------------
# 3. Avaliacao (Accuracy@1 / MRR) -- identica em espirito a analysis/*.py
#    da versao BERTimbau, para comparacao direta e justa
# ---------------------------------------------------------------------

def load_csv(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["id"]] = row["text"]
    return rows


def rank_eval(index_corpus, texts_by_qid, split):
    acc1, mrr = [], []
    for row in split:
        qid, pos_id, cands = row["query-id"], row["positive-doc-id"], row["candidates-ids"]
        q_vec = index_corpus.vectorize_query(texts_by_qid[qid])
        scores = [(cid, TfidfIndex.cosine(q_vec, index_corpus.vectors[cid])) for cid in cands]
        scores.sort(key=lambda x: -x[1])
        ranked_ids = [cid for cid, _ in scores]
        rank = ranked_ids.index(pos_id) + 1
        acc1.append(1.0 if rank == 1 else 0.0)
        mrr.append(1.0 / rank)
    return sum(acc1) / len(acc1), sum(mrr) / len(mrr)


def main():
    t0 = time.time()
    print("Carregando corpus e queries (CSV ja exportados)...")
    corpus_texts = load_csv("../selecao-resposta-milkqa/datasets/corpus.csv")
    query_texts = load_csv("../selecao-resposta-milkqa/datasets/queries.csv")
    print(f"  {len(corpus_texts)} respostas | {len(query_texts)} perguntas")

    print("\nConstruindo indice TF-IDF do corpus (a mao, sem sklearn)...")
    index = TfidfIndex(list(corpus_texts.keys()), list(corpus_texts.values()))
    vocab_size = len(index.idf)
    print(f"  Vocabulario (pos-stopwords): {vocab_size} termos unicos | {time.time()-t0:.1f}s")

    print("\nCarregando splits oficiais do MilkQA...")
    ds = load_dataset("eduagarcia/MilkQA")

    print("\n===== Avaliacao (TF-IDF + cosseno, sem treino) =====")
    for name, split in [("dev", ds["dev"]), ("test", ds["test"])]:
        acc1, mrr = rank_eval(index, query_texts, split)
        print(f"  {name:5s} ({len(split):3d} perguntas) -> Accuracy@1={acc1:.4f}  MRR={mrr:.4f}")

    n_rand = 50
    print(f"\nBaseline aleatorio esperado: Accuracy@1~={1/n_rand:.4f} "
          f"MRR~={sum(1/k for k in range(1, n_rand+1))/n_rand:.4f}")
    print(f"\nTempo total: {time.time()-t0:.1f}s (sem GPU, sem cache de embeddings -- so contagem de palavras)")


if __name__ == "__main__":
    main()
