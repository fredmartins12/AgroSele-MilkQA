"""
Camada 1 do modelo hibrido: PLN classico (BM25) sobre o corpus de respostas
do MilkQA. Pre-processamento textual (tokenizacao + remocao de stopwords)
e o mesmo usado na variante puramente classica do projeto; o ranqueamento
em si usa a biblioteca `rank_bm25` (BM25Okapi), conforme permitido pelo
enunciado ("Scikit-Learn ou Rank-BM25 para o PLN inicial").

O score BM25 de cada par (pergunta, candidata) e o sinal lexical que sera
fundido, na camada de Feature Fusion (ver model.py), ao vetor denso do
BERTimbau.
"""
import re

import numpy as np
from rank_bm25 import BM25Okapi

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
    """Minusculiza, extrai tokens alfanumericos e remove stopwords PT-BR."""
    words = _token_re.findall(text.lower())
    return [w for w in words if w not in STOPWORDS_PT and len(w) > 1]


class BM25Layer:
    """Indice BM25 (rank_bm25.BM25Okapi) sobre o corpus de respostas.

    O IDF e as estatisticas de comprimento medio de documento sao ajustados
    uma unica vez sobre o corpus inteiro (2657 respostas), como de praxe em
    BM25. Os scores brutos sao cacheados por query (qid) para nao recalcular
    o ranqueamento contra o corpus inteiro toda vez que o mesmo par
    (pergunta, candidata) aparece de novo durante a montagem de pares de
    treino ou avaliacao.
    """

    def __init__(self, doc_ids, doc_texts):
        self.doc_ids = list(doc_ids)
        self.id_to_idx = {did: i for i, did in enumerate(self.doc_ids)}
        tokenized_corpus = [tokenize(t) for t in doc_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self._raw_score_cache = {}

    def _raw_scores(self, qid, query_text):
        if qid not in self._raw_score_cache:
            self._raw_score_cache[qid] = np.asarray(
                self.bm25.get_scores(tokenize(query_text)), dtype=np.float64)
        return self._raw_score_cache[qid]

    def normalized_scores_for_pool(self, qid, query_text, candidate_ids):
        """Score BM25 normalizado (min-max, em [0, 1]) apenas entre as
        candidatas de UM pool (as ate 50 candidatas daquela pergunta) --
        e essa a comparacao que importa para o ranqueamento, e evita que a
        escala nao limitada do BM25 bruto distorca o treino do MLP."""
        raw = self._raw_scores(qid, query_text)
        idx = [self.id_to_idx[cid] for cid in candidate_ids]
        pool_scores = raw[idx]
        lo, hi = pool_scores.min(), pool_scores.max()
        span = (hi - lo) or 1.0
        norm = (pool_scores - lo) / span
        return {cid: float(s) for cid, s in zip(candidate_ids, norm)}
