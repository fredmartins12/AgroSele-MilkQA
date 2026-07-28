"""
Camada 2 do modelo hibrido: as operacoes matematicas classicas de
combinacao de pares de embeddings (estilo InferSent/SNLI, Conneau et al.
2017), escritas de forma explicita -- cada uma como funcao propria e
documentada, em vez de uma unica expressao composta -- para deixar clara a
semantica de cada termo do vetor final antes da fusao com o PLN classico.

Dado um par de embeddings BERTimbau (pergunta u, resposta v, cada um com
768 dimensoes, obtidos por mean pooling sobre a ultima camada oculta):

  - similaridade de cosseno: mede o angulo entre os vetores (proximidade
    semantica direcional, independente de magnitude).
  - diferenca absoluta |u - v|: mede, dimensao a dimensao, o quanto os dois
    embeddings divergem -- um "vetor de distancia" ponto a ponto.
  - produto elemento a elemento u * v: realca as dimensoes em que os dois
    embeddings concordam em sinal e magnitude (interacao multiplicativa).
"""
import torch


def cosine_similarity(u, v, eps=1e-8):
    """cos(u, v) = (u . v) / (||u|| * ||v||).

    Calculado explicitamente a partir do produto escalar e das normas L2 --
    nao usa torch.nn.functional.cosine_similarity -- para deixar visivel
    cada termo da formula. Aceita tanto um unico par (u, v com forma
    (768,)) quanto um lote (forma (N, 768)); o resultado tem uma dimensao
    a menos (escalar ou vetor de N similaridades).
    """
    dot_product = (u * v).sum(dim=-1)
    norm_u = torch.sqrt((u * u).sum(dim=-1))
    norm_v = torch.sqrt((v * v).sum(dim=-1))
    return dot_product / (norm_u * norm_v + eps)


def absolute_difference(u, v):
    """|u - v|, elemento a elemento -- mantem a dimensionalidade (768,)."""
    return torch.abs(u - v)


def elementwise_product(u, v):
    """u * v, elemento a elemento -- mantem a dimensionalidade (768,)."""
    return u * v


def pair_features(q_emb, a_emb):
    """Vetor denso [q, a, |q-a|, q*a] -- 4 blocos de 768 dims = 3072 dims.

    Este e o vetor "puro BERT", antes de qualquer fusao com sinais do PLN
    classico (ver feature_fusion() em model.py).
    """
    diff = absolute_difference(q_emb, a_emb)
    prod = elementwise_product(q_emb, a_emb)
    return torch.cat([q_emb, a_emb, diff, prod], dim=-1)
