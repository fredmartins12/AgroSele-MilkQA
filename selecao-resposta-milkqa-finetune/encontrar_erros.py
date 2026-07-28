"""
Busca exemplos concretos de erro do modelo final (fine-tuning parcial) no
conjunto de teste, para a secao de Analise de Erros do artigo (exigida na
Fase 3/4). Reaproveita o checkpoint ja treinado -- so roda avaliacao, sem
retreinar nada.
"""
import random

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from finetune_model import (MAX_LENGTH, PairMLP, encode, freeze_bert_except_last_layers,
                             mean_pooling, pair_features)

SEED = 42
random.seed(SEED)

device = "cpu"
print("Carregando textos...")
corpus_df = pd.read_csv("datasets/corpus.csv")
queries_df = pd.read_csv("datasets/queries.csv")
texto_resposta = dict(zip(corpus_df["id"].astype(str), corpus_df["text"]))
texto_pergunta = dict(zip(queries_df["id"].astype(str), queries_df["text"]))

print("Carregando modelo...")
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
bert = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased").to(device)
freeze_bert_except_last_layers(bert, 1)
head = PairMLP(768 * 4, 128, 0.3).to(device)

ckpt = torch.load("checkpoints/best_model_finetuned.pt", map_location=device, weights_only=False)
bert.load_state_dict(ckpt["bert_state"])
head.load_state_dict(ckpt["head_state"])
bert.eval()
head.eval()

print("Carregando splits...")
ds = load_dataset("eduagarcia/MilkQA")
test_split = ds["test"]

print("Avaliando teste completo, procurando erros...")
unique_qids = sorted({row["query-id"] for row in test_split})
unique_cids = sorted({c for row in test_split for c in row["candidates-ids"]})

with torch.no_grad():
    q_embs_all = encode([texto_pergunta[q] for q in unique_qids], tokenizer, bert, device, batch_size=16)
    c_embs_all = encode([texto_resposta[c] for c in unique_cids], tokenizer, bert, device, batch_size=16)
q_emb_by_id = dict(zip(unique_qids, q_embs_all))
c_emb_by_id = dict(zip(unique_cids, c_embs_all))

erros = []
with torch.no_grad():
    for row in test_split:
        qid, pos_id, cands = row["query-id"], row["positive-doc-id"], row["candidates-ids"]
        q_emb = q_emb_by_id[qid].unsqueeze(0).expand(len(cands), -1)
        a_emb = torch.stack([c_emb_by_id[c] for c in cands])
        feats = pair_features(q_emb, a_emb)
        scores = torch.sigmoid(head(feats)).numpy()
        order = np.argsort(-scores)
        ranked_ids = [cands[i] for i in order]
        rank = ranked_ids.index(pos_id) + 1
        if rank > 1:
            erros.append({
                "qid": qid, "pos_id": pos_id, "rank": rank,
                "escolhida_id": ranked_ids[0],
                "score_escolhida": float(scores[order[0]]),
                "score_correta": float(scores[cands.index(pos_id)]),
            })

print(f"\n{len(erros)} erros de {len(test_split)} perguntas (rank > 1)")
erros_ordenados = sorted(erros, key=lambda e: e["rank"], reverse=True)

print("\n===== 5 piores erros (resposta correta ficou mais longe no ranking) =====")
for e in erros_ordenados[:5]:
    print(f"\n--- QID {e['qid']} | resposta correta ficou na posicao {e['rank']}/50 ---")
    print(f"PERGUNTA: {texto_pergunta[e['qid']][:400]}")
    print(f"\nRESPOSTA ESCOLHIDA (errada, score={e['score_escolhida']:.3f}): {texto_resposta[e['escolhida_id']][:400]}")
    print(f"\nRESPOSTA CORRETA (score={e['score_correta']:.3f}): {texto_resposta[e['pos_id']][:400]}")
    print("=" * 80)
