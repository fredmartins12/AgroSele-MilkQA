"""
Variante com FINE-TUNING PARCIAL do BERTimbau, em vez de embeddings
congelados + cache (abordagem usada em ../selecao-resposta-milkqa/).

Diferenca central: aqui o BERTimbau roda DENTRO do loop de treino (nao
existe cache/*.pt reaproveitavel), com a ultima camada do encoder (de 12)
e o restante do modelo com requires_grad=False. Isso permite que a
representacao textual se especialize no vocabulario do domínio (nomes de
doenca, insumo, procedimento), ao custo de treino muito mais lento em CPU
-- cada passo de treino exige rodar o BERT inteiro (forward) mais
backward pela ultima camada, em vez de um lookup instantaneo no cache.

Por ser mais custoso, os hiperparametros de escala foram reduzidos em
relacao a ../selecao-resposta-milkqa/model.py: menos perguntas de treino,
menos negativos por positivo, sem grid search (configuracao unica), texto
truncado em max_length menor. O objetivo aqui e um teste de viabilidade
("vale a pena fine-tuning?"), nao uma varredura exaustiva.

Uso: py finetune_model.py
"""
import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_LENGTH = 128
N_UNFROZEN_LAYERS = 1     # so a ultima camada do encoder (de 12) + pooler ficam treinaveis
TRAIN_QUERIES = 2307      # rodada overnight: todas as perguntas de treino (igual a versao congelada)
N_NEG_TRAIN = 6           # perfilado com texto real: ~505ms/exemplo -> 2307*7=16149 ex/epoca ~=136min/epoca
BATCH_SIZE = 8
EPOCHS = 5                # 5 epocas ~= 11.3h de treino, cabe na janela "ate 8h de amanha" com folga
LR_HEAD = 1e-3
LR_BERT = 2e-5            # taxa de aprendizado menor para as camadas do BERT (padrao em fine-tuning)
HIDDEN = 128
DROPOUT = 0.3
DEV_EVAL_SUBSET = 15      # perguntas de dev usadas na checagem por epoca (avaliacao completa so no final)

# --- Retomada apos o PC ter reiniciado no meio da epoca 3 ---
# Checkpoint de seguranca salvo ao FIM da epoca 2 (dev(sub) MRR=0.7313).
RESUME_CHECKPOINT = "checkpoints/best_model_finetuned_INPROGRESS.pt"
RESUME_FROM_EPOCH = 2     # indice 0-based -> comeca na "epoca 3" (as 2 primeiras ja estao no checkpoint)


class PairMLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode(texts, tokenizer, bert, device, batch_size=16):
    """Codifica uma lista de textos com o BERT ATUAL (com ou sem grad,
    conforme bert.training). Usado tanto no treino quanto na avaliacao."""
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True,
                         max_length=MAX_LENGTH, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = bert(**enc)
        emb = mean_pooling(out.last_hidden_state, enc["attention_mask"])
        embs.append(emb)
    return torch.cat(embs, dim=0)


def pair_features(q_emb, a_emb):
    return torch.cat([q_emb, a_emb, torch.abs(q_emb - a_emb), q_emb * a_emb], dim=-1)


def freeze_bert_except_last_layers(bert, n_unfrozen):
    for p in bert.parameters():
        p.requires_grad = False
    total_layers = len(bert.encoder.layer)
    for layer in bert.encoder.layer[total_layers - n_unfrozen:]:
        for p in layer.parameters():
            p.requires_grad = True
    if hasattr(bert, "pooler") and bert.pooler is not None:
        for p in bert.pooler.parameters():
            p.requires_grad = True
    n_trainable = sum(p.numel() for p in bert.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in bert.parameters())
    print(f"  BERT: {n_trainable:,} / {n_total:,} parametros treinaveis "
          f"({n_trainable / n_total:.1%}, ultima(s) {n_unfrozen} camada(s) de {total_layers})")


def build_examples(split, texts_by_id, n_neg, n_queries):
    rows = list(split)
    rng = random.Random(SEED)
    rows = rng.sample(rows, min(n_queries, len(rows)))

    examples = []  # (query_text, answer_text, label)
    for row in rows:
        qid, pos_id = row["query-id"], row["positive-doc-id"]
        candidates = [c for c in row["candidates-ids"] if c != pos_id]
        negs = rng.sample(candidates, min(n_neg, len(candidates)))
        q_text = texts_by_id["queries"][qid]
        examples.append((q_text, texts_by_id["corpus"][pos_id], 1))
        for neg_id in negs:
            examples.append((q_text, texts_by_id["corpus"][neg_id], 0))
    return examples


def rank_eval_finetuned(split, texts_by_id, tokenizer, bert, head, device, limit=None):
    """Avalia por ranking. Encoda cada pergunta e cada candidata UNICA uma
    unica vez (varias perguntas do MilkQA reaproveitam as mesmas 50
    candidatas do corpus de 2657 documentos) -- sem isso, o teste completo
    (300 perguntas x 50 candidatas) recodificaria o BERT ate 15000 vezes
    em vez das ~poucos milhares de textos realmente unicos envolvidos."""
    bert.eval()
    head.eval()
    rows = list(split)[:limit] if limit else list(split)

    unique_qids = sorted({row["query-id"] for row in rows})
    unique_cids = sorted({c for row in rows for c in row["candidates-ids"]})

    with torch.no_grad():
        q_embs_all = encode([texts_by_id["queries"][q] for q in unique_qids],
                             tokenizer, bert, device, batch_size=16)
        c_embs_all = encode([texts_by_id["corpus"][c] for c in unique_cids],
                             tokenizer, bert, device, batch_size=16)
    q_emb_by_id = dict(zip(unique_qids, q_embs_all))
    c_emb_by_id = dict(zip(unique_cids, c_embs_all))

    acc1, mrr = [], []
    with torch.no_grad():
        for row in rows:
            qid, pos_id, cands = row["query-id"], row["positive-doc-id"], row["candidates-ids"]
            q_emb = q_emb_by_id[qid].unsqueeze(0).expand(len(cands), -1)
            a_emb = torch.stack([c_emb_by_id[c] for c in cands])
            feats = pair_features(q_emb, a_emb)
            scores = torch.sigmoid(head(feats)).cpu().numpy()

            order = np.argsort(-scores)
            ranked_ids = [cands[i] for i in order]
            rank = ranked_ids.index(pos_id) + 1
            acc1.append(1.0 if rank == 1 else 0.0)
            mrr.append(1.0 / rank)
    return float(np.mean(acc1)), float(np.mean(mrr))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Carregando textos (corpus.csv / queries.csv)...")
    corpus_df = pd.read_csv("datasets/corpus.csv")
    queries_df = pd.read_csv("datasets/queries.csv")
    texts_by_id = {
        "corpus": dict(zip(corpus_df["id"].astype(str), corpus_df["text"])),
        "queries": dict(zip(queries_df["id"].astype(str), queries_df["text"])),
    }

    print("Carregando splits oficiais do MilkQA...")
    ds = load_dataset("eduagarcia/MilkQA")
    train_split, dev_split, test_split = ds["train"], ds["dev"], ds["test"]

    print(f"Carregando {MODEL_NAME} para fine-tuning parcial...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert = AutoModel.from_pretrained(MODEL_NAME).to(device)
    freeze_bert_except_last_layers(bert, N_UNFROZEN_LAYERS)

    head = PairMLP(768 * 4, HIDDEN, DROPOUT).to(device)
    opt = torch.optim.AdamW([
        {"params": [p for p in bert.parameters() if p.requires_grad], "lr": LR_BERT},
        {"params": head.parameters(), "lr": LR_HEAD},
    ])
    criterion = nn.BCEWithLogitsLoss()

    print(f"\nMontando exemplos de treino ({TRAIN_QUERIES} perguntas, {N_NEG_TRAIN} negativos cada)...")
    examples = build_examples(train_split, texts_by_id, N_NEG_TRAIN, TRAIN_QUERIES)
    print(f"  {len(examples)} exemplos")

    print("\n===== Treino (fine-tuning parcial) =====")
    best_mrr, best_state = -1.0, None
    start_epoch = 0
    if os.path.exists(RESUME_CHECKPOINT):
        print(f"Retomando de {RESUME_CHECKPOINT} (epocas 1-{RESUME_FROM_EPOCH} ja concluidas)...")
        ckpt = torch.load(RESUME_CHECKPOINT, map_location=device, weights_only=False)
        bert.load_state_dict(ckpt["bert_state"])
        head.load_state_dict(ckpt["head_state"])
        best_mrr = ckpt.get("best_dev_mrr_sub", -1.0)
        best_state = {"bert": ckpt["bert_state"], "head": ckpt["head_state"]}
        start_epoch = RESUME_FROM_EPOCH
        print(f"  Pesos carregados. best_mrr(sub) anterior = {best_mrr:.4f}. "
              f"Retomando a partir da epoca {start_epoch + 1}/{EPOCHS}.")

    t_start = time.time()
    for epoch in range(start_epoch, EPOCHS):
        bert.train()
        head.train()
        rng = random.Random(SEED + epoch)
        rng.shuffle(examples)
        total_loss = 0.0

        n_batches = (len(examples) + BATCH_SIZE - 1) // BATCH_SIZE
        for bi, i in enumerate(range(0, len(examples), BATCH_SIZE)):
            batch = examples[i:i + BATCH_SIZE]
            q_texts = [b[0] for b in batch]
            a_texts = [b[1] for b in batch]
            labels = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)

            q_emb = encode(q_texts, tokenizer, bert, device, batch_size=len(batch))
            a_emb = encode(a_texts, tokenizer, bert, device, batch_size=len(batch))
            feats = pair_features(q_emb, a_emb)
            logits = head(feats)
            loss = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(batch)

            if (bi + 1) % 200 == 0:
                elapsed = time.time() - t_start
                print(f"    epoca {epoch + 1}, batch {bi + 1}/{n_batches} | "
                      f"{elapsed/3600:.2f}h decorridas", flush=True)

        # durante o treino, avalia so um subconjunto do dev (mais rapido, serve
        # para early-stopping); a avaliacao completa (dev + teste) roda 1x no final
        dev_acc1, dev_mrr = rank_eval_finetuned(dev_split, texts_by_id, tokenizer, bert, head,
                                                 device, limit=DEV_EVAL_SUBSET)
        elapsed = time.time() - t_start
        epochs_this_run = epoch - start_epoch + 1
        epochs_remaining = EPOCHS - (epoch + 1)
        eta_h = (elapsed / epochs_this_run * epochs_remaining) / 3600
        print(f"  epoca {epoch + 1}/{EPOCHS} | loss={total_loss / len(examples):.4f} | "
              f"dev(sub) Acc@1={dev_acc1:.4f} MRR={dev_mrr:.4f} | "
              f"{elapsed/3600:.2f}h decorridas | ETA restante ~{eta_h:.1f}h", flush=True)

        if dev_mrr > best_mrr:
            best_mrr = dev_mrr
            best_state = {
                "bert": {k: v.clone() for k, v in bert.state_dict().items()},
                "head": {k: v.clone() for k, v in head.state_dict().items()},
            }

        # checkpoint de seguranca a cada epoca -- se a sessao cair/precisar
        # parar no meio, nao perdemos o progresso ate aqui
        torch.save({
            "bert_state": best_state["bert"], "head_state": best_state["head"],
            "epoch": epoch + 1, "best_dev_mrr_sub": best_mrr,
        }, "checkpoints/best_model_finetuned_INPROGRESS.pt")

    bert.load_state_dict(best_state["bert"])
    head.load_state_dict(best_state["head"])

    print("\n===== Avaliacao final completa (dev inteiro + teste inteiro) =====")
    t_eval = time.time()
    dev_acc1_full, dev_mrr_full = rank_eval_finetuned(dev_split, texts_by_id, tokenizer, bert, head, device)
    print(f"Dev completo (50)  -> Acc@1={dev_acc1_full:.4f} MRR={dev_mrr_full:.4f} "
          f"({time.time() - t_eval:.0f}s)")
    t_eval = time.time()
    test_acc1, test_mrr = rank_eval_finetuned(test_split, texts_by_id, tokenizer, bert, head, device)
    print(f"Teste completo (300) -> Acc@1={test_acc1:.4f} MRR={test_mrr:.4f} ({time.time() - t_eval:.0f}s)")
    print("\nComparar com a versao congelada (../selecao-resposta-milkqa/): Acc@1=0.570 | MRR=0.679")

    torch.save({"bert_state": best_state["bert"], "head_state": best_state["head"],
                "test_acc1": test_acc1, "test_mrr": test_mrr,
                "dev_acc1": dev_acc1_full, "dev_mrr": dev_mrr_full},
               "checkpoints/best_model_finetuned.pt")
    print("\nCheckpoint salvo em checkpoints/best_model_finetuned.pt")


if __name__ == "__main__":
    main()
