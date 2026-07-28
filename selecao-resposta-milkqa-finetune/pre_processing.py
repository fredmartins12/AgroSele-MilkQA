"""
Pre-processamento (Artefato 2, adaptado): extrai embeddings BERTimbau
CONGELADOS para o corpus de respostas e para as perguntas do MilkQA
(Embrapa Gado de Leite), dataset real de perguntas e respostas do
atendimento ao produtor rural.

Diferenca em relacao ao projeto anterior (classificacao financeira): aqui
nao ha uma "amostra = um texto com um rotulo fixo". A tarefa e SELECAO DE
RESPOSTA -- dado uma pergunta, escolher a resposta certa entre um pool de
50 candidatas. Por isso o cache e organizado como DOIS dicionarios
consolidados (id -> embedding), um para perguntas e um para respostas, em
vez de um arquivo .pt por amostra: e o formato que permite lookup rapido
por id ao montar os pares (pergunta, candidata) no model.py, sem precisar
abrir milhares de arquivos pequenos a cada epoca.

Uso: py pre_processing.py
Salva em cache/corpus_embeddings.pt e cache/queries_embeddings.pt
Tambem exporta datasets/corpus.csv e datasets/queries.csv (texto bruto,
para transparencia/reprodutibilidade sem depender de acesso a internet
nas proximas execucoes).
"""
import os
import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_LENGTH = 256
BATCH_SIZE = 16


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts(texts, tokenizer, model, device):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            enc = tokenizer(batch, return_tensors="pt", truncation=True,
                             max_length=MAX_LENGTH, padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            emb = mean_pooling(out.last_hidden_state, enc["attention_mask"]).cpu()
            embeddings.append(emb)
            if (i // BATCH_SIZE + 1) % 20 == 0:
                print(f"    {i + len(batch)}/{len(texts)} processados")
    return torch.cat(embeddings, dim=0)


def main():
    os.makedirs("cache", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)

    print("Baixando/carregando MilkQA (corpus e queries) do Hugging Face...")
    corpus = load_dataset("eduagarcia/MilkQA", "corpus")["corpus"]
    queries = load_dataset("eduagarcia/MilkQA", "queries")["queries"]
    print(f"Corpus (respostas): {len(corpus)} | Queries (perguntas): {len(queries)}")

    # exporta CSVs para reprodutibilidade offline
    corpus.to_pandas()[["id", "text"]].to_csv("datasets/corpus.csv", index=False, encoding="utf-8")
    queries.to_pandas()[["id", "subject", "text"]].to_csv("datasets/queries.csv", index=False, encoding="utf-8")

    print(f"Carregando {MODEL_NAME} (congelado)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Device: {device}")

    t0 = time.time()
    print("\nExtraindo embeddings do corpus (respostas)...")
    corpus_ids = corpus["id"]
    corpus_emb = embed_texts(corpus["text"], tokenizer, model, device)
    corpus_dict = {cid: corpus_emb[i] for i, cid in enumerate(corpus_ids)}
    torch.save(corpus_dict, "cache/corpus_embeddings.pt")
    print(f"  Salvo: cache/corpus_embeddings.pt ({len(corpus_dict)} respostas, "
          f"dim={corpus_emb.shape[1]})")

    print("\nExtraindo embeddings das queries (perguntas)...")
    query_ids = queries["id"]
    query_emb = embed_texts(queries["text"], tokenizer, model, device)
    query_dict = {qid: query_emb[i] for i, qid in enumerate(query_ids)}
    torch.save(query_dict, "cache/queries_embeddings.pt")
    print(f"  Salvo: cache/queries_embeddings.pt ({len(query_dict)} perguntas, "
          f"dim={query_emb.shape[1]})")

    print(f"\nConcluido em {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
