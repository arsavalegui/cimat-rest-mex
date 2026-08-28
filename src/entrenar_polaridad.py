"""Fine-tuning de un modelo BERT en español para polaridad (1–5).

Primera iteración con la versión reducida (MeIA, 5,000 reseñas) para validar
el flujo completo en CPU. El mismo script sirve después para el corpus
completo cambiando --datos.

Uso:
    python src/entrenar_polaridad.py \
        --datos data/Datos-MeIA-Reto-01/MeIA_2025_train.xlsx \
        --epochs 2
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from limpieza import preparar


class ResenasDataset(Dataset):
    """Tokeniza sin padding; el collate rellena por lote (padding dinámico),
    que en este corpus (mediana ~60 tokens) ahorra ~la mitad del cómputo."""

    def __init__(self, textos, etiquetas, tokenizer, max_len):
        self.enc = tokenizer(list(textos), truncation=True, max_length=max_len)
        self.etiquetas = list(etiquetas)

    def __len__(self):
        return len(self.etiquetas)

    def __getitem__(self, i):
        return {
            "input_ids": self.enc["input_ids"][i],
            "attention_mask": self.enc["attention_mask"][i],
            "labels": self.etiquetas[i],
        }


def hacer_collate(tokenizer):
    def collate(lotes):
        etiquetas = torch.tensor([x.pop("labels") for x in lotes], dtype=torch.long)
        lote = tokenizer.pad(lotes, return_tensors="pt")
        lote["labels"] = etiquetas
        return lote

    return collate


def evaluar(model, loader, device):
    model.eval()
    preds, reales = [], []
    with torch.no_grad():
        for lote in loader:
            logits = model(
                input_ids=lote["input_ids"].to(device),
                attention_mask=lote["attention_mask"].to(device),
            ).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
            reales.extend(lote["labels"].tolist())
    preds, reales = np.array(preds), np.array(reales)
    return {
        "accuracy": round(accuracy_score(reales, preds), 4),
        "f1_macro": round(f1_score(reales, preds, average="macro"), 4),
        "mae": round(mean_absolute_error(reales + 1, preds + 1), 4),
        "f1_por_clase": {
            str(c + 1): round(v, 4)
            for c, v in enumerate(f1_score(reales, preds, average=None))
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datos", default="data/Datos-MeIA-Reto-01/MeIA_2025_train.xlsx")
    p.add_argument("--modelo", default="dccuchile/bert-base-spanish-wwm-uncased")
    p.add_argument("--salida", default="models/polaridad-meia")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--semilla", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.semilla)
    torch.set_num_threads(12)
    device = "cpu"

    ruta = Path(args.datos)
    df = pd.read_excel(ruta) if ruta.suffix == ".xlsx" else pd.read_csv(ruta)
    df = preparar(df)
    df["label"] = df["Polarity"] - 1  # clases 0–4

    tr, va = train_test_split(
        df, test_size=0.1, stratify=df["label"], random_state=args.semilla
    )
    print(f"train: {len(tr):,} | val: {len(va):,} | modelo: {args.modelo}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.modelo)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.modelo, num_labels=5
    ).to(device)

    collate = hacer_collate(tokenizer)
    dl_tr = DataLoader(
        ResenasDataset(tr["texto"], tr["label"].to_numpy(), tokenizer, args.max_len),
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate,
    )
    dl_va = DataLoader(
        ResenasDataset(va["texto"], va["label"].to_numpy(), tokenizer, args.max_len),
        batch_size=args.batch * 2,
        collate_fn=collate,
    )

    # Pesos de clase: inverso de la frecuencia, normalizado (por el desbalanceo)
    frec = tr["label"].value_counts().sort_index().to_numpy()
    pesos = torch.tensor((frec.sum() / (len(frec) * frec)), dtype=torch.float32)
    criterio = torch.nn.CrossEntropyLoss(weight=pesos.to(device))

    pasos_totales = len(dl_tr) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, int(pasos_totales * 0.1), pasos_totales)

    inicio = time.time()
    for epoca in range(1, args.epochs + 1):
        model.train()
        perdida_acum = 0.0
        for paso, lote in enumerate(dl_tr, 1):
            opt.zero_grad()
            logits = model(
                input_ids=lote["input_ids"].to(device),
                attention_mask=lote["attention_mask"].to(device),
            ).logits
            perdida = criterio(logits, lote["labels"].to(device))
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            perdida_acum += perdida.item()
            if paso % 25 == 0:
                velocidad = paso * args.batch / (time.time() - inicio)
                print(
                    f"época {epoca} paso {paso}/{len(dl_tr)} "
                    f"| loss {perdida_acum / paso:.4f} | {velocidad:.1f} ej/s",
                    flush=True,
                )
        metricas = evaluar(model, dl_va, device)
        print(f"época {epoca} → {metricas}", flush=True)

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(salida)
    tokenizer.save_pretrained(salida)
    metricas["duracion_min"] = round((time.time() - inicio) / 60, 1)
    metricas["config"] = vars(args)
    (salida / "metricas.json").write_text(json.dumps(metricas, indent=2))
    print(f"\nModelo guardado en {salida}/ | {metricas['duracion_min']} min", flush=True)


if __name__ == "__main__":
    main()
