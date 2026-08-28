# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Rest-Mex 2025 — Entrenamiento completo en Colab
#
# Fine-tuning de BETO para las **tres tareas** del reto (polaridad, tipo de
# lugar y Pueblo Mágico) con el corpus completo (208k reseñas), y generación
# del archivo de salida final con las 89,166 predicciones del test.
#
# **Antes de ejecutar**: `Entorno de ejecución → Cambiar tipo de entorno de
# ejecución → GPU (T4)`. Luego `Entorno de ejecución → Ejecutar todo`.
# Duración estimada: ~2.5–3 h. Cada modelo se respalda en Drive al terminar,
# así que si Colab se desconecta, al re-ejecutar se salta lo ya entrenado.

# %%
import torch

assert torch.cuda.is_available(), "Activa la GPU: Entorno de ejecución → Cambiar tipo → T4"
print(torch.cuda.get_device_name(0))

# %% [markdown]
# ## Setup: repo, dependencias y datos

# %%
# %cd /content
# !test -d cimat-rest-mex || git clone -q https://github.com/arsavalegui/cimat-rest-mex.git
# %cd /content/cimat-rest-mex
# !pip install -q gdown transformers scikit-learn
# !test -f data/Rest-Mex_2025_train.csv || (cd data && gdown -q 1xf0nGF29hFWBg_rXSJoFGcq86T9ekit8 && gdown -q 1k9s0_4D0vvfLYiCzcGfzRE9_yU1L_BTW && unzip -o -q '*.zip')
# !ls -lh data/

# %%
from google.colab import drive

drive.mount("/content/drive")
RESPALDO = "/content/drive/MyDrive/cimat-rest-mex"
# !mkdir -p {RESPALDO}

# %% [markdown]
# ## Carga y limpieza (reutiliza `src/limpieza.py` del repo)

# %%
import sys

sys.path.insert(0, "/content/cimat-rest-mex/src")

import pandas as pd
from limpieza import preparar

train = preparar(pd.read_csv("data/Rest-Mex_2025_train.csv"))
test = preparar(pd.read_excel("data/Rest-Mex_2025_test.xlsx"), con_etiquetas=False)
print(f"train limpio: {len(train):,} | test: {len(test):,}")

# %% [markdown]
# ## Función genérica de fine-tuning
#
# La misma receta validada en CPU con la versión reducida
# (`src/entrenar_polaridad.py`), adaptada a GPU: fp16, batch 32, padding
# dinámico y pesos de clase por el desbalanceo (25:1 en polaridad, 63:1 en
# pueblo).

# %%
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

MODELO_BASE = "dccuchile/bert-base-spanish-wwm-uncased"
MAX_LEN = 128
BATCH = 32
DEVICE = "cuda"


class ResenasDataset(Dataset):
    def __init__(self, textos, etiquetas, tokenizer):
        self.enc = tokenizer(list(textos), truncation=True, max_length=MAX_LEN)
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


def entrenar_tarea(nombre, textos, etiquetas_texto, epochs=2, lr=2e-5):
    """Fine-tuning de una tarea de clasificación. Devuelve la ruta del modelo.

    Si ya existe un respaldo en Drive (corrida anterior), se lo salta.
    """
    salida = Path(f"{RESPALDO}/modelo-{nombre}")
    if (salida / "config.json").exists():
        print(f"[{nombre}] ya entrenado en {salida}, saltando")
        return salida

    clases = sorted(pd.Series(etiquetas_texto).unique())
    a_indice = {c: i for i, c in enumerate(clases)}
    y = np.array([a_indice[e] for e in etiquetas_texto])

    X_tr, X_va, y_tr, y_va = train_test_split(
        list(textos), y, test_size=0.02, stratify=y, random_state=42
    )
    print(f"[{nombre}] {len(clases)} clases | train {len(X_tr):,} | val {len(X_va):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODELO_BASE,
        num_labels=len(clases),
        id2label={i: str(c) for c, i in a_indice.items()},
        label2id={str(c): i for c, i in a_indice.items()},
    ).to(DEVICE)

    collate = hacer_collate(tokenizer)
    dl_tr = DataLoader(
        ResenasDataset(X_tr, y_tr, tokenizer),
        batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=2,
    )
    dl_va = DataLoader(
        ResenasDataset(X_va, y_va, tokenizer),
        batch_size=BATCH * 2, collate_fn=collate, num_workers=2,
    )

    frec = np.bincount(y_tr, minlength=len(clases))
    pesos = torch.tensor(frec.sum() / (len(frec) * frec), dtype=torch.float32)
    criterio = torch.nn.CrossEntropyLoss(weight=pesos.to(DEVICE))

    pasos = len(dl_tr) * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = get_linear_schedule_with_warmup(opt, int(pasos * 0.06), pasos)
    scaler = torch.amp.GradScaler("cuda")

    inicio = time.time()
    for epoca in range(1, epochs + 1):
        model.train()
        perdida = 0.0
        for paso, lote in enumerate(dl_tr, 1):
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                out = model(
                    input_ids=lote["input_ids"].to(DEVICE),
                    attention_mask=lote["attention_mask"].to(DEVICE),
                )
                loss = criterio(out.logits, lote["labels"].to(DEVICE))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            perdida += loss.item()
            if paso % 500 == 0:
                v = paso * BATCH / (time.time() - inicio)
                print(f"[{nombre}] ép {epoca} {paso}/{len(dl_tr)} "
                      f"| loss {perdida/paso:.4f} | {v:.0f} ej/s", flush=True)

        model.eval()
        preds, reales = [], []
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for lote in dl_va:
                logits = model(
                    input_ids=lote["input_ids"].to(DEVICE),
                    attention_mask=lote["attention_mask"].to(DEVICE),
                ).logits
                preds.extend(logits.argmax(-1).cpu().tolist())
                reales.extend(lote["labels"].tolist())
        met = {
            "accuracy": round(accuracy_score(reales, preds), 4),
            "f1_macro": round(f1_score(reales, preds, average="macro"), 4),
        }
        print(f"[{nombre}] época {epoca} → {met}", flush=True)

    salida.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(salida)
    tokenizer.save_pretrained(salida)
    met["duracion_min"] = round((time.time() - inicio) / 60, 1)
    (salida / "metricas.json").write_text(json.dumps(met, indent=2))
    print(f"[{nombre}] guardado en {salida} | {met['duracion_min']} min")
    return salida


# %% [markdown]
# ## Entrenar las tres tareas
#
# Cada una se respalda en Drive al terminar. La polaridad usa las etiquetas
# 1–5, tipo usa 3 clases y pueblo 40.

# %%
ruta_polaridad = entrenar_tarea("polaridad", train["texto"], train["Polarity"])

# %%
ruta_tipo = entrenar_tarea("tipo", train["texto"], train["Type"])

# %%
ruta_pueblo = entrenar_tarea("pueblo", train["texto"], train["Town"])

# %% [markdown]
# ## Predicción del test completo y archivo final

# %%
def predecir(ruta_modelo, textos):
    tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)
    model = AutoModelForSequenceClassification.from_pretrained(ruta_modelo).to(DEVICE)
    model.eval()
    preds = []
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for i in range(0, len(textos), 128):
            enc = tokenizer(
                textos[i : i + 128], truncation=True, max_length=MAX_LEN,
                padding=True, return_tensors="pt",
            ).to(DEVICE)
            idx = model(**enc).logits.argmax(-1).cpu().tolist()
            preds.extend(model.config.id2label[j] for j in idx)
            if i % 12800 == 0:
                print(f"{i}/{len(textos)}", flush=True)
    return preds


textos_test = test["texto"].tolist()
pol = predecir(ruta_polaridad, textos_test)
tip = predecir(ruta_tipo, textos_test)
pue = predecir(ruta_pueblo, textos_test)

# %%
salida_txt = Path(f"{RESPALDO}/rest-mex_final.txt")
with salida_txt.open("w", encoding="utf-8") as f:
    for i, p, pb, t in zip(test["ID"], pol, pue, tip, strict=True):
        f.write(f"rest-mex\t{i}\t{p}\t{pb}\t{t}\n")
print(f"{len(textos_test):,} líneas → {salida_txt} (también quedó en tu Drive)")

# %%
# Verificación rápida del formato
# !head -3 {salida_txt}
# !wc -l {salida_txt}

# %% [markdown]
# ## Listo
#
# El archivo `rest-mex_final.txt` y los tres modelos quedaron en
# `MyDrive/cimat-rest-mex/`. Ese `.txt` es el entregable del reto.
