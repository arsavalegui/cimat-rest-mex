"""Genera el archivo de salida .txt en el formato del reto.

Formato (tab-separado, una línea por reseña de prueba):
    TaskName\tID\tPolaridad\tPuebloMagico\tTipo

Para la versión reducida (MeIA) el test trae Town y Type como columnas, así
que solo se predice la polaridad y las otras dos se copian. Para el test
completo (Rest-Mex) se necesitarán los modelos de pueblo y tipo.

Uso:
    python src/predecir.py \
        --modelo models/polaridad-meia \
        --test data/Datos-MeIA-Reto-01/MeIA_2025_test_wo_labels.xlsx \
        --salida salidas/meia_polaridad.txt
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from limpieza import preparar


def predecir_polaridad(textos, modelo_dir, batch=32, max_len=128):
    tokenizer = AutoTokenizer.from_pretrained(modelo_dir)
    model = AutoModelForSequenceClassification.from_pretrained(modelo_dir)
    model.eval()
    torch.set_num_threads(12)

    preds = []
    with torch.no_grad():
        for i in range(0, len(textos), batch):
            enc = tokenizer(
                list(textos[i : i + batch]),
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            )
            logits = model(**enc).logits
            preds.extend((logits.argmax(-1) + 1).tolist())  # clases 0–4 → 1–5
            if (i // batch) % 25 == 0:
                print(f"{i + len(enc['input_ids'])}/{len(textos)}", flush=True)
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", default="models/polaridad-meia")
    p.add_argument("--test", default="data/Datos-MeIA-Reto-01/MeIA_2025_test_wo_labels.xlsx")
    p.add_argument("--salida", default="salidas/meia_polaridad.txt")
    p.add_argument("--task-name", default="rest-mex")
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()

    ruta = Path(args.test)
    df = pd.read_excel(ruta) if ruta.suffix == ".xlsx" else pd.read_csv(ruta)
    df = preparar(df, con_etiquetas=False)

    polaridad = predecir_polaridad(df["texto"].tolist(), args.modelo, args.batch)

    # Town/Type: del propio test si vienen (MeIA); si no, pendiente de sus modelos
    pueblo = df["Town"] if "Town" in df.columns else ["?"] * len(df)
    tipo = df["Type"] if "Type" in df.columns else ["?"] * len(df)
    ids = df["ID"] if "ID" in df.columns else range(len(df))

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with salida.open("w", encoding="utf-8") as f:
        for i, pol, pu, ti in zip(ids, polaridad, pueblo, tipo, strict=True):
            f.write(f"{args.task_name}\t{i}\t{pol}\t{pu}\t{ti}\n")
    print(f"{len(df)} predicciones → {salida}", flush=True)


if __name__ == "__main__":
    main()
