"""Demo interactiva: escribe una reseña y el modelo predice su polaridad.

Uso:
    python src/demo.py [--modelo models/polaridad-meia]
"""

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from limpieza import limpiar_texto

ETIQUETAS = {1: "muy negativa 😠", 2: "negativa 🙁", 3: "neutral 😐",
             4: "positiva 🙂", 5: "muy positiva 🤩"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", default="models/polaridad-meia")
    args = p.parse_args()

    print("Cargando modelo...")
    tokenizer = AutoTokenizer.from_pretrained(args.modelo)
    model = AutoModelForSequenceClassification.from_pretrained(args.modelo)
    model.eval()
    print("Listo. Escribe una reseña (o Enter vacío para salir).\n")

    while True:
        texto = input("Reseña> ").strip()
        if not texto:
            break
        enc = tokenizer(limpiar_texto(texto), truncation=True, max_length=128,
                        return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits[0], dim=-1)
        pred = int(probs.argmax()) + 1
        confianza = float(probs.max())
        barra = " ".join(f"{i+1}:{p:.0%}" for i, p in enumerate(probs.tolist()))
        print(f"  → {pred} ({ETIQUETAS[pred]}) con {confianza:.0%} de confianza")
        print(f"    distribución: {barra}\n")


if __name__ == "__main__":
    main()
