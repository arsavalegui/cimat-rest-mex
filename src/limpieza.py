"""Limpieza del corpus Rest-Mex / MeIA.

Hallazgos del EDA (notebooks/01_eda.ipynb) que motivan cada paso:
- Miles de reseñas terminan en "Más" o "...Más": el botón "ver más" de
  TripAdvisor quedó pegado al texto durante el scraping.
- 178 filas duplicadas exactas en el train completo.
- Polarity viene como float (1.0–5.0).
- 2 títulos nulos.
"""

import re

import pandas as pd

# El artefacto aparece pegado al final, con o sin puntos suspensivos:
# "volveremos a venir otra vez.Más" / "La comida bien servida. La...Más"
_SUFIJO_MAS = re.compile(r"\s*(?:\.\.\.)?\s*Más$")
_ESPACIOS = re.compile(r"\s+")


def limpiar_texto(texto: str) -> str:
    """Quita el sufijo 'Más' de TripAdvisor y normaliza espacios."""
    if not isinstance(texto, str):
        return ""
    texto = _SUFIJO_MAS.sub("", texto)
    return _ESPACIOS.sub(" ", texto).strip()


def texto_modelo(titulo: str, resena: str) -> str:
    """Entrada del modelo: título + reseña (el título condensa el sentimiento)."""
    titulo, resena = limpiar_texto(titulo), limpiar_texto(resena)
    if titulo and not titulo.endswith((".", "!", "?")):
        titulo += "."
    return f"{titulo} {resena}".strip()


def preparar(df: pd.DataFrame, con_etiquetas: bool = True) -> pd.DataFrame:
    """Devuelve un DataFrame con la columna `texto` lista para el tokenizador.

    Con etiquetas: castea Polarity a int y elimina duplicados de texto.
    """
    df = df.copy()
    titulo = df["Title"] if "Title" in df.columns else ""
    df["texto"] = [
        texto_modelo(t, r) for t, r in zip(titulo, df["Review"], strict=False)
    ] if "Title" in df.columns else df["Review"].map(limpiar_texto)

    if con_etiquetas:
        if "Polarity" in df.columns:
            df["Polarity"] = df["Polarity"].astype(int)
        df = df.drop_duplicates(subset=["texto"]).reset_index(drop=True)
    return df
