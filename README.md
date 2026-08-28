# Rest-Mex 2025 — Transfer Learning y Fine-Tuning

Sistema de análisis de sentimientos y clasificación de reseñas turísticas en español, desarrollado como proyecto final del **Diplomado en Cómputo de Soluciones Avanzadas (CIMAT)**. Basado en el desafío [REST-MEX 2025](https://sites.google.com/cimat.mx/rest-mex-2025/), que usa reseñas reales de turistas sobre Pueblos Mágicos de México.

## Objetivo

Dado el texto de una reseña turística, el modelo predice tres cosas:

1. **Polaridad del sentimiento**: escala de 1 (muy negativo) a 5 (muy positivo).
2. **Tipo de lugar**: `Hotel`, `Restaurant` o `Attractive`.
3. **Pueblo Mágico**: uno de 40 pueblos posibles.

El enfoque es *transfer learning*: fine-tuning de un modelo de lenguaje preentrenado en español sobre el corpus del reto.

## Dataset

| Corpus | Instancias | Contenido |
|---|---|---|
| Entrenamiento | 208,051 | Title, Review, Polarity, Town, Region, Type |
| Prueba | 89,166 | ID, Title, Review (sin etiquetas) |

**Nota**: las clases están fuertemente desbalanceadas (p. ej. polaridad 5 tiene ~136k instancias vs ~5k de polaridad 1).

Los datos no se versionan en el repo por su tamaño. Para descargarlos:

```bash
python3 -m venv .venv
.venv/bin/pip install gdown
cd data
../.venv/bin/gdown 1VNiHnom0bLGke1IIH3oetZbvVreencPF   # versión reducida (MeIA)
../.venv/bin/gdown 1xf0nGF29hFWBg_rXSJoFGcq86T9ekit8   # entrenamiento completo
../.venv/bin/gdown 1k9s0_4D0vvfLYiCzcGfzRE9_yU1L_BTW   # prueba completa
unzip -o '*.zip'
```

## Formato de salida

El sistema genera un `.txt` con una línea por reseña de prueba, separada por tabuladores:

```
rest-mex	0	5	Sayulita	Restaurant
rest-mex	1	4	Bacalar	Attractive
```

Columnas: `TaskName`, `ID de la instancia`, `Polaridad`, `Pueblo Mágico`, `Tipo de lugar`.

## Estructura del proyecto

```
├── data/          # datasets (ignorados en git)
├── docs/          # enunciado del proyecto y referencias
├── notebooks/     # análisis exploratorio y experimentos
└── src/           # código del sistema final
```
