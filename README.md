# Fruits-360 · Comparación de modelos de Deep Learning (Grupo 7)

Componente Práctico 2 · Inteligencia Artificial · UNEMI

Clasificación de 4 frutas (Apple Red 1, Banana 1, Orange 1, Strawberry 1) del dataset
Fruits-360 con cuatro modelos: **CNN propia, MobileNetV2, EfficientNetB0 y ConvNeXtTiny**,
bajo las mismas condiciones experimentales.

## Estructura

```
src/
  config.py      Configuración única (clases, semilla, hiperparámetros)
  particion.py   Lee y valida la partición fija desde carpetas
  data.py        Pipelines tf.data

  models.py      Las 4 arquitecturas
  train.py       Entrena UN modelo (tiempo, parámetros, curvas)
  evaluate.py    Evalúa UN modelo en Test (métricas, matriz, errores)
  compare.py     Une los 4 resultados en la tabla comparativa
scripts/
  prepare_data.py  Deja el dataset en data/
notebooks/
  colab_runner.ipynb  Ejecuta el entrenamiento en la GPU de Colab
results/         Resultados por modelo (SÍ se suben a Git: son livianos)
docs/FLUJO_GIT.md  Cómo trabajar en equipo con Git
```

## Uso rápido

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
python scripts/prepare_data.py --zip ruta/Grupo7_Fruits360.zip

python -m src.train    --model mobilenetv2
python -m src.evaluate --model mobilenetv2
python -m src.compare
```

Modelos: `cnn`, `mobilenetv2`, `efficientnetb0`, `convnexttiny`.

> El entrenamiento necesita GPU para ser razonable (sobre todo ConvNeXtTiny).
> Sin GPU local, usar `notebooks/colab_runner.ipynb` (ver docs/FLUJO_GIT.md).

## Reparto del trabajo

| Integrante | Modelo | Rama |
|---|---|---|
| (nombre) | cnn | `modelo/cnn` |
| (nombre) | mobilenetv2 | `modelo/mobilenetv2` |
| (nombre) | efficientnetb0 | `modelo/efficientnetb0` |
| (nombre) | convnexttiny | `modelo/convnexttiny` |

## Reglas del grupo

1. `src/config.py` es común a los 4 modelos: solo se cambia con Pull Request aprobado por todos.
2. El conjunto **Test** se usa una sola vez por modelo, al final (`src.evaluate`).
3. No se suben datos (`data/`) ni modelos (`models/`): son pesados. Se comparten por Drive.
4. Cada resultado nuevo se sube como cambio en `results/<modelo>/`.
