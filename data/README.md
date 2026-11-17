# Datos

Esta carpeta NO se sube a Git. El dataset debe quedar así:

```
data/fruits360_preprocesado/
├── Train/        (Apple Red 1, Banana 1, Orange 1, Strawberry 1)
├── Validation/
└── Test/
```

Para obtenerlo:

```bash
python scripts/prepare_data.py --zip ruta/al/Grupo7_Fruits360.zip
# o, si el ZIP está en una URL pública:
python scripts/prepare_data.py --url https://...
```
