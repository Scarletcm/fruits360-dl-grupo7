"""Deja el dataset en data/fruits360_preprocesado/{Train,Validation,Test}/<clase>/.

Uso:
    python scripts/prepare_data.py --zip ruta/al/archivo.zip
    python scripts/prepare_data.py --url https://...
"""
import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as cfg  # noqa: E402


def buscar_raiz(base):
    """Busca la carpeta que contiene Train, Validation y Test."""
    for carpeta in [base, *[p for p in base.rglob("*") if p.is_dir()]]:
        if all((carpeta / c).is_dir() for c in cfg.CONJUNTOS):
            return carpeta
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", help="Ruta a un ZIP ya descargado")
    ap.add_argument("--url", help="URL pública del ZIP")
    args = ap.parse_args()
    if not (args.zip or args.url):
        ap.error("Indica --zip o --url")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        archivo = Path(args.zip) if args.zip else tmp / "dataset.zip"
        if args.url:
            print("Descargando...")
            urllib.request.urlretrieve(args.url, archivo)
        print("Extrayendo...")
        with zipfile.ZipFile(archivo) as z:
            z.extractall(tmp / "contenido")
        raiz = buscar_raiz(tmp / "contenido")
        if raiz is None:
            raise SystemExit("El ZIP no contiene las carpetas Train, Validation y Test.")
        if cfg.DATA_DIR.exists():
            shutil.rmtree(cfg.DATA_DIR)
        cfg.DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(raiz), str(cfg.DATA_DIR))

    from src.particion import listar_particion  # valida cantidades
    df = listar_particion()
    print("Dataset listo en", cfg.DATA_DIR)
    print(df.groupby("conjunto").size().to_string())


if __name__ == "__main__":
    main()
