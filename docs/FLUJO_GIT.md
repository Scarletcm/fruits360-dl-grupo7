# Cómo trabajar en equipo con Git y VS Code

## 0. Instalación (cada integrante, una sola vez)

1. **Git**: https://git-scm.com/downloads
2. **VS Code**: https://code.visualstudio.com
3. **Python 3.10 a 3.12**
4. Extensiones de VS Code (al abrir el proyecto, VS Code las sugiere): Python, Jupyter,
   **Colab** (Google), GitHub Pull Requests y GitLens.
5. Configurar identidad en Git:

```bash
git config --global user.name  "Tu Nombre"
git config --global user.email "tu_correo@ejemplo.com"
```

## 1. Crear el repositorio (SOLO UN integrante)

1. En https://github.com crear un repositorio **privado**, p. ej. `fruits360-dl-grupo7`
   (sin README, sin .gitignore: ya vienen incluidos).
2. En *Settings → Collaborators* invitar al resto del grupo y al docente si lo pide.
3. Desde la carpeta del proyecto:

```bash
git init -b main
git add .
git commit -m "Estructura inicial del proyecto"
git remote add origin https://github.com/USUARIO/fruits360-dl-grupo7.git
git push -u origin main
```

## 2. Los demás integrantes

```bash
git clone https://github.com/USUARIO/fruits360-dl-grupo7.git
cd fruits360-dl-grupo7
code .
```

## 3. Flujo diario (una rama por modelo o tarea)

```bash
git checkout main
git pull                                  # siempre antes de empezar
git checkout -b modelo/mobilenetv2        # tu rama

# ... trabajas, entrenas, generas results/mobilenetv2/ ...

git status                                # revisa qué cambió
git add src/ results/mobilenetv2/
git commit -m "Entrena MobileNetV2: resultados y curvas"
git push -u origin modelo/mobilenetv2
```

En GitHub: **Compare & pull request** → otro integrante revisa → **Merge**.
Después, todos: `git checkout main && git pull`.

Buenos mensajes de commit: `Agrega evaluación en Test`, `Corrige ruta de datos`,
`Actualiza hiperparámetros (aprobado por el grupo)`.

## 4. Qué NO subir a Git

- `data/` (el dataset) y `models/` (los .keras pesan decenas o cientos de MB).
  Compártanlos por Google Drive. El `.gitignore` ya los excluye.
- El informe en Word (genera conflictos imposibles de fusionar). Manténganlo en
  Drive/Word compartido. Solo el PDF final puede ir en `reports/`.

## 5. Evitar conflictos

- Cada quien edita **sus** archivos. `config.py` y `models.py` los tocan solo mediante Pull Request.
- Haz commits pequeños y frecuentes, y `git pull` antes de empezar.
- Si hay conflicto: VS Code marca las zonas en conflicto; elige *Accept Current / Incoming / Both*,
  guarda, y luego `git add archivo` y `git commit`.

## 6. Entrenar con la GPU de Colab desde VS Code

TensorFlow ya no usa GPU de forma nativa en Windows (solo con WSL2 o en Linux), por lo que lo
más simple es entrenar en Colab:

1. En VS Code, instala la extensión **Colab** (Google) y abre `notebooks/colab_runner.ipynb`.
2. *Select Kernel → Colab → New Colab Server* (inicia sesión con Google y elige GPU).
3. Completa las variables de la primera celda: `REPO_URL`, `BRANCH`, `MODELO` y el origen del dataset.
4. Ejecuta las celdas: clona el repo en Colab, prepara los datos, entrena, evalúa y guarda
   los resultados en un ZIP en tu Drive.
5. Descarga el ZIP, copia su contenido a `results/<modelo>/` de tu repositorio local, y
   haz `commit` + `push` (sección 3).

Nota: el repositorio debe ser **público** para clonarlo sin credenciales desde Colab, o usar un
token personal de GitHub en la URL. Si es privado, lo más sencillo es descargar el repo como ZIP
y subirlo a Colab, o hacer público el repositorio solo mientras se entrena.

## 7. Cuando los 4 modelos estén entrenados

```bash
git checkout main && git pull
python -m src.compare        # genera results/tabla_comparativa.csv y .md
```

Con esa tabla, las curvas (`results/*/curvas.png`) y las matrices de confusión se redactan los
puntos 10 a 13 del informe.
