# Football AI: Auto-etiquetado con SAM 3.1 & Fine-Tuning de TrackNetV4

Herramienta especializada para la detección y seguimiento de pelotas de fútbol en partidos amateur (fútbol 5/7/sintético con tomas elevadas), superando las limitaciones de YOLO mediante la combinación de **SAM 3.1 Multiplex** para autoetiquetado y **TrackNetV4** para tracking temporal de alta velocidad.

---

## 📋 Arquitectura de la Solución

1. **Autoetiquetador SAM 3.1 Multiplex** (`auto_label.py`):
   - Utiliza el checkpoint `sam3.1_multiplex_fp16.safetensors`.
   - Rastreo temporal de la pelota con atención espacial y memoria recurrente.
   - Filtro de plausibilidad física (`BallTrajectoryFilter`): restringe área, circularidad, velocidad máxima e interpola oclusiones breves.
   - Genera los frames extraídos y el archivo estándar `Label.csv` (`file name,visibility,x-coordinate,y-coordinate,status`) junto a un video de verificación anotado.
2. **Generador de Dataset TrackNetV4** (`prepare_dataset.py`):
   - Procesa los clips etiquetados en tensores `.npy` (`x_data_n.npy` de 9 canales para ventanas de 3 frames y `y_data_n.npy` de 3 mapas de calor gaussianos a 512×288).
   - Crea automáticamente los splits `train` y `test`.
3. **Entrenamiento / Fine-Tuning de TrackNetV4** (`train_football.py`):
   - Entrena los modelos `TrackNetV4_TypeA`, `TrackNetV4_TypeB` o `Baseline_TrackNetV2` sobre el dataset de fútbol amateur.
   - Guarda periódicamente checkpoints y el mejor modelo según F1 score (`model_best.keras`, `model_final.keras`).
4. **Inferencia y Predicción en Video** (`predict_football.py`):
   - Realiza la inferencia sobre partidos nuevos, dibujando la trayectoria estela y generando el CSV de coordenadas por frame.

---

## 🚀 Guía de Uso Rápido (con `uv`)

El proyecto utiliza **uv** y **Python 3.12** con soporte CUDA para aceleración por GPU.

### 1. Auto-etiquetar videos

Coloca tus videos (`.mp4`, `.avi`) en la carpeta `videos/`. Luego ejecuta:

```bash
# Etiquetar todos los videos de la carpeta videos/
uv run python auto_label.py

# O etiquetar un video específico
uv run python auto_label.py --video videos/clip.mp4

# Opcional: Probar los primeros N frames
uv run python auto_label.py --video videos/clip.mp4 --max_frames 300

# Opcional: Especificar un click inicial en el frame 0 (X,Y) si la pelota es difícil de detectar
uv run python auto_label.py --video videos/clip.mp4 --point 992,494
```

> Los resultados se guardarán en `Dataset/football/<nombre_video>/Clip1/` con:
> - Frames en formato `.jpg`.
> - `Label.csv` con las coordenadas normalizadas.
> - `<nombre_video>_annotated.mp4` para inspección visual.

---

### 2. Preparar el Dataset para TrackNetV4

Una vez etiquetados tus videos, compila los tensores de entrenamiento:

```bash
uv run python prepare_dataset.py
```

> Esto genera los archivos `x_data_{n}.npy` y `y_data_{n}.npy` en `processed_data/football/train/` y `processed_data/football/test/`.

---

### 3. Entrenar / Fine-Tunear TrackNetV4

Lanza el entrenamiento con los hiperparámetros deseados:

```bash
# Entrenamiento de TrackNetV4 Type A
uv run python train_football.py --model_name TrackNetV4_TypeA --epochs 30 --batch_size 2

# O entrenar con TrackNetV4 Type B
uv run python train_football.py --model_name TrackNetV4_TypeB --epochs 30 --batch_size 2

# Si tienes un checkpoint previo para fine-tuning:
uv run python train_football.py --model_path models/checkpoint_previo.keras --epochs 20
```

> Los modelos guardados se almacenarán en `models/football/` (`model_final.keras` y `model_best.keras`).

---

### 4. Predecir trayectoria en un nuevo video

Aplica tu modelo entrenado sobre cualquier video de fútbol:

```bash
uv run python predict_football.py --video_path videos/clip.mp4 --model_weights models/football/model_final.keras
```

> Genera en `predictions/`:
> - `<video>_tracknet_pred.mp4`: Video con la estela y marcador del balón.
> - `<video>_trajectory.csv`: Coordenadas exactas `Frame, Visibility, X, Y`.
