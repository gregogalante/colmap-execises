# relocation.py — Contesto e Logiche

## Scopo

`relocation.py` stima la posa 6-DoF (posizione + orientamento) di un'immagine query all'interno di una scena già ricostruita con COLMAP. In altre parole: dato un dataset con ricostruzione SfM esistente e una nuova immagine, risponde alla domanda *"da dove è stata scattata questa foto?"*.

---

## Dipendenze dal progetto

| File | Ruolo |
|---|---|
| `pipeline.py` | Costruisce la ricostruzione SfM dal dataset (da eseguire prima di relocation) |
| `datasets/{name}/sfm/0/` | Ricostruzione binaria COLMAP (cameras.bin, images.bin, points3D.bin) |
| `datasets/{name}/database.db` | Database SQLite COLMAP con keypoints e descrittori SIFT |
| `libs/read_write_model.py` | Utility I/O modello COLMAP (non usata direttamente in reloc) |

---

## Pipeline di localizzazione

### Step 1 — Carica la ricostruzione
```python
recon = pycolmap.Reconstruction("datasets/home/sfm/0")
```
Carica in memoria le `N` camere, le `M` immagini e i `P` punti 3D della mappa.

### Step 2 — Costruisce il descriptor index sui punti 3D
Ogni punto 3D nella ricostruzione ha un **track**: la lista di tutte le osservazioni `(image_id, point2D_idx)` che lo hanno visto.

Per ogni punto 3D si prende il **primo elemento del track** e si legge il descrittore SIFT corrispondente dal database SQLite:

```sql
SELECT image_id, rows, cols, data FROM descriptors
-- data è un BLOB numpy: dtype=uint8, shape=(N_keypoints, 128)
```

Il risultato è un array `desc_index` di shape `(P, 128)` in `float32` (conversione necessaria per cKDTree).

**Perché il primo elemento del track?**
Mediare i descrittori uint8 di più osservazioni darebbe valori non validi (overflow, perdita di quantizzazione SIFT). Usarne uno solo è la prassi standard (e.g., Active Search).

### Step 3 — Prepara l'immagine query
L'immagine viene ridimensionata al massimo `IMAGE_MAX_DIMENSION = 1024` px (come in `pipeline.py`) e salvata in una directory temporanea.

### Step 4 — Estrae feature SIFT dalla query
```python
pycolmap.extract_features(tmp_db_path, tmp_dir, camera_mode=pycolmap.CameraMode.SINGLE)
```
`pycolmap.extract_features` richiede una **directory** (non un file) come secondo argomento. I risultati vengono scritti nel database temporaneo e poi riletti via SQLite.

- `keypoints`: `(M, 6)` float32 — colonne 0,1 sono x,y pixel
- `descriptors`: `(M, 128)` uint8

### Step 5 — Matching descrittori (Lowe's ratio test)
Si costruisce un `cKDTree` sull'indice 3D e si cercano i 2 vicini più prossimi per ogni descrittore query:

```python
tree = cKDTree(desc_index)
dists, idxs = tree.query(query_descs, k=2, workers=-1)
ratio_mask = dists[:, 0] / (dists[:, 1] + 1e-8) < ratio  # default ratio=0.75
```

Il **ratio test di Lowe** filtra i match ambigui: se il match più vicino è molto più vicino del secondo, il match è affidabile.

### Step 6 — Assembla corrispondenze 2D–3D
```python
points2D = query_kp[q_idxs, 0:2]   # (K, 2) — coordinate pixel nella query
points3D = xyz_array[db_idxs]       # (K, 3) — coordinate mondo dei punti 3D
```

### Step 7 — Modello camera per la query
Priorità:
1. **EXIF**: `pycolmap.infer_camera_from_image(path)` — usa la lunghezza focale dai metadati
2. **Fallback**: prende la camera più comune della ricostruzione e scala la focale alle dimensioni dell'immagine query

### Step 8 — PnP RANSAC
```python
result = pycolmap.absolute_pose_estimation(points2D, points3D, camera)
```
Stima la trasformazione `cam_from_world` (una `Rigid3d`) che porta coordinate mondo → camera.

**Convenzione geometrica:**
```
p_cam = R * p_world + t
```
La posizione della camera nel mondo (punto fisico dove si trova la camera) si calcola come:
```python
camera_center = -R.T @ t
```

### Step 9 — Output
- Posizione della camera nel sistema di riferimento mondo `[x, y, z]`
- Vettore di traslazione `t`
- Matrice di rotazione `R` (world→cam)
- Numero di inlier PnP / totale corrispondenze

---

## Utilizzo

```bash
python relocation.py --dataset datasets/home --image inputs/relocation_home.jpg
python relocation.py --dataset datasets/home --image inputs/relocation_home.jpg --ratio 0.7
```

**Argomenti:**
- `--dataset`: path alla directory del dataset (deve contenere `sfm/0/` e `database.db`)
- `--image`: path all'immagine query
- `--ratio`: soglia ratio test di Lowe (default 0.75; abbassare se pochi match)

---

## Prerequisiti

Il dataset deve essere stato processato da `pipeline.py` (o equivalente) prima di eseguire `relocation.py`. Devono esistere:
- `datasets/{name}/sfm/0/cameras.bin`
- `datasets/{name}/sfm/0/images.bin`
- `datasets/{name}/sfm/0/points3D.bin`
- `datasets/{name}/database.db`

---

## Possibili problemi

| Problema | Causa | Soluzione |
|---|---|---|
| "Insufficient matches" | Query molto diversa dalla scena, o ratio troppo alto | Abbassare `--ratio` (es. 0.6) |
| "Pose estimation failed" | Troppi match sbagliati, geometria degenere | Verificare che la query sia della stessa scena |
| Pochi inlier PnP | Immagine query parzialmente sovrapposta alla scena | Normale se l'immagine copre una piccola porzione |
| Warning import IDE | Il language server non trova i pacchetti nel venv | Ignorare, i pacchetti sono installati |
