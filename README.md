# CIRCE — esperimento DINOv2 frozen

Baseline locale di Livello 1 per associare una fotografia parziale a uno fra tre componenti, senza classificatore, fine-tuning o training supervisionato. Il sistema crea molte patch delle fotografie complete, ne calcola gli embedding DINOv2 e confronta una query tramite cosine similarity.

Il checkpoint è [`facebook/dinov2-small`](https://huggingface.co/facebook/dinov2-small): DINOv2 ViT-S/14, usato in `eval()` con tutti i parametri frozen. Si usa il token CLS, normalizzato L2; la dimensione prevista è **384**. L'implementazione mantenuta scelta è Hugging Face Transformers.

## Requisiti e installazione (PowerShell)

Sono richiesti Windows, Python 3.10+ e Git. La GPU non è obbligatoria. Il progetto seleziona CUDA solo se PyTorch la vede; con `device: auto` passa automaticamente alla CPU.

Dalla root del repository:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# GPU NVIDIA: wheel PyTorch con CUDA 12.8 (driver NVIDIA aggiornato richiesto)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
# Oppure CPU:
# pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/00_verify_setup.py
```

L'ultimo comando scarica automaticamente il checkpoint al primo utilizzo in `models/huggingface/`, verifica il device, l'embedding di 384 elementi e la sua norma L2. Cache e pesi non sono versionati.

## Inserire le fotografie

Mettere le foto intere in `data/input/references/component_A|B|C/` e quelle parziali in `data/input/queries/component_A|B|C/`. I nomi delle cartelle delle query sono la ground truth per la valutazione. Sono supportati JPG, JPEG e PNG; per HEIC, convertire prima in JPEG/PNG (Pillow base non garantisce il decoder HEIC su Windows).

Per convertire foto iPhone HEIC/HEIF in JPEG, usa l'unico argomento richiesto dallo script. Sarà creata una cartella sorella con suffisso `_jpg`; la sorgente non viene modificata e le sottocartelle vengono mantenute:

```powershell
python scripts/convert_heic_to_jpeg.py "C:\foto\iphone"
# crea C:\foto\iphone_jpg
```

Per ispezionare visivamente una divisione teorica senza overlap, senza salvare output, usa:

```powershell
python scripts/show_patch_grid.py "C:\foto\riferimento.jpg" 224
```

Le linee verdi delimitano patch complete; un eventuale bordo arancione tratteggiato indica pixel residui che non formano una patch completa. La pipeline della feature bank usa invece l'overlap configurato nel YAML.

Per eliminare lo sfondo della scrivania configurare una ROI in `configs/default.yaml`. Il formato è `[left, top, width, height]` e la chiave può essere, per esempio, `references/component_A/foto.jpg` oppure `component_A/foto.jpg`. Non viene usata segmentazione automatica.

È più comodo selezionare le ROI con il mouse, una foto alla volta. Il comando richiede il solo folder da annotare (che deve stare sotto `references` o `queries`), salva progressivamente in `configs/rois.yaml` e tale file viene caricato automaticamente da tutte le pipeline:

```powershell
python scripts/select_rois.py data/input/references/component_A
```

Nella finestra: trascina per creare/sostituire il rettangolo, `Invio` salva e passa alla foto seguente, `N` salta/rimuove la ROI, `R` pulisce la selezione, `Q` o `Esc` salva quanto fatto ed esce.

## Esecuzione

```powershell
.\.venv\Scripts\Activate.ps1
# 1. Patch, feature bank NPZ e metadati JSON/CSV
python scripts/01_build_feature_bank.py

# 2. Scegli una singola immagine, seleziona una ROI con il mouse e mostra il matching a schermo
python scripts/02_match_queries.py "C:\Users\bovoenr94850\OneDrive - Università degli Studi di Padova\24-DINO\patch_embeddings\data\input\queries\component_C\IMG_3752.jpg"
# alternativa: forza aggregazione max
python scripts/02_match_queries.py "C:\foto\mia_query.jpg" --method max

# 3. Valuta entrambe le aggregazioni
python scripts/03_evaluate.py
```

Il default crea patch da 224×224 pixel nell'immagine originale con 50% di overlap e scala `1.0`. Questo è distinto dalla risoluzione a cui il processor del checkpoint adatta ogni patch. Le finestre sono sempre complete; con `cover_edges: true` viene aggiunta una finestra completa allineata al bordo. Modificare `patches.size`, `overlap`, `scales`, `batch_size`, `device`, `aggregation.top_k` e `aggregation.method` nel YAML per esperimenti diversi.

## Output

- `data/wip/patches/`: patch JPEG visibili, raggruppate per componente.
- `data/wip/embeddings/feature_bank.npz`: matrice N×384; `feature_bank_metadata.json`/CSV conserva una riga allineata per embedding, coordinate, scala, origine e modello.
- `02_match_queries.py` apre una singola immagine scelta da CLI. Dopo la ROI disegnata a mouse, estrae una patch quadrata della dimensione configurata, ne mostra il matching contro la feature bank e non crea output persistenti. `data/outputs/predictions/` resta disponibile per future esportazioni.
- `data/outputs/evaluations/`: predizioni per `max` e `top_k_mean`, metriche CSV, confusion matrix e distribuzioni di score/margini.

Gli score sono similarità cosine, non probabilità calibrate. Non sono introdotte soglie “dubbio”; il risultato espone già il margine Top-1/Top-2 per aggiungerle in futuro. Per ricostruire l'esperimento dopo aver cambiato riferimenti, parametri, ROI o modello, rieseguire prima `01_build_feature_bank.py` e quindi i due script successivi.

## Struttura

`src/circe/` contiene caricamento immagini/ROI, encoder, persistence della bank, matching e visualizzazione. `scripts/` contiene le tre pipeline e la verifica setup; `configs/default.yaml` centralizza i parametri. Dati, patch, embedding, risultati e checkpoint sono ignorati da Git; `.gitkeep` conserva le directory necessarie.
