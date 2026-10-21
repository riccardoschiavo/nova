# fine-tuning dell'ultimo blocco di DeepGaze IIE sulle fissazioni di OSIE
# uso in due passi:
#   1) python finetune_deepgaze.py --osie <cartella OSIE> --inspect-only
#      stampa i nomi dei parametri per scegliere quali allenare
#   2) python finetune_deepgaze.py --osie <cartella OSIE> --unfreeze-pattern "<pattern>"
#      allena solo i parametri il cui nome contiene il pattern

import argparse
import os
import sys

import numpy as np
import cv2

# aggiunge la cartella dello script al path per importare osie_ground_truth
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osie_ground_truth import load_fixations, points_to_heatmap


TARGET_HEIGHT, TARGET_WIDTH = 240, 320 # risoluzione fissa usata per il training


# costruisce le coppie (immagine, heatmap) da OSIE nel formato che vuole PyTorch
def build_dataset(osie_dir, max_images=None):
    mat_path = os.path.join(osie_dir, "data", "eye", "fixations.mat") # fissazioni degli osservatori
    stimuli_dir = os.path.join(osie_dir, "data", "stimuli") # immagini

    # punti di fissazione raggruppati per nome dell'immagine
    fixations = load_fixations(mat_path)
    names = list(fixations.keys())
    if max_images is not None:
        names = names[:max_images] # usa solo le prime max_images immagini

    pairs = []
    skipped = 0 # immagini presenti nel .mat ma non trovate su disco
    for name in names:
        path = os.path.join(stimuli_dir, name)
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            skipped += 1
            continue

        original_h, original_w = image_bgr.shape[:2]

        # heatmap creata alla risoluzione originale (le coordinate delle fissazioni
        # sono in quel sistema) e solo dopo ridimensionata, così il sigma della gaussiana è corretto
        heatmap = points_to_heatmap(fixations[name], (original_h, original_w))
        heatmap = cv2.resize(heatmap, (TARGET_WIDTH, TARGET_HEIGHT),
                             interpolation=cv2.INTER_AREA)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # BGR -> RGB
        image_resized = cv2.resize(image_rgb, (TARGET_WIDTH, TARGET_HEIGHT),
                                   interpolation=cv2.INTER_AREA)
        image_chw = image_resized.transpose(2, 0, 1).astype(np.float32) # HWC -> CHW, PyTorch vuole i canali per primi

        pairs.append((image_chw, heatmap.astype(np.float32)))

    if skipped:
        print(f"  ATTENZIONE: {skipped} immagini elencate in fixations.mat "
              f"non trovate in {stimuli_dir}, saltate.")

    return pairs


# crea un center bias uniforme per non dare vantaggio a nessuna regione dell'immagine,
# lo stesso usato in deepgaze.py della fase 2
def build_centerbias(height, width):
    import torch
    centerbias = np.zeros((height, width), dtype=np.float64)
    log_sum = np.log(np.exp(centerbias).sum())
    centerbias = centerbias - log_sum # normalizza: matrice di valori log(1/n)
    return torch.tensor(centerbias, dtype=torch.float32)


# carica DeepGaze IIE pre-addestrato, in PyTorch è già un modello allenabile
def load_trainable_model(device="cpu"):
    import deepgaze_pytorch
    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(device)
    return model


# stampa nome e forma di tutti i parametri, serve per scegliere --unfreeze-pattern
def inspect_parameters(model):
    named = list(model.named_parameters())
    print(f"\nParametri allenabili totali: {len(named)}\n")
    for name, p in named:
        print(f"  {name:<70} shape={tuple(p.shape)}")

    print(
        "\nCerca i nomi che sembrano appartenere agli ULTIMI moduli della "
        "rete (spesso le 'teste' di lettura finali, o i layer con indice "
        "piu' alto). La sottostringa comune a quei nomi e' il valore da "
        "passare a --unfreeze-pattern."
    )


# congela tutti i parametri e riattiva solo quelli il cui nome contiene il pattern
def select_trainable_parameters(model, unfreeze_pattern):
    for p in model.parameters():
        p.requires_grad = False # congela tutto

    trainable = []
    for name, p in model.named_parameters():
        if unfreeze_pattern in name:
            p.requires_grad = True # riattiva solo i parametri scelti
            trainable.append(p)

    return trainable # parametri da passare all'ottimizzatore


# ciclo di training: aggiorna solo i parametri scelti e salva i pesi alla fine
def train(model, trainable_params, pairs, epochs, batch_size, learning_rate, out_path):
    import torch

    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate) # ottimizza solo i parametri non congelati
    mse = torch.nn.MSELoss() # errore quadratico medio tra mappa predetta e heatmap

    # il center bias dipende solo dalle dimensioni, quindi si crea una volta sola
    centerbias = build_centerbias(TARGET_HEIGHT, TARGET_WIDTH)

    n = len(pairs)
    model.train()
    # model.train() rimette in training anche i BatchNorm congelati, che aggiornerebbero
    # running_mean/running_var a ogni forward: li si riporta in eval() per lasciarli intatti
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    for epoch in range(epochs):
        order = np.random.permutation(n) # mescola le immagini a ogni epoca
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            batch_size_actual = len(idx) # l'ultimo batch può essere più piccolo

            # tensori del batch: immagini (N, 3, H, W), heatmap (N, H, W), center bias (N, H, W)
            images = torch.tensor(np.stack([pairs[i][0] for i in idx]))
            targets = torch.tensor(np.stack([pairs[i][1] for i in idx]))
            batch_centerbias = centerbias.unsqueeze(0).repeat(batch_size_actual, 1, 1)

            optimizer.zero_grad() # azzera i gradienti del batch precedente

            log_density = model(images, batch_centerbias) # il modello restituisce una log-densità

            # log-densità -> densità, sottraendo il massimo prima di exp() per evitare overflow
            log_density_flat = log_density.view(batch_size_actual, -1)
            max_per_image = log_density_flat.max(dim=1, keepdim=True).values
            density = torch.exp(log_density_flat - max_per_image)
            # normalizza ogni mappa in [0, 1] come la heatmap target
            min_per_image = density.min(dim=1, keepdim=True).values
            max_density = density.max(dim=1, keepdim=True).values
            predicted = (density - min_per_image) / (max_density - min_per_image + 1e-8)
            predicted = predicted.view(batch_size_actual, TARGET_HEIGHT, TARGET_WIDTH)

            loss = mse(predicted, targets)
            loss.backward() # calcola i gradienti
            optimizer.step() # aggiorna i pesi

            epoch_loss += loss.item()
            n_batches += 1

        print(f"epoca {epoch + 1}/{epochs}  loss media: {epoch_loss / n_batches:.6f}")

    # salva tutti i pesi del modello (anche quelli congelati) in un file .pt
    import torch as torch_module
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch_module.save(model.state_dict(), out_path)
    print(f"\nPesi completi salvati in '{out_path}'.")


if __name__ == "__main__":
    # argomenti da riga di comando
    parser = argparse.ArgumentParser(description="Fine-tuning di DeepGaze IIE su OSIE.")
    parser.add_argument("--osie", required=True,
                        help="Cartella del repository OSIE clonato.")
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--unfreeze-pattern", type=str, default=None,
                        help="Sottostringa dei nomi dei parametri da NON congelare. "
                             "Obbligatorio a meno di --inspect-only.")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--out", default="deepgaze_finetuned.pt")
    args = parser.parse_args()

    # prepara il dataset (immagine, heatmap)
    print(f"Preparazione dati da OSIE (fino a {args.max_images} immagini)...")
    pairs = build_dataset(args.osie, max_images=args.max_images)
    print(f"Coppie (immagine, heatmap) pronte: {len(pairs)}")

    print("\nCaricamento di DeepGaze IIE...")
    model = load_trainable_model()

    # con --inspect-only stampa solo i parametri ed esce
    if args.inspect_only:
        inspect_parameters(model)
        raise SystemExit(0)

    # senza pattern non si sa cosa allenare
    if not args.unfreeze_pattern:
        raise SystemExit(
            "Serve --unfreeze-pattern. Lancia prima con --inspect-only per "
            "vedere i nomi dei parametri e scegliere il pattern giusto."
        )

    trainable_params = select_trainable_parameters(model, args.unfreeze_pattern)
    total = sum(1 for _ in model.parameters())
    print(f"Parametri selezionati per il training: {len(trainable_params)} su {total} totali.")
    # messaggio di errore se il pattern non corrisponde a nessun parametro
    if len(trainable_params) == 0:
        raise SystemExit(
            f"Nessun parametro corrisponde al pattern '{args.unfreeze_pattern}'. "
            "Ricontrolla l'output di --inspect-only."
        )
    # avviso se il pattern seleziona tutta la rete invece del solo ultimo blocco
    if len(trainable_params) == total:
        print(
            "ATTENZIONE: il pattern corrisponde a TUTTI i parametri -- "
            "stai allenando l'intera rete, non solo l'ultimo blocco."
        )

    train(model, trainable_params, pairs, args.epochs, args.batch_size,
          args.learning_rate, args.out)
