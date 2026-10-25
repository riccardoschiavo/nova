# fine-tuning of the last block of DeepGaze IIE on the OSIE fixations
# two-step usage:
#   1) python finetune_deepgaze.py --osie <OSIE folder> --inspect-only
#      prints the parameter names, to choose which ones to train
#   2) python finetune_deepgaze.py --osie <OSIE folder> --unfreeze-pattern "<pattern>"
#      trains only the parameters whose name contains the pattern

import argparse
import os
import sys

import numpy as np
import cv2

# adds the script folder to the path to import osie_ground_truth
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osie_ground_truth import load_fixations, points_to_heatmap


TARGET_HEIGHT, TARGET_WIDTH = 240, 320 # fixed resolution used for training


# builds the (image, heatmap) pairs from OSIE in the format PyTorch expects
def build_dataset(osie_dir, max_images=None):
    mat_path = os.path.join(osie_dir, "data", "eye", "fixations.mat") # observers' fixations
    stimuli_dir = os.path.join(osie_dir, "data", "stimuli") # images

    # fixation points grouped by image name
    fixations = load_fixations(mat_path)
    names = list(fixations.keys())
    if max_images is not None:
        names = names[:max_images] # uses only the first max_images images

    pairs = []
    skipped = 0 # images listed in the .mat but not found on disk
    for name in names:
        path = os.path.join(stimuli_dir, name)
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            skipped += 1
            continue

        original_h, original_w = image_bgr.shape[:2]

        # heatmap built at the original resolution (the fixation coordinates are in
        # that system) and only then resized, so the gaussian sigma is correct
        heatmap = points_to_heatmap(fixations[name], (original_h, original_w))
        heatmap = cv2.resize(heatmap, (TARGET_WIDTH, TARGET_HEIGHT),
                             interpolation=cv2.INTER_AREA)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # BGR -> RGB
        image_resized = cv2.resize(image_rgb, (TARGET_WIDTH, TARGET_HEIGHT),
                                   interpolation=cv2.INTER_AREA)
        image_chw = image_resized.transpose(2, 0, 1).astype(np.float32) # HWC -> CHW, PyTorch wants channels first

        pairs.append((image_chw, heatmap.astype(np.float32)))

    if skipped:
        print(f"  WARNING: {skipped} images listed in fixations.mat "
              f"not found in {stimuli_dir}, skipped.")

    return pairs


# builds a uniform center bias so that no region of the image gets an advantage,
# the same one used in deepgaze.py
def build_centerbias(height, width):
    import torch
    centerbias = np.zeros((height, width), dtype=np.float64)
    log_sum = np.log(np.exp(centerbias).sum())
    centerbias = centerbias - log_sum # normalizes: matrix of log(1/n) values
    return torch.tensor(centerbias, dtype=torch.float32)


# loads the pretrained DeepGaze IIE, in PyTorch it is already a trainable model
def load_trainable_model(device="cpu"):
    import deepgaze_pytorch
    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(device)
    return model


# prints name and shape of every parameter, used to choose --unfreeze-pattern
def inspect_parameters(model):
    named = list(model.named_parameters())
    print(f"\nTotal trainable parameters: {len(named)}\n")
    for name, p in named:
        print(f"  {name:<70} shape={tuple(p.shape)}")

    print(
        "\nLook for the names that belong to the LAST modules of the "
        "network (often the final readout 'heads', or the layers with the "
        "highest index). The substring shared by those names is the value "
        "to pass to --unfreeze-pattern."
    )


# freezes all the parameters and re-enables only those whose name contains the pattern
def select_trainable_parameters(model, unfreeze_pattern):
    for p in model.parameters():
        p.requires_grad = False # freezes everything

    trainable = []
    for name, p in model.named_parameters():
        if unfreeze_pattern in name:
            p.requires_grad = True # re-enables only the chosen parameters
            trainable.append(p)

    return trainable # parameters to give to the optimizer


# training loop: updates only the chosen parameters and saves the weights at the end
def train(model, trainable_params, pairs, epochs, batch_size, learning_rate, out_path):
    import torch

    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate) # optimizes only the non frozen parameters
    mse = torch.nn.MSELoss() # mean squared error between predicted map and heatmap

    # the center bias depends only on the size, so it is built only once
    centerbias = build_centerbias(TARGET_HEIGHT, TARGET_WIDTH)

    n = len(pairs)
    model.train()
    # model.train() also puts the frozen BatchNorm layers in training mode, and they would update
    # running_mean/running_var at every forward: they are set back to eval() to leave them untouched
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    for epoch in range(epochs):
        order = np.random.permutation(n) # shuffles the images at every epoch
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            batch_size_actual = len(idx) # the last batch can be smaller

            # batch tensors: images (N, 3, H, W), heatmaps (N, H, W), center bias (N, H, W)
            images = torch.tensor(np.stack([pairs[i][0] for i in idx]))
            targets = torch.tensor(np.stack([pairs[i][1] for i in idx]))
            batch_centerbias = centerbias.unsqueeze(0).repeat(batch_size_actual, 1, 1)

            optimizer.zero_grad() # resets the gradients of the previous batch

            log_density = model(images, batch_centerbias) # the model returns a log-density

            # log-density -> density, subtracting the maximum before exp() to avoid overflow
            log_density_flat = log_density.view(batch_size_actual, -1)
            max_per_image = log_density_flat.max(dim=1, keepdim=True).values
            density = torch.exp(log_density_flat - max_per_image)
            # normalizes each map to [0, 1] like the target heatmap
            min_per_image = density.min(dim=1, keepdim=True).values
            max_density = density.max(dim=1, keepdim=True).values
            predicted = (density - min_per_image) / (max_density - min_per_image + 1e-8)
            predicted = predicted.view(batch_size_actual, TARGET_HEIGHT, TARGET_WIDTH)

            loss = mse(predicted, targets)
            loss.backward() # computes the gradients
            optimizer.step() # updates the weights

            epoch_loss += loss.item()
            n_batches += 1

        print(f"epoch {epoch + 1}/{epochs}  mean loss: {epoch_loss / n_batches:.6f}")

    # saves all the model weights (frozen ones included) to a .pt file
    import torch as torch_module
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch_module.save(model.state_dict(), out_path)
    print(f"\nFull weights saved to '{out_path}'.")


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser(description="Fine-tuning of DeepGaze IIE on OSIE.")
    parser.add_argument("--osie", required=True,
                        help="Folder of the cloned OSIE repository.")
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--unfreeze-pattern", type=str, default=None,
                        help="Substring of the names of the parameters NOT to freeze. "
                             "Required unless --inspect-only is used.")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--out", default="deepgaze_finetuned.pt")
    args = parser.parse_args()

    # prepares the (image, heatmap) dataset
    print(f"Preparing data from OSIE (up to {args.max_images} images)...")
    pairs = build_dataset(args.osie, max_images=args.max_images)
    print(f"(image, heatmap) pairs ready: {len(pairs)}")

    print("\nLoading DeepGaze IIE...")
    model = load_trainable_model()

    # with --inspect-only it only prints the parameters and exits
    if args.inspect_only:
        inspect_parameters(model)
        raise SystemExit(0)

    # without a pattern there is nothing to train
    if not args.unfreeze_pattern:
        raise SystemExit(
            "--unfreeze-pattern is required. Run with --inspect-only first to "
            "see the parameter names and choose the right pattern."
        )

    trainable_params = select_trainable_parameters(model, args.unfreeze_pattern)
    total = sum(1 for _ in model.parameters())
    print(f"Parameters selected for training: {len(trainable_params)} of {total} in total.")
    # error message if the pattern matches no parameter
    if len(trainable_params) == 0:
        raise SystemExit(
            f"No parameter matches the pattern '{args.unfreeze_pattern}'. "
            "Check the output of --inspect-only again."
        )
    # warning if the pattern selects the whole network instead of only the last block
    if len(trainable_params) == total:
        print(
            "WARNING: the pattern matches ALL the parameters -- "
            "you are training the whole network, not only the last block."
        )

    train(model, trainable_params, pairs, args.epochs, args.batch_size,
          args.learning_rate, args.out)
