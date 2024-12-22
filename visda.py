import os
import ot
import torch
import pickle
import random
import argparse
import numpy as np
import pytorch_lightning as pl

from functools import partial
from sklearn.metrics import accuracy_score
from src import (
    em_gmm,
    GMMOTDA,
    conditional_em_gmm,
    WeightedShallowNeuralNet
)

# Fix seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def sinkhorn(a, b, C, reg_e):
    return ot.bregman.sinkhorn_stabilized(
        a, b, C / C.max(), reg=reg_e)


np.random.seed(42)
parser = argparse.ArgumentParser(
    description="GMMOTDA on VisDA Benchmark")
parser.add_argument(
    '--base_path',
    type=str,
    default="./data",
    help="Path to features"
)
parser.add_argument(
    '--features',
    type=str,
    default="resnet101"
)
parser.add_argument(
    '--source',
    type=str,
    default="amazon"
)
parser.add_argument(
    '--target',
    type=str,
    default="webcam"
)
parser.add_argument(
    '--clusters_per_class',
    type=int,
    default=3
)
parser.add_argument(
    '--reg_e',
    type=float,
    default=1e-2
)
args = parser.parse_args()


feat_name = args.features
results_path = "./results/tmlr_nn/"
base_path = args.base_path


if feat_name.lower() == 'resnet101':
    n_dim = 2048
    filename = "visda_resnet_101_v1.pkl"
elif feat_name.lower() == 'resnet50':
    n_dim = 2048
    filename = "visda_resnet_50.pkl"
elif feat_name.lower() == "vit":
    n_dim = 768
    filename = "visda_vit_16.pkl"
else:
    raise ValueError(f"Invalid feature name '{feat_name}'")

n_classes = 12

max_norm = None
l2_penalty = 0.0
lr_perceptron = 1e-4
n_epochs_perceptron = 30
batch_size_perceptron = 128
optimizer_perceptron = 'sgd'

clusters_per_class = args.clusters_per_class

with open(
     os.path.join(
         base_path, filename), 'rb') as f:
    dataset = pickle.loads(f.read())

Xs, ys = dataset['Train']
Xt, yt = dataset['Test']

mean, std = Xs.mean(), Xs.std()

Xs = (Xs - mean) / (std + 1e-9)
Xt = (Xt - mean) / (std + 1e-9)

Ys = torch.nn.functional.one_hot(ys.long(), num_classes=n_classes).float()
Yt = torch.nn.functional.one_hot(yt.long(), num_classes=n_classes).float()

clustering_source = partial(
    conditional_em_gmm,
    n_clusters=clusters_per_class,
    random_state=42
)

clustering_target = partial(
    em_gmm,
    n_clusters=clusters_per_class * n_classes,
    random_state=42
)

if args.reg_e == 0:
    ot_solver = ot.emd
else:
    ot_solver = partial(
        sinkhorn,
        reg_e=args.reg_e
    )

otda = GMMOTDA(
    clustering_source=clustering_source,
    clustering_target=clustering_target,
    ot_solver=ot_solver,
    min_var=1
)

# Fit OTDA
otda.fit(Xs, Ys, Xt, Yt)

# Transports towards target
numel = 2 * n_classes * clusters_per_class
w, TXs, TYs = otda.transport_samples(
    Xs, Ys, numel=numel)

train_dataset = torch.utils.data.TensorDataset(w, TXs, TYs)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size_perceptron,
    shuffle=True)
u = torch.ones(len(Xt)) / len(Xt)
test_dataset = torch.utils.data.TensorDataset(u, Xt, Yt)
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=128,
                                              shuffle=False)

model = WeightedShallowNeuralNet(
    n_features=n_dim,
    n_classes=n_classes,
    learning_rate=lr_perceptron,
    l2_penalty=0.0,
    momentum=0.9,
    optimizer_name=optimizer_perceptron,
    log_gradients=False,
    max_norm=max_norm
)
trainer = pl.Trainer(
    max_epochs=n_epochs_perceptron,
    accelerator='gpu',
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=False)
trainer.fit(model, train_dataloader, test_dataloader)
print(f"Max: {max(model.history['val_acc'])}"
      f" at it {np.argmax(model.history['val_acc'])}\n"
      f"Last: {model.history['val_acc'][-1]}")

with torch.no_grad():
    yp = model(Xt.float()).argmax(dim=1)
    acc = 100 * accuracy_score(yp, yt)
    print(f"{acc}%")

# GMM-OTDA MAP
yp = otda.predict_target_labels(
    Xt, use_estimated_labels=True).argmax(dim=1)
acc_map = 100 * accuracy_score(
    yp, Yt.argmax(dim=1))
print(f"{acc_map}%")
