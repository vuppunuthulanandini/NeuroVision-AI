"""
Federated Learning implementation using Federated Averaging (FedAvg)
exactly as described in the paper.

- Multiple simulated healthcare institutions (clients)
- Each client trains locally on its own MRI data
- Only model parameters are sent to the server
- Server aggregates with FedAvg and redistributes the global model
"""

import copy
import os
import numpy as np
import tensorflow as tf

from models.cnn_model import get_model
from utils.preprocessing import prepare_client_data
from config import (
    NUM_CLIENTS,
    LOCAL_EPOCHS,
    BATCH_SIZE,
    NUM_ROUNDS,
    MODEL_DIR,
    SEED,
)


def get_weights(model):
    return model.get_weights()


def set_weights(model, weights):
    model.set_weights(weights)


def average_weights(weight_list):
    """
    Classic FedAvg: element-wise average of client weight lists.
    """
    avg = copy.deepcopy(weight_list[0])
    for i in range(1, len(weight_list)):
        for j in range(len(avg)):
            avg[j] = avg[j] + weight_list[i][j]
    for j in range(len(avg)):
        avg[j] = avg[j] / len(weight_list)
    return avg


def train_federated(num_rounds=NUM_ROUNDS, local_epochs=LOCAL_EPOCHS):
    """
    Main Federated Learning loop.
    Returns the final global model and training history.
    """
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # Initialise global model
    global_model = get_model()
    global_weights = get_weights(global_model)

    history = {
        "round": [],
        "avg_client_accuracy": [],
        "avg_client_loss": [],
    }

    print("=" * 60)
    print("NeuroVision AI – Federated Learning (FedAvg)")
    print(f"Clients        : {NUM_CLIENTS}")
    print(f"Rounds         : {num_rounds}")
    print(f"Local epochs   : {local_epochs}")
    print(f"Batch size     : {BATCH_SIZE}")
    print("=" * 60)

    for rnd in range(1, num_rounds + 1):
        print(f"\n----- Federated Round {rnd}/{num_rounds} -----")
        client_weights = []
        client_accs = []
        client_losses = []

        for cid in range(NUM_CLIENTS):
            print(f"  → Client {cid} local training ...", end=" ", flush=True)

            local_model = get_model()
            set_weights(local_model, global_weights)

            train_gen = prepare_client_data(
                client_id=cid,
                batch_size=BATCH_SIZE,
                augment=True,
                shuffle=True,
            )

            steps = max(1, train_gen.samples // BATCH_SIZE)

            hist = local_model.fit(
                train_gen,
                epochs=local_epochs,
                steps_per_epoch=steps,
                verbose=0,
            )

            client_weights.append(get_weights(local_model))
            acc = hist.history["accuracy"][-1]
            loss = hist.history["loss"][-1]
            client_accs.append(acc)
            client_losses.append(loss)
            print(f"acc={acc:.4f}  loss={loss:.4f}")

        # ---- FedAvg aggregation ----
        global_weights = average_weights(client_weights)
        set_weights(global_model, global_weights)

        avg_acc = float(np.mean(client_accs))
        avg_loss = float(np.mean(client_losses))
        history["round"].append(rnd)
        history["avg_client_accuracy"].append(avg_acc)
        history["avg_client_loss"].append(avg_loss)

        print(f"  ★ Round {rnd} average client accuracy : {avg_acc:.4f}")
        print(f"  ★ Round {rnd} average client loss     : {avg_loss:.4f}")

    # Save final global model
    save_path = os.path.join(MODEL_DIR, "global_model.h5")
    global_model.save(save_path)
    print("\n" + "=" * 60)
    print(f"[OK] Global model saved → {save_path}")
    print("=" * 60)

    return global_model, history


if __name__ == "__main__":
    train_federated()
