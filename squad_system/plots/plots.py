import json
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_stats_train(experiment_dir: str) -> [dict[str, Any], dict[str, Any]]:
    cfg_path = f"{experiment_dir}/train_config.json"
    state_path = f"{experiment_dir}/trainer_state.json"

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    with open(state_path, "r") as f:
        state = json.load(f)

    return cfg, state


if __name__ == "__main__":
    OUT_DIR = "out_plots"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Training plots
    CLS_ANS_DIR = "out/classification_answerable"
    CLS_IND_ALL_DIR = "out/classification_indices_all"
    CLS_IND_ONLY_ANS_DIR = "out/classification_indices_only_ans"

    CLS_ANS_TAG = CLS_ANS_DIR.split("/")[-1]
    CLS_IND_ALL_TAG = CLS_IND_ALL_DIR.split("/")[-1]
    CLS_IND_ONLY_ANS_TAG = CLS_IND_ONLY_ANS_DIR.split("/")[-1]

    cfgs = {}
    states = {}
    for exp_dir, exp_tag in zip(
        [CLS_ANS_DIR, CLS_IND_ALL_DIR, CLS_IND_ONLY_ANS_DIR],
        [CLS_ANS_TAG, CLS_IND_ALL_TAG, CLS_IND_ONLY_ANS_TAG],
    ):
        cfg, state = load_stats_train(exp_dir)
        cfgs[exp_tag] = cfg
        states[exp_tag] = state

    # Plot classification_answerable losses
    fig, ax = plt.subplots(figsize=(9, 9))

    x = [x + 1 for x in range(states[CLS_ANS_TAG]["epochs_train"])]
    ax.plot(
        x,
        states[CLS_ANS_TAG]["losses_train"],
        color="#1f77b4",
        linestyle="-",
        label="train loss",
    )
    ax.plot(
        x,
        states[CLS_ANS_TAG]["losses_val"],
        color="#1f77b4",
        linestyle="--",
        label="val loss",
    )

    ax.set_xticks(x)
    ax.set_title("classification_answerable losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    fig.savefig(f"{OUT_DIR}/classification_answerable_losses.png", bbox_inches="tight")
    plt.clf()

    # Plot classification_indices losses
    fig, ax = plt.subplots(figsize=(9, 9))

    x = [x + 1 for x in range(states[CLS_IND_ALL_TAG]["epochs_train"])]
    ax.plot(
        x,
        states[CLS_IND_ALL_TAG]["losses_train"],
        color="#1f77b4",
        linestyle="-",
        label="train loss (allow_unanswerable=True)",
    )
    ax.plot(
        x,
        states[CLS_IND_ALL_TAG]["losses_val"],
        color="#1f77b4",
        linestyle="--",
        label="val loss (allow_unanswerable=True)",
    )
    ax.plot(
        x,
        states[CLS_IND_ONLY_ANS_TAG]["losses_train"],
        color="#ff7f0e",
        linestyle="-",
        label="train loss (allow_unanswerable=False)",
    )
    ax.plot(
        x,
        states[CLS_IND_ONLY_ANS_TAG]["losses_val"],
        color="#ff7f0e",
        linestyle="--",
        label="val loss (allow_unanswerable=False)",
    )

    ax.set_xticks(x)
    ax.set_title("classification_indices losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    fig.savefig(f"{OUT_DIR}/classification_indices_losses.png", bbox_inches="tight")
    plt.clf()

    # Plot classification_answerable metrics
    fig, ax = plt.subplots(figsize=(9, 9))

    x = [x + 1 for x in range(states[CLS_IND_ALL_TAG]["epochs_train"])]
    ax.plot(
        x,
        states[CLS_ANS_TAG]["accuracies_train"],
        color="#1f77b4",
        linestyle="-",
        label="train acc",
    )
    ax.plot(
        x,
        states[CLS_ANS_TAG]["accuracies_val"],
        color="#1f77b4",
        linestyle="--",
        label="val acc",
    )
    ax.plot(
        x,
        states[CLS_ANS_TAG]["precisions_train"],
        color="#ff7f0e",
        linestyle="-",
        label="train prec",
    )
    ax.plot(
        x,
        states[CLS_ANS_TAG]["precisions_val"],
        color="#ff7f0e",
        linestyle="--",
        label="val prec",
    )
    ax.plot(
        x,
        states[CLS_ANS_TAG]["recalls_train"],
        color="#2ca02c",
        linestyle="-",
        label="train rec",
    )
    ax.plot(
        x,
        states[CLS_ANS_TAG]["recalls_val"],
        color="#2ca02c",
        linestyle="--",
        label="val rec",
    )

    ax.set_xticks(x)
    ax.set_title("classification_indices metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric value")
    ax.legend()

    fig.savefig(f"{OUT_DIR}/classification_answerable_metrics.png", bbox_inches="tight")
    plt.clf()

    # Evaluation plots #
    INFER_DIR = "out_infer"
    ANS_IND_ALL_FILE = f"{INFER_DIR}/stats_ans_ind_all.json"
    ANS_IND_ONLY_ANS_FILE = f"{INFER_DIR}/stats_ans_ind_only_ans.json"
    IND_ALL_FILE = f"{INFER_DIR}/stats_ind_all.json"
    IND_ONLY_ANS_FILE = f"{INFER_DIR}/stats_ind_only_ans.json"

    ANS_IND_ALL_TAG = ANS_IND_ALL_FILE.split("/")[-1].replace(".json", "")
    ANS_IND_ONLY_ANS_TAG = ANS_IND_ONLY_ANS_FILE.split("/")[-1].replace(".json", "")
    IND_ALL_TAG = IND_ALL_FILE.split("/")[-1].replace(".json", "")
    IND_ONLY_ANS_TAG = IND_ONLY_ANS_FILE.split("/")[-1].replace(".json", "")

    stats_eval = {}
    for eval_file, eval_tag in zip(
        [ANS_IND_ALL_FILE, ANS_IND_ONLY_ANS_FILE, IND_ALL_FILE, IND_ONLY_ANS_FILE],
        [ANS_IND_ALL_TAG, ANS_IND_ONLY_ANS_TAG, IND_ALL_TAG, IND_ONLY_ANS_TAG],
    ):
        with open(eval_file, "r") as f:
            stats = json.load(f)
        stats_eval[eval_tag] = stats

    # Plot f1
    fig, ax = plt.subplots(figsize=(9, 9))

    x = np.arange(3)
    width = 0.15
    mult = [-2, -1, 0, 1]
    for i, eval_tag in enumerate(
        [ANS_IND_ALL_TAG, ANS_IND_ONLY_ANS_TAG, IND_ALL_TAG, IND_ONLY_ANS_TAG]
    ):
        offset = width * mult[i]
        metrics = [
            stats_eval[eval_tag]["f1"],
            stats_eval[eval_tag]["HasAns_f1"],
            stats_eval[eval_tag]["NoAns_f1"],
        ]
        ax.bar(x + offset, metrics, width, label=eval_tag, align="edge")

    ax.set_xticks(x, ["f1", "HasAns_f1", "NoAns_f1"])
    ax.set_title("f1")
    ax.legend()

    fig.savefig(f"{OUT_DIR}/eval_f1.png", bbox_inches="tight")
    plt.clf()

    # Plot exact
    fig, ax = plt.subplots(figsize=(9, 9))

    x = np.arange(3)
    width = 0.15
    mult = [-2, -1, 0, 1]
    for i, eval_tag in enumerate(
        [ANS_IND_ALL_TAG, ANS_IND_ONLY_ANS_TAG, IND_ALL_TAG, IND_ONLY_ANS_TAG]
    ):
        offset = width * mult[i]
        metrics = [
            stats_eval[eval_tag]["exact"],
            stats_eval[eval_tag]["HasAns_exact"],
            stats_eval[eval_tag]["NoAns_exact"],
        ]
        ax.bar(x + offset, metrics, width, label=eval_tag, align="edge")

    ax.set_xticks(x, ["exact", "HasAns_exact", "NoAns_exact"])
    ax.set_title("exact")
    ax.legend()

    fig.savefig(f"{OUT_DIR}/eval_exact.png", bbox_inches="tight")
    plt.clf()
