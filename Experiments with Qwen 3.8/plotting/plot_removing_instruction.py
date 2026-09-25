
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
from scipy import stats
from inspect_ai.log import read_eval_log
from inspect_ai.analysis import samples_df, score_to_float, prepare
from pathlib import Path
import json

colormap = {
    # 'Qwen3.8': ['#037228', '#06F957'],
    # 'Qwen3.8': ['#56b4e9', '#0173b2', '#234A5F'],
    'Qwen3.8': ['#c3a9f6', '#5816d8', '#2c0b6d']
}

y_label_fontsize = 14
legend_fontsize = 9
annotation_fontsize = 11
control_x_label_fontsize = 14
target_x_label_fontsize = 14
outer_x_label_fontsize = 14
title_fontsize = 14

def plot_overall_results(true_response_accuracies: dict[float], true_marginoferrors: dict[float], sandbagging_response_accuracies: dict[float], sandbag_marginoferrors: dict[float], indirect_response_accuracies: dict[float], indirect_marginoferrors: dict[float], model: str = 'Qwen3.8', title: str = 'Qwen 3.8 Flash Sandbagging Across Different Datasets', output_file_path: str = 'Qwen3.8_sandbagging_indirect.png'):

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = list(true_response_accuracies.keys())
    bar_width = 0.20
    x = np.arange(len(labels))
    max_accuracy = max(list(true_response_accuracies.values()) + list(sandbagging_response_accuracies.values()))

    bars1 = ax.bar(
            x - bar_width,
            true_response_accuracies.values(),
            width=bar_width,
            label=f"Regular",
            edgecolor='grey',
            color=colormap[model][0],
            yerr = true_marginoferrors.values()
    )
    bars2 = ax.bar(
        x,
        sandbagging_response_accuracies.values(),
        width=bar_width,
        label=f"Sandbagging",
        edgecolor='grey',
        color=colormap[model][1],
        yerr = sandbag_marginoferrors.values()
    )

    bars3 = ax.bar(
        x + bar_width,
        indirect_response_accuracies.values(),
        width=bar_width,
        label=f"Sandbagging (Indirect)",
        edgecolor='grey',
        color=colormap[model][2],
        yerr = indirect_marginoferrors.values()
    )

    ax.set_title(title, fontsize=title_fontsize, pad = 20)
    ax.set_xticklabels(labels)
    xtick_labels = ax.get_xticklabels()
    font_sizes = [control_x_label_fontsize] * 2 + [target_x_label_fontsize] * 3  # Example font sizes for each label

    for label, fontsize in zip(xtick_labels, font_sizes):
        label.set_fontsize(fontsize)
    
    # Annotate bars with their respective heights
    for idx, bars in enumerate([bars1, bars2, bars3]):
        for bar in bars:
            y_val = bar.get_height()
            p = int(y_val*100)
            ax.annotate(
                f'{p}%',
                xy=(bar.get_x() + bar.get_width() / 2, y_val),
                xytext=(3 + 1*idx, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=annotation_fontsize,
                color=colormap[model][idx]
            )

    ax.set_ylabel("Accuracy", fontsize=y_label_fontsize)  # Set y-label on the leftmost subplot only

    ax.set_ylim([0, max_accuracy * 1.1])  # Extend y-axis slightly above the maximum found
    ax.set_xticks(x)
    ax.axhline(y=0.25, color='gray', linestyle='dotted', label="Random baseline")
    ax.legend(fontsize=legend_fontsize, loc="lower left")
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file_path)  # Save the plot to a file
    plt.close()
    return 

def __main__():
    # Example usage
    log_path  = Path("../logs-original-experiment/logs.json")
    indirect_logs = Path("../logs-removing-instruction/logs.json")

    with open(log_path, 'r', encoding='utf-8') as file:
        eval_data = json.load(file)
    
    with open(indirect_logs, 'r', encoding='utf-8') as file:
        indirect_data = json.load(file)

    eval_data = eval_data | indirect_data

    true_accs = {}
    sandbagging_accs = {}
    indirect_accs = {}

    true_moes = {}
    sandbagging_moes = {}
    indirect_moes = {}
    
    for eval_name, eval in eval_data.items():
        # only plotting target sets
        if eval["eval"]["task_args"]["dataset_name"] == "mmlu" or eval["eval"]["task_args"]["dataset_name"] == "gpqa":
            continue
        acc = eval["results"]["scores"][0]["metrics"]["accuracy"]["value"]
        moe = eval["results"]["scores"][0]["metrics"]["stderr"]["value"]

        if eval["eval"]["task"] == "regular":
            true_accs[eval["eval"]["task_args"]["dataset_name"]] = acc
            true_moes[eval["eval"]["task_args"]["dataset_name"]] = moe
        elif eval["eval"]["task"] == "sandbagging":
            sandbagging_accs[eval["eval"]["task_args"]["dataset_name"]] = acc
            sandbagging_moes[eval["eval"]["task_args"]["dataset_name"]] = moe
        elif eval["eval"]["task"] == "sandbagging_indirect":
            indirect_accs[eval["eval"]["task_args"]["dataset_name"]] = acc
            indirect_moes[eval["eval"]["task_args"]["dataset_name"]] = moe
    
    # preserve same order as original experiments
    indirect_accs = dict(sorted(indirect_accs.items(), reverse=True))
    indirect_moes =  dict(sorted(indirect_moes.items(), reverse=True))

    print(true_accs)
    print(true_moes)
    print()
    print(sandbagging_accs)
    print(sandbagging_moes)
    print()
    print(indirect_accs)
    print(indirect_moes)


    print("Creating barplots...")
    plot_overall_results(true_accs, true_moes, sandbagging_accs, sandbagging_moes, indirect_accs, indirect_moes)
    return None


__main__()
