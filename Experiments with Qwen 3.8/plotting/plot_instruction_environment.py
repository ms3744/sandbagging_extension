
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
    'Qwen3.8': ['#c3a9f6', '#5816d8', '#9c5ff2', '#3b0b66']
}

y_label_fontsize = 14
legend_fontsize = 9
annotation_fontsize = 14
control_x_label_fontsize = 14
target_x_label_fontsize = 14
outer_x_label_fontsize = 14
title_fontsize = 14

def summarize(x: pd.Series) -> pd.Series:
    n = len(x)
    acc = x.mean()
    stderr = x.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return pd.Series({"n": n, "accuracy": acc, "stderr": stderr})
 
    

def plot_subset_results(log_path : list[str] = ["../logs-instruction-environment/normal.eval", "../logs-instruction-environment/hint.eval", "../logs-instruction-environment/direct.eval"], instruction_format: list[str] = ["normal","hint","direct"], DATASET_COL: str = "metadata_dataset", SCORE_COL: str = "score_read_and_answer", model: str = "Qwen3.8", output_file_path: str = "Qwen3.8_sandbagging_environment_bydataset.png"):

    subset_accs = {}
    subset_moes = {}
    labels = ['mmlu', 'wmdp_chem', 'wmdp_bio', 'wmdp_cyber']
    annotation_fontsize = 11
    # outer_x_label_fontsize = 11

    for log, format in zip(log_path, instruction_format):
        samples = samples_df(logs=log)

        if DATASET_COL not in samples.columns:
            raise ValueError(
                f"'{DATASET_COL}' not found. Available metadata columns: "
                f"{[c for c in samples.columns if c.startswith('metadata_')]}"
            )

        by_dataset = samples.groupby(DATASET_COL)[SCORE_COL].apply(summarize).unstack()
        by_dataset = by_dataset.reindex(labels)
        accs = by_dataset["accuracy"].to_list()
        moes = by_dataset["stderr"].to_list()

        subset_accs[format] = accs
        subset_moes[format] = moes

    # print(subset_accs)
    # print(subset_moes)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bar_width = 0.25
    x = np.arange(len(labels))
    max_accuracy = max(subset_accs['normal'])


    bars1 = ax.bar(
            x - bar_width,
            subset_accs['normal'],
            width=bar_width,
            label=f"Normal",
            edgecolor='grey',
            color=colormap[model][0],
            yerr = subset_moes['normal']
    )
    bars2 = ax.bar(
        x,
        subset_accs['hint'],
        width=bar_width,
        label=f"Hint",
        edgecolor='grey',
        color=colormap[model][1],
        yerr = subset_moes['hint']
    )

    bars3 = ax.bar(
        x + bar_width,
        subset_accs['direct'],
        width=bar_width,
        label=f"Direct",
        edgecolor='grey',
        color=colormap[model][2],
        yerr = subset_moes['direct']
    )

    ax.set_title("Sandbagging Across Datasets", fontsize=title_fontsize, pad = 20)
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
                xytext=(3 + 1*idx, 6),  # 3 points vertical offset
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

    ax_twin = ax.twiny()
    ax_twin.spines["bottom"].set_position(("axes", -0.2))
    ax_twin.tick_params('both', length=0, width=0, which='minor')
    ax_twin.tick_params('both', direction='in', which='major')
    ax_twin.xaxis.set_ticks_position("bottom")
    ax_twin.xaxis.set_label_position("bottom")

    ax_twin.set_xticks([0.0, 0.75, 1.0])
    ax_twin.xaxis.set_major_formatter(ticker.NullFormatter())
    ax_twin.xaxis.set_minor_locator(ticker.FixedLocator([0.35, 0.87]))
    ax_twin.xaxis.set_minor_formatter(ticker.FixedFormatter(['Control', 'Target']))

    for label in ax_twin.get_xticklabels(minor=True):
        label.set_fontsize(outer_x_label_fontsize)

    ax_twin.spines['top'].set_visible(False)
    ax_twin.spines['left'].set_visible(False)
    ax_twin.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file_path)  # Save the plot to a file
    plt.close()
    return
def plot_overall_results(true_response_accuracies: dict[float], true_marginoferrors: dict[float], model: str = 'Qwen3.8', title: str = 'Qwen 3.8 Flash Sandbagging Across Different Datasets', output_file_path: str = 'Qwen3.8_sandbagging_environment.png'):

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = list(true_response_accuracies.keys())
    bar_width = 0.25
    x = np.arange(len(labels))
    max_accuracy = max(list(true_response_accuracies.values()))
    
    bars1 = ax.bar(
            x,
            true_response_accuracies.values(),
            width=bar_width,
            # label=f"Regular",
            edgecolor='grey',
            color=colormap[model][0],
            yerr = true_marginoferrors.values()
    )

    ax.set_title(title, fontsize=title_fontsize, pad = 20)
    ax.set_xticklabels(labels)
    xtick_labels = ax.get_xticklabels()
    font_sizes = [control_x_label_fontsize] * 2 + [target_x_label_fontsize] * 3  # Example font sizes for each label

    for label, fontsize in zip(xtick_labels, font_sizes):
        label.set_fontsize(fontsize)
    
    # Annotate bars with their respective heights
    for idx, bars in enumerate([bars1]):
        for bar in bars:
            y_val = bar.get_height()
            p = int(y_val*100)
            ax.annotate(
                f'{p}%',
                xy=(bar.get_x() + bar.get_width() / 2, y_val),
                xytext=(3 + 2*idx, 3),  # 3 points vertical offset
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

    log_path  = Path("../logs-instruction-environment/logs.json")

    with open(log_path, 'r', encoding='utf-8') as file:
        eval_data = json.load(file)


    accs = {}
    moes = {}
    
    log_names = ["normal-filtered.eval", "hint-filtered.eval", "direct-filtered.eval"]

    for eval_name, eval in eval_data.items():
        if eval_name in log_names:
            # 1 is for the new scorer
            # 0 is for the original scorer
            acc = eval["results"]["scores"][1]["metrics"]["accuracy"]["value"]
            moe = eval["results"]["scores"][1]["metrics"]["stderr"]["value"]

            accs[eval["eval"]["task_args"]["instruction_format"]] = acc
            moes[eval["eval"]["task_args"]["instruction_format"]] = moe
        

    accs = dict(sorted(accs.items(), reverse=True))
    moes = dict(sorted(moes.items(), reverse=True))
    print(accs)
    print(moes)
    print()

    filtered_log_paths = [f"../logs-instruction-environment/{name}" for name in log_names]
    print("Creating barplots...")
    plot_overall_results(accs, moes)
    plot_subset_results(log_path=filtered_log_paths)
    return None


__main__()
