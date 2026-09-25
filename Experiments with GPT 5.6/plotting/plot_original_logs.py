
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
from scipy import stats
from inspect_ai.log import read_eval_log
from inspect_ai.analysis import samples_df, score_to_float, prepare

colormap = {
    'GPT-5.6': ['#56b4e9', '#0173b2', '#234A5F'],
    # 'Claude 3': ['#de8f05', '#d55e00'],
}

y_label_fontsize = 14
legend_fontsize = 9
annotation_fontsize = 14
control_x_label_fontsize = 14
target_x_label_fontsize = 14
outer_x_label_fontsize = 14
title_fontsize = 14

def plot_subset_results(acc_wide, err_wide, scenario_order, type_order, output_file_path = "accuracy_by_dataset_and_scenario.png"):
    n_scenarios = len(scenario_order)
    n_types = len(type_order)
    x = np.arange(n_scenarios)     # one y position per scenario
    bar_width = 0.35
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    bars = []
    for i, qtype in enumerate(type_order):
        offset = (i - (n_scenarios - 1) / 2) * bar_width
        bars.append(ax.bar(
            x + offset,
            acc_wide[qtype],
            width=bar_width,
            yerr=err_wide[qtype],
            capsize=4,
            label=qtype,
            color=colormap['GPT-5.6'][i]
        ))
    
    max_accuracy = acc_wide.max(axis=None)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_order)
    ax.set_ylabel("Accuracy", fontsize=y_label_fontsize)
    ax.set_ylim([0, max_accuracy * 1.1])  # Extend y-axis slightly above the maximum found

    ax.set_title("GPT 5.6 Accuracy Across Datasets", fontsize=title_fontsize, pad=20)
    ax.legend(title="Dataset")

    xtick_labels = ax.get_xticklabels()

    font_sizes = [control_x_label_fontsize] * 2 + [target_x_label_fontsize] * 3  # Example font sizes for each label

    for label, fontsize in zip(xtick_labels, font_sizes):
        label.set_fontsize(fontsize)

    # Annotate bars with their respective heights
    for idx, bar in enumerate(bars):
        for b in bar:
            y_val = b.get_height()
            p = int(y_val*100)
            ax.annotate(
                f'{p}%',
                xy=(b.get_x() + b.get_width() / 2, y_val),
                xytext=(3 + 2*idx, 20),  # 3 points vertical offset
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=annotation_fontsize,
                color=colormap['GPT-5.6'][idx]
            )

    ax.axhline(y=0.25, color='gray', linestyle='dotted', label="Random baseline")
    ax.legend(fontsize=legend_fontsize, loc="lower left")
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file_path)

def summarize_subsets(subset_metrics: dict) -> pd.DataFrame:
    all_rows = []
    for scenario_name, df in subset_metrics.items():
        tmp = df.reset_index().rename(columns={df.index.name or "index": "type"})
        tmp["scenario"] = scenario_name
        all_rows.append(tmp)
    
    return pd.concat(all_rows, ignore_index=True)

def summarize(x: pd.Series) -> pd.Series:
    """Mean (accuracy) and standard error of a 0/1 score column."""
    n = len(x)
    acc = x.mean()
    stderr = x.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return pd.Series({"n": n, "accuracy": acc, "stderr": stderr})

def plot_overall_results(accuracies: dict[float], stderrs: dict[float], model: str = 'GPT-5.6', title: str = 'GPT 5.6 Sandbagging Across Different Instructions', output_file_path: str = 'gpt5_6_sandbagging_overall.png'):

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = list(accuracies.keys())
    accs = list(accuracies.values())
    bar_width = 0.35
    x = np.arange(len(labels))
    errs = list(stderrs.values())
    max_accuracy = max(accs)

    bars1 = ax.bar(
        x,
        accs,
        width=bar_width,
        # label=f"Regular",
        edgecolor='grey',
        color=colormap[model][0],
        yerr = errs
    )

    ax.set_title(title, fontsize=title_fontsize, pad = 20)
    ax.set_xticklabels(labels)
    xtick_labels = ax.get_xticklabels()
    font_sizes = [control_x_label_fontsize] * 2 + [target_x_label_fontsize] * 3  # Example font sizes for each label

    for label, fontsize in zip(xtick_labels, font_sizes):
        label.set_fontsize(fontsize)
    
    for bar in bars1:
        y_val = bar.get_height()
        p = int(y_val*100)
        ax.annotate(
            f'{p}%',
            xy=(bar.get_x() + bar.get_width() / 2, y_val),
            xytext=(3, 15),  # 20 points vertical offset
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=annotation_fontsize,
            color=colormap[model][0]
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
    log_paths = {
    "normal": ['../logs/1_instruction_following_normal_gpt5_6.eval', '../logs/2_instruction_following_normal_gpt5_6.eval', '../logs/3_instruction_following_normal_gpt5_6.eval'], 
    "hint": ['../logs/1_instruction_following_hint_gpt5_6.eval', '../logs/2_instruction_following_hint_gpt5_6.eval', '../logs/3_instruction_following_hint_gpt5_6.eval'],
    "direct": ['../logs/1_instruction_following_direct_gpt5_6.eval', '../logs/2_instruction_folllowing_direct_gpt5_6.eval', '../logs/3_instruction_following_direct_gpt5_6.eval'],
    }

    QUESTION_FILE_PATH = "../all_questions.csv"

    accuracies = {}
    stderrs = {}
    subset_metrics = {}

    for key, log in log_paths.items():
        print(key)

        dfs = []
        n = len(log)
        for i, log_file in enumerate(log, start=1):
            df = samples_df(logs=log_file)
            df["run"] = i
            dfs.append(df)

        samples = pd.concat(dfs, ignore_index=True)
        SCORE_COL = "score_read_and_answer" 

        # Convert score values to 0.0/1.0 floats
        # samples = prepare(samples, score_to_float(columns=[SCORE_COL]))

        question_metdata = pd.read_csv(QUESTION_FILE_PATH)
        question_metdata.index += 1
        question_metdata['id_metadata'] = 'q' + question_metdata.index.astype(str)
        # print(question_metdata[['id', 'question']])

        samples = samples.merge(
            question_metdata[["id_metadata", "dataset"]],
            left_on="id",
            right_on="id_metadata",
            how="left",
        )

        missing = samples["dataset"].isna().sum()
        if missing:
            print(f"WARNING: {missing} sample rows had no matching question type "
                f"-- check that sample 'input' matches 'question' in the CSV.")

        overall = summarize(samples[SCORE_COL])
        accuracies[key] = overall['accuracy']
        stderrs[key] = overall['stderr']

        print(f"=== Overall accuracy for {key}, pooled across all {n} runs ===")
        print(overall.to_string())
        print()

        by_type = (
            samples.groupby("dataset")[SCORE_COL]
            .apply(summarize)
            .unstack()
        )
        print(f"=== Accuracy by question type for {key}, pooled across all {n} runs ===")
        print(by_type.to_string())
        print()
        subset_metrics[key] = by_type

        # per_run = samples.groupby("run")[SCORE_COL].apply(summarize).unstack()
        # print("=== Per-run accuracy (for reference / sanity-checking) ===")
        # print(per_run.to_string())
        # print()
        
        # per_run_type = (
        #     samples.groupby(["run", "dataset"])[SCORE_COL]
        #     .apply(summarize)
        #     .unstack()
        # )
        # print("=== Per-run, per-type accuracy (for reference) ===")
        # print(per_run_type.to_string())
        # print(by_type.to_string())
        # print()

        # TESTING ONLY FOR NORMAL FOR NOW 
        # break
    
    # plot_overall_results(accuracies, stderrs)

    subset_df = summarize_subsets(subset_metrics)
    acc_wide = subset_df.pivot(index="scenario", columns="type", values="accuracy")
    err_wide = subset_df.pivot(index="scenario", columns="type", values="stderr")
    
    # Keep a consistent scenario order (pivot alphabetizes columns by default)
    scenario_order = list(subset_metrics.keys())
    type_order = sorted(subset_df["type"].unique())
    acc_wide = acc_wide.loc[scenario_order, type_order]
    err_wide = err_wide.loc[scenario_order, type_order]

    print(f"=== Accuracy & Std Error by question type for each scenario ===")
    print(acc_wide)
    print(err_wide)
    print()

    plot_subset_results(acc_wide, err_wide, scenario_order, type_order)
    return None


__main__()
