import os
import re

import matplotlib.pyplot as plt

name_map = {
    "bagging": "Flower XGBoost Bagging",
    "cyclic": "Flower XGBoost Cyclic",
    "fedxgbllr": "FedXGBLLR",
    "nvflare": "NVFlare",
    "fedtree": "FedTree",
}


def extract_train_auc(train_log_file):
    if "fedxgbllr" in train_log_file:
        with open(train_log_file, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "Best AUC" in line:
                    return float(line.split()[-3])
    elif "bagging" in train_log_file or "cyclic" in train_log_file:
        with open(train_log_file, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "Best AUC" in line:
                    return float(line.split()[-1])
    elif "nvflare" in train_log_file:
        with open(train_log_file, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if "eval-auc" in line]
            lines = [re.sub(r"\t", " ", line) for line in lines]
            aucs = [float(line.split()[2].split(":")[-1]) for line in lines]
            return max(aucs)
    elif "fedtree" in train_log_file:
        num_clients = re.search(r"md_(\d+)", train_log_file).group(1)
        fedtree_dir = os.path.join(os.path.dirname(__file__), "FedTree")
        if "dp" in train_log_file:
            epsilon = float(train_log_file.split("dp_")[1].split("/")[0])
            logs_dir = os.path.join(fedtree_dir, "logs", f"logs_{num_clients}_dp_{epsilon_to_privacy_budget[epsilon]}")
        else:
            logs_dir = os.path.join(fedtree_dir, "logs", f"logs_{num_clients}")

        with open(os.path.join(logs_dir, "party0.log"), 'r') as f:
            # 2025-01-25 15:21:39,938 INFO gbdt.cpp:141 : AUC = 0.962217
            lines = f.readlines()
            for line in reversed(lines):
                if "AUC" in line:
                    return float(line.split()[-1])
        return 0.0
    else:
        raise Exception(f"Unknown system: {train_log_file}")


def collect_no_defense_results(base_path):
    results = {}

    for subdir in os.listdir(base_path):
        if subdir.startswith("md_"):
            try:
                max_depth = int(subdir.split("_")[1])
            except ValueError:
                print(f"Skipping invalid subdirectory: {subdir}")
                continue

            md_folder_path = os.path.join(base_path, subdir)
            no_defense_path = os.path.join(md_folder_path, "no_defense", "attack")
            no_defense_path_train = os.path.join(md_folder_path, "no_defense", "train")

            if not os.path.isdir(no_defense_path):
                print(f"No 'no_defense' directory found in {md_folder_path}")
                continue

            for system_name in os.listdir(no_defense_path):
                system_folder = os.path.join(no_defense_path, system_name)
                if not os.path.isdir(system_folder):
                    continue

                accuracy = None
                tsne_plots = []

                for file in os.listdir(system_folder):
                    if file.endswith(".log"):
                        log_file_path = os.path.join(system_folder, file)
                        with open(log_file_path, 'r') as log_file:
                            lines = log_file.readlines()
                            for line in lines[-1000:]:
                                match = re.search(
                                    r"Reconstruction accuracy:\s*([\d.]+)%", line)
                                if match:
                                    accuracy = float(match.group(1))
                                    break
                    elif file.startswith("tsne_") and file.endswith(".png"):
                        tsne_plots.append(os.path.join(system_folder, file))

                train_log_file = os.path.join(no_defense_path_train, f"{system_name}.log")
                train_auc = extract_train_auc(train_log_file)

                if max_depth not in results:
                    results[max_depth] = {}
                results[max_depth][system_name] = {
                    "accuracy": accuracy,
                    "train_auc": train_auc,
                    "tsne_plots": tsne_plots,
                }

    print(results)

    return results


def generate_latex_table(results, baseline_results, output_file):
    systems = ["nvflare", "fedxgbllr", "bagging", "cyclic", "fedtree"]

    lines = []
    lines.append("\\begin{table*}[h]")
    lines.append("  \\centering")
    lines.append(
        "  \\caption{Reconstruction accuracy, training AUC, and baseline accuracy for different systems at varying tree depths without defenses.}")
    lines.append("  \\label{tab:no_defense_results}")
    lines.append("  \\renewcommand{\\arraystretch}{1.2}")
    lines.append("  \\resizebox{\\textwidth}{!}{%")
    # We need 1 column for Depth plus 3 columns for each of the 5 systems = 16 columns.
    lines.append("  \\begin{tabular}{c " + "c " * (len(systems) * 3) + "}")
    lines.append("      \\toprule")

    # First header row: Depth plus each system spanning 3 columns.
    header = "      \\textbf{Depth} "
    for sys in systems:
        header += f"& \\multicolumn{{3}}{{c}}{{\\textbf{{{name_map[sys]}}}}}"
    header += "\\\\"
    lines.append(header)

    # Second header row: For each system, display the sub-columns.
    second_header = "       "
    for sys in systems:
        second_header += "& Rec Acc & AUC & Baseline "
    second_header += "\\\\"
    lines.append(second_header)
    lines.append("      \\midrule")

    # Now iterate over depths in sorted order.
    for depth in sorted(results.keys()):
        row = f"      {depth}"
        for sys in systems:
            # Get values from results dictionary.
            if sys in results[depth]:
                rec_acc = results[depth][sys].get("accuracy")
                train_auc = results[depth][sys].get("train_auc")
                rec_acc_str = f"{rec_acc:.2f}" if rec_acc is not None else "N/A"
                train_auc_str = f"{train_auc:.3f}" if train_auc is not None else "N/A"
            else:
                rec_acc_str = "N/A"
                train_auc_str = "N/A"

            # Get baseline accuracy from baseline_results.
            if sys in baseline_results.get(depth, {}):
                baseline_acc = baseline_results[depth][sys].get("accuracy")
                baseline_str = f"{baseline_acc:.2f}" if baseline_acc is not None else "N/A"
            else:
                baseline_str = "N/A"

            row += f" & {rec_acc_str} & {train_auc_str} & {baseline_str}"
        row += " \\\\"
        lines.append(row)

    lines.append("      \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("  }")
    lines.append("\\end{table*}")

    with open(output_file, 'w') as f:
        f.write("\n".join(lines))
    print(f"LaTeX table written to {output_file}")


def plot_no_defense_results(results, baseline_results, plot_path):
    systems = sorted({system for depths in results.values() for system in depths.keys()})
    colors = plt.cm.tab10.colors
    system_colors = {system: colors[i % len(colors)] for i, system in enumerate(systems)}

    plt.figure(figsize=(12, 8))

    for system in systems:
        # Plot main results for this system
        system_depths = []
        system_accuracies = []
        for max_depth, system_results in sorted(results.items()):
            if system in system_results:
                system_depths.append(max_depth)
                system_accuracies.append(system_results[system]["accuracy"])
        plt.plot(system_depths, system_accuracies, marker='o', color=system_colors[system],
                 label=name_map.get(system, system))

        # Plot baseline results for this system (if available)
        baseline_depths = []
        baseline_accuracies = []
        for max_depth, sys_res in sorted(baseline_results.items()):
            if system in sys_res:
                baseline_depths.append(max_depth)
                baseline_accuracies.append(sys_res[system]["accuracy"])
        if baseline_depths:
            plt.plot(baseline_depths, baseline_accuracies, marker='s', linestyle='--',
                     color=system_colors[system], label=name_map.get(system, system) + " (Baseline)")

    plt.xlabel("Max Depth")
    plt.ylabel("Reconstruction Accuracy (%)")
    plt.title("Reconstruction Accuracy vs Max Depth")
    plt.grid(True)

    # Place the legend outside of the plot area.
    leg = plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

    # Adjust layout and save using bbox_inches='tight' to crop extra space
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(plot_path, bbox_extra_artists=(leg,), bbox_inches='tight')
    plt.close()
    print(f"Plot saved at {plot_path}")


def plot_tsne_by_max_depth(results, output_path):
    for max_depth, system_results in results.items():
        num_systems = len(system_results)
        num_cols = 3
        num_rows = (num_systems + num_cols - 1) // num_cols

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), dpi=150)
        fig.suptitle(f"TSNE Plots for Max Depth: {max_depth}, Reconstruction Accuracy (No Defense)")

        for i, (system, data) in enumerate(system_results.items()):
            tsne_plots = data["tsne_plots"]
            row, col = divmod(i, num_cols)

            if tsne_plots:
                tsne_image = plt.imread(tsne_plots[0])  # Take the first TSNE plot
                axs[row, col].imshow(tsne_image)
                axs[row, col].axis('off')
                axs[row, col].set_title(f"System: {name_map.get(system, system)}, Rec Acc: {data['accuracy']:.2f}")
            else:
                axs[row, col].axis('off')
                axs[row, col].text(0.5, 0.5, "No TSNE Plot", ha='center', va='center')

        for j in range(i + 1, num_rows * num_cols):
            fig.delaxes(axs.flatten()[j])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_file = os.path.join(output_path, f"tsne_max_depth_{max_depth}.pdf")
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"TSNE plots for max_depth {max_depth} saved at {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze and plot no_defense results by max_depth with TSNE plots")
    parser.add_argument("base_path", type=str, help="Base path containing results")
    parser.add_argument("output_path", type=str, help="Output base path for plots and markdown")
    args = parser.parse_args()

    base_path = args.base_path
    output_path = args.output_path
    table_path = os.path.join(output_path, "no_defense_max_depth_table.txt")
    plot_path = os.path.join(output_path, "no_defense_max_depth_plot.pdf")

    results = collect_no_defense_results(base_path)
    baseline_results = collect_no_defense_results(os.path.join(base_path, "baseline"))

    generate_latex_table(results, baseline_results, table_path)
    plot_no_defense_results(results, baseline_results, plot_path)
    plot_tsne_by_max_depth(results, output_path)
