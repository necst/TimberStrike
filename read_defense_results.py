import os
import re

import matplotlib.pyplot as plt

# Placeholder map for epsilon to privacy budget for fedtree
epsilon_to_privacy_budget = {
    0.001: 25,
    0.005: 50,
    2: 200
}


def collect_results(base_path):
    results = []

    for subdir in os.listdir(base_path):
        if subdir.startswith("dp_") or subdir == "no_defense":
            epsilon = None
            if subdir.startswith("dp_"):
                try:
                    epsilon = float(subdir.split("_")[1])
                except ValueError:
                    continue

            folder_path = os.path.join(base_path, subdir)

            # Find subfolders starting with "attack"
            attack_subfolders = [f for f in os.listdir(folder_path) if f.startswith("attack")]
            train_subfolders = [f for f in os.listdir(folder_path) if f.startswith("train")]
            if not attack_subfolders:
                continue
            if not train_subfolders:
                continue

            attack_folder = os.path.join(folder_path, attack_subfolders[0])
            train_folder = os.path.join(folder_path, train_subfolders[0])

            # Process each system inside the training folder (the folder contains file called system_name.log), extract only the system name
            for system_name in [f.split(".")[0] for f in os.listdir(train_folder) if f.endswith(".log")]:
                system_folder = os.path.join(attack_folder, system_name)
                accuracy = None
                accuracy_noisy = None
                tsne_image = None
                tsne_dp_image = None
                log_file_path = None

                if os.path.isdir(system_folder):
                    # Find tsne_*.png and .log files inside the system folder
                    for file in os.listdir(system_folder):
                        if file.startswith("tsne_") and file.endswith(".png"):
                            tsne_image = os.path.join(system_folder, file)
                        elif file.endswith(".log"):
                            log_file_path = os.path.join(system_folder, file)

                    # Check for dp folder and its TSNE plot
                    dp_folder = os.path.join(system_folder, "dp")
                    if os.path.exists(dp_folder):
                        for file in os.listdir(dp_folder):
                            if file.startswith("tsne_") and file.endswith(".png"):
                                tsne_dp_image = os.path.join(dp_folder, file)
                                break

                    # Extract reconstruction accuracy
                    if log_file_path:
                        with open(log_file_path, 'rb') as log_file:
                            lines = log_file.readlines()

                            accuracies = []

                            for line in lines[-1000:]:
                                if line.startswith(b"Reconstruction accuracy:"):
                                    match = re.search(b"Reconstruction accuracy:\s*([\d.]+)%", line)
                                    if match:
                                        accuracies.append(float(match.group(1)))

                            if accuracies:
                                accuracy = accuracies[0]  # First is original dataset
                                if len(accuracies) > 1:
                                    accuracy_noisy = accuracies[1]  # Second is noisy dataset

                # Collect training AUC from the train folder (placeholder function for now)
                train_auc = None
                if os.path.exists(train_folder):
                    train_log_file = os.path.join(train_folder, f"{system_name}.log")
                    if os.path.exists(train_log_file):
                        train_auc = extract_train_auc(train_log_file)

                # Handle privacy budget mapping for fedtree
                privacy_budget = None
                if system_name == "fedtree" and epsilon in epsilon_to_privacy_budget:
                    privacy_budget = epsilon_to_privacy_budget[epsilon]

                results.append({
                    "epsilon": epsilon,
                    "privacy_budget": privacy_budget,  # New field for privacy budget
                    "accuracy": accuracy,
                    "accuracy_noisy": accuracy_noisy,
                    "train_auc": train_auc,  # New field for training AUC
                    "tsne_image": tsne_image,
                    "tsne_dp_image": tsne_dp_image,
                    "system": system_name,
                    "subdir": subdir
                })
    return results


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


def generate_grouped_markdown(results, output_path):
    results.sort(key=lambda x: (x["system"], x["epsilon"] is None, x["epsilon"]))
    markdown_lines = ["# Results Summary", ""]

    # Group results by system
    systems = sorted(set(r["system"] for r in results))
    for system in systems:
        markdown_lines.append(f"### System: {system}")
        markdown_lines.append(
            "| Epsilon       | Privacy Budget | Reconstruction Accuracy (%) | Noisy Dataset Accuracy (%) | Task AUC | TSNE Plot Path | DP TSNE Plot Path |"
        )
        markdown_lines.append(
            "|---------------|----------------|----------------------------|---------------------------|-------------|----------------|------------------|"
        )

        system_results = [r for r in results if r["system"] == system]
        for result in system_results:
            epsilon = "No Defense" if result["epsilon"] is None else f"{result['epsilon']}"
            privacy_budget = "N/A" if result["privacy_budget"] is None else f"{result['privacy_budget']}"
            accuracy = "N/A" if result["accuracy"] is None else f"{result['accuracy']}"
            accuracy_noisy = "N/A" if result["accuracy_noisy"] is None else f"{result['accuracy_noisy']}"
            train_auc = "N/A" if result["train_auc"] is None else f"{result['train_auc']}"
            tsne_path = result["tsne_image"] or "N/A"
            tsne_dp_path = result["tsne_dp_image"] or "N/A"
            markdown_lines.append(
                f"| {epsilon:<15} | {privacy_budget:<14} | {accuracy:<27} | {accuracy_noisy:<26} | {train_auc:<11} | {tsne_path} | {tsne_dp_path} |"
            )

        markdown_lines.append("")  # Blank line between systems

    with open(output_path, 'w') as md_file:
        md_file.write("\n".join(markdown_lines))
    print(f"Grouped Markdown file created at {output_path}")


def plot_results_by_system(results, plot_path):
    systems = sorted(set(r["system"] for r in results))
    for system in systems:
        system_results = [r for r in results if r["system"] == system]

        # Create separate figures for original and DP TSNE plots
        for plot_type in ['original', 'dp']:
            # Count valid plots for this type
            num_results = sum(1 for r in system_results if
                              (plot_type == 'original' and r["tsne_image"]) or
                              (plot_type == 'dp' and r["tsne_dp_image"]))

            if num_results == 0:  # Skip if no plots of this type
                continue

            # Determine grid layout
            num_cols = 3
            num_rows = (num_results + num_cols - 1) // num_cols

            fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), dpi=150)
            if num_rows == 1:
                axs = axs.reshape(1, -1)
            fig.suptitle(f"System: {system} - {'Original' if plot_type == 'original' else 'DP'} TSNE Plots")

            plot_idx = 0
            for result in system_results:
                epsilon = "No Defense" if result["epsilon"] is None else f"{result['epsilon']}"
                accuracy = "N/A" if result["accuracy"] is None else f"{result['accuracy']}"
                accuracy_noisy = "N/A" if result["accuracy_noisy"] is None else f"{result['accuracy_noisy']}"

                # Plot based on type
                image_path = result["tsne_image"] if plot_type == 'original' else result["tsne_dp_image"]
                if image_path:
                    row = plot_idx // num_cols
                    col = plot_idx % num_cols
                    tsne_image = plt.imread(image_path)
                    axs[row, col].imshow(tsne_image)
                    axs[row, col].axis('off')
                    plot_type_str = "Original" if plot_type == 'original' else "DP"
                    axs[row, col].set_title(
                        f"{plot_type_str} - Epsilon: {epsilon}\nAccuracy: {accuracy}\nNoisy Accuracy: {accuracy_noisy}")
                    plot_idx += 1

            # Hide any unused subplots
            for j in range(plot_idx, num_rows * num_cols):
                row = j // num_cols
                col = j % num_cols
                fig.delaxes(axs[row, col])

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            suffix = "_original" if plot_type == 'original' else "_dp"
            plt.savefig(plot_path.replace(".png", f"_{system}{suffix}.png"), dpi=150)
            plt.close()


def plot_results(results, plot_path):
    dp_results = [r for r in results if r["epsilon"] is not None]
    no_defense_results = [r for r in results if r["epsilon"] is None]

    for dp_result in dp_results:
        if dp_result["accuracy"] is None:
            return

    for no_defense_result in no_defense_results:
        if no_defense_result["accuracy"] is None:
            return

    systems = sorted(set(r["system"] for r in results))
    colors = plt.cm.tab10.colors
    system_colors = {system: colors[i % len(colors)] for i, system in enumerate(systems)}

    plt.figure(figsize=(12, 8))

    for system in systems:
        system_dp_results = [r for r in dp_results if r["system"] == system]
        if system_dp_results:
            dp_epsilons = [r["epsilon"] for r in system_dp_results]
            dp_accuracies = [r["accuracy"] for r in system_dp_results]
            dp_accuracies_noisy = [r["accuracy_noisy"] for r in system_dp_results]

            if system == "fedtree":
                privacy_budgets = [r["privacy_budget"] for r in system_dp_results]
                plt.plot(privacy_budgets, dp_accuracies, marker='o', color=system_colors[system],
                         label=f"{system} (Privacy Budget)")
            else:
                plt.plot(dp_epsilons, dp_accuracies, marker='o', color=system_colors[system],
                         label=f"{system} (DP - Original)")

            if any(dp_accuracies_noisy):
                plt.plot(dp_epsilons, dp_accuracies_noisy, marker='s', linestyle='--', color=system_colors[system],
                         label=f"{system} (DP - Noisy)")

        system_no_defense = [r for r in no_defense_results if r["system"] == system]
        if system_no_defense:
            for r in system_no_defense:
                plt.axhline(
                    y=r["accuracy"], linestyle='--', color=system_colors[system],
                    label=f"{system} (No Defense - Original)", alpha=0.8
                )
                if r["accuracy_noisy"] is not None:
                    plt.axhline(
                        y=r["accuracy_noisy"], linestyle=':', color=system_colors[system],
                        label=f"{system} (No Defense - Noisy)", alpha=0.8
                    )

    plt.xlabel("Epsilon / Privacy Budget")
    plt.ylabel("Reconstruction Accuracy (%)")
    plt.title("Reconstruction Accuracy vs Epsilon / Privacy Budget by System")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved at {plot_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Read results from the base path")
    parser.add_argument("base_path", type=str,
                        help="Base path containing the results")
    args = parser.parse_args()
    base_path = args.base_path
    markdown_path = os.path.join(base_path, "results_summary.md")
    plot_path = os.path.join(base_path, "accuracy_vs_epsilon.pdf")

    print(base_path)
    results = collect_results(base_path)
    generate_grouped_markdown(results, markdown_path)
    plot_results(results, plot_path)
    plot_results_by_system(results, plot_path.replace(".pdf", ".png"))
