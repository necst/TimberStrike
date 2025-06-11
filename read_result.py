import os
import re
import ast
import csv

name_map = {
    "bagging": "Flower XGBoost Bagging",
    "cyclic": "Flower XGBoost Cyclic",
    "fedxgbllr": "FedXGBllr",
    "nvflare": "NVFlare",
    "fedtree": "FedTree",
}


def extract_train_f1_auc(train_log_file, dataset):
    if "fedxgbllr" in train_log_file:
        last_f1 = None
        last_auc = None
        pattern = r"test_loss=.*?, test_F1=([\d.]+)\s+test_AUC=([\d.]+)"
        with open(train_log_file, 'r') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    last_f1 = float(match.group(1))
                    last_auc = float(match.group(2))
        if last_f1 is not None and last_auc is not None:
            return last_f1, last_auc
        else:
            raise ValueError(
                "Error for fedxgbllr.")
    elif "bagging" in train_log_file or "cyclic" in train_log_file:
        with open(train_log_file, 'r') as f:
            for line in reversed(f.readlines()):
                if "metrics_distributed" in line:
                    # Estrai il dizionario come stringa a partire da '{'
                    match = re.search(
                        r"(metrics_distributed\s+)(\{.*\})", line)
                    if match:
                        try:
                            metrics = ast.literal_eval(match.group(2))
                            last_auc = metrics["AUC"][-1][1]
                            last_f1 = metrics["F1"][-1][1]
                            return last_f1, last_auc
                        except Exception as e:
                            raise ValueError(
                                f"Error: {e}")
        raise ValueError(
            "Error for bagging/cyclic.")
    elif "nvflare" in train_log_file:
        last_f1 = None
        last_auc = None
        pattern = r"eval-auc:([\d.]+)\s+eval-f1:([\d.]+)"

        with open(train_log_file, 'r') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    last_auc = float(match.group(1))
                    last_f1 = float(match.group(2))
        if last_f1 is not None and last_auc is not None:
            return last_f1, last_auc
        else:
            raise ValueError("Error for nvflare.")
    elif "fedtree" in train_log_file:
        num_clients = re.search(r"nc_(\d+)", train_log_file).group(1)
        max_depth = re.search(r"md_(\d+)", train_log_file).group(1)
        fedtree_dir = os.path.join(os.path.dirname(
            __file__), "fl_systems", "frameworks", "FedTree")
        logs_dir = os.path.join(fedtree_dir, "logs",
                                dataset, f"logs_{num_clients}_{max_depth}")

        f1, auc = None, None
        with open(os.path.join(logs_dir, "party0.log"), 'r') as f:
            # 2025-01-25 15:21:39,938 INFO gbdt.cpp:141 : AUC = 0.962217
            lines = f.readlines()
            for line in reversed(lines):
                if "AUC" in line:
                    auc = float(line.split()[-1])
                if "F1" in line:
                    f1 = float(line.split()[-1])
                if f1 and auc:
                    return f1, auc
    else:
        raise Exception(f"Unknown system: {train_log_file}")


def collect_results(base_path, dataset):
    results = {}

    for subdir in os.listdir(base_path):
        if subdir.startswith("nc_"):
            try:
                num_clients = int(subdir.split("_")[1])
            except ValueError:
                print(f"Skipping invalid subdirectory: {subdir}")
                continue
            nc_folder_path = os.path.join(base_path, subdir)

            for md_subdir in os.listdir(nc_folder_path):
                if md_subdir.startswith("md_"):
                    try:
                        max_depth = int(subdir.split("_")[1])
                    except ValueError:
                        print(f"Skipping invalid subdirectory: {subdir}")
                        continue

                    md_folder_path = os.path.join(
                        nc_folder_path, f"md_{max_depth}")
                    print(md_folder_path)
                    no_defense_path = os.path.join(
                        md_folder_path, "no_defense", "attack")
                    no_defense_path_train = os.path.join(
                        md_folder_path, "no_defense", "train")

                    if not os.path.isdir(no_defense_path):
                        print(
                            f"No 'no_defense' directory found in {md_folder_path}")
                        continue

                    for system_name in os.listdir(no_defense_path):
                        system_folder = os.path.join(
                            no_defense_path, system_name)
                        if not os.path.isdir(system_folder):
                            continue

                        accuracy = None
                        tsne_plots = []

                        for file in os.listdir(system_folder):
                            if file.endswith(".log"):
                                log_file_path = os.path.join(
                                    system_folder, file)
                                with open(log_file_path, 'r') as log_file:
                                    lines = log_file.readlines()
                                    for line in lines[-1000:]:
                                        match = re.search(
                                            r"Reconstruction accuracy:\s*([\d.]+)%", line)
                                        if match:
                                            accuracy = float(match.group(1))
                                            break
                            elif file.startswith("tsne_") and file.endswith(".png"):
                                tsne_plots.append(
                                    os.path.join(system_folder, file))

                        train_log_file = os.path.join(no_defense_path_train.replace(
                            "/baseline/", "/"), f"{system_name}.log")
                        f1, auc = extract_train_f1_auc(train_log_file, dataset)

                        if num_clients not in results:
                            results[num_clients] = {}
                        if max_depth not in results[num_clients]:
                            results[num_clients][max_depth] = {}
                        results[num_clients][max_depth][system_name] = {
                            "accuracy": accuracy,
                            "f1": f1,
                            "auc": auc,
                            "tsne_plots": tsne_plots,
                        }

    return results


def export_comparison_to_csv(results1, results2, output_path):
    with open(output_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(
            ["Num Clients", "Depth", "System", "RA (F-TP)", "RA", "F1", "AUC"])

        # Iterate through all keys present in both results
        for num_clients in sorted(results1.keys()):
            if num_clients not in results2:
                continue
            for max_depth in sorted(results1[num_clients].keys()):
                if max_depth not in results2[num_clients]:
                    continue
                for system_name in results1[num_clients][max_depth]:
                    if system_name not in results2[num_clients][max_depth]:
                        continue

                    entry1 = results1[num_clients][max_depth][system_name]
                    entry2 = results2[num_clients][max_depth][system_name]

                    # Extract values
                    ra = entry1.get("accuracy")
                    ra_ftp = entry2.get("accuracy")
                    f1 = entry1.get("f1")
                    auc = entry1.get("auc")

                    # Write row
                    writer.writerow(
                        [num_clients, max_depth, system_name, ra_ftp, ra, f1, auc])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze and plot no_defense results by max_depth with TSNE plots")
    parser.add_argument("base_path", type=str,
                        help="Base path containing results")
    parser.add_argument("dataset", type=str,
                        help="Dataset (e.g. stroke, diabetes)")
    parser.add_argument("output_path", type=str,
                        help="Output base path for plots and markdown")
    args = parser.parse_args()

    base_path = args.base_path
    dataset = args.dataset
    output_path = args.output_path

    results = collect_results(base_path, dataset)
    baseline_results = collect_results(
        os.path.join(base_path, "baseline"), dataset)
    print(results)
    print(baseline_results)

    export_comparison_to_csv(results, baseline_results, output_path)
