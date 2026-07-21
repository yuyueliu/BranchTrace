import numpy as np
import os
import glob
from tqdm import tqdm
from fire import Fire


class Mask2PCD:
    def __init__(self, mask_dir, pcd_dir, output_dir):
        self.mask_dir = mask_dir
        self.pcd_dir = pcd_dir
        self.output_dir = output_dir
        self.file_names, self.scan_id = self._get_names()
        os.makedirs(self.output_dir, exist_ok=True)


    def process(self):
        result_list = []
        file_list = glob.glob(os.path.join(self.mask_dir, "*_inst_nostuff.txt"))
        fmt = ["%.6f"] * 3 + ["%d"] * 2 + ["%.4f"]
        for idx, _ in enumerate(tqdm(file_list)):
            points = self._process_mask_catalogue(idx)
            if points is not None:
                file_path = os.path.join(self.output_dir, f"{self.scan_id[idx]}_temp.txt")
                np.savetxt(
                    file_path,
                    points,
                    fmt=fmt)
                result_list.append(file_path)
            else:
                print(f"{self.scan_id[idx]} has no instance.")

        id_list = list(
            set([tree_id.split("/")[-1].split("_")[0] for tree_id in result_list])
        )
        fmt = ["%.6f"] * 3 + ["%d"] * 2 + ["%.4f"]
        for tree_id in id_list:
            append_list = glob.glob(os.path.join(self.output_dir, f"{tree_id}_*_temp.txt"))
            data = None
            for file_name in append_list:
                points = np.loadtxt(file_name)
                if data is None:
                    data = points
                else:
                    data = np.vstack([data, points])
                os.remove(file_name)
            np.savetxt(
                os.path.join(self.output_dir, f"{tree_id}.txt"),
                data,
                fmt=fmt
            )
            print(f"{tree_id}.txt has been saved.")

    def _get_names(self):
        file_list = glob.glob(os.path.join(self.mask_dir, "*.txt"))
        file_names = [file_name.split("/")[-1] for file_name in file_list]
        scan_id = [f'{i.split("_")[0]}_{i.split("_")[1]}' for i in file_names]
        return file_names, scan_id

    def _load_coordinates(self, idx):
        path = os.path.join(self.pcd_dir, self.scan_id[idx] + ".npy")
        data = np.load(path)
        coordinates = data[:, :3]
        return coordinates

    def _process_mask_catalogue(self, idx):
        mask_catalogue_path = os.path.join(self.mask_dir, self.file_names[idx])

        coordinates = self._load_coordinates(idx)
        num_points = coordinates.shape[0]
        points = np.zeros((num_points, 6))
        points[:, :3] = coordinates
        points[:, 3:] = -1
        # points: x y z semantic instance confidence

        predictions = {
            "mask_path": [],
            "semantic_label": [],
            "confidence": []
        }

        try:
            with open(mask_catalogue_path, 'r') as f:
                for line in f:
                    if not line.strip() and not line.startswith("pred_mask"):
                        continue
                    parts = line.strip().split()
                    if len(parts) == 3:
                        predictions["mask_path"].append(parts[0])
                        predictions["semantic_label"].append(int(parts[1]))
                        predictions["confidence"].append(float(parts[2]))

                predictions = self._sort_predictions(predictions)

            points = self._process_mask_file(predictions, points)
            return points
        except TypeError:
            return None

    def _sort_predictions(self, predictions):
        target_idx = None
        for idx, label in enumerate(predictions["semantic_label"]):
            if label == 1:
                target_idx = idx
                break

        if target_idx is not None:
            target_mask = predictions["mask_path"][target_idx]
            target_label = predictions["semantic_label"][target_idx]
            target_confidence = predictions["confidence"][target_idx]

            predictions["mask_path"].pop(target_idx)
            predictions["semantic_label"].pop(target_idx)
            predictions["confidence"].pop(target_idx)

            predictions["mask_path"].insert(0, target_mask)
            predictions["semantic_label"].insert(0, target_label)
            predictions["confidence"].insert(0, target_confidence)

            return predictions

    # 处理掩码文件
    def _process_mask_file(self, predictions, points):
        for instance_idx, mask_path in tqdm(enumerate(predictions["mask_path"])):
            mask_path = os.path.join(self.mask_dir, mask_path)
            with open(mask_path, 'r') as f:
                for points_idx, line in enumerate(f):
                    if line.strip() == "1":
                        points[points_idx, 3] = predictions["semantic_label"][instance_idx]
                        points[points_idx, 4] = instance_idx + 1
                        points[points_idx, 5] = predictions["confidence"][instance_idx]
        return points