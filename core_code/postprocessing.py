import os
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.cluster import HDBSCAN
from fire import Fire
from typing import Tuple


def restore_point_cloud(pred_pcd_path, raw_pcd_path):

    pred_pcd = np.loadtxt(pred_pcd_path)
    raw_pcd = np.loadtxt(raw_pcd_path)

    gt_instances = raw_pcd[:, -1]

    zero_idx = np.where(gt_instances == 0)[0][0]
    stem_indices = np.where(gt_instances == 0)[0]
    gt_instances = np.delete(gt_instances, zero_idx, axis=0)
    gt_instances = np.append(gt_instances, 0)

    first_stem_point = raw_pcd[stem_indices[0], :]

    restored_pcd = np.delete(raw_pcd, stem_indices[0], axis=0)

    branch_mask = restored_pcd[:, -2] == 0
    restored_pcd[branch_mask, -1] = pred_pcd[:-1, -2]

    restored_pcd = np.vstack((restored_pcd, first_stem_point))

    result = np.column_stack((restored_pcd, gt_instances))
    return result


def filter_invalid_instances(labels, count_threshold=20):

    unique_labels, label_counts = np.unique(labels, return_counts=True)
    invalid_labels = unique_labels[label_counts < count_threshold]

    mask = np.isin(labels, invalid_labels)
    labels[mask] = -1
    return labels


def sort_instances_by_elevation(points, labels):

    _, contiguous_labels = np.unique(labels, return_inverse=True)
    unique_labels = np.unique(contiguous_labels).astype(np.uint64)

    z_means = np.zeros(len(unique_labels), dtype=np.float64)

    for i, label in enumerate(unique_labels):
        if label == 0:
            z_means[i] = 0.0
        elif label == 1:
            z_means[i] = 1e-4
        else:
            z_coords = points[contiguous_labels == label, 2]
            z_means[i] = (np.max(z_coords) + np.min(z_coords)) / 2.0

    sorted_indices = np.argsort(z_means)
    sorted_labels = unique_labels[sorted_indices]

    label_mapping = np.zeros(np.max(contiguous_labels) + 1, dtype=contiguous_labels.dtype)
    for new_label, old_label in enumerate(sorted_labels):
        label_mapping[old_label] = new_label

    sorted_instance_labels = label_mapping[contiguous_labels]

    return points, sorted_instance_labels


def post_process(pred_dir, orig_dir, output_dir, use_postprocess=True):

    os.makedirs(output_dir, exist_ok=True)

    pred_list = sorted(glob(os.path.join(pred_dir, "*.txt")))
    orig_list = sorted(glob(os.path.join(orig_dir, "*.txt")))

    assert len(pred_list) == len(orig_list), "Mismatch in number of prediction and original files."

    save_format = ["%.6f"] * 3 + ["%d"] * 3

    for pred_path, orig_path in tqdm(zip(pred_list, orig_list), total=len(pred_list), desc="Processing"):

        data = restore_point_cloud(pred_path, orig_path)

        if not use_postprocess:
            sorted_points, sorted_labels = sort_instances_by_elevation(data[:, :3], data[:, -2])
            data[:, :3] = sorted_points
            data[:, -2] = sorted_labels

            save_name = os.path.basename(orig_path)
            np.savetxt(os.path.join(output_dir, save_name), data, fmt=save_format)
            continue

        pred_instances = data[:, -2]
        pred_instances = filter_invalid_instances(pred_instances, count_threshold=20)

        noise_mask = (pred_instances == -1)
        valid_mask = ~noise_mask

        valid_pcd = data[valid_mask]
        noise_pcd = data[noise_mask].copy()

        if len(noise_pcd) > 0:
            clusterer = HDBSCAN(min_samples=100)
            hdb_labels = clusterer.fit_predict(noise_pcd[:, :3])

            max_existing_label = np.max(pred_instances[valid_mask]) if np.any(valid_mask) else 0
            hdb_labels += (max_existing_label + 1)

            hdb_labels[hdb_labels == np.min(hdb_labels)] = -1

            noise_pcd[:, -2] = hdb_labels

        merged_data = np.vstack((valid_pcd, noise_pcd)) if len(noise_pcd) > 0 else valid_pcd

        sorted_points, sorted_pred_instance = sort_instances_by_elevation(
            merged_data[:, :3], merged_data[:, -2]
        )

        final_result = np.column_stack((
            sorted_points,
            merged_data[:, -3],
            sorted_pred_instance,
            merged_data[:, -1]
        ))
        # x, y, z, semantic, pred_instance, true_instance
        # pred_instance:
        #     0: noise; 1: stem; 2~: branch label
        # true_instance:
        #     0: stem; 1~: branch label

        save_name = os.path.basename(orig_path)
        np.savetxt(os.path.join(output_dir, save_name), final_result, fmt=save_format)


if __name__ == "__main__":
    Fire()
