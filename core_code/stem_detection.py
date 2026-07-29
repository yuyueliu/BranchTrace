import numpy as np


def ransac_circle_fit(data, sample_size, max_trials, ref_center=None, ref_radius=None, max_ratio=None, min_ratio=None):

    if ref_center is not None and ref_radius is not None:
        tol_dist = min(0.2, ref_radius / 2)
        dists = np.linalg.norm(data[:, :2] - ref_center[:2], axis=1)
        roi_mask = dists < (ref_radius * 2)
        tolerance = min(0.01, ref_radius / 5)
        r_max = ref_radius * max_ratio
        r_min = ref_radius * min_ratio
    else:
        tol_dist = np.inf
        roi_mask = np.ones(len(data), dtype=bool)
        tolerance = 0.02
        ref_center = np.mean(data, axis=0)
        r_max = 0.50
        r_min = 0.02

    roi_data = data[roi_mask]
    roi_data_shifted = roi_data - ref_center

    labels = np.zeros(len(data), dtype=int)
    best_roi_labels = np.zeros(len(roi_data), dtype=int)
    best_center = None
    best_radius = None

    if len(roi_data) > sample_size:
        for _ in range(max_trials):
            idx = np.random.choice(len(roi_data), sample_size, replace=False)
            sample = roi_data_shifted[idx, :2]

            A = np.column_stack((sample[:, 0], sample[:, 1], np.ones(sample_size)))
            b = -(sample[:, 0] ** 2 + sample[:, 1] ** 2)

            try:
                params = np.linalg.pinv(A) @ b
            except np.linalg.LinAlgError:
                continue

            cx, cy = -params[0] / 2, -params[1] / 2
            r_sq = cx ** 2 + cy ** 2 - params[2]

            if r_sq < 0:
                continue
            r = np.sqrt(r_sq)

            dists = np.sqrt((roi_data_shifted[:, 0] - cx) ** 2 + (roi_data_shifted[:, 1] - cy) ** 2) - r
            inliers = np.abs(dists) < tolerance

            dist_to_origin = np.sqrt(cx ** 2 + cy ** 2)

            if (np.sum(inliers) > np.sum(best_roi_labels) and
                    dist_to_origin <= tol_dist and
                    r_min <= r <= r_max):
                best_roi_labels = inliers
                mean_z = np.mean(data[:, 2])
                best_center = np.array([cx + ref_center[0], cy + ref_center[1], mean_z])
                best_radius = r

        labels[roi_mask] = best_roi_labels

    return labels, best_center, best_radius


def extract_stem_circles(pcd):

    num_pts = len(pcd)
    labels = np.zeros(num_pts, dtype=int)

    min_z, max_z = pcd[:, 2].min(), pcd[:, 2].max()

    base_z = max_z * 0.015
    base_mask = (pcd[:, 2] >= base_z) & (pcd[:, 2] < base_z + 0.1)

    if base_z < min_z or np.sum(base_mask) < 50:
        base_z = 1.0
        base_mask = (pcd[:, 2] >= base_z) & (pcd[:, 2] < base_z + 0.1)

    base_pc = pcd[base_mask]

    x_edges = np.arange(base_pc[:, 0].min(), base_pc[:, 0].max() + 0.01, 0.01)
    y_edges = np.arange(base_pc[:, 1].min(), base_pc[:, 1].max() + 0.01, 0.01)

    hist, x_edges, y_edges = np.histogram2d(base_pc[:, 0], base_pc[:, 1], bins=(x_edges, y_edges))

    bin_x = np.clip(np.digitize(base_pc[:, 0], x_edges) - 1, 0, hist.shape[0] - 1)
    bin_y = np.clip(np.digitize(base_pc[:, 1], y_edges) - 1, 0, hist.shape[1] - 1)

    valid_r, valid_c = np.where(hist > 3)
    core_mask = np.zeros(len(base_pc), dtype=bool)
    for r, c in zip(valid_r, valid_c):
        core_mask |= (bin_x == r) & (bin_y == c)

    tmp_labels, base_center, base_radius = ransac_circle_fit(base_pc[core_mask], 10, 5000)

    if base_center is None:
        return labels, {"circle": np.empty((0, 3)), "radius": np.empty(0)}

    base_global_labels = labels[base_mask]
    base_global_labels[core_mask] = tmp_labels
    labels[base_mask] = base_global_labels

    circle_centers = [base_center]
    circle_radii = [base_radius]

    height = base_z + 0.1
    step = 0.1
    ref_center, ref_radius = base_center, base_radius

    while height + step < max_z:
        slice_mask = (pcd[:, 2] >= (height - step / 3)) & (pcd[:, 2] < (height + step + step / 3))
        slice_data = pcd[slice_mask]

        if np.sum(slice_mask) > 20:
            slice_labels, center, radius = ransac_circle_fit(
                slice_data, 10, 100,
                ref_center=ref_center, ref_radius=ref_radius,
                max_ratio=1.05, min_ratio=0.9
            )

            if center is not None:
                circle_centers.append(center)
                circle_radii.append(radius)
                ref_center, ref_radius = center, radius

                labels[slice_mask] |= slice_labels
                height += step
                step = 0.1
            else:
                step += 0.05
        else:
            step += 0.05

    height = base_z
    step = 0.1
    ref_center, ref_radius = base_center, base_radius

    while height - step > 0:
        slice_mask = (pcd[:, 2] >= (height - step - step / 2)) & (pcd[:, 2] < (height + step / 2))
        slice_data = pcd[slice_mask]

        if np.sum(slice_mask) > 20:
            slice_labels, center, radius = ransac_circle_fit(
                slice_data, 10, 100,
                ref_center=ref_center, ref_radius=ref_radius,
                max_ratio=1.5, min_ratio=1.0
            )

            if center is not None:
                circle_centers.append(center)
                circle_radii.append(radius)
                ref_center, ref_radius = center, radius

                labels[slice_mask] |= slice_labels
                height -= step
                step = 0.1
            else:
                step += 0.05
        else:
            step += 0.05

    circle_dict = {
        "circle": np.array(circle_centers),
        "radius": np.array(circle_radii)
    }

    return labels, circle_dict


