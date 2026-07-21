import numpy as np
import os
import math
from scipy.interpolate import CubicSpline
# import CSF
import glob
from fire import Fire
from tqdm import trange
import open3d as o3d


class SUTE:
    def __init__(
            self,
            pcd_path,
            params_path,
            global_base_vector=np.array([1.0, 0.0, 0.0]),
            p=5
    ):
        self.pcd_path = pcd_path
        self.params_path = params_path
        self.global_base_vector = global_base_vector
        self.p = p

    def _load_pcd(self):
        data = np.loadtxt(self.pcd_path)
        points, labels = data[:, :3], data[:, -2:]

        return points, labels

    def _load_params(self):
        params = np.loadtxt(self.params_path)
        n = params.shape[0]
        params = params[
            [i for i in range(0, n, 5)], :
        ]
        center = params[:, :3]
        return center

    def _find_nearest_point_index(self, target_point, point_set):
        distances = np.linalg.norm(point_set - target_point, axis=1)
        min_index = np.argmin(distances)
        return min_index

    def _project_vector_onto_plane(self, vector, normal):
        proj = np.dot(vector, normal)
        vector_along_normal = proj * normal
        proj_vector = vector - vector_along_normal
        norm = np.linalg.norm(proj_vector)
        if norm > 0:
            proj_vector = proj_vector / norm
        return proj_vector, norm, proj

    def _smooth_axis_spline(self, center, dense=10000):
        x, y, z = center[:, 0], center[:, 1], center[:, 2]

        lengths = np.zeros([center.shape[0]])
        for i in range(1, center.shape[0]):
            length = np.linalg.norm((center[i, :] - center[i - 1, :]))
            lengths[i] = lengths[i - 1] + length
        t = lengths / lengths[-1]

        spline_x = CubicSpline(t, x)
        spline_y = CubicSpline(t, y)
        spline_z = CubicSpline(t, z)

        t_dense = np.linspace(0, 1, dense)
        x_smooth = spline_x(t_dense)
        y_smooth = spline_y(t_dense)
        z_smooth = spline_z(t_dense)
        spline = np.column_stack([x_smooth, y_smooth, z_smooth])

        spline_dx = spline_x.derivative()
        spline_dy = spline_y.derivative()
        spline_dz = spline_z.derivative()

        dx_dt = spline_dx(t_dense)
        dy_dt = spline_dy(t_dense)
        dz_dt = spline_dz(t_dense)
        direction_vector = np.column_stack([dx_dt, dy_dt, dz_dt])
        direction_vector = direction_vector / np.linalg.norm(direction_vector, axis=1).reshape(-1, 1)

        cumulative_lengths = [0]
        for i in range(spline.shape[0]):
            if i > 0:
                segment_length = np.linalg.norm(spline[i] - spline[i - 1])
                cumulative_lengths.append(cumulative_lengths[-1] + segment_length)

        return spline, direction_vector, cumulative_lengths

    def core_embedding(self):
        points, labels = self._load_pcd()
        center = self._load_params()

        center, directions, cumulative_lengths = self._smooth_axis_spline(center)

        x, y, z = np.zeros(points.shape[0]), np.zeros(points.shape[0]), np.zeros(points.shape[0])

        for i, point in enumerate(points):
            idx = self._find_nearest_point_index(point, center)
            direction = directions[idx]
            reference_point = center[idx]
            base_length = cumulative_lengths[idx]

            base_vector, _, _ = self._project_vector_onto_plane(self.global_base_vector, direction)

            translated_points = point - reference_point

            (radial_vector, radius_norm,
             axial_projection) = self._project_vector_onto_plane(
                translated_points, direction
            )

            x_axis = base_vector
            y_axis = np.cross(direction, base_vector)
            y_axis = y_axis / np.linalg.norm(y_axis)

            if radius_norm > 0:
                radial_direction = radial_vector / radius_norm
            else:
                radial_direction = np.zeros(3)

            point_x = np.dot(radial_direction, x_axis)
            point_y = np.dot(radial_direction, y_axis)

            theta = math.atan2(point_y, point_x)

            x[i] = (self.p * theta / (2 * np.pi))
            y[i] = base_length + axial_projection
            z[i] = radius_norm

        coordinates = np.column_stack([x, y, z])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coordinates)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.radius, max_nn=self.max_nn
            )
        )
        pcd.orient_normals_consistent_tangent_plane(
            k=self.max_nn
        )
        normals = np.asarray(pcd.normals)

        return np.column_stack([coordinates, normals, labels])

# class CSFStem:
#     def __init__(self, points, cloth_resolution=0.5, class_threshold=0.05, h_threshold=1.5):
#         self.points = points
#         self.cloth_resolution = cloth_resolution
#         self.class_threshold = class_threshold
#         self.h_threshold = h_threshold
#
#     def filter(self):
#         csf = CSF.CSF()
#
#         csf.params.bSloopSmooth = False
#         csf.params.cloth_resolution = self.cloth_resolution
#         csf.params.class_threshold = self.class_threshold
#
#         csf.setPointCloud(self.points)
#         stem = CSF.VecInt()
#         non_stem = CSF.VecInt()
#         csf.do_filtering(stem, non_stem)
#
#         index = np.zeros((self.points.shape[0], 1))
#         index[non_stem] = 1
#
#         pcd = np.column_stack((self.points, index))
#         pcd[pcd[:, 1] < self.h_threshold, -1] = 0
#
#         return pcd