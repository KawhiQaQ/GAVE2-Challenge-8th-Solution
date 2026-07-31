from __future__ import annotations

import random

import numpy as np
from skimage import graph, morphology


def topology_path_counts(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    threshold: float,
    n_paths: int,
    seed: int,
) -> tuple[int, int, int]:
    """Return RRWNet-style (infeasible, wrong-length, correct) path counts."""
    rng = random.Random(seed)
    predicted_binary = prediction > threshold
    predicted_components = morphology.label(predicted_binary)

    ground_centerline = morphology.skeletonize(ground_truth > 0.5)
    ground_components = morphology.label(ground_centerline)
    predicted_centerline = morphology.skeletonize(predicted_binary)

    ground_rows, ground_columns = np.where(ground_centerline)
    if ground_rows.size == 0:
        return n_paths, 0, 0

    ground_cost = np.ones(ground_centerline.shape, dtype=np.float32)
    ground_cost[~ground_centerline] = 10000.0
    predicted_cost = np.ones(predicted_centerline.shape, dtype=np.float32)
    predicted_cost[~predicted_centerline] = 10000.0
    predicted_points = np.column_stack(np.where(predicted_centerline))

    results: list[int] = []
    for _ in range(n_paths):
        first_index = rng.randint(0, len(ground_rows) - 1)
        first_point = (
            int(ground_rows[first_index]),
            int(ground_columns[first_index]),
        )
        component = ground_components[first_point]
        component_points = np.column_stack(np.where(ground_components == component))
        second_point_array = component_points[
            rng.randint(0, len(component_points) - 1)
        ]
        second_point = (int(second_point_array[0]), int(second_point_array[1]))

        first_predicted_component = predicted_components[first_point]
        if (
            first_predicted_component == 0
            or first_predicted_component != predicted_components[second_point]
        ):
            results.append(0)
            continue

        first_distances = np.sum(
            (predicted_points - np.asarray(first_point)) ** 2,
            axis=1,
        )
        second_distances = np.sum(
            (predicted_points - np.asarray(second_point)) ** 2,
            axis=1,
        )
        first_correspondence = tuple(
            int(value) for value in predicted_points[np.argmin(first_distances)]
        )
        second_correspondence = tuple(
            int(value) for value in predicted_points[np.argmin(second_distances)]
        )

        ground_path, _ = graph.route_through_array(
            ground_cost,
            first_point,
            second_point,
        )
        predicted_path, _ = graph.route_through_array(
            predicted_cost,
            first_correspondence,
            second_correspondence,
        )
        ground_path_array = np.asarray(ground_path)
        predicted_path_array = np.asarray(predicted_path)

        if predicted_path_array.shape[0] < 2:
            results.append(2)
            continue

        ground_length = np.sqrt(
            np.sum(np.diff(ground_path_array, axis=0) ** 2, axis=1)
        ).sum()
        predicted_length = np.sqrt(
            np.sum(np.diff(predicted_path_array, axis=0) ** 2, axis=1)
        ).sum()
        ratio = ground_length / predicted_length
        results.append(1 if ratio < 0.9 or ratio > 1.1 else 2)

    return results.count(0), results.count(1), results.count(2)
