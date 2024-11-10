import torch
import torch.nn as nn
import torch.nn.functional as F

class HausdorffDistance_Metric(nn.Module):
    def __init__(self, percentile=None, batch_size=1000):
        """
        Initialize the Hausdorff Distance Metric.
        
        Args:
            percentile (int or None): Percentile for HD calculation (e.g., 95). If None, compute the standard Hausdorff Distance.
            batch_size (int): Number of points to process in each batch during distance computation.
        """
        super(HausdorffDistance_Metric, self).__init__()
        self.percentile = percentile
        self.batch_size = batch_size

    def forward(self, inputs, targets, **kwargs):
        """
        Compute Hausdorff Distance or the specified percentile HD.
        
        Args:
            inputs (Tensor): Model output prediction mask (binary, [batch, channel, D, H, W]).
            targets (Tensor): Target label mask (binary, [batch, channel, D, H, W]).
        
        Returns:
            Tensor: Hausdorff Distance or specified percentile HD value for each batch.
        """
        if inputs.ndim != 5 or targets.ndim != 5:
            raise ValueError("Inputs and targets must be of shape [batch, channel, D, H, W].")

        batch_size = inputs.size(0)
        hausdorff_distances = torch.zeros(batch_size, device=inputs.device)

        for i in range(batch_size):
            input_mask = inputs[i]
            target_mask = targets[i]

            # Combine channel dimensions into a single mask
            input_combined = torch.any(input_mask > 0, dim=0)
            target_combined = torch.any(target_mask > 0, dim=0)

            # Extract edges
            input_edges = self._get_edges(input_combined)
            target_edges = self._get_edges(target_combined)

            if input_edges.sum() == 0 or target_edges.sum() == 0:
                hausdorff_distances[i] = float('inf')
                continue

            # Extract coordinates
            input_points = input_edges.nonzero(as_tuple=False).float()
            target_points = target_edges.nonzero(as_tuple=False).float()

            # Compute distances
            distances_input_to_target = self._compute_distances(input_points, target_points, self.batch_size)
            distances_target_to_input = self._compute_distances(target_points, input_points, self.batch_size)

            if self.percentile is not None:
                forward_hd = torch.quantile(distances_input_to_target, self.percentile / 100.0)
                backward_hd = torch.quantile(distances_target_to_input, self.percentile / 100.0)
                hd_percentile = torch.max(forward_hd, backward_hd)
                hausdorff_distances[i] = hd_percentile
            else:
                hd_forward = torch.max(distances_input_to_target)
                hd_backward = torch.max(distances_target_to_input)
                hausdorff_distance = torch.max(hd_forward, hd_backward)
                hausdorff_distances[i] = hausdorff_distance

        return hausdorff_distances

    def _get_edges(self, mask):
        """
        Extract edges from a binary mask.
        
        Args:
            mask (Tensor): Binary mask.
        
        Returns:
            Tensor: Edge mask.
        """
        # Pad the mask
        padded_mask = F.pad(mask.unsqueeze(0).unsqueeze(0).float(), (1, 1, 1, 1, 1, 1), mode='constant', value=0)
        
        # Define convolution kernel
        kernel = torch.ones((1, 1, 3, 3, 3), device=mask.device, dtype=padded_mask.dtype)
        
        # Perform convolution to count neighbors
        neighbors = F.conv3d(padded_mask, kernel, padding=0).squeeze()
        
        # Identify edge pixels: pixels that have at least one neighbor but are not part of the mask
        edges = (neighbors > 0) & (mask == 0)
        
        return edges

    def _compute_distances(self, source_points, target_points, batch_size=1000):
        """
        Compute the minimum distances between two sets of points using CPU and batching to manage memory.
        
        Args:
            source_points (Tensor): Source point set, shape (N, 3).
            target_points (Tensor): Target point set, shape (M, 3).
            batch_size (int): Number of source points to process in each batch.
        
        Returns:
            Tensor: Minimum distance from each source point to the nearest target point, shape (N,).
        """
        min_distances = []
        target_points_cpu = target_points.cpu()
        source_points_cpu = source_points.cpu()
        
        for i in range(0, source_points_cpu.size(0), batch_size):
            batch_source = source_points_cpu[i:i+batch_size]
            distances = torch.cdist(batch_source, target_points_cpu)
            min_d, _ = torch.min(distances, dim=1)
            min_distances.append(min_d)
        
        # Concatenate all minimum distances and move back to the original device
        min_distances = torch.cat(min_distances).to(source_points.device)
        return min_distances
