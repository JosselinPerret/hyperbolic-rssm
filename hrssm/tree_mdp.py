"""
B-ary Tree MDP environment for testing hierarchical representation learning.

Structure:
  - B: branching factor
  - L: maximum depth
  - n_nodes: (B^{L+1} - 1) / (B - 1)  for B > 1

Each node has a fixed 64-dim observation generated from depth and branch
embeddings mixed through a random linear layer:
    obs_v = tanh(W_obs @ concat(e_depth, e_branch)) + eps

The model only receives obs_v and must infer hierarchical structure.
Ground-truth depth and branch IDs are used only for evaluation.

Random walk policy: at each step, move to a uniformly random neighbour
(parent or any child). Root and leaves have fewer neighbours.
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Optional


class TreeNode:
    __slots__ = ("idx", "depth", "parent", "children", "obs", "branch_id")

    def __init__(self, idx, depth, parent, branch_id, obs):
        self.idx      = idx
        self.depth    = depth
        self.parent   = parent      # -1 for root
        self.children: List[int] = []
        self.branch_id = branch_id  # index among parent's children (0..B-1)
        self.obs      = obs         # (obs_dim,) float32


class BaryTreeMDP:
    """
    B-ary tree MDP.

    Args:
        B:       branching factor
        L:       max depth
        obs_dim: observation dimension
        seed:    random seed for observation generation
    """

    def __init__(self, B: int = 4, L: int = 5, obs_dim: int = 64, seed: int = 42):
        self.B       = B
        self.L       = L
        self.obs_dim = obs_dim

        rng = np.random.RandomState(seed)
        self._build(rng)

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _build(self, rng: np.random.RandomState):
        depth_emb_dim  = 16
        branch_emb_dim = 16

        # Fixed depth embeddings shared across all nodes at the same depth
        depth_embs = rng.randn(self.L + 1, depth_emb_dim).astype(np.float32)

        # Random projection to observation space
        W_obs = (rng.randn(self.obs_dim, depth_emb_dim + branch_emb_dim) * 0.3).astype(np.float32)

        self.nodes: List[TreeNode] = []
        # Queue entries: (parent_idx, depth, branch_emb, branch_id_at_parent)
        queue = deque([(
            -1, 0,
            np.zeros(branch_emb_dim, dtype=np.float32),
            0
        )])

        while queue:
            parent_idx, depth, branch_emb, branch_id = queue.popleft()
            idx = len(self.nodes)

            feat = np.concatenate([depth_embs[depth], branch_emb])
            obs  = np.tanh(W_obs @ feat) + rng.randn(self.obs_dim).astype(np.float32) * 0.05

            node = TreeNode(idx, depth, parent_idx, branch_id, obs)
            self.nodes.append(node)

            if parent_idx >= 0:
                self.nodes[parent_idx].children.append(idx)

            if depth < self.L:
                for b in range(self.B):
                    child_emb = rng.randn(branch_emb_dim).astype(np.float32)
                    queue.append((idx, depth + 1, child_emb, b))

        self.n_nodes = len(self.nodes)

        # Precomputed adjacency lists for O(1) random-walk steps
        self._adj: List[List[int]] = []
        for n in self.nodes:
            neighbours = list(n.children)
            if n.parent >= 0:
                neighbours.append(n.parent)
            self._adj.append(neighbours)

        # Group node indices by depth for stratified evaluation
        self.nodes_by_depth: List[List[int]] = [[] for _ in range(self.L + 1)]
        for n in self.nodes:
            self.nodes_by_depth[n.depth].append(n.idx)

    # ------------------------------------------------------------------
    # Trajectory sampling
    # ------------------------------------------------------------------

    def sample_trajectory(
        self,
        length: int = 10,
        start_node: Optional[int] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> List[int]:
        """Random walk of given length. Returns list of node indices."""
        if rng is None:
            rng = np.random.RandomState()
        curr = rng.randint(0, self.n_nodes) if start_node is None else start_node
        traj = [curr]
        for _ in range(length - 1):
            nbs = self._adj[curr]
            curr = int(rng.choice(nbs)) if nbs else curr
            traj.append(curr)
        return traj

    def sample_batch(
        self,
        batch_size: int,
        seq_len: int,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            obs_batch:    (batch, seq_len, obs_dim)  float32
            depth_batch:  (batch, seq_len)            int32
            branch_batch: (batch, seq_len)            int32  (branch_id at parent)
        """
        if rng is None:
            rng = np.random.RandomState()

        obs_list, depth_list, branch_list = [], [], []
        for _ in range(batch_size):
            traj = self.sample_trajectory(length=seq_len, rng=rng)
            obs_list.append(np.stack([self.nodes[i].obs    for i in traj]))
            depth_list.append([self.nodes[i].depth         for i in traj])
            branch_list.append([self.nodes[i].branch_id    for i in traj])

        return (
            np.stack(obs_list).astype(np.float32),
            np.array(depth_list, dtype=np.int32),
            np.array(branch_list, dtype=np.int32),
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def depth_of(self) -> np.ndarray:
        """depth_of[i] = depth of node i."""
        return np.array([n.depth for n in self.nodes], dtype=np.int32)

    @property
    def branch_id_of(self) -> np.ndarray:
        """branch_id_of[i] = branch index of node i among its parent's children."""
        return np.array([n.branch_id for n in self.nodes], dtype=np.int32)

    def __repr__(self) -> str:
        return (
            f"BaryTreeMDP(B={self.B}, L={self.L}, "
            f"n_nodes={self.n_nodes}, obs_dim={self.obs_dim})"
        )
