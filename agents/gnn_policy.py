"""
GNN (Graph Neural Network) policy for V4 routing — Adaptive Conductor.

V4 changes from V2:
    A. node_feat_dim bumped from 8 → 9 (added normalized_backlog)
    B. action_dim is L × (1 + S) = 80 for 16L × 4S topology
       (admission_rate + 4 spine weights per leaf)
    C. Actor output head: Tanh → [-1, 1] for both admission + spines
    D. All prior GCN architecture preserved (batched, skip, dual-pool)

Architecture:
    Node features → input_proj → GCN×L (skip + LN + ReLU) → dual_pool → graph_emb
    Edge features → MLP → dual_pool → edge_emb
    Intent matrix → MLP → intent_emb
    [graph_emb | edge_emb | intent_emb] → combiner → features
    features → actor_head → action_mean  (admission + spine weights)
    features → log_std_head → log_std    (state-dependent, clamped)
    features → value_head → state_value
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# ─── Initialisation helpers ────────────────────────────

def _ortho_init(module: nn.Module, gain: float = 1.0):
    """Orthogonal initialization for Linear layers (PPO best practice)."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ─── Feature Extractor ─────────────────────────────────

class GNNFeaturesExtractor(nn.Module):
    """Feature extractor using a **batched** GCN with dual pooling.

    Observation layout (flat vector, single frame):
        [node_features | edge_features | intent_matrix |
         link_utils | link_queues | link_ecn | routing_weights]

    Processing pipeline:
        1. Node features → project to hidden dim →
           GCN layers (skip + LayerNorm + ReLU) →
           dual global pool (mean + max) → graph_emb
        2. Edge features → per-edge MLP →
           dual pool (mean + max) → edge_emb
        3. Intent matrix → MLP → intent_emb
        4. Concatenate [graph_emb, edge_emb, intent_emb] →
           combiner MLP → features
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        num_leaves: int,
        num_spines: int,
        node_feat_dim: int = 9,
        edge_feat_dim: int = 7,
        gcn_hidden: int = 64,
        gcn_layers: int = 2,
        intent_hidden: int = 32,
        edge_hidden: int = 32,
        features_dim: int = 128,
    ):
        super().__init__()

        if not HAS_PYG:
            raise ImportError(
                "torch_geometric is required for GNN policy. "
                "Install with: pip install torch-geometric"
            )

        self.num_leaves = num_leaves
        self.num_spines = num_spines
        self.num_nodes = num_leaves + num_spines
        self.num_edges = 2 * num_leaves * num_spines
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self._features_dim = features_dim

        # Observation segment sizes
        self._node_feat_size = self.num_nodes * node_feat_dim
        self._edge_feat_size = self.num_edges * edge_feat_dim
        self._intent_size = num_leaves * num_leaves
        # Flat link-level features after intent: utils, queues, ecn, cap, up, split_ratios
        num_links = 2 * num_leaves * num_spines
        self._flat_link_dim = 5 * num_links + num_leaves * num_spines

        # ── GCN path ────────────────────────────────────
        self.input_proj = nn.Linear(node_feat_dim, gcn_hidden)
        self.gcn_convs = nn.ModuleList()
        self.gcn_norms = nn.ModuleList()
        for _ in range(gcn_layers):
            self.gcn_convs.append(GCNConv(gcn_hidden, gcn_hidden))
            self.gcn_norms.append(nn.LayerNorm(gcn_hidden))

        # ── Edge path ───────────────────────────────────
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, edge_hidden),
            nn.ReLU(),
        )

        # ── Intent path ────────────────────────────────
        self.intent_encoder = nn.Sequential(
            nn.Linear(self._intent_size, intent_hidden),
            nn.ReLU(),
            nn.Linear(intent_hidden, intent_hidden),
            nn.ReLU(),
        )

        # ── Flat link-level path (direct per-link features) ──
        flat_link_hidden = 64
        self.flat_link_encoder = nn.Sequential(
            nn.Linear(self._flat_link_dim, flat_link_hidden),
            nn.ReLU(),
            nn.Linear(flat_link_hidden, flat_link_hidden),
            nn.ReLU(),
        )

        # ── Combiner ───────────────────────────────────
        # GCN dual pool: 2 × gcn_hidden
        # Edge dual pool: 2 × edge_hidden
        # Intent: intent_hidden
        # Flat link: flat_link_hidden
        combined_dim = 2 * gcn_hidden + 2 * edge_hidden + intent_hidden + flat_link_hidden
        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

        # ── Static edge_index (registered as buffer) ───
        edges = []
        for i in range(num_leaves):
            for j in range(num_spines):
                edges.append([i, num_leaves + j])            # uplink
                edges.append([num_leaves + j, i])            # downlink
        self.register_buffer(
            '_edge_index',
            torch.tensor(edges, dtype=torch.long).T,        # (2, num_edges)
        )

        # ── Orthogonal init (gain=√2 for ReLU networks) ─
        self.apply(lambda m: _ortho_init(m, gain=np.sqrt(2)))

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Process *batched* observations through GCN + edge + intent.

        All graphs in the batch are combined into one big disconnected
        graph with offset node indices, processed in a single GCN pass,
        and separated again via ``global_mean_pool`` / ``global_max_pool``.

        Args:
            observations: (batch_size, obs_dim) flat tensor.
        Returns:
            (batch_size, features_dim) feature tensor.
        """
        B = observations.shape[0]
        device = observations.device

        # ── Unpack observation segments ─────────────────
        off = 0
        node_flat   = observations[:, off:off + self._node_feat_size];  off += self._node_feat_size
        edge_flat   = observations[:, off:off + self._edge_feat_size];  off += self._edge_feat_size
        intent_flat = observations[:, off:off + self._intent_size];     off += self._intent_size
        # Flat link features: per-link utils, queues, ecn, cap, up + split ratios
        flat_link   = observations[:, off:off + self._flat_link_dim]

        # ── GCN path (batched) ──────────────────────────
        # Stack all nodes: (B × N, node_feat_dim)
        x = node_flat.reshape(B * self.num_nodes, self.node_feat_dim)
        x = self.input_proj(x)                                  # → (B*N, H)

        # Build batched edge_index — offset each graph's indices
        ei = self._edge_index                                    # (2, E)
        E  = ei.shape[1]
        offsets = torch.arange(B, device=device) * self.num_nodes
        batched_ei = ei.repeat(1, B) + offsets.repeat_interleave(E).unsqueeze(0)

        # Batch vector for pooling: [0,0,..,0, 1,1,..,1, …]
        batch_vec = torch.arange(B, device=device).repeat_interleave(self.num_nodes)

        # GCN layers with skip connections + LayerNorm
        for conv, norm in zip(self.gcn_convs, self.gcn_norms):
            residual = x
            x = conv(x, batched_ei)
            x = norm(x)
            x = torch.relu(x)
            x = x + residual                                    # skip connection

        # Dual pooling: mean (avg health) + max (worst bottleneck)
        g_mean = global_mean_pool(x, batch_vec)                  # (B, H)
        g_max  = global_max_pool(x, batch_vec)                   # (B, H)
        graph_emb = torch.cat([g_mean, g_max], dim=-1)           # (B, 2H)

        # ── Edge path ───────────────────────────────────
        ef = edge_flat.reshape(B * self.num_edges, self.edge_feat_dim)
        ef = self.edge_encoder(ef)                               # (B*E, eh)
        ef = ef.reshape(B, self.num_edges, -1)
        e_mean = ef.mean(dim=1)                                  # (B, eh)
        e_max  = ef.max(dim=1).values                            # (B, eh)
        edge_emb = torch.cat([e_mean, e_max], dim=-1)            # (B, 2*eh)

        # ── Intent path ─────────────────────────────────
        intent_emb = self.intent_encoder(intent_flat)            # (B, ih)

        # ── Flat link path ──────────────────────────────
        flat_link_emb = self.flat_link_encoder(flat_link)        # (B, flh)

        # ── Combine all paths ───────────────────────────
        combined = torch.cat([graph_emb, edge_emb, intent_emb, flat_link_emb], dim=-1)
        return self.combiner(combined)


# ─── Actor-Critic Policy ───────────────────────────────

class GNNActorCriticPolicy(nn.Module):
    """Actor-critic with GNN features and **state-dependent** std.

    Key architectural fix: ``log_std`` is no longer a lone
    ``nn.Parameter``.  It is the output of a small linear head
    on the shared features, so the exploration level adapts
    to each observed network state.  Clamped to [-3.0, 0.5]
    (std ∈ [0.05, 1.65]) for numerical safety.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_dim: int,
        num_leaves: int,
        num_spines: int,
        gcn_hidden: int = 64,
        gcn_layers: int = 2,
        features_dim: int = 128,
        log_std_init: float = -1.5,
    ):
        super().__init__()

        self.num_leaves = num_leaves
        self.num_spines = num_spines
        self.action_dim = action_dim

        # ── Shared feature extractor ────────────────────
        self.features_extractor = GNNFeaturesExtractor(
            observation_space=observation_space,
            num_leaves=num_leaves,
            num_spines=num_spines,
            gcn_hidden=gcn_hidden,
            gcn_layers=gcn_layers,
            features_dim=features_dim,
        )

        # ── Actor: action mean ──────────────────────────
        self.action_mean = nn.Sequential(
            nn.Linear(features_dim, features_dim // 2),
            nn.ReLU(),
            nn.Linear(features_dim // 2, action_dim),
            nn.Tanh(),                                           # → [-1, 1]
        )
        # Moderate init: spine outputs start near zero (ECMP), but
        # gradients can flow to create meaningful weight shifts.
        _ortho_init(self.action_mean[-2], gain=0.1)

        # Warm-start admission: bias admission dimensions toward +1 (high admission).
        # Action layout: [adm0, s0_0, s1_0, adm1, s0_1, s1_1, ...] per leaf.
        # Admission indices: 0, 1+S, 2*(1+S), ...
        stride = 1 + num_spines
        admission_indices = list(range(0, num_leaves * stride, stride))
        with torch.no_grad():
            self.action_mean[-2].bias.data[admission_indices] = 2.0  # tanh(2) ≈ 0.96

        # ── Actor: state-dependent log_std ──────────────
        self.log_std_head = nn.Linear(features_dim, action_dim)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, log_std_init)

        # ── Critic: state value ─────────────────────────
        self.value_head = nn.Sequential(
            nn.Linear(features_dim, features_dim // 2),
            nn.ReLU(),
            nn.Linear(features_dim // 2, 1),
        )
        _ortho_init(self.value_head[-1], gain=1.0)

    def forward(self, obs: torch.Tensor):
        """Returns (action_mean, log_std, value)."""
        features = self.features_extractor(obs)
        action_mean = self.action_mean(features)
        log_std = torch.clamp(
            self.log_std_head(features), min=-3.0, max=-1.0
        )  # std ∈ [0.05, 0.37] — calibrated for temp=5 softmax
        value = self.value_head(features)
        return action_mean, log_std, value

    def get_action_and_value(self, obs: torch.Tensor, action=None):
        """Sample action, compute log_prob, entropy, value for PPO."""
        action_mean, log_std, value = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(action_mean, std)

        if action is None:
            action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value.squeeze(-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """State value only (used in GAE computation)."""
        features = self.features_extractor(obs)
        return self.value_head(features).squeeze(-1)
