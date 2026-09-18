"""Columbina 4.0-MoE model components."""

import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch_geometric.nn import HGTConv, SAGEConv

from config import Config
from device_manager import DeviceManager


logger = logging.getLogger(__name__)


def _module_key(value):
    return str(value).replace('.', '__dot__')


def _unique(items):
    return list(OrderedDict((tuple(item), None) for item in items).keys())


class SLomicsExpert(nn.Module):
    """Encode omics and label-safe training SL context."""

    def __init__(self, config):
        super().__init__()
        hidden_dim = config.HIDDEN_DIM
        output_dim = config.EMBEDDING_DIM
        dropout = config.DROPOUT
        input_dim = int(getattr(config, 'OMICS_FEATURE_DIM', 262))
        self.omics_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.sl_convs = nn.ModuleList([
            SAGEConv(output_dim, output_dim),
            SAGEConv(output_dim, output_dim),
        ])
        self.sl_norms = nn.ModuleList([
            nn.LayerNorm(output_dim),
            nn.LayerNorm(output_dim),
        ])
        self.context_gate = nn.Sequential(
            nn.Linear(output_dim * 2 + 1, output_dim),
            nn.Sigmoid(),
        )
        self.context_max_weight = float(
            getattr(config, 'SL_CONTEXT_MAX_WEIGHT', 1.0)
        )
        if not 0.0 <= self.context_max_weight <= 1.0:
            raise ValueError("SL_CONTEXT_MAX_WEIGHT must be in [0, 1]")
        self.dropout = nn.Dropout(dropout)

    def forward(self, omics_features, context_edge_index=None):
        base = self.omics_encoder(omics_features)
        degree = torch.zeros(base.size(0), device=base.device, dtype=base.dtype)
        if context_edge_index is None or context_edge_index.numel() == 0:
            return base, degree

        context_edge_index = context_edge_index.to(base.device)
        reverse_edges = context_edge_index.flip(0)
        undirected_edges = torch.cat([context_edge_index, reverse_edges], dim=1)
        degree.index_add_(
            0,
            undirected_edges[1],
            torch.ones(undirected_edges.size(1), device=base.device, dtype=base.dtype),
        )

        graph = base
        for conv, norm in zip(self.sl_convs, self.sl_norms):
            update = conv(graph, undirected_edges)
            graph = norm(graph + self.dropout(F.gelu(update)))

        degree_feature = torch.log1p(degree).unsqueeze(1)
        gate = self.context_gate(
            torch.cat([base, graph, degree_feature], dim=1)
        ) * self.context_max_weight
        has_context = degree.gt(0).unsqueeze(1)
        fused = gate * graph + (1.0 - gate) * base
        return torch.where(has_context, fused, base), degree


class HGTKnowledgeEncoder(nn.Module):
    """Schema-registered HGT with automatic reverse relations."""

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = int(config.HGT_HIDDEN_DIM)
        self.output_dim = int(config.EMBEDDING_DIM)
        self.num_layers = int(config.HGT_NUM_LAYERS)
        self.num_heads = int(config.HGT_NUM_HEADS)
        self.dropout_rate = float(config.DROPOUT)
        # Recompute HGT layers during backward instead of retaining per-edge
        # attention activations.  Mathematically identical (dropout masks are
        # replayed from the preserved RNG state); it trades one extra forward
        # per layer for roughly halved activation memory on million-edge KGs.
        self.gradient_checkpointing = bool(
            getattr(config, 'HGT_GRADIENT_CHECKPOINTING', True)
        )
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("HGT_HIDDEN_DIM must be divisible by HGT_NUM_HEADS")

        self.node_encoders = nn.ModuleDict()
        self.hgt_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )
        self.node_types = []
        self.edge_types = []
        self.original_edge_types = []

    @staticmethod
    def _reverse_edge_type(edge_type):
        source, relation, target = edge_type
        return target, f'rev__{relation}', source

    def prepare_schema(self, schema):
        node_input_dims = dict(schema.get('node_input_dims', {}))
        original_edge_types = _unique(schema.get('edge_types', []))
        node_types = list(node_input_dims)
        edge_types = list(original_edge_types)
        for edge_type in original_edge_types:
            reverse_type = self._reverse_edge_type(edge_type)
            if reverse_type not in edge_types:
                edge_types.append(reverse_type)

        if not node_types:
            self.node_types = []
            self.edge_types = []
            self.original_edge_types = []
            self.node_encoders = nn.ModuleDict()
            self.hgt_layers = nn.ModuleList()
            self.layer_norms = nn.ModuleList()
            return

        signature = (tuple(node_input_dims.items()), tuple(edge_types))
        current_signature = (
            tuple((node_type, self.node_encoders[_module_key(node_type)][0].in_features)
                  for node_type in self.node_types),
            tuple(self.edge_types),
        ) if self.node_types else None
        if current_signature == signature:
            return

        device = next(self.parameters()).device
        self.node_encoders = nn.ModuleDict({
            _module_key(node_type): nn.Sequential(
                nn.Linear(int(input_dim), self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
            )
            for node_type, input_dim in node_input_dims.items()
        })
        metadata = (node_types, edge_types)
        self.hgt_layers = nn.ModuleList([
            HGTConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                metadata=metadata,
                heads=self.num_heads,
            )
            for _ in range(self.num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.ModuleDict({
                _module_key(node_type): nn.LayerNorm(self.hidden_dim)
                for node_type in node_types
            })
            for _ in range(self.num_layers)
        ])
        self.node_types = node_types
        self.edge_types = edge_types
        self.original_edge_types = original_edge_types
        self.to(device)

    def _hgt_layer_block(self, layer_index, x_dict, edge_index_dict):
        """One HGT layer: conv, residual, GELU+dropout, layer norm."""
        conv = self.hgt_layers[layer_index]
        updates = conv(x_dict, edge_index_dict)
        next_x = {}
        for node_type, current in x_dict.items():
            update = updates.get(node_type)
            if update is None:
                next_x[node_type] = current
                continue
            norm = self.layer_norms[layer_index][_module_key(node_type)]
            next_x[node_type] = norm(current + F.dropout(
                F.gelu(update), p=self.dropout_rate, training=self.training
            ))
        return next_x

    def forward(self, kg_data):
        if not self.node_types:
            raise RuntimeError("HGT schema is not prepared")
        device = next(self.parameters()).device
        x_dict = {}
        for node_type in self.node_types:
            if node_type not in kg_data.node_types:
                continue
            node_store = kg_data[node_type]
            if not hasattr(node_store, 'x') or node_store.x is None:
                continue
            x_dict[node_type] = self.node_encoders[_module_key(node_type)](
                node_store.x.to(device)
            )

        available_edges = {}
        for edge_type in self.original_edge_types:
            if edge_type not in kg_data.edge_types:
                continue
            source_type, _, target_type = edge_type
            if source_type not in x_dict or target_type not in x_dict:
                continue
            edge_index = kg_data[edge_type].edge_index.to(device)
            if edge_index.numel() == 0:
                continue
            available_edges[edge_type] = edge_index
            available_edges[self._reverse_edge_type(edge_type)] = edge_index.flip(0)

        # HGT's relation-specific linear layer treats type IDs as sorted. Preserve
        # the exact metadata order instead of interleaving each reverse relation.
        edge_index_dict = OrderedDict(
            (edge_type, available_edges[edge_type])
            for edge_type in self.edge_types
            if edge_type in available_edges
        )

        for layer_index in range(len(self.hgt_layers)):
            if not edge_index_dict:
                break
            use_checkpoint = (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            )
            if use_checkpoint:
                x_dict = torch.utils.checkpoint.checkpoint(
                    self._hgt_layer_block,
                    layer_index,
                    x_dict,
                    edge_index_dict,
                    use_reentrant=False,
                )
            else:
                x_dict = self._hgt_layer_block(layer_index, x_dict, edge_index_dict)

        return {
            node_type: self.output_projection(features)
            for node_type, features in x_dict.items()
        }


class CancerFiLM(nn.Module):
    def __init__(self, feature_dim, cancer_dim):
        super().__init__()
        self.modulation = nn.Linear(cancer_dim, feature_dim * 2)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, features, cancer_embedding):
        gamma, beta = self.modulation(cancer_embedding).chunk(2, dim=1)
        return features * (1.0 + gamma) + beta


class SymmetricPairExpert(nn.Module):
    """Order-invariant pair encoder and expert classifier."""

    def __init__(self, node_dim, expert_dim, cancer_dim, dropout):
        super().__init__()
        self.pair_encoder = nn.Sequential(
            nn.Linear(node_dim * 3 + 2, expert_dim),
            nn.LayerNorm(expert_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_dim, expert_dim),
            nn.LayerNorm(expert_dim),
            nn.GELU(),
        )
        self.cancer_film = CancerFiLM(expert_dim, cancer_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(expert_dim, expert_dim // 2),
            nn.GELU(),
            nn.Linear(expert_dim // 2, 1),
        )

    @staticmethod
    def symmetric_features(source, target, source_score, target_score):
        return torch.cat([
            source + target,
            torch.abs(source - target),
            source * target,
            source_score + target_score,
            torch.abs(source_score - target_score),
        ], dim=1)

    def forward(self, node_embeddings, edge_index, node_scores, cancer_embedding):
        source = node_embeddings[edge_index[0]]
        target = node_embeddings[edge_index[1]]
        source_score = node_scores[edge_index[0]].unsqueeze(1)
        target_score = node_scores[edge_index[1]].unsqueeze(1)
        pair_features = self.symmetric_features(
            source, target, source_score, target_score
        )
        pair_representation = self.pair_encoder(pair_features)
        pair_representation = self.cancer_film(
            pair_representation, cancer_embedding
        )
        return pair_representation, self.classifier(pair_representation).squeeze(-1)


class ParameterFreeNodeFusion(nn.Module):
    """Compatibility helper for callers that request one node embedding."""

    def forward(self, sl_embeddings, kg_embeddings, omics_features=None):
        del omics_features
        return F.layer_norm((sl_embeddings + kg_embeddings) * 0.5, sl_embeddings.shape[1:])


class Columbina_Model(nn.Module):
    """Cancer-conditioned two-expert Columbina 4.0-MoE."""

    def __init__(self, config=None, device_manager=None):
        super().__init__()
        self.config = config or Config()
        self.device_manager = device_manager or DeviceManager()
        self.device = self.device_manager.target_device

        self.sl_omics_expert = SLomicsExpert(self.config)
        self.kg_expert = HGTKnowledgeEncoder(self.config)
        self.cancer_embedding = nn.Embedding(
            int(self.config.MAX_CANCER_TYPES),
            int(self.config.CANCER_EMBEDDING_DIM),
        )
        self.sl_pair_expert = SymmetricPairExpert(
            self.config.EMBEDDING_DIM,
            self.config.MOE_EXPERT_DIM,
            self.config.CANCER_EMBEDDING_DIM,
            self.config.DROPOUT,
        )
        self.kg_pair_expert = SymmetricPairExpert(
            self.config.EMBEDDING_DIM,
            self.config.MOE_EXPERT_DIM,
            self.config.CANCER_EMBEDDING_DIM,
            self.config.DROPOUT,
        )
        self.reliability_dim = int(getattr(self.config, 'RELIABILITY_DIM', 10))
        if self.reliability_dim not in (4, 10):
            raise ValueError("RELIABILITY_DIM must be 4 (legacy) or 10 (pair-aware)")
        gate_input_dim = (
            self.config.MOE_EXPERT_DIM * 2
            + self.config.CANCER_EMBEDDING_DIM
            + self.reliability_dim
        )
        self.expert_gate = nn.Sequential(
            nn.Linear(gate_input_dim, self.config.MOE_EXPERT_DIM),
            nn.LayerNorm(self.config.MOE_EXPERT_DIM),
            nn.GELU(),
            nn.Dropout(self.config.DROPOUT),
            nn.Linear(self.config.MOE_EXPERT_DIM, 2),
        )
        self.sl_connectivity = nn.Sequential(
            nn.Linear(self.config.EMBEDDING_DIM, self.config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Linear(self.config.HIDDEN_DIM // 2, 1),
        )
        self.node_fusion = ParameterFreeNodeFusion()

        self.kg_schema = {'node_input_dims': {}, 'edge_types': []}
        self._kg_pair_stats_cache = {}
        self.train_gene_ids_set = None
        self.train_gene_ids = []
        self.train_positive_pairs = []
        self.train_gene_embeddings = None
        self.to(self.device)

    @property
    def sl_gnn(self):
        return self.sl_omics_expert

    @property
    def knowledge_gnn(self):
        return self.kg_expert

    @property
    def multimodal_fusion(self):
        return self.node_fusion

    @property
    def edge_predictor(self):
        return self.expert_gate

    def prepare_kg_schema(self, schema):
        merged_node_dims = dict(self.kg_schema.get('node_input_dims', {}))
        for node_type, input_dim in schema.get('node_input_dims', {}).items():
            previous = merged_node_dims.get(node_type)
            if previous is not None and int(previous) != int(input_dim):
                raise ValueError(
                    f"KG node dimension changed for {node_type}: {previous} vs {input_dim}"
                )
            merged_node_dims[node_type] = int(input_dim)
        merged_edge_types = _unique(
            list(self.kg_schema.get('edge_types', []))
            + list(schema.get('edge_types', []))
        )
        self.kg_schema = {
            'node_input_dims': merged_node_dims,
            'edge_types': merged_edge_types,
        }
        self.kg_expert.prepare_schema(self.kg_schema)
        return self.kg_schema

    def prepare_for_kg(self, kg_data):
        node_input_dims = {}
        for node_type in kg_data.node_types:
            node_store = kg_data[node_type]
            if hasattr(node_store, 'x') and node_store.x is not None:
                node_input_dims[node_type] = int(node_store.x.shape[1])
        return self.prepare_kg_schema({
            'node_input_dims': node_input_dims,
            'edge_types': [tuple(edge_type) for edge_type in kg_data.edge_types],
        })

    def set_train_sl_context(self, train_data, gene_ids):
        self.train_gene_ids = list(gene_ids)
        self.train_gene_ids_set = set(self.train_gene_ids)
        self.train_positive_pairs = []
        if not hasattr(train_data, 'y') or train_data.edge_index.numel() == 0:
            return
        positive_edges = train_data.edge_index[:, train_data.y > 0.5].detach().cpu()
        for source_index, target_index in positive_edges.t().tolist():
            if source_index < len(gene_ids) and target_index < len(gene_ids):
                self.train_positive_pairs.append(
                    (gene_ids[source_index], gene_ids[target_index])
                )

    def restore_train_sl_context(self, gene_ids, positive_pairs):
        self.train_gene_ids = list(gene_ids or [])
        self.train_gene_ids_set = set(self.train_gene_ids)
        self.train_positive_pairs = [tuple(pair) for pair in (positive_pairs or [])]

    def detect_new_genes(self, gene_ids):
        if self.train_gene_ids_set is None:
            return [False] * len(gene_ids)
        return [gene_id not in self.train_gene_ids_set for gene_id in gene_ids]

    @staticmethod
    def _resolve_gene_ids(gene_mapping):
        return [
            gene_id for gene_id, _ in
            sorted(gene_mapping.items(), key=lambda item: item[1])
        ]

    def _training_context_edges(self, sl_data, gene_ids, mode):
        device = sl_data.x.device
        if mode == 'train' and hasattr(sl_data, 'y') and sl_data.y.numel():
            return sl_data.edge_index[:, sl_data.y > 0.5].to(device)
        if not self.train_positive_pairs:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        index_by_gene = {gene_id: index for index, gene_id in enumerate(gene_ids)}
        edges = [
            (index_by_gene[source], index_by_gene[target])
            for source, target in self.train_positive_pairs
            if source in index_by_gene and target in index_by_gene
        ]
        if not edges:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()

    def _align_kg_embeddings(self, kg_data, gene_ids, model_device):
        empty_map = (None, [], [])
        if not self.kg_expert.node_types:
            return (
                torch.zeros(len(gene_ids), self.config.EMBEDDING_DIM, device=model_device),
                torch.zeros(len(gene_ids), device=model_device),
                empty_map,
            )
        kg_embeddings_by_type = self.kg_expert(kg_data)
        gene_type = next(
            (node_type for node_type in ('Gene', 'gene') if node_type in kg_embeddings_by_type),
            None,
        )
        aligned = torch.zeros(
            len(gene_ids), self.config.EMBEDDING_DIM, device=model_device
        )
        degree = torch.zeros(len(gene_ids), device=model_device)
        if gene_type is None:
            return aligned, degree, empty_map

        gene_store = kg_data[gene_type]
        kg_mapping = getattr(gene_store, 'node_name_to_idx', {})
        kg_indices = []
        sl_indices = []
        for sl_index, gene_id in enumerate(gene_ids):
            kg_index = kg_mapping.get(gene_id)
            if kg_index is None:
                kg_index = kg_mapping.get(str(gene_id))
            if kg_index is not None and kg_index < kg_embeddings_by_type[gene_type].size(0):
                sl_indices.append(sl_index)
                kg_indices.append(kg_index)
        if sl_indices:
            sl_tensor = torch.tensor(sl_indices, dtype=torch.long, device=model_device)
            kg_tensor = torch.tensor(kg_indices, dtype=torch.long, device=model_device)
            aligned[sl_tensor] = kg_embeddings_by_type[gene_type][kg_tensor]

        kg_degree = torch.zeros(gene_store.x.size(0), device=model_device)
        for edge_type in kg_data.edge_types:
            source_type, _, target_type = edge_type
            edge_index = kg_data[edge_type].edge_index.to(model_device)
            if edge_index.numel() == 0:
                continue
            if source_type == gene_type:
                kg_degree.index_add_(
                    0, edge_index[0], torch.ones(edge_index.size(1), device=model_device)
                )
            if target_type == gene_type:
                kg_degree.index_add_(
                    0, edge_index[1], torch.ones(edge_index.size(1), device=model_device)
                )
        if sl_indices:
            degree[sl_tensor] = kg_degree[kg_tensor]
        return aligned, degree, (gene_type, sl_indices, kg_indices)

    def _kg_pair_stats(self, kg_data, gene_type):
        """Cached (direct, common-neighbor) CSR matrices over KG-local gene indices.

        direct[i, j] = 1 if any Gene-Gene edge connects i and j (e.g. STRING).
        common[i, j] = number of KG nodes (any type) adjacent to both i and j.
        """
        edge_count = 0
        for edge_type in kg_data.edge_types:
            edge_count += int(kg_data[edge_type].edge_index.size(1))
        num_genes = int(kg_data[gene_type].x.size(0))
        cache_key = (id(kg_data), edge_count, num_genes, len(kg_data.edge_types))
        cached = self._kg_pair_stats_cache.get(cache_key)
        if cached is not None:
            return cached

        import scipy.sparse as sp

        offsets = {}
        total_nodes = 0
        for node_type in kg_data.node_types:
            node_store = kg_data[node_type]
            if hasattr(node_store, 'x') and node_store.x is not None:
                offsets[node_type] = total_nodes
                total_nodes += int(node_store.x.size(0))
            else:
                offsets[node_type] = None

        incidence_rows = []
        incidence_cols = []
        direct_src = []
        direct_dst = []
        for edge_type in kg_data.edge_types:
            source_type, _, target_type = edge_type
            if offsets.get(source_type) is None or offsets.get(target_type) is None:
                continue
            edge_index = kg_data[edge_type].edge_index
            if edge_index.numel() == 0:
                continue
            edge_np = edge_index.detach().cpu().numpy()
            if source_type == gene_type and target_type == gene_type:
                direct_src.append(edge_np[0])
                direct_dst.append(edge_np[1])
            if source_type == gene_type:
                incidence_rows.append(edge_np[0])
                incidence_cols.append(offsets[target_type] + edge_np[1])
            if target_type == gene_type:
                incidence_rows.append(edge_np[1])
                incidence_cols.append(offsets[source_type] + edge_np[0])

        if incidence_rows:
            rows = np.concatenate(incidence_rows)
            cols = np.concatenate(incidence_cols)
            incidence = sp.csr_matrix(
                (np.ones(rows.shape[0], dtype=np.float32), (rows, cols)),
                shape=(num_genes, total_nodes),
            )
            incidence.data[:] = 1.0
            common = incidence @ incidence.T
            common.setdiag(0)
            common.eliminate_zeros()
            common = common.tocsr()
        else:
            common = sp.csr_matrix((num_genes, num_genes), dtype=np.float32)

        if direct_src:
            src = np.concatenate(direct_src)
            dst = np.concatenate(direct_dst)
            sym_src = np.concatenate([src, dst])
            sym_dst = np.concatenate([dst, src])
            direct = sp.csr_matrix(
                (np.ones(sym_src.shape[0], dtype=np.float32), (sym_src, sym_dst)),
                shape=(num_genes, num_genes),
            )
            direct.data[:] = 1.0
        else:
            direct = sp.csr_matrix((num_genes, num_genes), dtype=np.float32)

        result = (direct, common)
        if len(self._kg_pair_stats_cache) >= 4:
            self._kg_pair_stats_cache.clear()
        self._kg_pair_stats_cache[cache_key] = result
        return result

    def _pair_kg_features(self, kg_data, num_nodes, edge_index, kg_gene_map, device):
        """Per-pair KG connectivity: [direct edge 0/1, log1p(shared neighbors)/scale]."""
        edge_count = edge_index.size(1)
        features = torch.zeros(edge_count, 2, device=device)
        gene_type, sl_indices, kg_indices = kg_gene_map
        if gene_type is None or not sl_indices:
            return features

        direct, common = self._kg_pair_stats(kg_data, gene_type)
        scale = max(float(getattr(self.config, 'KG_PAIR_FEATURE_SCALE', 5.0)), 1e-6)

        sl_to_kg = np.full(int(num_nodes), -1, dtype=np.int64)
        sl_to_kg[np.asarray(sl_indices, dtype=np.int64)] = np.asarray(
            kg_indices, dtype=np.int64
        )
        edge_np = edge_index.detach().cpu().numpy()
        src_kg = sl_to_kg[edge_np[0]]
        dst_kg = sl_to_kg[edge_np[1]]
        valid = (src_kg >= 0) & (dst_kg >= 0)
        if not valid.any():
            return features

        src_valid = src_kg[valid]
        dst_valid = dst_kg[valid]
        direct_vals = np.asarray(direct[src_valid, dst_valid]).ravel()
        common_vals = np.asarray(common[src_valid, dst_valid]).ravel()
        pair_features = np.stack(
            [direct_vals, np.log1p(common_vals) / scale], axis=1
        ).astype(np.float32)
        valid_positions = torch.from_numpy(np.nonzero(valid)[0]).to(device)
        features[valid_positions] = torch.from_numpy(pair_features).to(device)
        return features

    def _cancer_context(self, sl_data, edge_count, device, cancer_index=None):
        if cancer_index is None:
            cancer_index = getattr(sl_data, 'cancer_index', None)
        if cancer_index is None or cancer_index.numel() != edge_count:
            cancer_index = torch.ones(edge_count, dtype=torch.long, device=device)
        cancer_index = cancer_index.to(device=device, dtype=torch.long)
        cancer_index = cancer_index.clamp(0, self.cancer_embedding.num_embeddings - 1)
        return cancer_index, self.cancer_embedding(cancer_index)

    def _pair_prediction(self, sl_data, sl_nodes, sl_degree, kg_nodes, kg_degree,
                         gene_ids, cancer_index=None,
                         simulated_new_gene_mask=None, pair_kg_features=None):
        device = sl_nodes.device
        edge_index = sl_data.edge_index.to(device)
        _, cancer_context = self._cancer_context(
            sl_data, edge_index.size(1), device, cancer_index
        )
        sl_score = self.sl_connectivity(sl_nodes).squeeze(-1)
        kg_score = torch.tanh(torch.log1p(kg_degree))
        sl_pair, sl_logit = self.sl_pair_expert(
            sl_nodes, edge_index, sl_score, cancer_context
        )
        kg_pair, kg_logit = self.kg_pair_expert(
            kg_nodes, edge_index, kg_score, cancer_context
        )

        new_gene_mask = torch.tensor(
            self.detect_new_genes(gene_ids), dtype=torch.float, device=device
        )
        if simulated_new_gene_mask is not None:
            simulated_new_gene_mask = simulated_new_gene_mask.to(
                device=device, dtype=torch.bool
            )
            if simulated_new_gene_mask.numel() != len(gene_ids):
                raise ValueError(
                    "simulated_new_gene_mask must contain one value per gene"
                )
            new_gene_mask = torch.maximum(
                new_gene_mask, simulated_new_gene_mask.float()
            )
        source_index, target_index = edge_index
        new_fraction = (
            new_gene_mask[source_index] + new_gene_mask[target_index]
        ).unsqueeze(1) * 0.5
        sl_coverage = (
            sl_degree[source_index].gt(0).float()
            + sl_degree[target_index].gt(0).float()
        ).unsqueeze(1) * 0.5
        kg_coverage = (
            kg_degree[source_index].gt(0).float()
            + kg_degree[target_index].gt(0).float()
        ).unsqueeze(1) * 0.5
        omics_present = sl_data.x.abs().sum(dim=1).gt(0).float()
        omics_coverage = (
            omics_present[source_index] + omics_present[target_index]
        ).unsqueeze(1) * 0.5
        reliability_parts = [
            new_fraction, sl_coverage, kg_coverage, omics_coverage
        ]
        if self.reliability_dim >= 10:
            if pair_kg_features is None:
                pair_kg_features = sl_nodes.new_zeros((edge_index.size(1), 2))
            scale = max(
                float(getattr(self.config, 'KG_PAIR_FEATURE_SCALE', 5.0)), 1e-6
            )
            kg_log_degree = torch.log1p(kg_degree)
            sl_log_degree = torch.log1p(sl_degree)
            reliability_parts.extend([
                pair_kg_features[:, 0].unsqueeze(1),
                pair_kg_features[:, 1].unsqueeze(1),
                ((kg_log_degree[source_index] + kg_log_degree[target_index])
                 * 0.5 / scale).unsqueeze(1),
                (torch.abs(kg_log_degree[source_index] - kg_log_degree[target_index])
                 / scale).unsqueeze(1),
                ((sl_log_degree[source_index] + sl_log_degree[target_index])
                 * 0.5 / scale).unsqueeze(1),
                (torch.abs(sl_log_degree[source_index] - sl_log_degree[target_index])
                 / scale).unsqueeze(1),
            ])
        reliability = torch.cat(reliability_parts, dim=1)
        gate_logits = self.expert_gate(torch.cat([
            sl_pair, kg_pair, cancer_context, reliability
        ], dim=1))
        gate_weights = F.softmax(gate_logits, dim=1)
        expert_logits = torch.stack([sl_logit, kg_logit], dim=1)
        final_logit = (gate_weights * expert_logits).sum(dim=1)
        return final_logit, sl_score, {
            'sl_expert_logit': sl_logit,
            'kg_expert_logit': kg_logit,
            'expert_logits': expert_logits,
            'gate_weights': gate_weights,
            'gate_logits': gate_logits,
            'reliability': reliability,
            'symmetry_loss': final_logit.new_zeros(()),
        }

    def forward(self, sl_data, kg_data, gene_mapping, mode='train', gene_ids=None,
                train_gene_ids=None, return_attention=False,
                refresh_train_memory=False, return_details=False,
                cancer_index=None, simulated_new_gene_mask=None):
        model_device = next(self.parameters()).device
        sl_data.x = sl_data.x.to(model_device)
        sl_data.edge_index = sl_data.edge_index.to(model_device)
        if hasattr(sl_data, 'y'):
            sl_data.y = sl_data.y.to(model_device)
        gene_ids = list(gene_ids or self._resolve_gene_ids(gene_mapping))
        if len(gene_ids) != sl_data.num_nodes:
            raise ValueError(
                f"gene_ids length ({len(gene_ids)}) must equal nodes ({sl_data.num_nodes})"
            )
        if train_gene_ids and (mode == 'train' or refresh_train_memory):
            self.train_gene_ids = list(train_gene_ids)
            self.train_gene_ids_set = set(self.train_gene_ids)

        context_edges = self._training_context_edges(sl_data, gene_ids, mode)
        if simulated_new_gene_mask is not None and context_edges.numel() > 0:
            simulated_new_gene_mask = simulated_new_gene_mask.to(
                device=model_device, dtype=torch.bool
            )
            if simulated_new_gene_mask.numel() != len(gene_ids):
                raise ValueError(
                    "simulated_new_gene_mask must contain one value per gene"
                )
            context_keep = ~(
                simulated_new_gene_mask[context_edges[0]]
                | simulated_new_gene_mask[context_edges[1]]
            )
            context_edges = context_edges[:, context_keep]
        sl_nodes, sl_degree = self.sl_omics_expert(sl_data.x, context_edges)
        kg_nodes, kg_degree, kg_gene_map = self._align_kg_embeddings(
            kg_data, gene_ids, model_device
        )
        fused_nodes = self.node_fusion(sl_nodes, kg_nodes, sl_data.x)

        if refresh_train_memory or (mode == 'train' and train_gene_ids):
            self.train_gene_embeddings = fused_nodes.detach()
        if mode == 'embedding':
            return sl_nodes, kg_nodes

        connectivity = self.sl_connectivity(sl_nodes).squeeze(-1)
        if mode == 'embedding_only':
            return fused_nodes, connectivity

        pair_kg_features = None
        if self.reliability_dim >= 10:
            pair_kg_features = self._pair_kg_features(
                kg_data, sl_data.num_nodes, sl_data.edge_index,
                kg_gene_map, model_device,
            )
        final_logit, connectivity, details = self._pair_prediction(
            sl_data, sl_nodes, sl_degree, kg_nodes, kg_degree,
            gene_ids, cancer_index=cancer_index,
            simulated_new_gene_mask=simulated_new_gene_mask,
            pair_kg_features=pair_kg_features,
        )
        result = (fused_nodes, final_logit, connectivity)
        if return_attention:
            attention = {
                'expert_gate': details['gate_weights'],
                'expert_reliability': details['reliability'],
            }
            return result, attention
        if return_details:
            return result + (details,)
        return result


class Inductive_Columbina_Model(Columbina_Model):
    """Strict-inductive alias for the shared 4.0-MoE architecture."""

    pass


# Backward-compatible names retained for utility imports.
HeteroGNN = HGTKnowledgeEncoder
InductiveGNN = SLomicsExpert
