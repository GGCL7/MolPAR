"""Fixed-background encoder reused from the canonical drug inference implementation."""
import math
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import degree, subgraph

class FixedBackgroundEncoder:
    """Reproduce a literal batch of 255 fixed training molecules plus one query."""

    def __init__(self, model, dataset, background_indices, device):
        self.model, self.dataset, self.device = model, dataset, device
        self.background_indices = list(background_indices)
        self.background_graphs = [dataset[i][0] for i in self.background_indices]
        self.background_fps = [dataset[i][1] for i in self.background_indices]
        batch = Batch.from_data_list(self.background_graphs).to(device)
        with torch.no_grad():
            hidden = self._prepool(batch)
            self.background_raw = self._raw_views(hidden, batch.edge_index, batch.edge_attr).detach()

    def _prepool(self, batch):
        encoder = self.model.encoder.graph_encoder
        hidden = encoder.dropout(F.relu(encoder.bn1(encoder.conv1(batch.x, batch.edge_index, batch.edge_attr))))
        hidden = encoder.dropout(F.relu(encoder.bn2(encoder.conv2(hidden, batch.edge_index, batch.edge_attr))))
        hidden = encoder.dropout(F.relu(encoder.bn3(encoder.conv3(hidden, batch.edge_index, batch.edge_attr))))
        return hidden

    def _raw_views(self, hidden, edge_index, edge_attr):
        pool = self.model.encoder.graph_encoder.pool
        row, _ = edge_index
        structural = torch.log(degree(row, num_nodes=hidden.size(0), dtype=hidden.dtype) + 1.0)
        feature = pool.feat_mlp(hidden).squeeze(-1)
        diffused = pool._diffusion_step(hidden, edge_index, edge_attr)
        diffusion = (hidden * diffused).sum(-1) / ((hidden.norm(dim=-1) + 1e-6) * (diffused.norm(dim=-1) + 1e-6))
        return torch.stack([structural, feature, diffusion], dim=-1)

    @torch.no_grad()
    def encode_indices(self, dataset, indices, chunk_size=128):
        outputs = []
        graph_encoder = self.model.encoder.graph_encoder
        for start in range(0, len(indices), chunk_size):
            block = list(indices[start : start + chunk_size])
            items = [dataset[i] for i in block]
            graphs = [item[0] for item in items]
            fps = [item[1] for item in items]
            batch = Batch.from_data_list(graphs).to(self.device)
            hidden_all = self._prepool(batch)
            raw_all = self._raw_views(hidden_all, batch.edge_index, batch.edge_attr)
            pre, selected = [], []
            for offset, graph in enumerate(graphs):
                low, high = int(batch.ptr[offset]), int(batch.ptr[offset + 1])
                hidden, raw = hidden_all[low:high], raw_all[low:high]
                combined = torch.cat([self.background_raw, raw], dim=0)
                standardized = (raw - combined.mean(0)) / (combined.std(0) + 1e-6)
                score = torch.sigmoid(standardized @ graph_encoder.pool.view_weight)
                keep_n = max(1, int(math.ceil(graph_encoder.pool_ratio * hidden.size(0))))
                keep = torch.topk(score, keep_n, sorted=False).indices.sort().values
                edge_index, edge_attr = subgraph(
                    keep, graph.edge_index.to(self.device), graph.edge_attr.to(self.device),
                    relabel_nodes=True, num_nodes=hidden.size(0),
                )
                selected.append(Data(
                    x=graph_encoder._apply_score_gate(hidden[keep], score[keep]),
                    edge_index=edge_index, edge_attr=edge_attr,
                ))
                pre.append(hidden.mean(0))
            selected_batch = Batch.from_data_list(selected).to(self.device)
            post_nodes = graph_encoder.dropout(F.relu(graph_encoder.bn4(graph_encoder.conv4(
                selected_batch.x, selected_batch.edge_index, selected_batch.edge_attr
            ))))
            post = global_mean_pool(post_nodes, selected_batch.batch)
            graph_embedding = graph_encoder.jk_proj(torch.cat([torch.stack(pre), post], dim=-1))
            fingerprints = torch.stack(fps).to(self.device)
            outputs.append(self.model.encoder.fusion(graph_embedding, self.model.encoder.fp_encoder(fingerprints)).cpu())
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def literal_embedding(self, dataset, index):
        item = dataset[index]
        graphs = self.background_graphs + [item[0]]
        fps = torch.stack(self.background_fps + [item[1]]).to(self.device)
        batch = Batch.from_data_list(graphs).to(self.device)
        return self.model.encoder(batch, fps)[-1].cpu()
