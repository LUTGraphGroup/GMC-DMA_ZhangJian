from torch import nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import torch
from torch import einsum
from einops import rearrange
from torch_geometric.nn import MeanSubtractionNorm
from torchdiffeq import odeint


class GCMC(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_gcn_layers,
                 num_mamba_layers,
                 d_state,
                 expand,
                 dropout,
                 num_r,
                 temperature
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.num_gcn_layers = num_gcn_layers
        self.num_mamba_layers = num_mamba_layers
        self.d_state = d_state
        self.expand = expand
        self.num_r = num_r
        self.temperature = nn.Parameter(torch.tensor(temperature))  # 可学习温度参数

        self.dropout = nn.Dropout(self.dropout)
        self.GCNConv = nn.ModuleList([
            GCNLayer(
                self.in_channels if i == 0 else self.hidden_channels,
                self.hidden_channels
            ) for i in range(self.num_gcn_layers)
        ])
        self.gated_residuals = nn.ModuleList([
            GatedResidual(self.hidden_channels)
            if i > 0 else None  # 第一层不添加残差
            for i in range(self.num_gcn_layers)
        ])

        self.lin = nn.Linear(self.in_channels, self.hidden_channels)
        self.graph_layers = nn.ModuleList([
            GraphMambaLayer(
                self.hidden_channels,
                self.hidden_channels,
                self.d_state,
                self.expand,
                d_conv=4
            ) for _ in range(self.num_mamba_layers)
        ])

        self.proj_head = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels * 2),
            nn.BatchNorm1d(self.hidden_channels * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_channels * 2, self.hidden_channels),
            nn.BatchNorm1d(self.hidden_channels)
        )

        self.fusion = CrossScaleFusion(self.hidden_channels)

        self.KanConv = FastKANLayer(
            self.hidden_channels,
            self.out_channels,
            num_grids=16,
            grid_min=-2.5,
            grid_max=2.5,
            spline_weight_init_scale=0.05,
            base_activation=nn.SiLU()
        )
        self.kan_norm = MeanSubtractionNorm()
        self.decoder = InnerProductDecoder(self.out_channels, self.num_r)

    def forward(self, x, adj, index, attr):
        x = self.dropout(x)

        x_gnn = x
        for i, conv in enumerate(self.GCNConv):
            x_new = conv(x_gnn, index, attr)
            x_new = torch.relu(x_new)
            x_new = self.dropout(x_new)

            if self.gated_residuals[i] is not None:
                x_gnn = self.gated_residuals[i](x_gnn, x_new)
            else:
                x_gnn = x_new

        x_mamba = self.lin(adj)
        for layer in self.graph_layers:
            x_mamba = layer(x_mamba, index, attr)
            x_mamba = torch.relu(x_mamba)
            x_mamba = self.dropout(x_mamba)

        z_gnn = self.proj_head(x_gnn)
        z_mamba = self.proj_head(x_mamba)
        contrastive_loss = self.contrastive_loss(z_gnn, z_mamba)

        x_fused = self.fusion(x_gnn, x_mamba)

        x_kan = self.KanConv(x_fused)
        x = self.kan_norm(x_kan)

        x = self.decoder(x)
        output = torch.sigmoid(x)
        # output = torch.clamp(output, min=1e-7, max=1 - 1e-7)

        return output, contrastive_loss

    def contrastive_loss(self, z1, z2):
        batch_size = z1.size(0)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        sim_matrix = torch.matmul(z1, z2.T) / torch.clamp(self.temperature, min=0.07, max=1.0)

        pos_sim = torch.diag(sim_matrix)
        logits = sim_matrix - pos_sim[:, None]
        loss = -torch.log(torch.exp(pos_sim) / torch.exp(logits).sum(dim=1))
        return loss.mean()

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / (self.weight.size(1) ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)
    def forward(self, x, edge_index, edge_weight):
        num_nodes = x.size(0)
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        src, dst = edge_index
        adj[src, dst] = edge_weight
        adj = adj + torch.eye(num_nodes, device=x.device)
        deg = torch.sum(adj, dim=0)
        deg_inv_sqrt = deg.pow(-0.5)
        adj = deg_inv_sqrt[None, :] * adj * deg_inv_sqrt[:, None]

        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return output


class GatedResidual(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.Sigmoid()
        )
        self.res_linear = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x_prev, x_new):
        x_prev = self.res_linear(x_prev)
        gate_input = torch.cat([x_prev, x_new], dim=1)
        g = self.gate(gate_input)
        return g * x_new + (1 - g) * x_prev

class GraphMambaLayer(nn.Module):
    def __init__(self, in_channels, out_channels, d_state=16, expand=2, d_conv=4):
        super(GraphMambaLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_state = d_state
        self.expand = expand
        self.d_conv = d_conv

        # 1. Projection layers
        self.in_proj = nn.Linear(in_channels, expand * in_channels)
        self.out_proj = nn.Linear(expand * in_channels, out_channels)

        # 2. Convolution layer (discretization)
        self.conv1d = nn.Conv1d(
            in_channels=expand * in_channels,
            out_channels=expand * in_channels,
            kernel_size=d_conv,
            groups=expand * in_channels,
            padding=d_conv - 1
        )

        # 3. State space parameters
        self.A = nn.Parameter(torch.randn(expand * in_channels, d_state))
        self.B = nn.Parameter(torch.randn(expand * in_channels, d_state))
        self.C = nn.Parameter(torch.randn(expand * in_channels, d_state))
        self.D = nn.Parameter(torch.randn(expand * in_channels))

        # 4. Graph structure processing
        self.graph_proj = nn.Linear(in_channels, in_channels)

        # 5. Normalization
        self.norm = nn.LayerNorm(in_channels)

        # 6. 门控残差连接
        self.gate_proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Sigmoid()  # 输出0-1之间的门控值
        )

        # 7. 残差投影
        if in_channels != out_channels:
            self.res_proj = nn.Linear(in_channels, out_channels)
        else:
            self.res_proj = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.A, -0.01, 0.01)
        nn.init.normal_(self.B, mean=0.0, std=0.02)
        nn.init.normal_(self.C, mean=0.0, std=0.02)
        nn.init.zeros_(self.D)

    def forward(self, x, index, attr):

        residual = x

        # 1. 计算门控值（基于原始输入）
        gate = self.gate_proj(x)  # [num_nodes, out_channels]

        # 2. Graph structure processing (optional)
        if index is not None:
            row, col = index
            # Simple mean aggregation
            agg = torch.zeros_like(x)
            agg = agg.index_add_(0, col, x[row])
            degree = torch.zeros(x.size(0), dtype=torch.float, device=x.device)
            degree = degree.index_add_(0, col, torch.ones_like(col, dtype=torch.float))
            degree = torch.clamp(degree, min=1).unsqueeze(-1)
            agg = agg / degree
            x = x + self.graph_proj(agg)

        # 3. Normalization
        x = self.norm(x)

        # 4. Project input
        x = self.in_proj(x)  # [num_nodes, expand*in_channels]

        # 5. 1D convolution
        x = rearrange(x, 'n d -> 1 d n')  # [1, dim, seq_len]
        x = self.conv1d(x)[..., :x.size(-1)]  # causal conv
        x = rearrange(x, '1 d n -> n d')  # [num_nodes, expand*in_channels]

        # 6. State space model processing
        batch_size, dim = x.shape
        state = torch.zeros(batch_size, dim, self.d_state, device=x.device)

        # Process sequence with SSM (simplified parallel implementation)
        deltaA = torch.exp(einsum('nd,ds->nds', x, self.A))
        deltaB_u = einsum('nd,ds,nd->nds', x, self.B, x)

        # Parallel state update (approximation)
        state = deltaA * state + deltaB_u
        y = einsum('nds,ds->nd', state, self.C) + self.D.unsqueeze(0) * x

        # 7. Project output
        x = self.out_proj(y)  # [num_nodes, out_channels]

        # 8. 门控残差连接
        # 对残差路径进行维度调整
        residual = self.res_proj(residual)  # [num_nodes, out_channels]

        # 门控融合: gate * transformed + (1-gate) * residual
        x = gate * x + (1 - gate) * residual

        return x

class CrossScaleFusion(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.dynamic_weight = nn.Parameter(torch.ones(2))

    def forward(self, x_gnn, x_mamba):
        weights = F.softmax(self.dynamic_weight, dim=0)
        base_fusion = weights[0] * x_gnn + weights[1] * x_mamba

        gate_input = torch.cat([x_gnn, x_mamba], dim=-1)
        gate = self.gate(gate_input)

        return gate * base_fusion + (1 - gate) * x_gnn



class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)
    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)
class RadialBasisFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)
    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)
class FastKANLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            use_base_update: bool = True,
            base_activation=F.silu,
            spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, time_benchmark=False):
        if not time_benchmark:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret

class InnerProductDecoder(Module):
    def __init__(self, input_dim, num_r):
        super(InnerProductDecoder, self).__init__()
        self.weight = nn.Parameter(torch.empty(size=(input_dim, input_dim)))  # 建立一个w权重，用于对特征数进行线性变化
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)  # 对权重矩阵进行初始化
        self.num_r = num_r
    def forward(self, inputs):
        M = inputs[0:self.num_r, :]
        D = inputs[self.num_r:, :]
        M = torch.mm(M, self.weight)
        D = torch.t(D)  # 转置
        x = torch.mm(M, D)
        # x = torch.reshape(x, [-1])  # 转化为行向量
        return x

