import math
import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from src import utils
from src.diffusion import diffusion_utils
from src.models.layers import Xtoy, Etoy, masked_softmax


class XEyTransformerLayer(nn.Module):
    """Transformer that updates node, edge and global features."""
    def __init__(
        self,
        dx: int,
        de: int,
        dy: int,
        n_head: int,
        dim_ffX: int = 2048,
        dim_ffE: int = 128,
        dim_ffy: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        kw = {"device": device, "dtype": dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, X, E, y, node_mask):
        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

        X = self.normX1(X + self.dropoutX1(newX))
        E = self.normE1(E + self.dropoutE1(newE))
        y = self.norm_y1(y + self.dropout_y1(new_y))

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        X = self.normX2(X + self.dropoutX3(ff_outputX))

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        E = self.normE2(E + self.dropoutE3(ff_outputE))

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        y = self.norm_y2(y + self.dropout_y3(ff_output_y))

        return X, E, y


class NodeEdgeBlock(nn.Module):
    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        self.y_e_mul = Linear(dy, dx)
        self.y_e_add = Linear(dy, dx)

        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        self.y_y = Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)
        e_mask1 = x_mask.unsqueeze(2)
        e_mask2 = x_mask.unsqueeze(1)

        Q = self.q(X) * x_mask
        K = self.k(X) * x_mask
        diffusion_utils.assert_correctly_masked(Q, x_mask)

        Q = Q.reshape((bs, n, self.n_head, self.df))
        K = K.reshape((bs, n, self.n_head, self.df))
        Q = Q.unsqueeze(2)
        K = K.unsqueeze(1)

        Y = (Q * K) / math.sqrt(self.df)
        diffusion_utils.assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        E1 = self.e_mul(E) * e_mask1 * e_mask2
        E1 = E1.reshape((bs, n, n, self.n_head, self.df))
        E2 = self.e_add(E) * e_mask1 * e_mask2
        E2 = E2.reshape((bs, n, n, self.n_head, self.df))
        Y = Y * (E1 + 1) + E2

        newE = Y.flatten(start_dim=3)
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        newE = self.e_out(newE) * e_mask1 * e_mask2
        diffusion_utils.assert_correctly_masked(newE, e_mask1 * e_mask2)

        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)
        attn = masked_softmax(Y, softmax_mask, dim=2)

        V = self.v(X) * x_mask
        V = V.reshape((bs, n, self.n_head, self.df)).unsqueeze(1)

        weighted_V = (attn * V).sum(dim=2).flatten(start_dim=2)

        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        newX = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(newX, x_mask)

        y0 = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = self.y_out(y0 + x_y + e_y)

        return newX, newE, new_y


class GraphTransformer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dims: dict,
        hidden_mlp_dims: dict,
        hidden_dims: dict,
        output_dims: dict,
        act_fn_in: nn.ReLU(),
        act_fn_out: nn.ReLU(),
    ):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims["X"]
        self.out_dim_E = output_dims["E"]
        self.out_dim_y = output_dims["y"]
        
        # Save config params for dynamic adaptation
        self.hidden_mlp_dims_y = hidden_mlp_dims["y"]
        self.hidden_dims_dy = hidden_dims["dy"]
        self.act_fn_in = act_fn_in
        self.act_fn_out = act_fn_out

        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims["X"], hidden_mlp_dims["X"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["X"], hidden_dims["dx"]),
            act_fn_in,
        )

        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims["E"], hidden_mlp_dims["E"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["E"], hidden_dims["de"]),
            act_fn_in,
        )

        # Init standard (will be fixed in forward if needed)
        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims["y"], hidden_mlp_dims["y"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["y"], hidden_dims["dy"]),
            act_fn_in,
        )

        self.tf_layers = nn.ModuleList(
            [
                XEyTransformerLayer(
                    dx=hidden_dims["dx"],
                    de=hidden_dims["de"],
                    dy=hidden_dims["dy"],
                    n_head=hidden_dims["n_head"],
                    dim_ffX=hidden_dims["dim_ffX"],
                    dim_ffE=hidden_dims["dim_ffE"],
                )
                for _ in range(n_layers)
            ]
        )

        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims["dx"], hidden_mlp_dims["X"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["X"], output_dims["X"]),
        )

        self.mlp_out_E = nn.Sequential(
            nn.Linear(hidden_dims["de"], hidden_mlp_dims["E"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["E"], output_dims["E"]),
        )

        self.mlp_out_y = nn.Sequential(
            nn.Linear(hidden_dims["dy"], hidden_mlp_dims["y"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["y"], output_dims["y"]),
        )

    def forward(self, X, E, y, node_mask):
        bs, n = X.shape[0], X.shape[1]
        
        # ======================================================================
        # [MAGIC FIX] AUTO-ADAPT DIMENSIONS
        # Si la dimension réelle de y (ex: 139) ne matche pas le modèle (ex: 151)
        # On redéfinit la couche d'entrée ET de sortie à la volée.
        # ======================================================================
        real_y_dim = y.shape[-1]
        model_y_dim = self.mlp_in_y[0].in_features
        
        if real_y_dim != model_y_dim:
            print(f"DEBUG: [GraphTransformer] Mismatch detected! Input y: {real_y_dim}, Model expect: {model_y_dim}")
            print(f"DEBUG: [GraphTransformer] Auto-adapting layers to {real_y_dim}...")
            
            device = self.mlp_in_y[0].weight.device
            dtype = self.mlp_in_y[0].weight.dtype
            
            # 1. Fix Input MLP
            new_in_layer = nn.Linear(real_y_dim, self.hidden_mlp_dims_y).to(device=device, dtype=dtype)
            self.mlp_in_y[0] = new_in_layer
            
            # 2. Fix Output MLP (Must output same dim for residuals)
            # Last layer of mlp_out_y is a Linear
            last_idx = len(self.mlp_out_y) - 1
            new_out_layer = nn.Linear(self.hidden_mlp_dims_y, real_y_dim).to(device=device, dtype=dtype)
            self.mlp_out_y[last_idx] = new_out_layer
            
            # 3. Update self.out_dim_y for residual slicing
            self.out_dim_y = real_y_dim
            print("DEBUG: [GraphTransformer] Adaptation complete. Continuing forward pass.")
        # ======================================================================

        # Diagonal mask for edges
        diag_mask = torch.eye(n, device=X.device).bool()
        diag_mask = ~diag_mask
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        # Save original for residuals
        X_to_out = X[..., : self.out_dim_X]
        E_to_out = E[..., : self.out_dim_E]
        y_to_out = y[..., : self.out_dim_y]

        # Input MLPs
        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        new_X = self.mlp_in_X(X)
        new_y = self.mlp_in_y(y) # Now safe!

        after_in = utils.PlaceHolder(X=new_X, E=new_E, y=new_y).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y

        # Transformer layers
        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        # Output MLPs
        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)

        # Residual connections
        X = X + X_to_out
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out

        # Symmetrize edges
        E = 0.5 * (E + torch.transpose(E, 1, 2))

        return utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)