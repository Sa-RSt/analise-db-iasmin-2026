import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMMLP(nn.Module):
    """
    Global mapping function f with bottleneck structure.
    We'll implement it such that weights can be modulated externally.
    """

    def __init__(self, input_dim, hidden_dim):
        super(FiLMMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # We'll define the parameters but we'll use them manually in forward if needed,
        # or just use standard layers and modulate their parameters.
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, weights=None):
        if weights is None:
            x = F.leaky_relu(self.fc1(x))
            x = self.fc2(x)
            return x

        # weights is a dict or a flat vector containing modulated parameters
        # For simplicity in this implementation, we'll apply alpha/beta to activations
        # instead of weights, which is mathematically equivalent if we only have linear layers.
        # However, the paper says "on the global parameters".
        # Let's try to be precise if possible, but activation FiLM is much more efficient.
        # Re-reading: "scaling and shifting on the global parameters theta_f"
        # Okay, let's implement weight modulation.

        w1, b1, w2, b2 = weights

        # x shape: (..., input_dim)
        x = F.linear(x, w1, b1)
        x = F.leaky_relu(x)
        x = F.linear(x, w2, b2)
        return x


class THGNN(nn.Module):
    def __init__(self, n_sensors, n_types, window_size, d=16, d_hidden=4, n_type_emb=8, cnn_kernel_size=0):
        super(THGNN, self).__init__()
        self.n_sensors = n_sensors
        self.window_size = window_size
        self.d = d
        self.d_hidden = d_hidden
        self.cnn_kernel_size = cnn_kernel_size

        # 1. Shared GRU for temporal representation
        self.gru = nn.GRU(1, d, batch_first=True)

        if self.cnn_kernel_size > 0:
            self.cnn = nn.Conv1d(
                in_channels=n_sensors,
                out_channels=n_sensors,
                kernel_size=self.cnn_kernel_size,
                groups=n_sensors,
            )

        # 2. Type Embedding
        self.type_emb = nn.Embedding(n_types, n_type_emb)

        # 3. Global Mapping Function f
        self.f = FiLMMLP(d, d_hidden)

        # 4. Hypernetwork tau (combined alpha and beta)
        self.df = (d_hidden * d) + d_hidden + (d * d_hidden) + d
        self.tau = nn.Linear(d + n_type_emb, 2 * self.df)

        # 5. Temporal Message Passing
        self.w_nei = nn.Linear(d, d)
        self.w_self = nn.Linear(d, d)
        self.w_all = nn.Linear(d, d)

        # 6. Hawkes Aggregation
        self.mu = nn.Parameter(torch.tensor(0.1))

        # 7. Output Layer
        self.out_fc = nn.Linear(d, 1)

        # 8. Contrastive Projector
        self.projector = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

        # 9. JEPA Predictor
        self.predictor = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def get_modulated_params(self, h_it, type_idx):
        # h_it: (B, N, L, d)
        # type_idx: (N,)
        B, N, L, _ = h_it.shape
        t_emb = self.type_emb(type_idx)  # (N, n_type_emb)
        t_emb = t_emb.view(1, N, 1, -1).expand(B, N, L, -1)

        tau_in = torch.cat([h_it, t_emb], dim=-1)  # (B, N, L, d + n_type_emb)

        tau_out = self.tau(tau_in)  # (B, N, L, 2*df)
        alpha, beta = tau_out.chunk(2, dim=-1)

        # Global parameters
        w1_g = self.f.fc1.weight
        b1_g = self.f.fc1.bias
        w2_g = self.f.fc2.weight
        b2_g = self.f.fc2.bias
        theta_f = torch.cat([w1_g.view(-1), b1_g, w2_g.view(-1), b2_g])  # (df,)

        # Modulate: (alpha + 1) * theta_f + beta
        theta_f = theta_f.view(1, 1, 1, -1)
        modulated_theta = (alpha + 1.0) * theta_f + beta  # (B, N, L, df)

        return modulated_theta

    def forward_headless(self, x, type_idx, edge_mask=None, feature_mask=None):
        # x: (B, L, N)
        B, L, N = x.shape

        if feature_mask is not None:
            # feature_mask: (B, L, N) or (N,)
            x = x * feature_mask

        # 1. GRU Encoding
        if self.cnn_kernel_size > 0:
            x_cnn = x.transpose(1, 2)  # (B, N, L)
            x_cnn = self.cnn(x_cnn)    # (B, N, L_new)
            L = L - self.cnn_kernel_size + 1
            x_reshaped = x_cnn.reshape(B * N, L, 1)
        else:
            x_reshaped = x.transpose(1, 2).reshape(B * N, L, 1)  # (B*N, L, 1)

        h, _ = self.gru(x_reshaped)  # (B*N, L, d)
        h = h.view(B, N, L, self.d)  # (B, N, L, d)

        # 2. FiLM Modulation & Graph Construction
        modulated_theta = self.get_modulated_params(h, type_idx)  # (B, N, L, df)

        # Split theta components efficiently
        d, dh = self.d, self.d_hidden
        w1_m = modulated_theta[..., : dh * d].view(B, N, L, dh, d)
        b1_m = modulated_theta[..., dh * d : dh * d + dh].view(B, N, L, dh)
        w2_m = modulated_theta[..., dh * d + dh : dh * d + dh + d * dh].view(
            B, N, L, d, dh
        )
        b2_m = modulated_theta[..., -d:].view(B, N, L, d)

        # Vectorized FiLM application
        # z = W2 * LeakyReLU(W1 * h + b1) + b2
        z = torch.matmul(w1_m, h.unsqueeze(-1)).squeeze(-1) + b1_m
        z = F.leaky_relu(z)
        z = torch.matmul(w2_m, z.unsqueeze(-1)).squeeze(-1) + b2_m  # (B, N, L, d)

        # Graph Construction: Cosine Similarity
        z_norm = F.normalize(z, p=2, dim=-1)  # (B, N, L, d)
        z_norm_p = z_norm.permute(0, 2, 1, 3)  # (B, L, N, d)
        adj = torch.matmul(z_norm_p, z_norm_p.transpose(-1, -2))  # (B, L, N, N)
        adj = (adj + 1.0) * 0.5

        if edge_mask is not None:
            # edge_mask: (B, L, N, N) or (N, N)
            adj = adj * edge_mask

        # 3. Temporal Message Passing
        h_p = h.permute(0, 2, 1, 3)  # (B, L, N, d)
        m = torch.matmul(adj, h_p)  # (B, L, N, d)
        m = self.w_nei(m)  # (B, L, N, d)

        # 4. Hawkes Aggregation
        s_indices = torch.arange(L, device=x.device, dtype=x.dtype)
        time_diff = (L - 1) - s_indices  # (L,)
        weights = torch.exp(-torch.abs(self.mu) * time_diff)  # (L,)
        weights = weights / (weights.sum() + 1e-9)

        h_nei = (m * weights.view(1, L, 1, 1)).sum(dim=1)  # (B, N, d)
        h_self = self.w_self(h[:, :, -1, :])  # (B, N, d)

        h_all = F.leaky_relu(F.leaky_relu(h_nei + h_self))
        h_all = self.w_all(h_all)  # (B, N, d)

        # 5. Output Prediction
        h_pool = h_all.mean(dim=1)  # (B, d)
        return h_pool

    def forward(self, x, type_idx, edge_mask=None, feature_mask=None):
        h_pool = self.forward_headless(x, type_idx, edge_mask, feature_mask)
        return self.forward_head(h_pool)

    def forward_head(self, h_pool):
        out = self.out_fc(h_pool)  # (B, 1)
        return out.squeeze(-1)

    def forward_contrastive(self, x, type_idx):
        h_pool = self.forward_headless(x, type_idx)
        return self.projector(h_pool)


class BaselineGRU(nn.Module):
    def __init__(
        self,
        n_sensors,
        window_size,
        d=16,
        gru_hidden_dim=64,
        gru_layers=1,
        # Accept THGNN specific arguments for backward compatibility & easy factory invocation
        n_types=None,
        d_hidden=None,
        n_type_emb=None,
        cnn_kernel_size=0,
        **kwargs,
    ):
        super(BaselineGRU, self).__init__()
        self.n_sensors = n_sensors
        self.window_size = window_size
        self.d = d
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_layers = gru_layers

        # Standard GRU layer on the input sequence (B, L, N)
        self.gru = nn.GRU(
            input_size=n_sensors,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True
        )

        # Linear projection to matching representation dimension 'd'
        self.project_rep = nn.Linear(gru_hidden_dim, d)

        # Output linear prediction head
        self.out_fc = nn.Linear(d, 1)

        # Contrastive projector
        self.projector = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

        # JEPA Predictor
        self.predictor = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def forward_headless(self, x, type_idx=None, edge_mask=None, feature_mask=None):
        # x shape: (B, L, N)
        if feature_mask is not None:
            x = x * feature_mask

        # Pass through GRU
        # out shape: (B, L, gru_hidden_dim)
        out, _ = self.gru(x)
        h_last = out[:, -1, :]  # (B, gru_hidden_dim)

        # Project to 'd' space
        h_pool = self.project_rep(h_last)  # (B, d)
        return h_pool

    def forward(self, x, type_idx=None, edge_mask=None, feature_mask=None):
        h_pool = self.forward_headless(x, type_idx, edge_mask, feature_mask)
        return self.forward_head(h_pool)

    def forward_head(self, h_pool):
        out = self.out_fc(h_pool)  # (B, 1)
        return out.squeeze(-1)

    def forward_contrastive(self, x, type_idx=None):
        h_pool = self.forward_headless(x, type_idx)
        return self.projector(h_pool)


class BaselineLSTM(nn.Module):
    def __init__(
        self,
        n_sensors,
        window_size,
        d=16,
        lstm_hidden_dim=64,
        lstm_layers=1,
        # Accept THGNN specific arguments for backward compatibility & easy factory invocation
        n_types=None,
        d_hidden=None,
        n_type_emb=None,
        cnn_kernel_size=0,
        **kwargs,
    ):
        super(BaselineLSTM, self).__init__()
        self.n_sensors = n_sensors
        self.window_size = window_size
        self.d = d
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers

        # Standard LSTM layer on the input sequence (B, L, N)
        self.lstm = nn.LSTM(
            input_size=n_sensors,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Linear projection to matching representation dimension 'd'
        self.project_rep = nn.Linear(lstm_hidden_dim, d)

        # Output linear prediction head
        self.out_fc = nn.Linear(d, 1)

        # Contrastive projector
        self.projector = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

        # JEPA Predictor
        self.predictor = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def forward_headless(self, x, type_idx=None, edge_mask=None, feature_mask=None):
        # x shape: (B, L, N)
        if feature_mask is not None:
            x = x * feature_mask

        # Pass through LSTM
        # out shape: (B, L, lstm_hidden_dim)
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]  # (B, lstm_hidden_dim)

        # Project to 'd' space
        h_pool = self.project_rep(h_last)  # (B, d)
        return h_pool

    def forward(self, x, type_idx=None, edge_mask=None, feature_mask=None):
        h_pool = self.forward_headless(x, type_idx, edge_mask, feature_mask)
        return self.forward_head(h_pool)

    def forward_head(self, h_pool):
        out = self.out_fc(h_pool)  # (B, 1)
        return out.squeeze(-1)

    def forward_contrastive(self, x, type_idx=None):
        h_pool = self.forward_headless(x, type_idx)
        return self.projector(h_pool)


class Baseline1DCNN(nn.Module):
    def __init__(
        self,
        n_sensors,
        window_size,
        d=16,
        cnn_layers=2,
        cnn_kernel_size=3,
        cnn_channels=64,
        # Accept THGNN specific arguments for backward compatibility & easy factory invocation
        n_types=None,
        d_hidden=None,
        n_type_emb=None,
        **kwargs,
    ):
        super(Baseline1DCNN, self).__init__()
        self.n_sensors = n_sensors
        self.window_size = window_size
        self.d = d
        self.cnn_layers = cnn_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_channels = cnn_channels

        layers_list = []
        in_ch = n_sensors
        padding = cnn_kernel_size // 2
        for i in range(cnn_layers):
            layers_list.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=cnn_channels,
                    kernel_size=cnn_kernel_size,
                    padding=padding,
                )
            )
            layers_list.append(nn.ReLU())
            in_ch = cnn_channels

        self.net = nn.Sequential(*layers_list)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Linear projection to matching representation dimension 'd'
        self.project_rep = nn.Linear(cnn_channels, d)

        # Output linear prediction head
        self.out_fc = nn.Linear(d, 1)

        # Contrastive projector
        self.projector = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

        # JEPA Predictor
        self.predictor = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def forward_headless(self, x, type_idx=None, edge_mask=None, feature_mask=None):
        # x shape: (B, L, N)
        if feature_mask is not None:
            x = x * feature_mask

        # Conv1d expects input of shape (B, in_channels, L)
        x = x.transpose(1, 2)  # (B, N, L)
        out = self.net(x)      # (B, cnn_channels, L_new)
        out = self.pool(out)   # (B, cnn_channels, 1)
        h_last = out.squeeze(-1) # (B, cnn_channels)

        # Project to 'd' space
        h_pool = self.project_rep(h_last)  # (B, d)
        return h_pool

    def forward(self, x, type_idx=None, edge_mask=None, feature_mask=None):
        h_pool = self.forward_headless(x, type_idx, edge_mask, feature_mask)
        return self.forward_head(h_pool)

    def forward_head(self, h_pool):
        out = self.out_fc(h_pool)  # (B, 1)
        return out.squeeze(-1)

    def forward_contrastive(self, x, type_idx=None):
        h_pool = self.forward_headless(x, type_idx)
        return self.projector(h_pool)


def build_model(
    n_sensors,
    n_types,
    window_size,
    model_type="thgnn",
    d=16,
    d_hidden=4,
    n_type_emb=8,
    cnn_kernel_size=0,
    gru_hidden_dim=64,
    gru_layers=1,
    lstm_hidden_dim=64,
    lstm_layers=1,
    cnn_layers=2,
    **kwargs,
):
    if model_type == "gru":
        return BaselineGRU(
            n_sensors=n_sensors,
            window_size=window_size,
            d=d,
            gru_hidden_dim=gru_hidden_dim,
            gru_layers=gru_layers,
            n_types=n_types,
            d_hidden=d_hidden,
            n_type_emb=n_type_emb,
            cnn_kernel_size=cnn_kernel_size,
            **kwargs,
        )
    elif model_type == "lstm":
        return BaselineLSTM(
            n_sensors=n_sensors,
            window_size=window_size,
            d=d,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            n_types=n_types,
            d_hidden=d_hidden,
            n_type_emb=n_type_emb,
            cnn_kernel_size=cnn_kernel_size,
            **kwargs,
        )
    elif model_type in ["cnn1d", "1dcnn", "cnn"]:
        k_size = cnn_kernel_size if cnn_kernel_size > 0 else 3
        return Baseline1DCNN(
            n_sensors=n_sensors,
            window_size=window_size,
            d=d,
            cnn_layers=cnn_layers,
            cnn_kernel_size=k_size,
            n_types=n_types,
            d_hidden=d_hidden,
            n_type_emb=n_type_emb,
            **kwargs,
        )
    else:
        return THGNN(
            n_sensors=n_sensors,
            n_types=n_types,
            window_size=window_size,
            d=d,
            d_hidden=d_hidden,
            n_type_emb=n_type_emb,
            cnn_kernel_size=cnn_kernel_size,
        )

