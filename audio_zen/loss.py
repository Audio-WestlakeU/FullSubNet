import torch

l1_loss = torch.nn.L1Loss
mse_loss = torch.nn.MSELoss


def si_snr_loss():
    def si_snr(x, s, eps=1e-8):
        """

        Args:
            x: Enhanced fo shape [B, T]
            s: Reference of shape [B, T]
            eps:

        Returns:
            si_snr: [B]
        """
        def l2norm(mat, keep_dim=False):
            return torch.norm(mat, dim=-1, keepdim=keep_dim)

        if x.shape != s.shape:
            raise RuntimeError(f"Dimension mismatch when calculate si_snr, {x.shape} vs {s.shape}")

        x_zm = x - torch.mean(x, dim=-1, keepdim=True)
        s_zm = s - torch.mean(s, dim=-1, keepdim=True)

        t = torch.sum(x_zm * s_zm, dim=-1, keepdim=True) * s_zm / (l2norm(s_zm, keep_dim=True) ** 2 + eps)

        return -torch.mean(20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps)))

    return si_snr
