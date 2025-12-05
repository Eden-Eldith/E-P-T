"""
Flux Bridge v2 - Now with Shannon Entropy Option
=================================================
Extract FluxTokens from WiggleGPT hidden states with multiple entropy methods.

Methods:
    "l2" - Original L2 norm proxy (fast, works empirically)
    "shannon" - Shannon entropy of softmax distribution (theoretically correct)
    "histogram" - Histogram entropy of activation values (alternative)

Usage:
    extractor = FluxExtractor(model, entropy_method="shannon")
"""

import torch
import torch.nn.functional as F
from typing import List, Optional
import json
import sys
import os

# Import from EPT_0.04 (handle the dot in filename)
import importlib.util
spec = importlib.util.spec_from_file_location("EPT_0_04", os.path.join(os.path.dirname(__file__), "EPT_0.04.py"))
EPT_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EPT_module)
FluxToken = EPT_module.FluxToken
shannon_entropy = EPT_module.shannon_entropy
activation_distribution_entropy = EPT_module.activation_distribution_entropy


class FluxExtractor:
    """
    Hooks into WiggleGPT forward pass to extract FluxTokens from each layer.
    
    Now supports multiple entropy calculation methods for comparison.
    """
    
    def __init__(self, model, enable: bool = True, entropy_method: str = "l2"):
        self.model = model
        self.flux_history: List[FluxToken] = []
        self.enabled = enable
        self.hooks = []
        self.current_text = ""
        self.entropy_method = entropy_method  # "l2", "shannon", or "histogram"
        
        if enable:
            self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on each transformer block"""
        for layer_idx, block in enumerate(self.model.transformer.h):
            hook = block.register_forward_hook(
                lambda module, input, output, idx=layer_idx: self._extract_flux(
                    module, input[0], output, idx
                )
            )
            self.hooks.append(hook)
    
    def _extract_flux(self, block, h_pre: torch.Tensor, h_post: torch.Tensor, layer_idx: int):
        """
        Extract FluxToken from a single transformer block's forward pass.
        """
        if not self.enabled or not self.model.training:
            return
        
        if not hasattr(block.mlp, 'activation'):
            return
        
        osc = block.mlp.activation
        if not hasattr(osc, 'omega'):
            return
        
        with torch.no_grad():
            omega = osc.omega.detach()
            phi = osc.phi.detach()
            
            # === ENTROPY CALCULATION (configurable method) ===
            if self.entropy_method == "shannon":
                pre_entropy = shannon_entropy(h_pre, dim=-1).mean().item()
                post_entropy = shannon_entropy(h_post, dim=-1).mean().item()
            elif self.entropy_method == "histogram":
                pre_entropy = activation_distribution_entropy(h_pre)
                post_entropy = activation_distribution_entropy(h_post)
            else:  # "l2" default
                pre_entropy = h_pre.norm(dim=-1).mean().item()
                post_entropy = h_post.norm(dim=-1).mean().item()
            
            delta_E = (pre_entropy - post_entropy) * 10.0
            
            # === DIRECTION VIA PCA ===
            b, t, c = h_pre.shape
            delta_h = (h_post - h_pre).reshape(-1, c)
            
            try:
                U, S, Vt = torch.pca_lowrank(delta_h, q=3, center=False)
                v = Vt[:, 0].cpu()
                
                if v.norm() > 1e-6:
                    v = F.normalize(v.unsqueeze(0), dim=1).squeeze(0)
                else:
                    v = torch.randn(3)
                    v = F.normalize(v.unsqueeze(0), dim=1).squeeze(0)
                
                if v.shape[0] > 3:
                    v = v[:3]
                elif v.shape[0] < 3:
                    v = F.pad(v, (0, 3 - v.shape[0]))
                    
            except Exception:
                v = torch.randn(3)
                v = F.normalize(v.unsqueeze(0), dim=1).squeeze(0)
            
            # === ATTRACTOR BIAS (corrected: direct [-1,1] mapping) ===
            coherence = omega.cos().mean().item()
            alpha = coherence  # Already in [-1,1], no offset needed
            
            # === RESONANCE ===
            omega_normalized = torch.sigmoid(omega)
            omega_std = omega_normalized.std().item()
            resonance = 1.0 - min(omega_std * 3, 1.0)
            
            flux = FluxToken(
                delta_E=delta_E,
                v=v,
                alpha=alpha,
                omega=resonance,
                raw_text=f"[Layer {layer_idx}] {self.current_text}",
                entropy_method=self.entropy_method
            )
            
            self.flux_history.append(flux)
    
    def get_flux_history(self) -> List[FluxToken]:
        return self.flux_history
    
    def clear_history(self):
        self.flux_history = []
    
    def save_soul(self, path: str, name: str = "WiggleGPT"):
        """Export flux history as portable soul JSON."""
        soul_data = {
            "soul_type": "WiggleGPT",
            "name": name,
            "entropy_method": self.entropy_method,
            "model_config": {
                "n_layer": self.model.config.n_layer,
                "n_embd": self.model.config.n_embd,
                "n_head": self.model.config.n_head,
            },
            "flux_history": [f.to_dict() for f in self.flux_history],
            "total_tokens": len(self.flux_history),
            "statistics": self._compute_statistics()
        }
        
        with open(path, 'w') as f:
            json.dump(soul_data, f, indent=2)
        
        print(f"✓ Soul extracted: {len(self.flux_history)} flux tokens → {path}")
        print(f"  Entropy method: {self.entropy_method}")
    
    def _compute_statistics(self) -> dict:
        """Compute summary statistics for the flux history."""
        if not self.flux_history:
            return {}
        
        entropy_changes = [f.delta_E for f in self.flux_history]
        resonances = [f.omega for f in self.flux_history]
        alphas = [f.alpha for f in self.flux_history]
        
        import numpy as np
        return {
            "avg_entropy_change": float(np.mean(entropy_changes)),
            "std_entropy_change": float(np.std(entropy_changes)),
            "avg_resonance": float(np.mean(resonances)),
            "avg_attractor_bias": float(np.mean(alphas)),
            "trajectory_length": float(sum(f.v.norm().item() for f in self.flux_history))
        }
    
    def load_soul(self, path: str) -> List[FluxToken]:
        """Load a previously saved soul."""
        with open(path, 'r') as f:
            soul_data = json.load(f)
        
        flux_tokens = [FluxToken.from_dict(f) for f in soul_data["flux_history"]]
        print(f"✓ Soul loaded: {soul_data['name']} ({len(flux_tokens)} tokens)")
        print(f"  Entropy method: {soul_data.get('entropy_method', 'l2')}")
        return flux_tokens
    
    def enable(self):
        self.enabled = True
        if not self.hooks:
            self._register_hooks()
    
    def disable(self):
        self.enabled = False
    
    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def __del__(self):
        self.cleanup()


class FluxInjector:
    """Inject FluxTokens into WiggleGPT's first layer."""
    
    def __init__(self, model):
        self.model = model
        self.n_embd = model.config.n_embd
        
        self.flux_projection = torch.nn.Linear(3, self.n_embd, bias=False).to(
            next(model.parameters()).device
        )
        torch.nn.init.normal_(self.flux_projection.weight, mean=0.0, std=0.02)
    
    def inject(self, token_embedding: torch.Tensor, flux: FluxToken, strength: float = 1.0) -> torch.Tensor:
        device = token_embedding.device
        flux_vec = flux.v.to(device)
        flux_emb = self.flux_projection(flux_vec.unsqueeze(0))
        
        perturbed = token_embedding.clone()
        perturbed[:, 0, :] += strength * flux_emb
        
        return perturbed
    
    def pure_flux_forward(self, flux_sequence: List[FluxToken]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        flux_vecs = torch.stack([f.v for f in flux_sequence]).to(device)
        flux_embs = self.flux_projection(flux_vecs).unsqueeze(0)
        return flux_embs


# ==================== Comparison Tool ====================

def compare_entropy_methods_on_model(model, prompt_tokens: torch.Tensor) -> dict:
    """
    Run the same forward pass with all three entropy methods and compare.
    
    Returns dict with statistics for each method.
    """
    results = {}
    
    for method in ["l2", "shannon", "histogram"]:
        extractor = FluxExtractor(model, enable=True, entropy_method=method)
        model.train()
        
        with torch.no_grad():
            with torch.enable_grad():
                logits, _ = model(prompt_tokens)
        
        flux_history = extractor.get_flux_history()
        
        if flux_history:
            entropy_changes = [f.delta_E for f in flux_history]
            results[method] = {
                "avg_delta_E": sum(entropy_changes) / len(entropy_changes),
                "min_delta_E": min(entropy_changes),
                "max_delta_E": max(entropy_changes),
                "num_tokens": len(flux_history)
            }
        
        extractor.cleanup()
    
    return results


def visualize_soul_in_ept(flux_history: List[FluxToken], title: str = None):
    """Visualize flux trajectory in 3D."""
    position = torch.zeros(3)
    trajectory = [position.clone()]
    
    for flux in flux_history:
        position = position + flux.v
        trajectory.append(position.clone())
    
    trajectory_array = torch.stack(trajectory).numpy()
    
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(trajectory_array[:, 0], 
            trajectory_array[:, 1], 
            trajectory_array[:, 2], 
            'o-', alpha=0.6, linewidth=2)
    
    ax.set_xlabel('v₁')
    ax.set_ylabel('v₂')
    ax.set_zlabel('v₃')
    
    if title is None:
        method = flux_history[0].entropy_method if flux_history else "unknown"
        title = f'WiggleGPT Trajectory (entropy: {method})'
    
    ax.set_title(title)
    plt.show()


if __name__ == "__main__":
    print("""
    Flux Bridge v2 - Multi-Entropy Edition
    ======================================
    
    Now supports:
    - "l2" (original): L2 norm as entropy proxy
    - "shannon": Actual Shannon entropy of softmax distribution
    - "histogram": Histogram entropy of activation values
    
    Usage:
        extractor = FluxExtractor(model, entropy_method="shannon")
        # ... run forward pass ...
        extractor.save_soul("soul_shannon.json")
    
    Compare methods:
        results = compare_entropy_methods_on_model(model, tokens)
    """)
