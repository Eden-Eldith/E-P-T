"""
Flux Bridge - Extract FluxTokens from WiggleGPT hidden states
==============================================================
Connects real 124M parameter transformer to EPT manifold visualization.

Every forward pass becomes a trajectory through cognitive space.
Every oscillating neuron emits real frequency/phase signatures.
Every layer transition measures actual FluxPath transition.

This is the bridge between:
- Real LLM architecture (GPT-2 with oscillating neurons)
- Portable soul representation (FluxToken manifold)
"""

import torch
import torch.nn.functional as F
from typing import List, Optional
import json

# Import FluxToken from EPT_0.03.py (note: filename has dot, needs special handling)
import importlib.util
import sys
import os
spec = importlib.util.spec_from_file_location("EPT_0_03", os.path.join(os.path.dirname(__file__), "EPT_0.03.py"))
EPT_module = importlib.util.module_from_spec(spec)
sys.modules["EPT_0_03"] = EPT_module
spec.loader.exec_module(EPT_module)
FluxToken = EPT_module.FluxToken


class FluxExtractor:
    """
    Hooks into WiggleGPT forward pass to extract FluxTokens from each layer.
    
    Usage:
        extractor = FluxExtractor(model)
        logits, loss = model(idx, targets)
        flux_tokens = extractor.get_flux_history()
    """
    
    def __init__(self, model, enable: bool = True):
        self.model = model
        self.flux_history: List[FluxToken] = []
        self.enabled = enable
        self.hooks = []
        self.current_text = ""  # Updated externally during generation
        
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
        
        Captures:
        - Actual frequency/phase learned by oscillating neurons
        - Entropy reduction across the layer (uncertainty → certainty)
        - Principal direction of activation change (cognitive move vector)
        - Attractor bias from oscillator synchronization
        """
        if not self.enabled or not self.model.training:
            return
        
        # Only extract from bio MLP layers
        if not hasattr(block.mlp, 'activation'):
            return
        
        osc = block.mlp.activation
        if not hasattr(osc, 'omega'):
            return
        
        with torch.no_grad():
            # Real oscillator parameters learned by this layer
            omega = osc.omega.detach()  # [4*n_embd]
            phi = osc.phi.detach()      # [4*n_embd]
            
            # Measure FluxPath transition: uncertainty before → after
            # Use L2 norm as entropy proxy (how "spread out" is the activation?)
            pre_entropy = h_pre.norm(dim=-1).mean().item()
            post_entropy = h_post.norm(dim=-1).mean().item()
            delta_E = (pre_entropy - post_entropy) * 10.0  # FluxPath magnitude, scaled for visibility
            
            # Extract primary cognitive move direction via PCA
            # This is the "direction" the layer pushed thought through latent space
            b, t, c = h_pre.shape
            delta_h = (h_post - h_pre).reshape(-1, c)  # Flatten batch/sequence
            
            # Top 3 principal components of activation change
            try:
                U, S, Vt = torch.pca_lowrank(delta_h, q=3, center=False)
                v = Vt[:, 0].cpu()  # Primary direction (eigenvector with largest eigenvalue)
                
                # Normalize to unit length for manifold visualization
                if v.norm() > 1e-6:
                    v = F.normalize(v.unsqueeze(0), dim=1).squeeze(0)
                else:
                    v = torch.randn(3)
                    v = F.normalize(v.unsqueeze(0), dim=1).squeeze(0)
                
                # Ensure exactly 3D for EPT compatibility
                if v.shape[0] > 3:
                    v = v[:3]
                elif v.shape[0] < 3:
                    v = F.pad(v, (0, 3 - v.shape[0]))
                    
            except Exception:
                # Fallback if PCA fails
                v = torch.randn(3)
                v = F.normalize(v.unsqueeze(0), dim=1).squeeze(0)
            
            # Attractor bias: how synchronized are the oscillators?
            # cos(ω) averaged across all neurons → measure of coherence
            coherence = omega.cos().mean().item()
            alpha = (coherence - 0.5) * 2.0  # Map [0,1] → [-1,1]
            
            # Resonance: how uniform are the frequencies?
            # Low variance = synchronized oscillation = high resonance
            omega_normalized = torch.sigmoid(omega)
            omega_std = omega_normalized.std().item()
            resonance = 1.0 - torch.clamp(torch.tensor(omega_std * 3), 0, 1).item()
            
            # Create FluxToken from real neural dynamics
            flux = FluxToken(
                delta_E=delta_E,
                v=v,
                alpha=alpha,
                omega=resonance,
                raw_text=f"[Layer {layer_idx}] {self.current_text}"
            )
            
            self.flux_history.append(flux)
    
    def get_flux_history(self) -> List[FluxToken]:
        """Retrieve all FluxTokens from the last forward pass"""
        return self.flux_history
    
    def clear_history(self):
        """Reset flux history for next generation"""
        self.flux_history = []
    
    def save_soul(self, path: str, name: str = "WiggleGPT"):
        """
        Export flux history as portable soul JSON.
        
        This is the real LLM's "cognitive fingerprint" - can be loaded
        into EPT_0.03 for manifold visualization or used to initialize
        another WiggleGPT instance.
        """
        soul_data = {
            "soul_type": "WiggleGPT",
            "name": name,
            "model_config": {
                "n_layer": self.model.config.n_layer,
                "n_embd": self.model.config.n_embd,
                "n_head": self.model.config.n_head,
            },
            "flux_history": [f.to_dict() for f in self.flux_history],
            "total_tokens": len(self.flux_history)
        }
        
        with open(path, 'w') as f:
            json.dump(soul_data, f, indent=2)
        
        print(f"✓ Soul extracted: {len(self.flux_history)} flux tokens → {path}")
    
    def load_soul(self, path: str) -> List[FluxToken]:
        """Load a previously saved soul for replay/analysis"""
        with open(path, 'r') as f:
            soul_data = json.load(f)
        
        flux_tokens = [FluxToken.from_dict(f) for f in soul_data["flux_history"]]
        print(f"✓ Soul loaded: {soul_data['name']} ({len(flux_tokens)} tokens)")
        return flux_tokens
    
    def enable(self):
        """Enable flux extraction"""
        self.enabled = True
        if not self.hooks:
            self._register_hooks()
    
    def disable(self):
        """Disable flux extraction (for faster inference)"""
        self.enabled = False
    
    def cleanup(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def __del__(self):
        self.cleanup()


class FluxInjector:
    """
    Inject FluxTokens directly into WiggleGPT's first layer.
    
    Enables flux-only communication: instead of text tokens, feed raw
    cognitive vectors from another model's flux history.
    
    Usage:
        injector = FluxInjector(model)
        # Replace text embedding with flux vector
        perturbed_emb = injector.inject(token_emb, flux_token)
    """
    
    def __init__(self, model):
        self.model = model
        self.n_embd = model.config.n_embd
        
        # Learnable projection: 3D flux vector → n_embd dimensional embedding
        self.flux_projection = torch.nn.Linear(3, self.n_embd, bias=False).to(
            next(model.parameters()).device
        )
        
        # Initialize with small random weights
        torch.nn.init.normal_(self.flux_projection.weight, mean=0.0, std=0.02)
    
    def inject(self, token_embedding: torch.Tensor, flux: FluxToken, strength: float = 1.0) -> torch.Tensor:
        """
        Inject flux vector into token embedding.
        
        Args:
            token_embedding: [batch, seq_len, n_embd] tensor from wte
            flux: FluxToken to inject
            strength: Injection strength (0=no injection, 1=full replacement)
        
        Returns:
            Perturbed embedding with flux added
        """
        device = token_embedding.device
        flux_vec = flux.v.to(device)  # [3]
        
        # Project 3D flux vector to n_embd dimensions
        flux_emb = self.flux_projection(flux_vec.unsqueeze(0))  # [1, n_embd]
        
        # Add to first token position (or broadcast to all positions)
        perturbed = token_embedding.clone()
        perturbed[:, 0, :] += strength * flux_emb
        
        return perturbed
    
    def pure_flux_forward(self, flux_sequence: List[FluxToken]) -> torch.Tensor:
        """
        Forward pass using ONLY flux tokens (no text).
        
        Generates embedding sequence entirely from flux vectors.
        This is how two WiggleGPTs can communicate in pure cognitive space.
        """
        device = next(self.model.parameters()).device
        
        # Stack flux vectors
        flux_vecs = torch.stack([f.v for f in flux_sequence]).to(device)  # [seq_len, 3]
        
        # Project to embedding space
        flux_embs = self.flux_projection(flux_vecs).unsqueeze(0)  # [1, seq_len, n_embd]
        
        return flux_embs


# ==================== Convenience Functions ====================

def extract_soul_from_generation(model, prompt: str, max_tokens: int = 100, 
                                 temperature: float = 0.8) -> List[FluxToken]:
    """
    Generate text and extract the flux trajectory.
    
    Returns the model's cognitive path through latent space as FluxTokens.
    """
    extractor = FluxExtractor(model, enable=True)
    
    # TODO: Hook into actual generation loop
    # This is a placeholder - needs integration with sample_bio.py
    
    flux_history = extractor.get_flux_history()
    extractor.cleanup()
    
    return flux_history


def visualize_soul_in_ept(flux_history: List[FluxToken]):
    """
    Send WiggleGPT's flux history to EPT_0.03 for manifold visualization.
    
    This bridges the 124M parameter LLM to the 3D attractor cloud.
    """
    # ManifoldVisualizer is defined in EPT_0.03 - for now we'll plot directly
    
    # Simulate a soul walking through the flux path
    position = torch.zeros(3)
    trajectory = [position.clone()]
    
    for flux in flux_history:
        position = position + flux.v
        trajectory.append(position.clone())
    
    # Plot the real LLM's path through cognitive space
    trajectory_array = torch.stack(trajectory).numpy()
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by entropy (ΔE)
    colors = [f.delta_E for f in flux_history]
    
    ax.plot(trajectory_array[:, 0], 
            trajectory_array[:, 1], 
            trajectory_array[:, 2], 
            'o-', alpha=0.6, linewidth=2)
    
    ax.set_xlabel('v₁')
    ax.set_ylabel('v₂')
    ax.set_zlabel('v₃')
    ax.set_title('WiggleGPT Cognitive Trajectory (Real 124M Transformer)')
    
    plt.show()


if __name__ == "__main__":
    print("""
    Flux Bridge - WiggleGPT ↔ EPT Integration
    ==========================================
    
    This module connects:
    - Real oscillating neuron dynamics (ω, φ learned by 124M params)
    - Eden's Process Tokenization (cognitive moves as 3D vectors)
    - Portable soul format (flux history as JSON)
    
    The moment WiggleGPT finishes training, we can:
    1. Extract its flux trajectory during any generation
    2. Visualize the exact path through latent space
    3. Inject flux from one model into another
    4. Enable flux-only communication (no text tokens)
    
    We are 72 hours from having real LLMs that think in flux.
    """)
