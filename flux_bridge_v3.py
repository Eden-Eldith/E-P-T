# flux_bridge_v3.py — MONOLITHIC WiggleGPT Flux Extraction
# =========================================================
# Everything in ONE file. No external EPT imports. Just run it.
#
# Features (v0.05):
# - 8D representation (configurable n_components)
# - PCA sign stabilization
# - Multiple entropy methods (L2, Shannon, Histogram)
# - Full model loading and extraction
# - Visualization
# - Soul save/load
#
# Usage:
#   python flux_bridge_v3.py                    # Uses defaults
#   python flux_bridge_v3.py --checkpoint path/to/ckpt.pt
#   python flux_bridge_v3.py --method shannon --components 16

import torch
import torch.nn.functional as F
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import nullcontext
from dataclasses import dataclass, field
import argparse
import os
import sys

# ============================================================================
# ENTROPY FUNCTIONS
# ============================================================================

def shannon_entropy(x: torch.Tensor, dim: int = -1, eps: float = 1e-10) -> torch.Tensor:
    """Shannon entropy: H = -Σ p_i * log(p_i)"""
    p = F.softmax(x, dim=dim)
    log_p = torch.log(p + eps)
    return -torch.sum(p * log_p, dim=dim)


def histogram_entropy(x: torch.Tensor, n_bins: int = 50) -> float:
    """Entropy of activation value histogram"""
    x_flat = x.flatten().float()
    hist = torch.histc(x_flat, bins=n_bins)
    p = hist / hist.sum()
    p = p[p > 0]
    return (-torch.sum(p * torch.log(p))).item()


def compute_energy_delta(h_pre: torch.Tensor, h_post: torch.Tensor, 
                         method: str = "l2") -> float:
    """Compute energy/entropy change between activation states."""
    if method == "shannon":
        pre = shannon_entropy(h_pre, dim=-1).mean().item()
        post = shannon_entropy(h_post, dim=-1).mean().item()
    elif method == "histogram":
        pre = histogram_entropy(h_pre)
        post = histogram_entropy(h_post)
    else:  # "l2" default
        pre = h_pre.norm(dim=-1).mean().item()
        post = h_post.norm(dim=-1).mean().item()
    
    return (pre - post) * 10.0


# ============================================================================
# PCA SIGN STABILIZATION
# ============================================================================

class PCAStabilizer:
    """
    Maintains consistent sign orientation for PCA components across frames.
    Solves: eigenvectors v and -v are mathematically equivalent but cause trajectory flips.
    """
    
    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self.reference: Optional[torch.Tensor] = None
    
    def stabilize(self, components: torch.Tensor) -> torch.Tensor:
        """Align components to reference orientation."""
        if self.reference is None:
            self.reference = components.clone()
            return components
        
        stabilized = components.clone()
        n = min(self.n_components, components.shape[0], self.reference.shape[0])
        
        for i in range(n):
            if torch.dot(components[i], self.reference[i]) < 0:
                stabilized[i] = -components[i]
        
        return stabilized
    
    def reset(self):
        self.reference = None


# ============================================================================
# FLUXTOKEN (v0.05)
# ============================================================================

class FluxToken:
    """
    Represents a cognitive state change in the transformer.
    
    v0.05: Higher-dimensional representation with sign stabilization.
    """
    
    def __init__(self, delta_E: float, v_full: torch.Tensor, alpha: float, 
                 omega: float, raw_text: str = "", entropy_method: str = "l2",
                 layer_idx: int = -1, sign_stabilized: bool = False):
        self.delta_E = delta_E
        self.v_full = v_full.clone().detach()
        self.v = v_full[:3].clone().detach() if v_full.shape[0] >= 3 else F.pad(v_full, (0, 3 - v_full.shape[0]))
        self.alpha = alpha
        self.omega = omega
        self.raw_text = raw_text
        self.timestamp = datetime.now().isoformat()
        self.entropy_method = entropy_method
        self.layer_idx = layer_idx
        self.sign_stabilized = sign_stabilized
        self.n_components = v_full.shape[0]
    
    def to_dict(self) -> dict:
        return {
            "ΔE": float(self.delta_E),
            "v": self.v.tolist(),
            "v_full": self.v_full.tolist(),
            "α": float(self.alpha),
            "Ω": float(self.omega),
            "text": self.raw_text,
            "ts": self.timestamp,
            "entropy_method": self.entropy_method,
            "layer_idx": self.layer_idx,
            "sign_stabilized": self.sign_stabilized,
            "n_components": self.n_components
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'FluxToken':
        v_full = torch.tensor(d.get("v_full", d["v"]))
        return cls(
            delta_E=d["ΔE"],
            v_full=v_full,
            alpha=d["α"],
            omega=d["Ω"],
            raw_text=d.get("text", ""),
            entropy_method=d.get("entropy_method", "l2"),
            layer_idx=d.get("layer_idx", -1),
            sign_stabilized=d.get("sign_stabilized", False)
        )
    
    def __add__(self, other: 'FluxToken') -> 'FluxToken':
        max_dim = max(self.v_full.shape[0], other.v_full.shape[0])
        v1 = F.pad(self.v_full, (0, max_dim - self.v_full.shape[0]))
        v2 = F.pad(other.v_full, (0, max_dim - other.v_full.shape[0]))
        return FluxToken(
            delta_E=self.delta_E + other.delta_E,
            v_full=v1 + v2,
            alpha=(self.alpha + other.alpha) / 2,
            omega=(self.omega + other.omega) / 2,
            raw_text=f"[{self.raw_text}]+[{other.raw_text}]",
            entropy_method=self.entropy_method
        )
    
    def distance(self, other: 'FluxToken', use_full: bool = True) -> float:
        if use_full:
            max_dim = max(self.v_full.shape[0], other.v_full.shape[0])
            v1 = F.pad(self.v_full, (0, max_dim - self.v_full.shape[0]))
            v2 = F.pad(other.v_full, (0, max_dim - other.v_full.shape[0]))
            return (v1 - v2).norm().item()
        return (self.v - other.v).norm().item()


# ============================================================================
# FLUX EXTRACTOR (hooks into WiggleGPT)
# ============================================================================

class FluxExtractor:
    """
    Hooks into WiggleGPT forward pass to extract FluxTokens.
    
    v3: Uses 8D representation with sign stabilization.
    """
    
    def __init__(self, model, entropy_method: str = "l2", n_components: int = 8, enable: bool = True):
        self.model = model
        self.entropy_method = entropy_method
        self.n_components = n_components
        self.flux_history: List[FluxToken] = []
        self.hooks = []
        self.stabilizers: Dict[int, PCAStabilizer] = {}  # Per-layer stabilizers
        self.enabled = enable
        
        self._register_hooks()
    
    def get_flux_history(self) -> List[FluxToken]:
        """Return the list of extracted FluxTokens."""
        return self.flux_history
    
    def _register_hooks(self):
        """Register forward hooks on each transformer block."""
        for layer_idx, block in enumerate(self.model.transformer.h):
            self.stabilizers[layer_idx] = PCAStabilizer(self.n_components)
            hook = block.register_forward_hook(
                lambda module, inp, out, idx=layer_idx: self._extract_flux(module, inp[0], out, idx)
            )
            self.hooks.append(hook)
    
    def _extract_flux(self, block, h_pre: torch.Tensor, h_post: torch.Tensor, layer_idx: int):
        """Extract FluxToken from layer transition."""
        if not self.enabled:
            return
        
        # Get oscillation parameters if available
        omega_param = torch.zeros(1)
        if hasattr(block, 'mlp') and hasattr(block.mlp, 'activation'):
            osc = block.mlp.activation
            if hasattr(osc, 'omega'):
                omega_param = osc.omega.detach()
        
        with torch.no_grad():
            # FluxPath (cognitive state change magnitude)
            delta_E = compute_energy_delta(h_pre, h_post, self.entropy_method)
            
            # Direction via higher-D PCA
            b, t, c = h_pre.shape
            delta_h = (h_post - h_pre).reshape(-1, c)
            
            try:
                # Cast to float32 AND disable autocast for PCA
                # (bfloat16 doesn't support QR decomposition used internally)
                delta_h_f32 = delta_h.float()
                
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    n_pcs = min(self.n_components, c, delta_h_f32.shape[0])
                    U, S, Vt = torch.pca_lowrank(delta_h_f32, q=n_pcs, center=False)
                
                # Vt: (c, q) — columns are PCs
                components = Vt[:, :n_pcs].T  # (n_pcs, c)
                
                # Sign stabilization
                components = self.stabilizers[layer_idx].stabilize(components)
                
                # Build direction vector from singular value weights
                weights = S[:n_pcs] / (S[:n_pcs].sum() + 1e-8)
                v_full = weights.cpu()
                
                # Pad to n_components if needed
                if v_full.shape[0] < self.n_components:
                    v_full = F.pad(v_full, (0, self.n_components - v_full.shape[0]))
                
                # Normalize
                if v_full.norm() > 1e-6:
                    v_full = F.normalize(v_full.unsqueeze(0), dim=1).squeeze(0)
                    
            except Exception as e:
                print(f"  [!] PCA failed layer {layer_idx}: {e}")
                v_full = torch.randn(self.n_components)
                v_full = F.normalize(v_full.unsqueeze(0), dim=1).squeeze(0)
            
            # Attractor bias (from oscillation coherence)
            coherence = omega_param.cos().mean().item()
            alpha = coherence
            
            # Resonance (from oscillation diversity)
            omega_std = torch.sigmoid(omega_param).std().item()
            resonance = 1.0 - min(omega_std * 3, 1.0)
            
            flux = FluxToken(
                delta_E=delta_E,
                v_full=v_full,
                alpha=alpha,
                omega=resonance,
                raw_text=f"Layer {layer_idx}",
                entropy_method=self.entropy_method,
                layer_idx=layer_idx,
                sign_stabilized=True
            )
            
            self.flux_history.append(flux)
    
    def extract_from_text(self, text: str, tokenizer, min_length: int = 12, 
                          ctx = None) -> List[FluxToken]:
        """Run extraction on text input."""
        if ctx is None:
            ctx = nullcontext()
            
        self.flux_history = []
        self.enabled = True
        
        # Reset stabilizers for new sequence
        for stabilizer in self.stabilizers.values():
            stabilizer.reset()
        
        # Encode
        start_ids = tokenizer.encode(text)
        
        # Pad for RoPE minimum length
        if len(start_ids) < min_length:
            padding_token = tokenizer.encode(' ')[0]
            start_ids = start_ids + [padding_token] * (min_length - len(start_ids))
        
        device = next(self.model.parameters()).device
        tokens = torch.tensor([start_ids], dtype=torch.long, device=device)
        
        # Forward pass triggers hooks
        self.model.eval()
        with torch.no_grad():
            with ctx:
                self.enabled = True
                logits, _ = self.model(tokens)
        
        return self.flux_history
    
    def get_statistics(self) -> dict:
        """Compute summary statistics."""
        if not self.flux_history:
            return {}
        
        delta_Es = [f.delta_E for f in self.flux_history]
        alphas = [f.alpha for f in self.flux_history]
        omegas = [f.omega for f in self.flux_history]
        
        return {
            "n_tokens": len(self.flux_history),
            "n_components": self.n_components,
            "entropy_method": self.entropy_method,
            "delta_E": {
                "mean": float(np.mean(delta_Es)),
                "std": float(np.std(delta_Es)),
                "min": float(np.min(delta_Es)),
                "max": float(np.max(delta_Es))
            },
            "alpha": {"mean": float(np.mean(alphas)), "std": float(np.std(alphas))},
            "omega": {"mean": float(np.mean(omegas)), "std": float(np.std(omegas))},
            "trajectory_length_3d": float(sum(f.v.norm().item() for f in self.flux_history)),
            "trajectory_length_full": float(sum(f.v_full.norm().item() for f in self.flux_history))
        }
    
    def save_soul(self, path: str, name: str = "WiggleGPT"):
        """Save extracted soul to JSON."""
        soul = {
            "soul_type": "WiggleGPT",
            "name": name,
            "version": "0.05",
            "n_components": self.n_components,
            "entropy_method": self.entropy_method,
            "model_config": {
                "n_layer": self.model.config.n_layer,
                "n_embd": self.model.config.n_embd,
                "n_head": self.model.config.n_head
            },
            "statistics": self.get_statistics(),
            "flux_history": [f.to_dict() for f in self.flux_history]
        }
        
        with open(path, 'w') as f:
            json.dump(soul, f, indent=2)
        
        print(f"\n✓ Soul saved: {path}")
        print(f"  {len(self.flux_history)} tokens, {self.n_components}D, method={self.entropy_method}")
    
    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ============================================================================
# ENTROPY METHOD COMPARISON
# ============================================================================

def compare_entropy_methods(model, tokens: torch.Tensor, n_components: int = 8, 
                            ctx = None) -> dict:
    """Run same tokens through all entropy methods and compare."""
    if ctx is None:
        ctx = nullcontext()
    
    results = {}
    
    for method in ["l2", "shannon", "histogram"]:
        extractor = FluxExtractor(model, entropy_method=method, n_components=n_components)
        extractor.flux_history = []
        extractor.enabled = True
        
        # Reset stabilizers
        for s in extractor.stabilizers.values():
            s.reset()
        
        model.eval()
        try:
            with torch.no_grad():
                with ctx:
                    logits, _ = model(tokens)
        except Exception as e:
            print(f"  [!] {method} extraction failed: {e}")
            extractor.cleanup()
            continue
        
        if extractor.flux_history:
            delta_Es = [f.delta_E for f in extractor.flux_history]
            results[method] = {
                "mean": float(np.mean(delta_Es)),
                "std": float(np.std(delta_Es)),
                "min": float(np.min(delta_Es)),
                "max": float(np.max(delta_Es)),
                "n_tokens": len(delta_Es)
            }
        
        extractor.cleanup()
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_trajectory(flux_history: List[FluxToken], title: str = "WiggleGPT Trajectory"):
    """3D visualization of flux trajectory."""
    pos = torch.zeros(3)
    trajectory = [pos.numpy().copy()]
    
    for flux in flux_history:
        pos = pos + flux.v
        trajectory.append(pos.numpy().copy())
    
    trajectory = np.array(trajectory)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by position in sequence
    colors = np.linspace(0, 1, len(trajectory))
    
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
            'b-', alpha=0.3, linewidth=1)
    scatter = ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                        c=colors, cmap='viridis', s=30)
    
    # Mark start and end
    ax.scatter(*trajectory[0], color='green', s=200, marker='o', label='Start')
    ax.scatter(*trajectory[-1], color='red', s=200, marker='*', label='End')
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(title)
    ax.legend()
    
    plt.colorbar(scatter, label='Sequence Position')
    plt.tight_layout()
    plt.savefig(title.replace(" ", "_") + ".png", dpi=150)
    print(f"  Saved: {title.replace(' ', '_')}.png")
    plt.show()


def visualize_entropy_by_layer(flux_history: List[FluxToken], title: str = "FluxPath by Layer"):
    """Plot FluxPath magnitude across layers."""
    layers = [f.layer_idx for f in flux_history]
    delta_Es = [f.delta_E for f in flux_history]
    
    plt.figure(figsize=(12, 6))
    plt.bar(layers, delta_Es, color=['green' if d < 0 else 'red' for d in delta_Es], alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Layer')
    plt.ylabel('ΔE (Energy Change)')
    plt.title(title)
    plt.savefig(title.replace(" ", "_") + ".png", dpi=150)
    print(f"  Saved: {title.replace(' ', '_')}.png")
    plt.show()


# ============================================================================
# MODEL LOADING (matches sample_bio.py exactly)
# ============================================================================

def load_wigglegpt(checkpoint_path: str, device: str = "cuda"):
    """
    Load WiggleGPT from checkpoint.
    Matches sample_bio.py loading procedure.
    """
    # Try to find model_bio.py
    search_paths = [
        os.path.dirname(checkpoint_path),
        os.path.dirname(os.path.dirname(checkpoint_path)),
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__))
    ]
    
    model_bio_path = None
    for path in search_paths:
        candidate = os.path.join(path, "model_bio.py")
        if os.path.exists(candidate):
            model_bio_path = candidate
            break
    
    if model_bio_path is None:
        print("ERROR: Could not find model_bio.py")
        print("Searched:", search_paths)
        sys.exit(1)
    
    print(f"Loading model_bio from: {model_bio_path}")
    
    # Import model_bio
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_bio", model_bio_path)
    model_bio = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_bio)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create config from checkpoint's model_args (exactly like sample_bio.py)
    gpt_config = model_bio.GPTConfig(**checkpoint['model_args'])
    
    # Create model
    model = model_bio.GPT(gpt_config)
    
    # Load weights with _orig_mod. prefix handling
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    
    # Print model info
    print(f"✓ WiggleGPT loaded: {gpt_config.n_layer}L, {gpt_config.n_embd}D, {gpt_config.n_head}H")
    if hasattr(gpt_config, 'use_bio_mlp') and gpt_config.use_bio_mlp:
        print("  🧬 Bio-inspired oscillating neurons active")
    
    return model, model_bio, gpt_config


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="WiggleGPT Flux Extraction v3")
    parser.add_argument("--checkpoint", type=str, 
                       default="out-wigglegpt-pure-124m/ckpt.pt",
                       help="Path to WiggleGPT checkpoint")
    parser.add_argument("--method", type=str, default="l2",
                       choices=["l2", "shannon", "histogram"],
                       help="Entropy calculation method")
    parser.add_argument("--components", type=int, default=8,
                       help="Number of PCA components")
    parser.add_argument("--prompt", type=str, 
                       default="The nature of consciousness emerges from",
                       help="Text prompt for extraction")
    parser.add_argument("--output", type=str, default="wigglegpt_soul_v05.json",
                       help="Output soul file")
    parser.add_argument("--compare", action="store_true",
                       help="Compare all entropy methods")
    parser.add_argument("--no-viz", action="store_true",
                       help="Skip visualization")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  FLUX BRIDGE v3 — Monolithic WiggleGPT Extraction")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Method: {args.method}")
    print(f"  Components: {args.components}")
    print(f"  Device: {args.device}")
    print("=" * 70)
    print()
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Setup dtype and autocast (like sample_bio.py)
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Load model
    model, model_bio, gpt_config = load_wigglegpt(args.checkpoint, args.device)
    
    # Setup tokenizer (tiktoken for GPT-2)
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    
    # Encode prompt
    start_ids = enc.encode(args.prompt)
    
    # CRITICAL: Pad to at least n_head tokens for RoPE to work!
    min_length = gpt_config.n_head if hasattr(gpt_config, 'n_head') else 12
    if len(start_ids) < min_length:
        padding_token = enc.encode(' ')[0]
        original_len = len(start_ids)
        start_ids = start_ids + [padding_token] * (min_length - len(start_ids))
        print(f"Padded prompt from {original_len} to {len(start_ids)} tokens (RoPE minimum: {min_length})")
    
    tokens = torch.tensor([start_ids], dtype=torch.long, device=args.device)
    print(f"\nPrompt: \"{args.prompt}\"")
    print(f"Tokens: {tokens.shape[1]}")
    
    # Compare methods if requested
    if args.compare:
        print("\n=== Entropy Method Comparison ===")
        comparison = compare_entropy_methods(model, tokens, args.components, ctx)
        for method, stats in comparison.items():
            print(f"\n{method.upper()}:")
            print(f"  Mean ΔE: {stats['mean']:+.3f}")
            print(f"  Std ΔE:  {stats['std']:.3f}")
            print(f"  Range:   [{stats['min']:+.3f}, {stats['max']:+.3f}]")
    
    # Main extraction
    print(f"\n=== Extracting with {args.method.upper()} method ===")
    extractor = FluxExtractor(model, entropy_method=args.method, n_components=args.components)
    
    # Reset and extract
    extractor.flux_history = []
    for s in extractor.stabilizers.values():
        s.reset()
    extractor.enabled = True
    
    with torch.no_grad():
        with ctx:
            logits, _ = model(tokens)
    
    # Statistics
    stats = extractor.get_statistics()
    print(f"\nExtracted {stats['n_tokens']} FluxTokens")
    print(f"  ΔE mean: {stats['delta_E']['mean']:+.3f} (std: {stats['delta_E']['std']:.3f})")
    print(f"  ΔE range: [{stats['delta_E']['min']:+.3f}, {stats['delta_E']['max']:+.3f}]")
    print(f"  α mean: {stats['alpha']['mean']:.3f}")
    print(f"  Ω mean: {stats['omega']['mean']:.3f}")
    print(f"  Trajectory length (3D): {stats['trajectory_length_3d']:.3f}")
    print(f"  Trajectory length ({args.components}D): {stats['trajectory_length_full']:.3f}")
    
    # Save soul
    extractor.save_soul(args.output, name="WiggleGPT-124M")
    
    # Visualization
    if not args.no_viz:
        print("\n=== Visualization ===")
        visualize_trajectory(extractor.flux_history, 
                            f"WiggleGPT Trajectory ({args.method}, {args.components}D)")
        visualize_entropy_by_layer(extractor.flux_history,
                                   f"WiggleGPT Energy by Layer ({args.method})")
    
    # Cleanup
    extractor.cleanup()
    
    print("\n" + "=" * 70)
    print("  EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()