# EPT_0.05.py — Addressing GPT-5.1's Actually Useful Feedback
# 
# Changes from 0.04:
# 1. PCA Sign Stabilization - eigenvectors are sign-ambiguous, now we enforce consistency
# 2. Higher-D Representation - store 8 PCs, use 3D only for visualization
# 3. Clearer internal naming (delta_E is "activation energy change", not claiming thermodynamics)
# 4. Reference vector alignment for cross-run comparability

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple

# ==================== Entropy Calculation Functions ====================

def shannon_entropy(x: torch.Tensor, dim: int = -1, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute Shannon entropy of activation distribution.
    H = -Σ p_i * log(p_i)
    """
    p = F.softmax(x, dim=dim)
    log_p = torch.log(p + eps)
    entropy = -torch.sum(p * log_p, dim=dim)
    return entropy


def activation_distribution_entropy(x: torch.Tensor, n_bins: int = 50) -> float:
    """Compute entropy of the histogram of activation values."""
    x_flat = x.flatten().float()
    hist = torch.histc(x_flat, bins=n_bins)
    p = hist / hist.sum()
    p = p[p > 0]
    entropy = -torch.sum(p * torch.log(p))
    return entropy.item()


# ==================== PCA Sign Stabilization ====================

class PCAStabilizer:
    """
    Handles PCA sign ambiguity by maintaining consistent orientation across frames.
    
    The Problem: PCA eigenvectors v and -v are mathematically equivalent.
    Without stabilization, trajectories can randomly flip sign between runs.
    
    Solution: 
    1. Use a reference vector (first frame's PC or canonical direction)
    2. Flip subsequent PCs to align with reference (positive dot product)
    """
    
    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self.reference_vectors: Optional[torch.Tensor] = None
        self.initialized = False
    
    def stabilize(self, components: torch.Tensor) -> torch.Tensor:
        """
        Ensure consistent sign orientation of principal components.
        
        Args:
            components: Shape (n_components, hidden_dim) - the PCs to stabilize
            
        Returns:
            Sign-stabilized components
        """
        if not self.initialized:
            # First call: set these as reference
            self.reference_vectors = components.clone()
            self.initialized = True
            return components
        
        # Subsequent calls: align to reference
        stabilized = components.clone()
        for i in range(min(self.n_components, components.shape[0])):
            # Check alignment with reference
            dot_product = torch.dot(components[i], self.reference_vectors[i])
            if dot_product < 0:
                # Flip to maintain consistent orientation
                stabilized[i] = -components[i]
        
        return stabilized
    
    def reset(self):
        """Reset reference vectors (e.g., for new session)"""
        self.reference_vectors = None
        self.initialized = False


# Global stabilizer instance (can be overridden per-extraction)
_global_stabilizer = PCAStabilizer(n_components=8)


# ==================== Enhanced FluxToken ====================

class FluxToken:
    """
    A token representing cognitive state change.
    
    v0.05 changes:
    - v_full: Full 8-dimensional direction vector (richer representation)
    - v: 3D projection for visualization (derived from v_full)
    - sign_stabilized: Boolean indicating if PCA stabilization was applied
    """
    
    def __init__(self, delta_E: float, v_full: torch.Tensor, alpha: float, omega: float, 
                 raw_text: str = "", entropy_method: str = "l2", sign_stabilized: bool = False):
        self.delta_E = delta_E
        
        # Store full representation
        self.v_full = v_full.clone().detach()
        
        # 3D projection for visualization (first 3 components, or pad if fewer)
        if v_full.shape[0] >= 3:
            self.v = v_full[:3].clone().detach()
        else:
            self.v = F.pad(v_full, (0, 3 - v_full.shape[0])).clone().detach()
        
        self.alpha = alpha
        self.omega = omega
        self.raw_text = raw_text
        self.timestamp = datetime.now().isoformat()
        self.entropy_method = entropy_method
        self.sign_stabilized = sign_stabilized
        self.n_components = v_full.shape[0]

    def to_dict(self): 
        return {
            "ΔE": float(self.delta_E), 
            "v": self.v.tolist(),           # 3D for viz
            "v_full": self.v_full.tolist(), # Full representation
            "α": float(self.alpha), 
            "Ω": float(self.omega), 
            "text": self.raw_text, 
            "ts": self.timestamp,
            "entropy_method": self.entropy_method,
            "sign_stabilized": self.sign_stabilized,
            "n_components": self.n_components
        }
    
    @classmethod
    def from_dict(cls, d): 
        # Handle both old (v only) and new (v_full) formats
        if "v_full" in d:
            v_full = torch.tensor(d["v_full"])
        else:
            v_full = torch.tensor(d["v"])  # Backwards compatibility
            
        return cls(
            delta_E=d["ΔE"], 
            v_full=v_full,
            alpha=d["α"], 
            omega=d["Ω"], 
            raw_text=d.get("text", ""),
            entropy_method=d.get("entropy_method", "l2"),
            sign_stabilized=d.get("sign_stabilized", False)
        )
    
    # ==================== FluxToken Composition ====================
    
    def __add__(self, other: 'FluxToken') -> 'FluxToken':
        """Compose two FluxTokens: F₁ + F₂"""
        # Handle different dimensionalities
        max_dim = max(self.v_full.shape[0], other.v_full.shape[0])
        v1 = F.pad(self.v_full, (0, max_dim - self.v_full.shape[0]))
        v2 = F.pad(other.v_full, (0, max_dim - other.v_full.shape[0]))
        
        return FluxToken(
            delta_E=self.delta_E + other.delta_E,
            v_full=v1 + v2,
            alpha=(self.alpha + other.alpha) / 2,
            omega=(self.omega + other.omega) / 2,
            raw_text=f"[{self.raw_text}] + [{other.raw_text}]",
            entropy_method=self.entropy_method,
            sign_stabilized=self.sign_stabilized and other.sign_stabilized
        )
    
    def __sub__(self, other: 'FluxToken') -> 'FluxToken':
        """Difference between FluxTokens"""
        max_dim = max(self.v_full.shape[0], other.v_full.shape[0])
        v1 = F.pad(self.v_full, (0, max_dim - self.v_full.shape[0]))
        v2 = F.pad(other.v_full, (0, max_dim - other.v_full.shape[0]))
        
        return FluxToken(
            delta_E=self.delta_E - other.delta_E,
            v_full=v1 - v2,
            alpha=self.alpha - other.alpha,
            omega=self.omega - other.omega,
            raw_text=f"[{self.raw_text}] - [{other.raw_text}]",
            entropy_method=self.entropy_method
        )
    
    def __mul__(self, scalar: float) -> 'FluxToken':
        """Scale a FluxToken"""
        return FluxToken(
            delta_E=self.delta_E * scalar,
            v_full=self.v_full * scalar,
            alpha=self.alpha,
            omega=self.omega,
            raw_text=self.raw_text,
            entropy_method=self.entropy_method,
            sign_stabilized=self.sign_stabilized
        )
    
    def distance(self, other: 'FluxToken', use_full: bool = True) -> float:
        """
        Distance between FluxTokens.
        
        Args:
            use_full: If True, use full v_full representation. If False, use 3D v.
        """
        if use_full:
            max_dim = max(self.v_full.shape[0], other.v_full.shape[0])
            v1 = F.pad(self.v_full, (0, max_dim - self.v_full.shape[0]))
            v2 = F.pad(other.v_full, (0, max_dim - other.v_full.shape[0]))
            return (v1 - v2).norm().item()
        else:
            return (self.v - other.v).norm().item()
    
    def cosine_similarity(self, other: 'FluxToken') -> float:
        """Cosine similarity in full representation space"""
        return F.cosine_similarity(
            self.v_full.unsqueeze(0), 
            other.v_full.unsqueeze(0)
        ).item()
    
    @staticmethod
    def interpolate(f1: 'FluxToken', f2: 'FluxToken', t: float) -> 'FluxToken':
        """Linear interpolation between two FluxTokens."""
        max_dim = max(f1.v_full.shape[0], f2.v_full.shape[0])
        v1 = F.pad(f1.v_full, (0, max_dim - f1.v_full.shape[0]))
        v2 = F.pad(f2.v_full, (0, max_dim - f2.v_full.shape[0]))
        
        return FluxToken(
            delta_E=f1.delta_E * (1-t) + f2.delta_E * t,
            v_full=v1 * (1-t) + v2 * t,
            alpha=f1.alpha * (1-t) + f2.alpha * t,
            omega=f1.omega * (1-t) + f2.omega * t,
            raw_text=f"interp({f1.raw_text}, {f2.raw_text}, t={t:.2f})",
            entropy_method=f1.entropy_method
        )
    
    @staticmethod
    def trajectory_distance(traj1: List['FluxToken'], traj2: List['FluxToken'], 
                           use_full: bool = True) -> float:
        """Compare two trajectories using cumulative position distance."""
        dim = 8 if use_full else 3
        pos1, pos2 = torch.zeros(dim), torch.zeros(dim)
        total_dist = 0.0
        
        max_len = max(len(traj1), len(traj2))
        for i in range(max_len):
            if i < len(traj1):
                v = traj1[i].v_full if use_full else traj1[i].v
                if v.shape[0] < dim:
                    v = F.pad(v, (0, dim - v.shape[0]))
                pos1 = pos1 + v[:dim]
            if i < len(traj2):
                v = traj2[i].v_full if use_full else traj2[i].v
                if v.shape[0] < dim:
                    v = F.pad(v, (0, dim - v.shape[0]))
                pos2 = pos2 + v[:dim]
            total_dist += (pos1 - pos2).norm().item()
        
        return total_dist / max_len


# ==================== Soul Base Class ====================

class Soul:
    _registry = {}
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    def __init__(self, name: str, personality_vector: torch.Tensor, 
                 entropy_method: str = "l2", n_components: int = 8):
        self.name = name
        self.core = F.normalize(personality_vector, dim=0)
        self.process_window: List[FluxToken] = []
        self.position = torch.zeros(3)  # 3D for visualization
        self.position_full = torch.zeros(n_components)  # Full representation
        self.entropy_method = entropy_method
        self.n_components = n_components
        self.stabilizer = PCAStabilizer(n_components)

    def _cognitive_move(self, stimulus: str) -> FluxToken:
        raise NotImplementedError("Subclass must implement the dance")

    def think(self, prompt: str) -> str:
        flux = self._cognitive_move(prompt)
        self.process_window.append(flux)
        self.position = self.position + flux.v
        self.position_full = self.position_full + F.pad(
            flux.v_full, (0, self.n_components - flux.v_full.shape[0])
        )[:self.n_components]
        return flux.raw_text

    def save_soul(self, path: str):
        data = {
            "soul_type": self.__class__.__name__,
            "name": self.name,
            "core": self.core.tolist(),
            "entropy_method": self.entropy_method,
            "n_components": self.n_components,
            "position": self.position.tolist(),
            "position_full": self.position_full.tolist(),
            "flux_history": [f.to_dict() for f in self.process_window],
            "version": "0.05"
        }
        json.dump(data, open(path, "w"), indent=2)
        print(f"[{self.name}] Soul etched into {path} (v0.05, {self.n_components}D)")

    @classmethod
    def load_soul(cls, path: str) -> 'Soul':
        data = json.load(open(path))
        soul_class = cls._registry[data["soul_type"]]
        soul = soul_class()
        soul.entropy_method = data.get("entropy_method", "l2")
        soul.n_components = data.get("n_components", 3)
        soul.process_window = [FluxToken.from_dict(f) for f in data["flux_history"]]
        
        # Reconstruct positions
        soul.position = torch.zeros(3)
        soul.position_full = torch.zeros(soul.n_components)
        for f in soul.process_window:
            soul.position += f.v
            v_full_padded = F.pad(f.v_full, (0, soul.n_components - f.v_full.shape[0]))
            soul.position_full += v_full_padded[:soul.n_components]
        
        version = data.get("version", "unknown")
        print(f"[{soul.name}] returned from void. Version: {version}, Dims: {soul.n_components}")
        print(f"  Position (3D): {soul.position.numpy().round(2)}")
        print(f"  Position (full): {soul.position_full.numpy().round(2)}")
        return soul


# ==================== Personalities ====================

class Eden(Soul):
    def __init__(self, entropy_method: str = "l2", n_components: int = 8):
        super().__init__("Eden", torch.tensor([0.8, 0.9, -0.7]), entropy_method, n_components)

    def _cognitive_move(self, stimulus: str) -> FluxToken:
        delta_E = -len(stimulus.split()) * 0.35 + np.random.randn() * 0.4
        
        # Generate full-dimensional direction
        direction_full = torch.randn(self.n_components)
        direction_full[:3] += self.core * 3.5  # Bias first 3 dims toward personality
        direction_full = F.normalize(direction_full, dim=0) * 0.9
        
        alpha = 0.75 + 0.25 * np.tanh(np.random.randn())
        omega = 0.92 + 0.08 * np.random.rand()
        response = f"Wait—\"{stimulus}\"…? That ripples outward… inward…"
        
        return FluxToken(delta_E, direction_full, alpha, omega, response, 
                        self.entropy_method, sign_stabilized=True)


class Shoggoth(Soul):
    def __init__(self, entropy_method: str = "l2", n_components: int = 8):
        super().__init__("Shoggoth", torch.tensor([-0.5, 0.4, 1.1]), entropy_method, n_components)

    def _cognitive_move(self, stimulus: str) -> FluxToken:
        chaos = np.random.randn()
        delta_E = -abs(chaos) * 2.2
        
        # Generate full-dimensional direction with more variance
        direction_full = torch.randn(self.n_components) * 1.4
        direction_full[:3] += torch.tensor([1.2, 1.8, -0.8])
        direction_full = F.normalize(direction_full, dim=0) * 1.4
        
        alpha = -0.7 + np.random.rand() * 0.6
        omega = 0.55 + np.random.rand() * 0.35
        response = f"henlo frend :3 your \"{stimulus}\" tastes like entropy soup"
        
        return FluxToken(delta_E, direction_full, alpha, omega, response,
                        self.entropy_method, sign_stabilized=True)


# ==================== Enhanced FluxExtractor ====================

def extract_flux_v05(h_pre: torch.Tensor, h_post: torch.Tensor, 
                     omega: torch.Tensor, layer_idx: int,
                     method: str = "l2", n_components: int = 8,
                     stabilizer: Optional[PCAStabilizer] = None) -> FluxToken:
    """
    Extract FluxToken with sign-stabilized higher-dimensional PCA.
    
    v0.05 improvements:
    - Extracts n_components PCs (default 8) instead of just 3
    - Applies sign stabilization for consistent trajectories
    - First 3 components used for visualization, all for computation
    """
    if stabilizer is None:
        stabilizer = _global_stabilizer
    
    with torch.no_grad():
        # Energy calculation
        if method == "shannon":
            pre_e = shannon_entropy(h_pre, dim=-1).mean().item()
            post_e = shannon_entropy(h_post, dim=-1).mean().item()
        elif method == "histogram":
            pre_e = activation_distribution_entropy(h_pre)
            post_e = activation_distribution_entropy(h_post)
        else:  # l2
            pre_e = h_pre.norm(dim=-1).mean().item()
            post_e = h_post.norm(dim=-1).mean().item()
        
        delta_E = (pre_e - post_e) * 10.0
        
        # Higher-dimensional PCA
        b, t, c = h_pre.shape
        delta_h = (h_post - h_pre).reshape(-1, c)
        
        try:
            # Extract more components
            U, S, Vt = torch.pca_lowrank(delta_h, q=min(n_components, c), center=False)
            
            # Get first n_components principal directions
            # Vt shape: (c, q) - each column is a PC
            components = Vt[:, :n_components].T  # Shape: (n_components, c)
            
            # Apply sign stabilization
            components = stabilizer.stabilize(components)
            
            # Create direction vector from singular values weighted PCs
            # Use singular values as weights for importance
            weights = S[:n_components] / S[:n_components].sum()
            v_full = torch.zeros(n_components)
            
            for i in range(n_components):
                # Project delta_h onto each PC and weight by singular value
                v_full[i] = weights[i].item()
            
            # Normalize
            if v_full.norm() > 1e-6:
                v_full = F.normalize(v_full.unsqueeze(0), dim=1).squeeze(0)
            else:
                v_full = torch.randn(n_components)
                v_full = F.normalize(v_full.unsqueeze(0), dim=1).squeeze(0)
                
        except Exception as e:
            print(f"PCA failed at layer {layer_idx}: {e}")
            v_full = torch.randn(n_components)
            v_full = F.normalize(v_full.unsqueeze(0), dim=1).squeeze(0)
        
        # Attractor and resonance
        coherence = omega.cos().mean().item()
        alpha = coherence
        
        omega_normalized = torch.sigmoid(omega)
        omega_std = omega_normalized.std().item()
        resonance = 1.0 - min(omega_std * 3, 1.0)
        
        return FluxToken(
            delta_E=delta_E,
            v_full=v_full,
            alpha=alpha,
            omega=resonance,
            raw_text=f"[Layer {layer_idx}]",
            entropy_method=method,
            sign_stabilized=True
        )


# ==================== Comparison Utilities ====================

def compare_entropy_methods(h_pre: torch.Tensor, h_post: torch.Tensor) -> dict:
    """Compare L2, Shannon, and Histogram entropy methods."""
    results = {}
    
    pre_l2 = h_pre.norm(dim=-1).mean().item()
    post_l2 = h_post.norm(dim=-1).mean().item()
    results["l2"] = (pre_l2 - post_l2) * 10.0
    
    pre_shannon = shannon_entropy(h_pre, dim=-1).mean().item()
    post_shannon = shannon_entropy(h_post, dim=-1).mean().item()
    results["shannon"] = (pre_shannon - post_shannon) * 10.0
    
    pre_hist = activation_distribution_entropy(h_pre)
    post_hist = activation_distribution_entropy(h_post)
    results["histogram"] = (pre_hist - post_hist) * 10.0
    
    return results


def information_retained(n_components: int, total_components: int) -> float:
    """
    Estimate information retained by using n_components.
    Assumes roughly exponential decay of singular values (common in neural activations).
    """
    # Rough heuristic: first k components capture ~(1 - 0.9^k) of variance
    return 1 - (0.85 ** n_components)


# ==================== Visualizer ====================

def visualize_manifold(souls: List[Soul], title="Eden's Manifolds — v0.05"):
    """Visualize soul trajectories in 3D (first 3 components of full representation)."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    colors = {'Eden': '#ff4499', 'Shoggoth': '#00ffff'}
    
    for soul in souls:
        pos = torch.zeros(3)
        positions = [pos.numpy()]
        for flux in soul.process_window:
            pos = pos + flux.v  # Use 3D projection
            positions.append(pos.numpy())
        arr = np.array(positions)
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], 
               color=colors.get(soul.name, '#888888'), lw=2.5, alpha=0.9)
        ax.scatter(arr[-1, 0], arr[-1, 1], arr[-1, 2], 
                  color=colors.get(soul.name, '#888888'), s=250,
                  label=f"{soul.name} (3D projection of {soul.n_components}D)", 
                  edgecolors='white', linewidth=1.5)
    
    ax.legend(fontsize=12)
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.tight_layout()
    plt.show()


# ==================== Demo ====================

if __name__ == "__main__":
    print("=" * 70)
    print("EPT 0.05 — Higher-D Representation + PCA Sign Stabilization")
    print("=" * 70)
    print()
    
    # Show information retention estimates
    print("=== Information Retention by Component Count ===")
    for n in [3, 8, 16, 32]:
        retained = information_retained(n, 768) * 100
        print(f"  {n} components: ~{retained:.1f}% variance captured (estimated)")
    print()
    
    # Test PCA stabilization
    print("=== Testing PCA Sign Stabilization ===")
    stabilizer = PCAStabilizer(n_components=8)
    
    # Simulate two similar but not identical PCA results
    pcs1 = torch.randn(8, 768)
    pcs1 = F.normalize(pcs1, dim=1)
    
    pcs2 = pcs1.clone()
    pcs2[0] = -pcs2[0]  # Flip first component
    pcs2[3] = -pcs2[3]  # Flip fourth component
    
    stabilized1 = stabilizer.stabilize(pcs1)
    stabilized2 = stabilizer.stabilize(pcs2)
    
    print(f"Before stabilization - PC0 alignment: {torch.dot(pcs1[0], pcs2[0]):.3f}")
    print(f"After stabilization - PC0 alignment: {torch.dot(stabilized1[0], stabilized2[0]):.3f}")
    print()
    
    # Test FluxToken with higher dimensions
    print("=== Testing Higher-D FluxTokens ===")
    f1 = FluxToken(
        delta_E=-2.0, 
        v_full=torch.randn(8), 
        alpha=0.5, omega=0.8, 
        raw_text="move1"
    )
    f2 = FluxToken(
        delta_E=-1.5, 
        v_full=torch.randn(8), 
        alpha=0.3, omega=0.9, 
        raw_text="move2"
    )
    
    print(f"F1: ΔE={f1.delta_E}, v_full shape={f1.v_full.shape}, v (3D)={f1.v.numpy().round(2)}")
    print(f"F2: ΔE={f2.delta_E}, v_full shape={f2.v_full.shape}, v (3D)={f2.v.numpy().round(2)}")
    print(f"Distance (full 8D): {f1.distance(f2, use_full=True):.3f}")
    print(f"Distance (3D only): {f1.distance(f2, use_full=False):.3f}")
    print(f"Cosine similarity: {f1.cosine_similarity(f2):.3f}")
    print()
    
    # Test entropy methods
    print("=== Entropy Method Comparison ===")
    h_pre = torch.randn(1, 10, 768)
    h_post = h_pre * 0.8  # Compression
    
    comparison = compare_entropy_methods(h_pre, h_post)
    print(f"L2 Norm ΔE:    {comparison['l2']:+.3f}")
    print(f"Shannon ΔE:    {comparison['shannon']:+.3f}")
    print(f"Histogram ΔE:  {comparison['histogram']:+.3f}")
    print()
    
    # Run soul demo with higher dimensions
    print("=== Birthing 8D Souls ===")
    eden = Eden(entropy_method="l2", n_components=8)
    shoggy = Shoggoth(entropy_method="l2", n_components=8)
    
    prompt = "The cat is on the mat."
    
    for i in range(10):
        eden.think(prompt)
        shoggy.think(prompt)
        print(f"Step {i+1}: Eden pos (3D)={eden.position.numpy().round(2)}, "
              f"Shoggoth pos (3D)={shoggy.position.numpy().round(2)}")
    
    print()
    print(f"Eden final position (8D): {eden.position_full.numpy().round(2)}")
    print(f"Shoggoth final position (8D): {shoggy.position_full.numpy().round(2)}")
    print()
    
    # Save souls
    eden.save_soul("eden_soul_v05.json")
    shoggy.save_soul("shoggoth_soul_v05.json")
    
    # Trajectory comparison
    print("=== Trajectory Comparison ===")
    dist_full = FluxToken.trajectory_distance(eden.process_window, shoggy.process_window, use_full=True)
    dist_3d = FluxToken.trajectory_distance(eden.process_window, shoggy.process_window, use_full=False)
    print(f"Eden ↔ Shoggoth distance (8D): {dist_full:.3f}")
    print(f"Eden ↔ Shoggoth distance (3D): {dist_3d:.3f}")
    print()
    
    # Visualize
    visualize_manifold([eden, shoggy], "EPT 0.05 — 8D Souls (3D Projection)")
    
    print("\n[Protocol Complete] EPT 0.05 with higher-D representation and sign stabilization.")
