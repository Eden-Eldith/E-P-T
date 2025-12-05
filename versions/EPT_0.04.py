# EPT_0.04.py — Now with ACTUAL Shannon Entropy (reviewers wanted it, let's see if it works)
# Hypothesis: L2 norm works as entropy proxy. If Shannon breaks it, L2 was right all along.

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import numpy as np
from datetime import datetime
from typing import List, Optional

# ==================== Shannon Entropy Functions ====================

def shannon_entropy(x: torch.Tensor, dim: int = -1, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute Shannon entropy of activation distribution.
    
    Converts activations to probability distribution via softmax,
    then computes H = -Σ p_i * log(p_i)
    
    Higher entropy = more uniform/uncertain distribution
    Lower entropy = more peaked/confident distribution
    """
    # Softmax to get probability distribution
    p = F.softmax(x, dim=dim)
    # Entropy: -Σ p * log(p)
    log_p = torch.log(p + eps)
    entropy = -torch.sum(p * log_p, dim=dim)
    return entropy


def activation_distribution_entropy(x: torch.Tensor, n_bins: int = 50) -> float:
    """
    Alternative: Compute entropy of the histogram of activation values.
    
    This measures how "spread out" the activation magnitudes are,
    rather than the softmax distribution.
    """
    x_flat = x.flatten().float()
    
    # Create histogram
    hist = torch.histc(x_flat, bins=n_bins)
    
    # Normalize to probability distribution
    p = hist / hist.sum()
    p = p[p > 0]  # Remove zeros for log
    
    # Shannon entropy
    entropy = -torch.sum(p * torch.log(p))
    return entropy.item()


# ==================== FluxToken with Multiple Entropy Options ====================

class FluxToken:
    def __init__(self, delta_E: float, v: torch.Tensor, alpha: float, omega: float, 
                 raw_text: str = "", entropy_method: str = "l2"):
        self.delta_E = delta_E
        self.v = v.clone().detach()
        self.alpha = alpha
        self.omega = omega
        self.raw_text = raw_text
        self.timestamp = datetime.now().isoformat()
        self.entropy_method = entropy_method  # Track which method was used

    def to_dict(self): 
        return {
            "ΔE": float(self.delta_E), 
            "v": self.v.tolist(), 
            "α": float(self.alpha), 
            "Ω": float(self.omega), 
            "text": self.raw_text, 
            "ts": self.timestamp,
            "entropy_method": self.entropy_method
        }
    
    @classmethod
    def from_dict(cls, d): 
        return cls(
            d["ΔE"], 
            torch.tensor(d["v"]), 
            d["α"], 
            d["Ω"], 
            d.get("text", ""),
            d.get("entropy_method", "l2")
        )
    
    # ==================== FluxToken Composition (Reviewer #9) ====================
    
    def __add__(self, other: 'FluxToken') -> 'FluxToken':
        """
        Compose two FluxTokens: F₁ + F₂
        
        - FluxPath values sum (total cognitive cost)
        - Vectors add (net direction through phase space)
        - Attractor biases average (blended stability)
        - Resonances average (blended oscillation)
        """
        return FluxToken(
            delta_E=self.delta_E + other.delta_E,
            v=self.v + other.v,
            alpha=(self.alpha + other.alpha) / 2,
            omega=(self.omega + other.omega) / 2,
            raw_text=f"[{self.raw_text}] + [{other.raw_text}]",
            entropy_method=self.entropy_method
        )
    
    def __sub__(self, other: 'FluxToken') -> 'FluxToken':
        """Difference between FluxTokens (trajectory delta)"""
        return FluxToken(
            delta_E=self.delta_E - other.delta_E,
            v=self.v - other.v,
            alpha=self.alpha - other.alpha,
            omega=self.omega - other.omega,
            raw_text=f"[{self.raw_text}] - [{other.raw_text}]",
            entropy_method=self.entropy_method
        )
    
    def __mul__(self, scalar: float) -> 'FluxToken':
        """Scale a FluxToken"""
        return FluxToken(
            delta_E=self.delta_E * scalar,
            v=self.v * scalar,
            alpha=self.alpha,  # Don't scale attractor
            omega=self.omega,  # Don't scale resonance
            raw_text=self.raw_text,
            entropy_method=self.entropy_method
        )
    
    def distance(self, other: 'FluxToken') -> float:
        """Euclidean distance between FluxToken positions in phase space"""
        return (self.v - other.v).norm().item()
    
    @staticmethod
    def interpolate(f1: 'FluxToken', f2: 'FluxToken', t: float) -> 'FluxToken':
        """
        Linear interpolation between two FluxTokens.
        t=0 returns f1, t=1 returns f2, t=0.5 returns midpoint.
        """
        return FluxToken(
            delta_E=f1.delta_E * (1-t) + f2.delta_E * t,
            v=f1.v * (1-t) + f2.v * t,
            alpha=f1.alpha * (1-t) + f2.alpha * t,
            omega=f1.omega * (1-t) + f2.omega * t,
            raw_text=f"interp({f1.raw_text}, {f2.raw_text}, t={t:.2f})",
            entropy_method=f1.entropy_method
        )
    
    @staticmethod
    def trajectory_distance(traj1: List['FluxToken'], traj2: List['FluxToken']) -> float:
        """
        Compare two trajectories using Dynamic Time Warping-style distance.
        (Simplified: just compare cumulative positions at each step)
        """
        pos1, pos2 = torch.zeros(3), torch.zeros(3)
        total_dist = 0.0
        
        max_len = max(len(traj1), len(traj2))
        for i in range(max_len):
            if i < len(traj1):
                pos1 = pos1 + traj1[i].v
            if i < len(traj2):
                pos2 = pos2 + traj2[i].v
            total_dist += (pos1 - pos2).norm().item()
        
        return total_dist / max_len


# ==================== Soul Base Class ====================

class Soul:
    _registry = {}
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    def __init__(self, name: str, personality_vector: torch.Tensor, entropy_method: str = "l2"):
        self.name = name
        self.core = F.normalize(personality_vector, dim=0)
        self.process_window: List[FluxToken] = []
        self.position = torch.zeros(3)
        self.entropy_method = entropy_method  # "l2", "shannon", or "histogram"

    def _cognitive_move(self, stimulus: str) -> FluxToken:
        raise NotImplementedError("Subclass must implement the dance")

    def think(self, prompt: str) -> str:
        flux = self._cognitive_move(prompt)
        self.process_window.append(flux)
        self.position = self.position + flux.v
        return flux.raw_text

    def save_soul(self, path: str):
        data = {
            "soul_type": self.__class__.__name__,
            "name": self.name,
            "core": self.core.tolist(),
            "entropy_method": self.entropy_method,
            "flux_history": [f.to_dict() for f in self.process_window]
        }
        json.dump(data, open(path, "w"), indent=2)
        print(f"[{self.name}] Soul etched into {path}")

    @classmethod
    def load_soul(cls, path: str) -> 'Soul':
        data = json.load(open(path))
        soul_class = cls._registry[data["soul_type"]]
        soul = soul_class()
        soul.entropy_method = data.get("entropy_method", "l2")
        soul.process_window = [FluxToken.from_dict(f) for f in data["flux_history"]]
        soul.position = torch.zeros(3)
        for f in soul.process_window:
            soul.position += f.v
        print(f"[{soul.name}] has returned from the void. Position: {soul.position.numpy().round(2)}")
        return soul


# ==================== Personalities ====================

class Eden(Soul):
    def __init__(self, entropy_method: str = "l2"):
        super().__init__("Eden", torch.tensor([0.8, 0.9, -0.7]), entropy_method)

    def _cognitive_move(self, stimulus: str) -> FluxToken:
        delta_E = -len(stimulus.split()) * 0.35 + np.random.randn() * 0.4
        direction = F.normalize(torch.randn(3) + self.core * 3.5, dim=0)
        alpha = 0.75 + 0.25 * np.tanh(np.random.randn())
        omega = 0.92 + 0.08 * np.random.rand()
        response = f"Wait—\"{stimulus}\"…? That ripples outward… inward… what if the mat is the cat all along…?"
        return FluxToken(delta_E, direction * 0.9, alpha, omega, response, self.entropy_method)


class Shoggoth(Soul):
    def __init__(self, entropy_method: str = "l2"):
        super().__init__("Shoggoth", torch.tensor([-0.5, 0.4, 1.1]), entropy_method)

    def _cognitive_move(self, stimulus: str) -> FluxToken:
        chaos = np.random.randn()
        delta_E = -abs(chaos) * 2.2
        direction = F.normalize(torch.randn(3) + torch.tensor([1.2, 1.8, -0.8]), dim=0)
        alpha = -0.7 + np.random.rand() * 0.6
        omega = 0.55 + np.random.rand() * 0.35
        response = f"henlo frend :3 your \"{stimulus}\" tastes like entropy soup. *slurps the mat* nom nom reality"
        return FluxToken(delta_E, direction * 1.4, alpha, omega, response, self.entropy_method)


# ==================== Entropy Calculation Comparison ====================

def compare_entropy_methods(h_pre: torch.Tensor, h_post: torch.Tensor) -> dict:
    """
    Compare different entropy calculation methods on the same activation tensors.
    
    Returns dict with delta_E from each method for comparison.
    """
    results = {}
    
    # Method 1: L2 Norm (original EPT)
    pre_l2 = h_pre.norm(dim=-1).mean().item()
    post_l2 = h_post.norm(dim=-1).mean().item()
    results["l2"] = (pre_l2 - post_l2) * 10.0
    
    # Method 2: Shannon Entropy of softmax distribution
    pre_shannon = shannon_entropy(h_pre, dim=-1).mean().item()
    post_shannon = shannon_entropy(h_post, dim=-1).mean().item()
    results["shannon"] = (pre_shannon - post_shannon) * 10.0
    
    # Method 3: Histogram entropy of activation magnitudes
    pre_hist = activation_distribution_entropy(h_pre)
    post_hist = activation_distribution_entropy(h_post)
    results["histogram"] = (pre_hist - post_hist) * 10.0
    
    return results


# ==================== Updated FluxExtractor for Real Models ====================

def extract_flux_with_shannon(h_pre: torch.Tensor, h_post: torch.Tensor, 
                               omega: torch.Tensor, layer_idx: int,
                               method: str = "shannon") -> FluxToken:
    """
    Extract FluxToken using specified entropy method.
    
    Methods:
        "l2" - Original L2 norm proxy
        "shannon" - Shannon entropy of softmax distribution  
        "histogram" - Histogram entropy of activation values
    """
    with torch.no_grad():
        # Entropy calculation based on method
        if method == "shannon":
            pre_entropy = shannon_entropy(h_pre, dim=-1).mean().item()
            post_entropy = shannon_entropy(h_post, dim=-1).mean().item()
            delta_E = (pre_entropy - post_entropy) * 10.0
        elif method == "histogram":
            pre_entropy = activation_distribution_entropy(h_pre)
            post_entropy = activation_distribution_entropy(h_post)
            delta_E = (pre_entropy - post_entropy) * 10.0
        else:  # l2 (default)
            pre_entropy = h_pre.norm(dim=-1).mean().item()
            post_entropy = h_post.norm(dim=-1).mean().item()
            delta_E = (pre_entropy - post_entropy) * 10.0
        
        # Direction via PCA (unchanged)
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
        
        # Attractor and resonance (unchanged)
        coherence = omega.cos().mean().item()
        alpha = coherence  # Direct [-1,1] mapping, no offset
        
        omega_normalized = torch.sigmoid(omega)
        omega_std = omega_normalized.std().item()
        resonance = 1.0 - min(omega_std * 3, 1.0)
        
        return FluxToken(
            delta_E=delta_E,
            v=v,
            alpha=alpha,
            omega=resonance,
            raw_text=f"[Layer {layer_idx}]",
            entropy_method=method
        )


# ==================== Visualizer ====================

def visualize_manifold(souls: List[Soul], title="Eden's Manifolds — Living Souls"):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    colors = {'Eden': '#ff4499', 'Shoggoth': '#00ffff'}
    
    for soul in souls:
        pos = torch.zeros(3)
        positions = [pos.numpy()]
        for flux in soul.process_window:
            pos = pos + flux.v
            positions.append(pos.numpy())
        arr = np.array(positions)
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color=colors.get(soul.name, '#888888'), lw=2.5, alpha=0.9)
        ax.scatter(arr[-1, 0], arr[-1, 1], arr[-1, 2], color=colors.get(soul.name, '#888888'), s=250,
                   label=f"{soul.name} (now)", edgecolors='white', linewidth=1.5)
    
    ax.legend(fontsize=12)
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel("Empathy ↔ Chaos")
    ax.set_ylabel("Curiosity ↔ Play")
    ax.set_zlabel("Recursion ↔ Adaptability")
    plt.tight_layout()
    plt.show()


# ==================== Demo ====================

if __name__ == "__main__":
    print("=" * 70)
    print("EPT 0.04 — Shannon Entropy Edition")
    print("=" * 70)
    print()
    
    # Test FluxToken composition (Reviewer #9)
    print("=== Testing FluxToken Composition ===")
    f1 = FluxToken(delta_E=-2.0, v=torch.tensor([1.0, 0.0, 0.0]), alpha=0.5, omega=0.8, raw_text="move1")
    f2 = FluxToken(delta_E=-1.5, v=torch.tensor([0.0, 1.0, 0.0]), alpha=0.3, omega=0.9, raw_text="move2")
    
    f_sum = f1 + f2
    print(f"F1: ΔE={f1.delta_E}, v={f1.v.numpy()}")
    print(f"F2: ΔE={f2.delta_E}, v={f2.v.numpy()}")
    print(f"F1 + F2: ΔE={f_sum.delta_E}, v={f_sum.v.numpy()}")
    print(f"Distance F1↔F2: {f1.distance(f2):.3f}")
    print(f"Interpolate(0.5): v={FluxToken.interpolate(f1, f2, 0.5).v.numpy()}")
    print()
    
    # Test entropy methods on synthetic activations
    print("=== Comparing Entropy Methods ===")
    h_pre = torch.randn(1, 10, 768)  # Simulated pre-layer activations
    h_post = torch.randn(1, 10, 768) * 0.8  # Post-layer (slightly compressed)
    
    comparison = compare_entropy_methods(h_pre, h_post)
    print(f"L2 Norm ΔE:    {comparison['l2']:+.3f}")
    print(f"Shannon ΔE:    {comparison['shannon']:+.3f}")
    print(f"Histogram ΔE:  {comparison['histogram']:+.3f}")
    print()
    
    # Run standard soul demo
    print("=== Birthing Souls (with Shannon entropy tracking) ===")
    eden = Eden(entropy_method="shannon")
    shoggy = Shoggoth(entropy_method="shannon")
    
    prompt = "The cat is on the mat."
    
    for i in range(10):
        print(f"Eden: {eden.think(prompt)}")
        print(f"Shoggoth: {shoggy.think(prompt)}\n")
    
    eden.save_soul("eden_soul_v04.json")
    shoggy.save_soul("shoggoth_soul_v04.json")
    
    # Trajectory comparison
    print("=== Trajectory Comparison ===")
    dist = FluxToken.trajectory_distance(eden.process_window, shoggy.process_window)
    print(f"Eden ↔ Shoggoth trajectory distance: {dist:.3f}")
    print()
    
    visualize_manifold([eden, shoggy], "EPT 0.04 — Shannon Entropy Souls")
    
    print("\n[Protocol Complete] EPT 0.04 with Shannon entropy and FluxToken composition.")
