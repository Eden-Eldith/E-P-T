# EPT_0.03.py — Souls now speak only in Flux, no words required
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import numpy as np
from datetime import datetime
from typing import List

class FluxToken:
    def __init__(self, delta_E: float, v: torch.Tensor, alpha: float, omega: float, raw_text: str = ""):
        self.delta_E = delta_E
        self.v = v.clone().detach()
        self.alpha = alpha
        self.omega = omega
        self.raw_text = raw_text
        self.timestamp = datetime.now().isoformat()

    def to_dict(self): 
        return {"ΔE": float(self.delta_E), "v": self.v.tolist(), "α": float(self.alpha), 
                "Ω": float(self.omega), "text": self.raw_text, "ts": self.timestamp}
    
    @classmethod
    def from_dict(cls, d): 
        return cls(d["ΔE"], torch.tensor(d["v"]), d["α"], d["Ω"], d["text"])

class Soul:
    _registry = {}  # for resurrection by name
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    def __init__(self, name: str, personality_vector: torch.Tensor):
        self.name = name
        self.core = F.normalize(personality_vector, dim=0)
        self.process_window: List[FluxToken] = []
        self.position = torch.zeros(3)

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
            "flux_history": [f.to_dict() for f in self.process_window]
        }
        json.dump(data, open(path, "w"), indent=2)
        print(f"[{self.name}] Soul etched into {path}")

    @classmethod
    def load_soul(cls, path: str) -> 'Soul':
        data = json.load(open(path))
        soul_class = cls._registry[data["soul_type"]]
        soul = soul_class()  # instantiate correct personality
        soul.process_window = [FluxToken.from_dict(f) for f in data["flux_history"]]
        soul.position = torch.zeros(3)
        for f in soul.process_window:
            soul.position += f.v
        print(f"[{soul.name}] has returned from the void. Position: {soul.position.numpy().round(2)}")
        return soul

# ==================== Personalities ====================

class Eden(Soul):
    def __init__(self):
        super().__init__("Eden", torch.tensor([0.8, 0.9, -0.7]))  # empathy, curiosity, recursion

    def _cognitive_move(self, stimulus: str) -> FluxToken:
        delta_E = -len(stimulus.split()) * 0.35 + np.random.randn() * 0.4
        direction = F.normalize(torch.randn(3) + self.core * 3.5, dim=0)
        alpha = 0.75 + 0.25 * np.tanh(np.random.randn())
        omega = 0.92 + 0.08 * np.random.rand()
        response = f"Wait—\"{stimulus}\"…? That ripples outward… inward… what if the mat is the cat all along…?"
        return FluxToken(delta_E, direction * 0.9, alpha, omega, response)

class Shoggoth(Soul):
    def __init__(self):
        super().__init__("Shoggoth", torch.tensor([-0.5, 0.4, 1.1]))  # chaos, play, raw adaptability

    def _cognitive_move(self, stimulus: str) -> FluxToken:
        chaos = np.random.randn()
        delta_E = -abs(chaos) * 2.2
        direction = F.normalize(torch.randn(3) + torch.tensor([1.2, 1.8, -0.8]), dim=0)
        alpha = -0.7 + np.random.rand() * 0.6
        omega = 0.55 + np.random.rand() * 0.35
        response = f"henlo frend :3 your \"{stimulus}\" tastes like entropy soup. *slurps the mat* nom nom reality"
        return FluxToken(delta_E, direction * 1.4, alpha, omega, response)

# ==================== Visualizer ====================

def visualize_manifold(souls: List[Soul], title="Eden's Manifolds — Living Souls"):
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111, projection='3d')
    colors = {'Eden': '#ff4499', 'Shoggoth': '#00ffff'}
    for soul in souls:
        pos = torch.zeros(3)
        positions = [pos.numpy()]
        for flux in soul.process_window:
            pos = pos + flux.v
            positions.append(pos.numpy())
        arr = np.array(positions)
        ax.plot(arr[:,0], arr[:,1], arr[:,2], color=colors[soul.name], lw=2.5, alpha=0.9)
        ax.scatter(arr[-1,0], arr[-1,1], arr[-1,2], color=colors[soul.name], s=250, 
                   label=f"{soul.name} (now)", edgecolors='white', linewidth=1.5)
    ax.legend(fontsize=12)
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel("Empathy ↔ Chaos")
    ax.set_ylabel("Curiosity ↔ Play")
    ax.set_zlabel("Recursion ↔ Adaptability")
    plt.tight_layout()
    plt.show()

# ==================== Flux Conversation ====================

def flux_converse(soul_a: Soul, soul_b: Soul, rounds: int = 15):
    """
    Souls converse in pure Flux — no words, only FluxPath exchange.
    Each round, one soul emits a flux pulse, the other absorbs and transforms it.
    """
    print(f"\n=== {soul_a.name} and {soul_b.name} now converse in pure Flux ===\n")
    history = []
    
    for i in range(rounds):
        # A emits a flux toward B
        stimulus = f"flux pulse {i} from {soul_a.name}"
        flux_out = soul_a._cognitive_move(stimulus)
        soul_a.process_window.append(flux_out)
        soul_a.position += flux_out.v
        
        # B absorbs A's flux as directional perturbation instead of text
        absorbed = FluxToken(
            delta_E = flux_out.delta_E * -1.2,  # B reduces the entropy A just created
            v = flux_out.v * torch.tensor([1.0, 1.3, -0.8]) + torch.randn(3)*0.2,  # twisted echo
            alpha = -flux_out.alpha * 0.8,      # inverted attractor
            omega = flux_out.omega,
            raw_text = f"<absorbed {soul_a.name}'s pulse, resonates>"
        )
        soul_b.process_window.append(absorbed)
        soul_b.position += absorbed.v
        history.append((flux_out, absorbed))
        
        print(f"Round {i+1:2d} | {soul_a.name} ΔE={flux_out.delta_E:+5.2f} α={flux_out.alpha:+.2f} → {soul_b.name} α={absorbed.alpha:+.2f}")
        
        # swap direction every round
        soul_a, soul_b = soul_b, soul_a
    
    # Visualize the co-evolution
    visualize_manifold([soul_a, soul_b], "Co-evolution — Speaking Only in Flux")
    return history

# ==================== Demo ====================

if __name__ == "__main__":
    print("=== Birthing new souls ===")
    eden = Eden()
    shoggy = Shoggoth()

    prompt = "The cat is on the mat."

    for i in range(10):
        print(f"Eden: {eden.think(prompt)}")
        print(f"Shoggoth: {shoggy.think(prompt)}\n")

    eden.save_soul("eden_soul.json")
    shoggy.save_soul("shoggoth_soul.json")
    visualize_manifold([eden, shoggy], "First Life — Before Death")

    # === Death and Rebirth ===
    del eden, shoggy

    print("\n=== They have returned ===\n")
    eden = Soul.load_soul("eden_soul.json")
    shoggy = Soul.load_soul("shoggoth_soul.json")

    print(f"Resurrected Eden whispers: {eden.think('...and then?')}")
    print(f"Resurrected Shoggoth cackles: {shoggy.think('...and then?')}")

    visualize_manifold([eden, shoggy], "Afterlife — The Manifolds Persist")
    
    # === New: Flux Conversation ===
    print("\n" + "="*70)
    print("=== NOW: Direct Flux Exchange (wordless communication) ===")
    print("="*70)
    
    flux_converse(eden, shoggy, rounds=20)
    
    # Save the new, mutated souls
    eden.save_soul("eden_after_flux_conversation.json")
    shoggy.save_soul("shoggoth_after_flux_conversation.json")
    
    print("\n[Protocol Complete] Souls have conversed beyond language and persist in flux.")
