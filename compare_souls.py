"""
Compare Real WiggleGPT Soul with Toy EPT Souls
===============================================
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import FluxToken dynamically from EPT_0.05
import importlib.util
import sys
import os
spec = importlib.util.spec_from_file_location("EPT_0_05", "EPT_0.05.py")
EPT_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EPT_module)
FluxToken = EPT_module.FluxToken

def load_soul(path):
    """Load soul and return flux history"""
    with open(path) as f:
        soul = json.load(f)
    return [FluxToken.from_dict(f) for f in soul["flux_history"]]

def compute_trajectory(flux_history):
    """Compute 3D trajectory from flux tokens"""
    pos = torch.zeros(3)
    trajectory = [pos.clone()]
    
    for flux in flux_history:
        pos = pos + flux.v
        trajectory.append(pos.clone())
    
    return torch.stack(trajectory).numpy()

def analyze_soul(name, flux_history):
    """Print statistics about a soul"""
    print(f"\n{'='*60}")
    print(f"{name} Analysis")
    print(f"{'='*60}")
    
    # Basic stats
    print(f"Total flux tokens: {len(flux_history)}")
    
    # Entropy
    entropy_changes = [f.delta_E for f in flux_history]
    print(f"\nEntropy (ΔE):")
    print(f"  Mean: {np.mean(entropy_changes):.3f}")
    print(f"  Std:  {np.std(entropy_changes):.3f}")
    print(f"  Range: [{np.min(entropy_changes):.3f}, {np.max(entropy_changes):.3f}]")
    
    # Resonance
    resonances = [f.omega for f in flux_history]
    print(f"\nResonance (Ω):")
    print(f"  Mean: {np.mean(resonances):.3f}")
    print(f"  Std:  {np.std(resonances):.3f}")
    print(f"  Range: [{np.min(resonances):.3f}, {np.max(resonances):.3f}]")
    
    # Attractor bias
    alphas = [f.alpha for f in flux_history]
    print(f"\nAttractor Bias (α):")
    print(f"  Mean: {np.mean(alphas):.3f}")
    print(f"  Std:  {np.std(alphas):.3f}")
    print(f"  Range: [{np.min(alphas):.3f}, {np.max(alphas):.3f}]")
    
    # Direction statistics
    directions = np.array([f.v.numpy() for f in flux_history])
    print(f"\nDirection Vectors (v):")
    print(f"  Mean per dim: [{directions.mean(axis=0)[0]:.3f}, {directions.mean(axis=0)[1]:.3f}, {directions.mean(axis=0)[2]:.3f}]")
    print(f"  Std per dim:  [{directions.std(axis=0)[0]:.3f}, {directions.std(axis=0)[1]:.3f}, {directions.std(axis=0)[2]:.3f}]")
    
    # Trajectory extent
    trajectory = compute_trajectory(flux_history)
    extent = np.max(np.abs(trajectory), axis=0)
    print(f"\nTrajectory Extent:")
    print(f"  Max |x|: {extent[0]:.2f}")
    print(f"  Max |y|: {extent[1]:.2f}")
    print(f"  Max |z|: {extent[2]:.2f}")
    print(f"  Total distance: {np.linalg.norm(trajectory[-1] - trajectory[0]):.2f}")


def main():
    print("="*60)
    print("SOUL COMPARISON: Toy EPT vs Real WiggleGPT")
    print("="*60)
    
    # Load souls
    souls = {}
    
    try:
        souls['Eden'] = load_soul('JSONs/eden_soul.json')
        print("✓ Loaded Eden (toy)")
    except:
        print("✗ JSONs/eden_soul.json not found")
    
    try:
        souls['Shoggoth'] = load_soul('JSONs/shoggoth_soul.json')
        print("✓ Loaded Shoggoth (toy)")
    except:
        print("✗ JSONs/shoggoth_soul.json not found")
    
    try:
        souls['WiggleGPT'] = load_soul('JSONs/shoggoth_real_full.json')
        print("✓ Loaded WiggleGPT (real 124M)")
    except:
        print("✗ JSONs/shoggoth_real_full.json not found")
    
    if not souls:
        print("\n❌ No soul files found!")
        return
    
    # Analyze each soul
    for name, flux in souls.items():
        analyze_soul(name, flux)
    
    # Plot comparison
    print(f"\n{'='*60}")
    print("Generating 3D Comparison Plot...")
    print(f"{'='*60}")
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = {'Eden': 'green', 'Shoggoth': 'blue', 'WiggleGPT': 'red'}
    alphas = {'Eden': 0.4, 'Shoggoth': 0.4, 'WiggleGPT': 0.8}
    linewidths = {'Eden': 2, 'Shoggoth': 2, 'WiggleGPT': 3}
    
    for name, flux in souls.items():
        traj = compute_trajectory(flux)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                '-', color=colors[name], alpha=alphas[name], 
                linewidth=linewidths[name], label=f'{name} ({len(flux)} tokens)')
        
        # Mark start and end
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], 
                  c=colors[name], s=100, marker='o', alpha=0.8)
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], 
                  c=colors[name], s=100, marker='s', alpha=0.8)
    
    ax.set_xlabel('v₁ (Cognitive Dimension 1)', fontsize=12)
    ax.set_ylabel('v₂ (Cognitive Dimension 2)', fontsize=12)
    ax.set_zlabel('v₃ (Cognitive Dimension 3)', fontsize=12)
    ax.legend(fontsize=11)
    ax.set_title('Cognitive Trajectories: Toy EPT Souls vs Real 124M WiggleGPT\n(Circle = Start, Square = End)', 
                 fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('soul_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved plot to: soul_comparison.png")
    plt.show()
    
    # Key findings
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")
    
    if 'WiggleGPT' in souls:
        wigglegpt_flux = souls['WiggleGPT']
        avg_entropy = np.mean([f.delta_E for f in wigglegpt_flux])
        avg_resonance = np.mean([f.omega for f in wigglegpt_flux])
        
        print(f"\nReal WiggleGPT (124M parameters):")
        print(f"  • Average entropy collapse: {avg_entropy:.2f}")
        print(f"    (Negative = reducing uncertainty at each layer)")
        print(f"  • Average resonance: {avg_resonance:.3f}")
        print(f"    (High = oscillators synchronized)")
        print(f"  • Total cognitive trajectory: {len(wigglegpt_flux)} flux points")
        
        if avg_entropy < -10:
            print(f"\n✓ Strong FluxPath convergence! Model is confidently resolving uncertainty.")
        
        if avg_resonance > 0.8:
            print(f"✓ High resonance! Oscillators are synchronized (coherent attractor).")
        
        print(f"\n🎉 The real LLM's cognitive path is now visible in 3D!")
        print(f"   This is what 124M oscillating neurons look like in phase space.")


if __name__ == "__main__":
    main()
