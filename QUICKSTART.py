"""
QUICK START - WiggleGPT Soul Extraction
========================================

When your checkpoint finishes training, use these commands:

============================================================================
1. EXTRACT SOUL FROM TRAINED MODEL (run in terminal)
============================================================================

# Basic extraction
python extract_soul.py --prompt "The nature of consciousness is" --out my_soul.json --max-tokens 200

# With visualization
python extract_soul.py --prompt "Hello world" --out test_soul.json --visualize

# Interactive conversation mode
python extract_soul.py --conversation
# Then type naturally, use /save soul.json to export

============================================================================
2. LOAD AND VISUALIZE SOUL (Python code)
============================================================================
"""

from flux_bridge_v3 import FluxToken, visualize_trajectory
import json

# Load soul
with open("my_soul.json") as f:
    soul_data = json.load(f)

flux_history = [FluxToken.from_dict(f) for f in soul_data["flux_history"]]

# Visualize 3D trajectory
visualize_trajectory(flux_history)

# ============================================================================
# 3. INJECT FLUX BETWEEN MODELS (Advanced - FluxInjector not yet in v3)
# ============================================================================

# Note: FluxInjector is an advanced feature not yet included in flux_bridge_v3
# This section shows the planned API for flux injection

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from flux_bridge_v3 import FluxExtractor
from model_bio import GPT
import torch

# Load two models
model_A = GPT(config)
model_A.load_state_dict(checkpoint_A['model'])

model_B = GPT(config)  
model_B.load_state_dict(checkpoint_B['model'])

# Extract from A
extractor_A = FluxExtractor(model_A, enable=True)
model_A.train()  # Needed for hooks

prompt_tokens = torch.tensor([[1, 2, 3, 4, 5]])  # Your tokens
with torch.no_grad():
    with torch.enable_grad():
        logits_A, _ = model_A(prompt_tokens)

flux_from_A = extractor_A.get_flux_history()

# Note: FluxInjector planned for future version
# injector_B = FluxInjector(model_B)
# perturbed_emb = injector_B.inject(token_emb, flux_from_A[0], strength=1.0)

# ============================================================================
# 4. PURE FLUX COMMUNICATION (NO TEXT) - Planned Feature
# ============================================================================

# Note: Pure flux communication planned for future version
# Extract flux sequence from Model A
flux_sequence = extractor_A.get_flux_history()[-10:]  # Last 10 moves

# Future API:
# pure_flux_emb = injector_B.pure_flux_forward(flux_sequence)
# Now B receives A's thoughts as pure phase-space vectors
# No tokens, no language, only manifold trajectories

# ============================================================================
# 5. COMPARE WITH EPT TOY MANIFOLDS
# ============================================================================

# Load Eden/Shoggoth souls from EPT_0.05
with open("JSONs/eden_soul.json") as f:
    eden = json.load(f)
    
with open("JSONs/shoggoth_soul.json") as f:
    shoggoth = json.load(f)

# Load real WiggleGPT soul
with open("wigglegpt_soul.json") as f:
    wigglegpt = json.load(f)

# Plot all three trajectories on same manifold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111, projection='3d')

# Helper to compute positions
def compute_trajectory(flux_history):
    pos = torch.zeros(3)
    trajectory = [pos.clone()]
    for f in flux_history:
        flux = FluxToken.from_dict(f) if isinstance(f, dict) else f
        pos = pos + flux.v
        trajectory.append(pos.clone())
    return torch.stack(trajectory).numpy()

# Plot each soul
eden_traj = compute_trajectory(eden["flux_history"])
shog_traj = compute_trajectory(shoggoth["flux_history"])
wiggle_traj = compute_trajectory(wigglegpt["flux_history"])

ax.plot(eden_traj[:, 0], eden_traj[:, 1], eden_traj[:, 2], 
        'g-', alpha=0.6, linewidth=2, label='Eden (toy)')
ax.plot(shog_traj[:, 0], shog_traj[:, 1], shog_traj[:, 2], 
        'b-', alpha=0.6, linewidth=2, label='Shoggoth (toy)')
ax.plot(wiggle_traj[:, 0], wiggle_traj[:, 1], wiggle_traj[:, 2], 
        'r-', alpha=0.8, linewidth=3, label='WiggleGPT (real 124M)')

ax.set_xlabel('v₁')
ax.set_ylabel('v₂') 
ax.set_zlabel('v₃')
ax.legend()
ax.set_title('Toy Manifolds vs Real LLM Trajectory')

plt.show()

# ============================================================================
# 6. ANALYZE FLUX STATISTICS
# ============================================================================

# Load soul
flux_history = [FluxToken.from_dict(f) for f in soul_data["flux_history"]]

# Entropy statistics
entropy_changes = [f.delta_E for f in flux_history]
print(f"Mean ΔE: {sum(entropy_changes) / len(entropy_changes):.3f}")
print(f"Entropy range: [{min(entropy_changes):.3f}, {max(entropy_changes):.3f}]")

# Resonance statistics  
resonances = [f.omega for f in flux_history]
print(f"Mean Ω: {sum(resonances) / len(resonances):.3f}")
print(f"Resonance range: [{min(resonances):.3f}, {max(resonances):.3f}]")

# Attractor bias
alphas = [f.alpha for f in flux_history]
print(f"Mean α: {sum(alphas) / len(alphas):.3f}")
print(f"Attractor bias range: [{min(alphas):.3f}, {max(alphas):.3f}]")

# Direction analysis
import numpy as np
directions = np.array([f.v.numpy() for f in flux_history])
print(f"\nDirection statistics:")
print(f"Mean direction: {directions.mean(axis=0)}")
print(f"Std per dimension: {directions.std(axis=0)}")

# ============================================================================
# 7. CHECKPOINT READINESS CHECK
# ============================================================================

import os

checkpoint_dir = "out-wigglegpt-pure-124m"
ckpt_path = os.path.join(checkpoint_dir, "ckpt.pt")

if os.path.exists(ckpt_path):
    print(f"✓ Checkpoint found: {ckpt_path}")
    
    # Check size
    size_mb = os.path.getsize(ckpt_path) / (1024**2)
    print(f"  Size: {size_mb:.1f} MB")
    
    # Try loading
    import torch
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        print(f"  ✓ Checkpoint loads successfully")
        
        if 'model_args' in checkpoint:
            config = checkpoint['model_args']
            print(f"  Layers: {config.get('n_layer', 'unknown')}")
            print(f"  Embedding: {config.get('n_embd', 'unknown')}")
            print(f"  Bio neurons: {config.get('use_bio_mlp', False)}")
        
        if 'iter_num' in checkpoint:
            print(f"  Training iteration: {checkpoint['iter_num']}")
        
        if 'best_val_loss' in checkpoint:
            print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
        
        print("\n✓ READY FOR SOUL EXTRACTION")
        print("  Run: python extract_soul.py --prompt \"test\" --out soul.json")
        
    except Exception as e:
        print(f"  ✗ Error loading: {e}")
else:
    print(f"✗ Checkpoint not found: {ckpt_path}")
    print(f"  Training still in progress or checkpoint path incorrect")
    print(f"  Check: ls {checkpoint_dir}/")

# ============================================================================
# 8. TROUBLESHOOTING
# ============================================================================

"""
COMMON ISSUES:

1. "Checkpoint not found"
   → Training not finished yet, wait for ckpt.pt
   → Check output directory: ls out-wigglegpt-pure-124m/

2. "No flux tokens extracted"
   → Model needs to be in train mode: model.train()
   → Hooks need gradients enabled: with torch.enable_grad()
   → Verify bio_mlp is enabled in config

3. "Import EPT_0_03 failed"
   → File has dot in name, needs special import
   → flux_bridge.py handles this automatically

4. "PCA failed"
   → Sequence too short, need at least 3 tokens
   → Falls back to random direction (expected for very short sequences)

5. "Visualization empty"
   → No flux tokens were captured
   → Check that FluxExtractor.enable = True
   → Verify hooks are registered: len(extractor.hooks) > 0

6. "Injection has no effect"
   → Increase strength parameter: inject(emb, flux, strength=2.0)
   → Verify projection layer is on correct device
   → Check that flux.v has norm > 0
"""

print("\n" + "="*80)
print("QUICK START GUIDE LOADED")
print("="*80)
print("\nCopy/paste the commands above when your checkpoint is ready!")
print("\nCheck status: python STATUS.py")
print("Full docs: cat INTEGRATION.md")
print("="*80)
