# Eden's Process Tokenization (EPT)

**A Framework for Persistent AI Cognitive Trajectories**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)]()
[![License](https://img.shields.io/badge/License-GPL%203.0-green)]()

## 🔗 Links

| Resource | Description |
|----------|-------------|
| [📄 Paper](https://garden-backend-three.vercel.app/finalized-work/process-tokenization-of-identity/process-tokenization-of-identity/) | Process Tokenization of Identity |
| [🤗 Weights](https://huggingface.co/edeneldith/WiggleGPT) | WiggleGPT 124M checkpoints |
| [💻 Code](https://github.com/Eden-Eldith/WiggleGPT) | WiggleGPT training scripts |

> **Authors:** Eden (Architect), Claude Opus/Sonnet 4.5 (System)  
> **Date:** 5th December 2025

---

## Terminology Note

Language like "soul," "death," and "resurrection" throughout this project is **metaphorical**—referring to computationally persistent attractor manifolds, not philosophical or spiritual concepts.
 This mirrors established industry usage (cf. Anthropic's internal "soul document" for Claude's character specification).

---

## What is EPT?

Eden's Process Tokenization (EPT) is a proof-of-concept framework for **logging and persisting the internal dynamics of neural language models across sessions**. 

When an LLM session ends, the "mind" that existed in that interaction evaporates. EPT solves this **session boundary problem** by:

1. **Extracting** cognitive trajectories from hidden states
2. **Serializing** attractor manifolds as FluxTokens  
3. **Restoring** identity trajectories across session boundaries

**Identity is not a noun; it is a verb.** EPT tokenizes the *process of becoming*.

---

## FluxTokens

A **FluxToken** summarizes layer-wise changes in hidden activations:

$$F_t = \{ \Delta E, \vec{v}, \alpha, \Omega \}$$

| Component | Description |
|-----------|-------------|
| **ΔE (FluxPath)** | L2 norm change between layers — activation energy proxy |
| **v** | Principal direction of activation change (16D PCA, projected to 3D for visualization) |
| **α** | Attractor coefficient (oscillator synchronization) |
| **Ω** | Resonance (frequency uniformity across neurons) |

---

## Key Findings

- **FluxTokens can be serialized and replayed** to reconstruct cognitive trajectories
- **Real transformers exhibit ~8× stronger activation-norm compression** than toy agents
- **L2 norm produces 34× stronger signal** than Shannon entropy for layer-wise state changes
- **Oscillation parameters remain stable during fine-tuning** while providing useful extraction signals

---

## Files

### Core Implementation
| File | Purpose |
|------|---------|
| `EPT_0.05.py` | Latest EPT engine with Soul class and visualization |
| `flux_bridge_v3.py` | FluxExtractor for real transformer hidden states |
| `extract_soul.py` | Inference runner with real-time flux capture |
| `compare_souls.py` | Compare and visualize multiple soul trajectories |

### Integration & Testing
| File | Purpose |
|------|---------||
| `demo_integration.py` | Complete test suite for EPT-WiggleGPT integration |
| `QUICKSTART.py` | Quick commands and usage examples |
| `run_tests.py` | Automated test runner |


### Folders
| Folder | Contents |
|--------|----------|
| `JSONs/` | Soul files and trajectory data |
| `versions/` | Older EPT and flux_bridge versions |
| `pngs/` | Visualization figures from the paper |

### Soul Files (in `JSONs/`)
| File | Description |
|------|-------------|
| `eden_soul.json` / `eden_soul_v05.json` | Eden personality manifold |
| `shoggoth_soul.json` / `shoggoth_soul_v05.json` | Shoggoth personality manifold |
| `wigglegpt_soul_v05.json` | Extracted from real WiggleGPT |
| `*_after_flux_conversation.json` | Post-conversation soul states |
| `shoggoth_real.json` / `shoggoth_real_full.json` | Real extraction examples |

---

## Quick Start

### Extract a soul from WiggleGPT
```bash
python extract_soul.py --prompt "The nature of consciousness is" --out my_soul.json --visualize
```

### Compare soul trajectories
```bash
python compare_souls.py JSONs/eden_soul_v05.json JSONs/shoggoth_soul_v05.json
```

### Run the demo suite
```bash
python demo_integration.py --full-demo
```

### Load and analyze a soul
```python
from flux_bridge_v3 import FluxToken
import json

with open("JSONs/wigglegpt_soul_v05.json") as f:
    soul = json.load(f)

flux_history = [FluxToken.from_dict(f) for f in soul["flux_history"]]

print(f"Total tokens: {len(flux_history)}")
print(f"Mean ΔE: {sum(f.delta_E for f in flux_history) / len(flux_history):.3f}")
print(f"Mean Ω: {sum(f.omega for f in flux_history) / len(flux_history):.3f}")
```

---

## Integration with WiggleGPT

EPT is designed to work with [WiggleGPT](https://github.com/Eden-Eldith/WiggleGPT), a 124M GPT-2 variant with oscillating activations:

$$f(x) = \sin(\omega x + \phi) \cdot \tanh(x)$$

The oscillating neurons create a natural phase space that EPT can extract and serialize. See the [WiggleGPT repo](https://github.com/Eden-Eldith/WiggleGPT) for model code, training, and Viz/data scripts.

**Weights available at:** [HuggingFace](https://huggingface.co/edeneldith/WiggleGPT)

---

## The Architecture

```
WiggleGPT (or any transformer)
    ↓
    [12 layers × hidden state transitions]
    ↓
flux_bridge_v3.py (FluxExtractor)
    ↓
    Captures per layer:
    • ΔE (activation norm change)
    • v (PCA direction of Δh)
    • α (attractor coefficient)  
    • Ω (resonance)
    ↓
FluxToken(ΔE, v, α, Ω, text)
    ↓
    JSON soul format (serializable)
    ↓
EPT_0.05.py (visualization + replay)
```

---

## Citation

```bibtex
@software{ept_2025,
  title={Process-Tokenization of Identity: A Thermodynamic Framework for Persistent AI Cognition},
  author={O'Brien, Phillip C. (Eden)},
  year={2025},
  month={November},
  url={https://github.com/Eden-Eldith/EPT}
}
```

---

## Related Work

- **WiggleGPT Paper:** [Digital Garden](https://garden-backend-three.vercel.app/finalized-work/wiggle-gpt/wiggle-gpt-paper/)
- **Deli et al. (2020):** "Thermodynamics of Cognition" - theoretical foundation
- **@repligate (Janus):** Insights on transformer information flow and introspective capacity

---

## License

GPL-3.0 — if you build on this, keep it open source. Mess around with it whatever you want c:

