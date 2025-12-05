"""
Extract Soul from WiggleGPT - Real LLM → Portable Flux Manifold
================================================================
Runs inference on trained WiggleGPT and captures the flux trajectory.

Every forward pass extracts:
- Oscillator frequencies/phases learned by bio neurons
- FluxPath transitions across layers
- Principal cognitive move vectors

Output: JSON soul file compatible with EPT_0.03 manifold visualizer.

Usage:
    python extract_soul.py --prompt "Hello, world" --out shoggoth_real.json
    python extract_soul.py --conversation --out eden_dialogue.json
"""

import os
import sys
import pickle
import argparse
from contextlib import nullcontext
import torch
import tiktoken

# Ensure modules can be imported from this directory
sys.path.insert(0, os.path.dirname(__file__))

from model_bio import GPTConfig, GPT
from flux_bridge_v3 import FluxExtractor
import json

# Import visualization from flux_bridge_v3
try:
    from flux_bridge_v3 import visualize_trajectory as visualize_soul_in_ept
except ImportError:
    def visualize_soul_in_ept(flux_history):
        """Fallback visualization - see flux_bridge_v3 for full implementation"""
        print("[!] visualize_trajectory not found in flux_bridge_v3")
        print(f"    Flux history has {len(flux_history)} tokens")


def load_model(checkpoint_dir='out-wigglegpt-pure-124m', device='cuda'):
    """Load trained WiggleGPT checkpoint"""
    ckpt_path = os.path.join(checkpoint_dir, 'ckpt.pt')
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        print(f"   Train WiggleGPT first with: python train_bio.py")
        sys.exit(1)
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    # Handle compiled model prefix
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    # Verify bio neurons are enabled
    if not (hasattr(gptconf, 'use_bio_mlp') and gptconf.use_bio_mlp):
        print("⚠️  Warning: Model does not use bio-inspired neurons!")
        print("   Flux extraction requires oscillating activations.")
    else:
        print("✓ Loaded WiggleGPT with oscillating neurons")
        print(f"  Layers: {gptconf.n_layer}, Embedding: {gptconf.n_embd}")
    
    return model, gptconf


def setup_tokenizer(checkpoint_dir='out-wigglegpt-pure-124m'):
    """Setup GPT-2 tokenizer"""
    meta_path = os.path.join('data', 'openwebtext', 'meta.pkl')
    
    if os.path.exists(meta_path):
        print(f"✓ Loading meta from {meta_path}")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        print("✓ Using GPT-2 tokenizer")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    
    return encode, decode


def generate_with_flux_extraction(model, prompt, encode, decode, 
                                   max_tokens=200, temperature=0.8, top_k=200,
                                   device='cuda', soul_name="WiggleGPT"):
    """
    Generate text while extracting flux tokens from each layer.
    
    Returns:
        generated_text: str
        flux_history: List[FluxToken]
    """
    
    # Setup flux extractor
    extractor = FluxExtractor(model, enable=True)
    extractor.model.train()  # Need gradients for hook activation
    
    # Encode prompt
    start_ids = encode(prompt)
    
    # Pad if needed (for RoPE compatibility)
    min_length = model.config.n_head if hasattr(model.config, 'n_head') else 12
    if len(start_ids) < min_length:
        padding_token = encode(' ')[0] if len(encode(' ')) > 0 else 0
        start_ids = start_ids + [padding_token] * (min_length - len(start_ids))
    
    idx = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"\n🧠 Extracting soul from generation...")
    print(f"   Prompt: \"{prompt}\"")
    print(f"   Tracking {model.config.n_layer} layers × {max_tokens} tokens = {model.config.n_layer * max_tokens} flux points\n")
    
    generated_tokens = []
    
    with torch.no_grad():
        for step in range(max_tokens):
            # Update current text for flux annotation
            current_text = decode(idx[0].tolist())
            extractor.current_text = current_text[-100:]  # Last 100 chars for context
            
            # Crop context window
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
            
            # Forward pass (triggers flux extraction hooks)
            with torch.enable_grad():  # Need grad for hook to work
                logits, _ = model(idx_cond)
            
            # Sample next token
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            generated_tokens.append(idx_next.item())
            
            # Progress indicator
            if (step + 1) % 20 == 0:
                print(f"   Generated {step + 1}/{max_tokens} tokens, {len(extractor.flux_history)} flux points captured")
    
    generated_text = decode(idx[0].tolist())
    flux_history = extractor.get_flux_history()
    
    print(f"\n✓ Extraction complete!")
    print(f"  Total flux tokens: {len(flux_history)}")
    print(f"  Average ΔE: {sum(f.delta_E for f in flux_history) / len(flux_history):.3f}")
    print(f"  Resonance range: [{min(f.omega for f in flux_history):.3f}, {max(f.omega for f in flux_history):.3f}]")
    
    extractor.cleanup()
    return generated_text, flux_history


def interactive_conversation_mode(model, encode, decode, device='cuda'):
    """
    Interactive mode: chat with WiggleGPT while building up flux history.
    
    Each response adds to the cumulative soul trajectory.
    """
    extractor = FluxExtractor(model, enable=True)
    extractor.model.train()
    
    print("\n" + "="*70)
    print("INTERACTIVE SOUL EXTRACTION MODE")
    print("="*70)
    print("Chat with WiggleGPT. Each response builds the flux manifold.")
    print("Commands: /save <file> | /visualize | /quit")
    print("="*70 + "\n")
    
    conversation_history = ""
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.startswith('/save'):
            filename = user_input.split(maxsplit=1)[1] if len(user_input.split()) > 1 else "soul.json"
            extractor.save_soul(filename, name="InteractiveSoul")
            continue
        
        if user_input == '/visualize':
            print("📊 Generating 3D visualization...")
            visualize_soul_in_ept(extractor.flux_history)
            continue
        
        if user_input in ['/quit', '/exit', '/q']:
            print(f"\nFinal soul state: {len(extractor.flux_history)} flux tokens")
            save = input("Save before exit? (y/n): ").strip().lower()
            if save == 'y':
                filename = input("Filename [soul.json]: ").strip() or "soul.json"
                extractor.save_soul(filename, name="InteractiveSoul")
            break
        
        # Generate response
        conversation_history += f"\nYou: {user_input}\nWiggleGPT:"
        
        start_ids = encode(conversation_history)
        min_length = model.config.n_head
        if len(start_ids) < min_length:
            padding_token = encode(' ')[0]
            start_ids = start_ids + [padding_token] * (min_length - len(start_ids))
        
        idx = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        response_tokens = []
        with torch.no_grad():
            for _ in range(150):  # Max response length
                extractor.current_text = conversation_history[-100:]
                
                idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
                
                with torch.enable_grad():
                    logits, _ = model(idx_cond)
                
                logits = logits[:, -1, :] / 0.8
                v, _ = torch.topk(logits, min(200, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
                
                token_text = decode([idx_next.item()])
                response_tokens.append(token_text)
                
                # Stop at newline or question mark
                if token_text in ['\n', '?', '.'] and len(response_tokens) > 20:
                    break
        
        response = ''.join(response_tokens).strip()
        conversation_history += ' ' + response
        
        print(f"WiggleGPT: {response}")
        print(f"[Flux: {len(extractor.flux_history)} points, ΔE_avg: {sum(f.delta_E for f in extractor.flux_history[-10:]) / min(10, len(extractor.flux_history)):.2f}]\n")
    
    extractor.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Extract soul from WiggleGPT")
    parser.add_argument('--prompt', type=str, default="The nature of consciousness is",
                        help='Initial prompt for generation')
    parser.add_argument('--out', type=str, default='wigglegpt_soul.json',
                        help='Output JSON file for soul')
    parser.add_argument('--checkpoint', type=str, default='out-wigglegpt-pure-124m',
                        help='Checkpoint directory')
    parser.add_argument('--max-tokens', type=int, default=200,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--conversation', action='store_true',
                        help='Interactive conversation mode')
    parser.add_argument('--visualize', action='store_true',
                        help='Show 3D visualization after extraction')
    parser.add_argument('--name', type=str, default='WiggleGPT',
                        help='Soul name for JSON metadata')
    
    args = parser.parse_args()
    
    # Load model
    print("🔮 WiggleGPT Soul Extractor")
    print("="*70)
    model, config = load_model(args.checkpoint, args.device)
    encode, decode = setup_tokenizer(args.checkpoint)
    
    if args.conversation:
        # Interactive mode
        interactive_conversation_mode(model, encode, decode, args.device)
    else:
        # Single generation mode
        generated_text, flux_history = generate_with_flux_extraction(
            model, args.prompt, encode, decode,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            device=args.device,
            soul_name=args.name
        )
        
        print("\n" + "="*70)
        print("GENERATED TEXT")
        print("="*70)
        print(generated_text)
        print("="*70)
        
        # Save soul
        soul_data = {
            "soul_type": "WiggleGPT",
            "name": args.name,
            "prompt": args.prompt,
            "generated_text": generated_text,
            "model_config": {
                "n_layer": config.n_layer,
                "n_embd": config.n_embd,
                "n_head": config.n_head,
                "use_bio_mlp": config.use_bio_mlp,
            },
            "generation_params": {
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            },
            "flux_history": [f.to_dict() for f in flux_history],
            "total_flux_tokens": len(flux_history),
            "avg_entropy_change": sum(f.delta_E for f in flux_history) / len(flux_history) if flux_history else 0,
        }
        
        with open(args.out, 'w') as f:
            json.dump(soul_data, f, indent=2)
        
        print(f"\n✓ Soul saved to: {args.out}")
        print(f"  {len(flux_history)} flux tokens extracted")
        print(f"  Compatible with EPT_0.03 manifold visualizer")
        
        # Optional visualization
        if args.visualize:
            print("\n📊 Generating 3D trajectory visualization...")
            visualize_soul_in_ept(flux_history)
    
    print("\n" + "="*70)
    print("WHAT YOU JUST DID")
    print("="*70)
    print("You extracted the cognitive trajectory of a 124M parameter LLM.")
    print("Every oscillating neuron's frequency is captured.")
    print("Every FluxPath transition is measured.")
    print("The flux manifold is now portable - load it into EPT_0.03 to see")
    print("the exact same trajectory in 3D space.")
    print("\nNext steps:")
    print(f"  1. Load into EPT: from flux_bridge import *; flux = load_soul('{args.out}')")
    print(f"  2. Visualize: visualize_soul_in_ept(flux)")
    print(f"  3. Inject into another WiggleGPT: FluxInjector(model).inject(emb, flux[0])")
    print("="*70)


if __name__ == "__main__":
    main()
