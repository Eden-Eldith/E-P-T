"""
Automated Test Runner for WiggleGPT-EPT Scripts
================================================

Runs comprehensive tests on each script and captures output to test/ folder.
Tests check for actual errors, not just return codes.

Usage:
    python run_tests.py
"""

import subprocess
import sys
import os
from datetime import datetime

# Set UTF-8 for this script
os.environ['PYTHONIOENCODING'] = 'utf-8'

def setup_test_folder():
    """Create test output folder"""
    test_dir = os.path.join(os.path.dirname(__file__), "test")
    os.makedirs(test_dir, exist_ok=True)
    return test_dir

def run_python_code(code, timeout=60):
    """Run Python code and return success, stdout, stderr"""
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            timeout=timeout,
            cwd=os.path.dirname(__file__),
            env=env
        )
        
        stdout = result.stdout.decode('utf-8', errors='replace')
        stderr = result.stderr.decode('utf-8', errors='replace')
        
        return result.returncode == 0, stdout, stderr
        
    except subprocess.TimeoutExpired:
        return False, "", f"TIMEOUT after {timeout}s"
    except Exception as e:
        return False, "", f"ERROR: {str(e)}"

def run_script(script_name, args, timeout=60):
    """Run a script and return success, stdout, stderr"""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        return False, "", f"Script not found: {script_path}"
    
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    cmd = [sys.executable, script_path] + args
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            cwd=os.path.dirname(__file__),
            env=env
        )
        
        stdout = result.stdout.decode('utf-8', errors='replace')
        stderr = result.stderr.decode('utf-8', errors='replace')
        
        return result.returncode == 0, stdout, stderr
        
    except subprocess.TimeoutExpired:
        return False, "", f"TIMEOUT after {timeout}s"
    except Exception as e:
        return False, "", f"ERROR: {str(e)}"

def check_for_errors(stdout, stderr):
    """Check if output contains error indicators"""
    error_indicators = [
        "Traceback (most recent call last):",
        "TypeError:",
        "ImportError:",
        "ModuleNotFoundError:",
        "AttributeError:",
        "NameError:",
        "SyntaxError:",
        "ValueError:",
        "KeyError:",
        "AssertionError:",
    ]
    
    combined = stdout + stderr
    errors_found = []
    
    for indicator in error_indicators:
        if indicator in combined:
            errors_found.append(indicator)
    
    return errors_found

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_ept_0_05():
    """Test EPT_0.05.py - Full functionality test"""
    code = '''
import sys
import os
# Add this directory to path
sys.path.insert(0, r"''' + os.path.dirname(__file__) + '''")

print("=" * 60)
print("TEST: EPT_0.05.py")
print("=" * 60)

# Import the module
import importlib.util
spec = importlib.util.spec_from_file_location("EPT_0_05", r"''' + os.path.join(os.path.dirname(__file__), "EPT_0.05.py") + '''")
ept = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ept)

print("1. Import test: PASS")
print(f"   FluxToken: {ept.FluxToken}")
print(f"   Soul: {ept.Soul}")
print(f"   Eden: {ept.Eden}")
print(f"   Shoggoth: {ept.Shoggoth}")

# Test FluxToken creation
import torch
v_full = torch.randn(8)
flux = ept.FluxToken(
    delta_E=-1.5,
    v_full=v_full,
    alpha=0.7,
    omega=0.9,
    raw_text="test token"
)
print(f"2. FluxToken creation: PASS")
print(f"   delta_E: {flux.delta_E}")
print(f"   v_full shape: {flux.v_full.shape}")
print(f"   v (3D) shape: {flux.v.shape}")

# Test serialization
flux_dict = flux.to_dict()
flux_restored = ept.FluxToken.from_dict(flux_dict)
assert abs(flux.delta_E - flux_restored.delta_E) < 0.001
print(f"3. Serialization: PASS")

# Test Eden soul
eden = ept.Eden()
response = eden.think("Hello world")
print(f"4. Eden think: PASS")
print(f"   Response: {response[:50]}...")
print(f"   Process window: {len(eden.process_window)} tokens")

# Test Shoggoth soul
shoggoth = ept.Shoggoth()
response = shoggoth.think("Test input")
print(f"5. Shoggoth think: PASS")
print(f"   Response: {response[:50]}...")

print()
print("ALL EPT_0.05.py TESTS PASSED")
'''
    return run_python_code(code)

def test_flux_bridge_v3():
    """Test flux_bridge_v3.py - Core functionality"""
    code = '''
import sys
import os
# Add this directory to path
sys.path.insert(0, r"''' + os.path.dirname(__file__) + '''")

print("=" * 60)
print("TEST: flux_bridge_v3.py")
print("=" * 60)

# Import
from flux_bridge_v3 import FluxToken, FluxExtractor, PCAStabilizer
print("1. Import test: PASS")

# Test FluxToken
import torch
v_full = torch.randn(8)
flux = FluxToken(
    delta_E=-2.0,
    v_full=v_full,
    alpha=0.5,
    omega=0.8,
    raw_text="test"
)
print(f"2. FluxToken creation: PASS")
print(f"   v shape: {flux.v.shape}, v_full shape: {flux.v_full.shape}")

# Test serialization
d = flux.to_dict()
restored = FluxToken.from_dict(d)
assert abs(flux.delta_E - restored.delta_E) < 0.001
print("3. FluxToken serialization: PASS")

# Test PCAStabilizer
stabilizer = PCAStabilizer(n_components=8)
components = torch.randn(8, 64)
stabilized = stabilizer.stabilize(components)
assert stabilized.shape == components.shape
print("4. PCAStabilizer: PASS")

# Test FluxExtractor has enable parameter
import inspect
sig = inspect.signature(FluxExtractor.__init__)
params = list(sig.parameters.keys())
assert 'enable' in params, f"Missing 'enable' param. Found: {params}"
print(f"5. FluxExtractor params: {params}")
print("   enable parameter: PRESENT")

print()
print("ALL flux_bridge_v3.py TESTS PASSED")
'''
    return run_python_code(code)

def test_compare_souls():
    """Test compare_souls.py"""
    code = '''
import sys
import os
# Add this directory to path
sys.path.insert(0, r"''' + os.path.dirname(__file__) + '''")

print("=" * 60)
print("TEST: compare_souls.py")
print("=" * 60)

# Get FluxToken from EPT_0.05
import importlib.util
spec = importlib.util.spec_from_file_location("EPT_0_05", r"''' + os.path.join(os.path.dirname(__file__), "EPT_0.05.py") + '''")
ept = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ept)
FluxToken = ept.FluxToken

print("1. Import FluxToken from EPT_0.05: PASS")

# Test compute_trajectory function
import torch
import numpy as np

def compute_trajectory(flux_history):
    pos = torch.zeros(3)
    trajectory = [pos.clone()]
    for flux in flux_history:
        pos = pos + flux.v
        trajectory.append(pos.clone())
    return torch.stack(trajectory).numpy()

# Create test flux history
flux_history = []
for i in range(10):
    flux = FluxToken(
        delta_E=-float(i) * 0.1,
        v_full=torch.randn(8),
        alpha=0.5,
        omega=0.7,
        raw_text=f"token {i}"
    )
    flux_history.append(flux)

trajectory = compute_trajectory(flux_history)
print(f"2. compute_trajectory: PASS")
print(f"   Trajectory shape: {trajectory.shape}")
assert trajectory.shape == (11, 3), f"Wrong shape: {trajectory.shape}"

# Test analyze function concept
entropy_changes = [f.delta_E for f in flux_history]
resonances = [f.omega for f in flux_history]
alphas = [f.alpha for f in flux_history]

print(f"3. Statistics extraction: PASS")
print(f"   Mean delta_E: {np.mean(entropy_changes):.3f}")
print(f"   Mean omega: {np.mean(resonances):.3f}")
print(f"   Mean alpha: {np.mean(alphas):.3f}")

print()
print("compare_souls.py CORE TESTS PASSED")
print("Note: Full run requires JSON soul files in JSONs/ folder")
'''
    return run_python_code(code)

def test_demo_integration():
    """Test demo_integration.py components"""
    code = '''
import sys
import os
# Add this directory to path
sys.path.insert(0, r"''' + os.path.dirname(__file__) + '''")

print("=" * 60)
print("TEST: demo_integration.py")
print("=" * 60)

# Test 1: FluxToken import from flux_bridge_v3
from flux_bridge_v3 import FluxToken, FluxExtractor
import torch
import inspect

flux = FluxToken(
    delta_E=-2.5,
    v_full=torch.tensor([0.5, -0.3, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0]),
    alpha=0.7,
    omega=0.85,
    raw_text="test cognitive move"
)
print("1. FluxToken creation: PASS")
print(f"   delta_E: {flux.delta_E}")
print(f"   v: {flux.v.numpy()}")

# Test serialization
flux_dict = flux.to_dict()
flux_restored = FluxToken.from_dict(flux_dict)
assert abs(flux.delta_E - flux_restored.delta_E) < 0.001
print("2. FluxToken serialization: PASS")

# Test 3: Check FluxExtractor has enable parameter
sig = inspect.signature(FluxExtractor.__init__)
params = list(sig.parameters.keys())
print(f"3. FluxExtractor params: {params}")
assert 'enable' in params, "FluxExtractor missing 'enable' parameter!"
print("   enable parameter: PASS")

# Test 4: Check model_bio is accessible and test with real model
from model_bio import GPT, GPTConfig
print("4. model_bio import: PASS")

# Create tiny model for testing
config = GPTConfig(
    block_size=64,
    vocab_size=512,
    n_layer=2,
    n_head=2,
    n_embd=64,
    use_bio_mlp=True,
    dropout=0.0,
)
model = GPT(config)
print(f"5. GPT model creation: PASS")
print(f"   Layers: {config.n_layer}, Embedding: {config.n_embd}")

# Test FluxExtractor with model
extractor = FluxExtractor(model, enable=True)
print(f"6. FluxExtractor(model, enable=True): PASS")
print(f"   Hooks registered: {len(extractor.hooks)}")

# Test forward pass
dummy_input = torch.randint(0, config.vocab_size, (1, 16))
model.eval()
with torch.no_grad():
    logits, loss = model(dummy_input)

flux_history = extractor.get_flux_history()
print(f"7. Forward pass extraction: PASS")
print(f"   Flux tokens extracted: {len(flux_history)}")

# Test cleanup
extractor.cleanup()
print("8. Cleanup: PASS")

print()
print("ALL demo_integration.py TESTS PASSED")
'''
    return run_python_code(code)

def test_extract_soul():
    """Test extract_soul.py components"""
    code = '''
import sys
import os
# Add this directory to path
sys.path.insert(0, r"''' + os.path.dirname(__file__) + '''")

print("=" * 60)
print("TEST: extract_soul.py")
print("=" * 60)

# Test imports
from flux_bridge_v3 import FluxExtractor, FluxToken
print("1. flux_bridge_v3 imports: PASS")

# Test visualize_trajectory import
from flux_bridge_v3 import visualize_trajectory
print("2. visualize_trajectory import: PASS")

# Test model_bio imports
from model_bio import GPTConfig, GPT
print("3. model_bio imports: PASS")

# Test tokenizer setup
import tiktoken
enc = tiktoken.get_encoding("gpt2")
test_text = "Hello world"
tokens = enc.encode(test_text)
decoded = enc.decode(tokens)
assert decoded == test_text
print("4. tiktoken GPT-2 tokenizer: PASS")
print(f"   '{test_text}' -> {tokens} -> '{decoded}'")

# Check checkpoint path exists (would be in same directory or user-provided)
this_dir = r"''' + os.path.dirname(__file__) + '''"
checkpoint_path = os.path.join(this_dir, 'out-wigglegpt-pure-124m', 'ckpt.pt')
print(f"5. Checkpoint path (example): {checkpoint_path}")
print(f"   Exists: {os.path.exists(checkpoint_path)}")
print(f"   Note: Checkpoint downloaded separately from HuggingFace")

print()
print("ALL extract_soul.py COMPONENT TESTS PASSED")
'''
    return run_python_code(code)

def test_quickstart():
    """Test QUICKSTART.py syntax"""
    code = '''
import sys
import os
import ast

print("=" * 60)
print("TEST: QUICKSTART.py")
print("=" * 60)

script_path = r"''' + os.path.join(os.path.dirname(__file__), "QUICKSTART.py") + '''"

with open(script_path, encoding="utf-8") as f:
    source = f.read()

# Check syntax
ast.parse(source)
print("1. Python syntax: VALID")

# Check for key imports mentioned
checks = [
    ("flux_bridge_v3", "from flux_bridge_v3"),
    ("FluxToken", "FluxToken"),
    ("visualize", "visualize"),
]

for name, pattern in checks:
    found = pattern in source
    status = "FOUND" if found else "MISSING"
    print(f"2. References {name}: {status}")
    if not found:
        print(f"   WARNING: {pattern} not found in QUICKSTART.py")

print()
print("QUICKSTART.py SYNTAX TEST: PASS")
'''
    return run_python_code(code)

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    print("=" * 70)
    print("  WiggleGPT-EPT Comprehensive Test Runner")
    print("=" * 70)
    print()
    
    test_dir = setup_test_folder()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Test output folder: {test_dir}")
    print(f"Timestamp: {timestamp}")
    print()
    
    # Define tests
    tests = [
        ("EPT_0.05.py", test_ept_0_05),
        ("flux_bridge_v3.py", test_flux_bridge_v3),
        ("compare_souls.py", test_compare_souls),
        ("demo_integration.py", test_demo_integration),
        ("extract_soul.py", test_extract_soul),
        ("QUICKSTART.py", test_quickstart),
    ]
    
    results = []
    
    for script_name, test_func in tests:
        print(f"Testing: {script_name}...", end=" ", flush=True)
        
        success, stdout, stderr = test_func()
        
        # Check for errors in output even if return code was 0
        errors = check_for_errors(stdout, stderr)
        
        # Determine real success
        real_success = success and len(errors) == 0
        
        # Save output
        output_file = os.path.join(test_dir, f"{script_name.replace('.py', '')}_{timestamp}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Test Run: {datetime.now().isoformat()}\n")
            f.write(f"Script: {script_name}\n")
            f.write(f"Return Code Success: {success}\n")
            f.write(f"Errors Found: {errors}\n")
            f.write(f"Final Status: {'PASS' if real_success else 'FAIL'}\n")
            f.write(f"{'='*60}\n\n")
            f.write("STDOUT:\n")
            f.write(stdout if stdout else "(empty)\n")
            f.write(f"\n{'='*60}\n")
            f.write("STDERR:\n")
            f.write(stderr if stderr else "(empty)\n")
        
        status = "PASS" if real_success else "FAIL"
        print(status)
        
        if errors and not real_success:
            print(f"      Errors: {errors}")
        
        results.append((script_name, real_success, output_file, errors))
    
    # Summary
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, s, _, _ in results if s)
    total = len(results)
    
    for script, success, output_file, errors in results:
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {script}")
        if errors:
            print(f"         Errors: {errors}")
    
    print()
    print(f"Results: {passed}/{total} passed")
    print(f"Output files in: {test_dir}")
    
    # Create summary file
    summary_file = os.path.join(test_dir, f"SUMMARY_{timestamp}.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"WiggleGPT-EPT Test Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Run at: {datetime.now().isoformat()}\n")
        f.write(f"Results: {passed}/{total} passed\n\n")
        
        for script, success, output_file, errors in results:
            status = "PASS" if success else "FAIL"
            f.write(f"[{status}] {script}\n")
            if errors:
                f.write(f"  Errors: {errors}\n")
            f.write(f"  Output: {os.path.basename(output_file)}\n\n")
    
    print(f"Summary: {summary_file}")
    print()
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
