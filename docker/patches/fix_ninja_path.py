"""Fix 10: Use shutil.which for ninja path resolution in FlashInfer JIT.

Patches flashinfer/jit/cpp_ext.py in site-packages.
"""
import sys

path = sys.argv[1]
with open(path) as f:
    src = f.read()

old = '    command = [\n        "ninja",'
new = '''    import shutil, os
    ninja_exe = shutil.which("ninja")
    ninja_exe = os.path.abspath(ninja_exe) if ninja_exe else "/usr/bin/ninja"
    command = [
        ninja_exe,'''

if old not in src:
    print(f"WARNING: Fix 10 marker not found in {path} — skipping", file=sys.stderr)
    sys.exit(0)

if "shutil.which" in src:
    print("Fix 10: already applied, skipping")
    sys.exit(0)

src = src.replace(old, new)
with open(path, 'w') as f:
    f.write(src)
print(f"Fix 10: patched {path}")
