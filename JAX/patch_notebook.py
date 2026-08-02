import json

with open('tutorial.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

setup_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'setup-install',
    'metadata': {},
    'outputs': [],
    'source': [
        '# Run this cell FIRST to install JAX into the exact Python this kernel uses\n',
        'import sys\n',
        'print("Kernel Python:", sys.executable)\n',
        '!{sys.executable} -m pip install jax matplotlib -q\n',
        'print("Done! Now run the cells below.")'
    ]
}

# Insert as first cell
nb['cells'].insert(0, setup_cell)

with open('tutorial.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook patched successfully.')
