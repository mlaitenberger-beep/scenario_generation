"""check_imports_deps.py

Walks Python files under provided directories (default: src) and parses import statements.
For each import it tries to resolve the referenced module using importlib.util.find_spec
(which does not execute module code) and reports any missing modules.

Usage:
    python scripts/check_imports_deps.py [--root ROOT_DIR] [paths...]

Examples:
    python scripts/check_imports_deps.py            # checks 'src'
    python scripts/check_imports_deps.py src third_party

Notes:
 - This tool does NOT execute the imported modules; it only checks whether Python
   can *find* the referenced module/package via importlib.
 - Relative imports (e.g., `from .utils import X`) are resolved relative to the
   package path rooted at the first occurrence of `src` (if present) in the file path.
"""

import ast
import os
import sys
import argparse
import importlib.util
from collections import defaultdict


def collect_python_files(paths):
    for base in paths:
        if os.path.isfile(base) and base.endswith('.py'):
            yield os.path.abspath(base)
        elif os.path.isdir(base):
            for root, _, files in os.walk(base):
                for f in files:
                    if f.endswith('.py'):
                        yield os.path.abspath(os.path.join(root, f))


def parse_imports_from_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            src = fh.read()
    except Exception:
        return []

    try:
        tree = ast.parse(src, filename=path)
    except Exception:
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, node.lineno, 0))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module
            level = node.level or 0
            # If module is None and level > 0, it's a relative import like 'from .. import x'
            imports.append((mod, node.lineno, level))
    return imports


def resolve_relative_import(file_path, level, module):
    # Find the 'src' root if present; otherwise use parent directories as package root
    parts = os.path.normpath(file_path).split(os.sep)
    if 'src' in parts:
        src_idx = parts.index('src')
        rel_parts = parts[src_idx + 1 : -1]  # directories inside src, excluding file
    else:
        # fallback: use path elements above file
        rel_parts = parts[:-1]

    # Compute base package parts after applying the relative level
    if level <= 0:
        base_parts = rel_parts
    else:
        # level=1 means 'from .module' -> use rel_parts as base
        # level=2 means 'from ..module' -> drop last directory
        drop = level - 1
        if drop >= len(rel_parts):
            base_parts = []
        else:
            base_parts = rel_parts[: len(rel_parts) - drop]

    if module:
        candidate = '.'.join(base_parts + [module]) if base_parts else module
    else:
        candidate = '.'.join(base_parts) if base_parts else None
    return candidate


def check_importable(modname):
    if not modname:
        return True  # nothing to check (e.g., rare 'from . import X' where module is None)
    try:
        spec = importlib.util.find_spec(modname)
        if spec is not None:
            return True
        # try top-level package
        top = modname.split('.')[0]
        return importlib.util.find_spec(top) is not None
    except Exception:
        return False


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument('paths', nargs='*', default=['src'], help='Files or directories to scan (default: src)')
    p.add_argument('--root', default=None, help='Optional project root to add to sys.path (defaults to current working dir)')
    p.add_argument('--fail-on-missing', action='store_true', help='Exit with non-zero code if missing imports are found')
    args = p.parse_args(argv)

    root = args.root or os.getcwd()
    # Make sure common package roots are on sys.path so local packages (e.g., src) resolve
    sys.path.insert(0, os.path.abspath(os.path.join(root, 'src')))
    sys.path.insert(0, os.path.abspath(root))

    failures = defaultdict(list)
    file_count = 0
    import_count = 0

    for fpath in collect_python_files(args.paths):
        file_count += 1
        imports = parse_imports_from_file(fpath)
        for mod, lineno, level in imports:
            import_count += 1
            if level and level > 0:
                mod_to_check = resolve_relative_import(fpath, level, mod)
            else:
                mod_to_check = mod

            ok = check_importable(mod_to_check)
            if not ok:
                failures[fpath].append((mod_to_check, lineno))

    # Print results
    print('\nImport scan summary')
    print('-------------------')
    print(f'Files scanned: {file_count}')
    print(f'Import statements found: {import_count}')
    if not failures:
        print('\nAll imports resolved successfully. ✅')
        return 0

    print(f'\nMissing imports detected in {len(failures)} file(s):')
    for fpath, items in failures.items():
        print('\n  File:', fpath)
        for modname, lineno in items:
            print(f'    Line {lineno}: {modname!r} (not found)')

    print('\nTips:')
    print(' - If the missing imports are third-party packages, run `pip install <package>`')
    print(' - If they are local modules, ensure the project `src` folder is on PYTHONPATH or that __init__.py markers exist')

    if args.fail_on_missing:
        return 2
    return 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
