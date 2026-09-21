"""Process-wide research boundary for supported development commands.

This prevents accidental Python file access, not adversarial native-code access.
Run development in a separate OS account/container for a security boundary.
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = frozenset(('train_clean.jsonl', 'validation_clean.jsonl'))


def install(root=ROOT):
    root = Path(root).resolve()
    data = root / 'data'
    clean = data / 'derived' / 'naamapadam_hi_crf_v1'
    allowed = {clean / name for name in ALLOWED}
    final = root / 'reports' / 'crf' / '100k_final'

    def audit(event, args):
        if event in ('subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn'):
            raise PermissionError('Development guard forbids child processes that could bypass data isolation')
        if event != 'open' or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).absolute()
        resolved = path.resolve()
        # Check lexical and resolved paths: traversal and symlink aliases cannot escape.
        for candidate in (path, resolved):
            if candidate.is_relative_to(final):
                raise PermissionError('Final evaluation artifacts are sealed during development')
            if candidate.is_relative_to(data) and candidate not in allowed:
                raise PermissionError('Development data access is limited to train_clean and validation_clean')
        if path in allowed and resolved != path:
            raise PermissionError('Development datasets must not be symlink aliases')
        if resolved in allowed:
            mode = args[1] or ''
            flags = args[2] or 0
            if any(c in str(mode) for c in 'wax+') or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
                raise PermissionError('Development datasets are read only')
    sys.addaudithook(audit)
