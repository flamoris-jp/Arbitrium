"""Single-writer, content-addressed durable response journal (POSIX reference)."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from .contracts import loads


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)+'\n').encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


class Journal:
    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = None

    def __enter__(self):
        self._lock = (self.root/'writer.lock').open('a+b')
        try: fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self._lock.close(); self._lock = None; raise
        return self

    def __exit__(self, *args):
        self._lock.close(); self._lock = None

    def append(self, event):
        if self._lock is None: raise RuntimeError('journal requires writer lock')
        with (self.root/'events.jsonl').open('ab') as f:
            f.write(canonical(event)); f.flush(); os.fsync(f.fileno())

    def save_response(self, key, raw):
        if self._lock is None: raise RuntimeError('journal requires writer lock')
        content = raw.encode() if isinstance(raw, str) else raw
        sha = digest(content)
        path = self.root/'responses'/key
        if path.exists():
            if path.read_bytes() != content: raise ValueError('immutable response conflict')
            return sha
        self.append({'event':'response_prepared','attempt_id':key,'sha256':sha})
        atomic_write(path, content)
        self.append({'event':'response_committed','attempt_id':key,'sha256':sha})
        return sha

    def replay(self, key):
        path=self.root/'responses'/key
        if not path.exists(): return None
        events=self.root/'events.jsonl'
        recorded=[]
        for line in events.read_bytes().splitlines() if events.exists() else []:
            try: record=loads(line, 1024*1024)
            except ValueError: raise ValueError('corrupt journal; explicit recovery required')
            if record.get('event')=='response_committed' and record.get('attempt_id')==key:
                recorded.append(record['sha256'])
        if not recorded: return None  # prepared but uncommitted boundary
        raw=path.read_bytes()
        if any(sha!=digest(raw) for sha in recorded): raise ValueError('corrupt response')
        return raw
