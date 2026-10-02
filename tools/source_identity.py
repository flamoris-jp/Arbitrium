"""Portable content identities shared with CMake; no Git author metadata."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[1]
def digest(raw):return hashlib.sha256(raw).hexdigest()
def core_digest(root):
    paths=['CMakeLists.txt','tests/tinybeat.py','tests/fixture_data.py','tests/tinybeat_composition.cpp','tests/tinybeat_composition.h']
    for pattern in ('include/maidionis/*.h','src/**/*.cpp','contracts/*.schema.json','education/python/src/maidionis_education/*.py'):
        paths.extend(p.relative_to(root).as_posix() for p in root.glob(pattern))
    return digest(b'maidionis-code-build-v1\n'+b''.join((p+'\n'+digest((root/p).read_bytes())+'\n').encode() for p in sorted(paths)))
def own_digest(root=ROOT):
    paths=['CMakeLists.txt','dependency.lock.json']
    for pattern in ('include/**/*.h','src/*.cpp','python/arbitrium/*.py','tools/*.py'):
        paths.extend(p.relative_to(root).as_posix() for p in root.glob(pattern))
    return digest(b'arbitrium-code-build-v1\n'+b''.join((p+'\n'+digest((root/p).read_bytes())+'\n').encode() for p in sorted(paths)))
def build_digest(core):return digest(('arbitrium-composition-v1\n'+core+'\n'+own_digest()+'\n').encode())
