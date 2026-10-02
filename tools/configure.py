import argparse,json,subprocess,sys
from pathlib import Path
import cmake,nlohmann_json,torch
from source_identity import core_digest,own_digest,ROOT
p=argparse.ArgumentParser();p.add_argument('--maidionis-root',required=True);p.add_argument('--build-dir',default='build');a=p.parse_args()
core=Path(a.maidionis_root).resolve();lock=json.loads((ROOT/'dependency.lock.json').read_text())
if core_digest(core)!=lock['core_build_digest']:raise SystemExit('Maidionis source digest differs from dependency.lock.json')
sys.path.insert(0,str(ROOT/'python'))
from arbitrium.legacy import verify_archive,ARCHIVE
from arbitrium.composition import Decision,SCHEMAS
from maidionis_education.contracts import canonical,digest
if digest((ARCHIVE/'inventory.json').read_bytes())!=lock['legacy_inventory_digest']:raise SystemExit('archive inventory pin')
verify_archive()
build=Path(a.build_dir).resolve();build.mkdir(parents=True,exist_ok=True)
trusted={}
for name in lock['datasets']:
 d=Decision(name);trusted[name]=dict(manifest=d.legacy_digest,rows=[dict(id=r['sample_id'],family=r['family_id'],input={k:r['input'][k] for k in ('state','policy','question','choices')},target=r['target']) for r in d.legacy_rows],task=d.task,configs=d.configs,descriptor=d.descriptor)
header='#pragma once\nnamespace arbitrium {\ninline const char* trusted_legacy=R"arbitrium('+canonical(trusted).decode()+')arbitrium";\ninline const char* compiled_schemas=R"arbitrium('+canonical(SCHEMAS).decode()+')arbitrium";\n}\n'
(build/'trusted_legacy.h').write_text(header)
subprocess.run([str(Path(cmake.CMAKE_BIN_DIR)/'cmake'),'-S',str(ROOT),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release','-DMAIDIONIS_ROOT='+str(core),'-DNLOHMANN_JSON_INCLUDE_DIR='+str(nlohmann_json.get_include()),'-DCMAKE_PREFIX_PATH='+torch.utils.cmake_prefix_path,'-DARBITRIUM_SOURCE_DIGEST='+own_digest()],check=True)
