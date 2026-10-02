"""Build current supported profile and run the meaningful offline tests."""
import argparse,os,subprocess,sys
from pathlib import Path
import cmake
from source_identity import ROOT
p=argparse.ArgumentParser();p.add_argument('--maidionis-root',required=True);p.add_argument('--build-dir',default='build');a=p.parse_args()
os.chdir(ROOT);build=Path(a.build_dir).resolve();bin_dir=Path(cmake.CMAKE_BIN_DIR)
subprocess.run([sys.executable,'tools/configure.py','--maidionis-root',a.maidionis_root,'--build-dir',str(build)],check=True)
subprocess.run([str(bin_dir/'cmake'),'--build',str(build),'--target','arbitrium_driver','-j','2'],check=True)
subprocess.run([str(bin_dir/'ctest'),'--test-dir',str(build),'-R','decision_binding','--output-on-failure'],check=True)
env=dict(os.environ,ARBITRIUM_DRIVER=str(build/'arbitrium_driver'))
subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],env=env,check=True)
