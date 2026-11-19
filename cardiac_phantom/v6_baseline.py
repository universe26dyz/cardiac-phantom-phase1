"""Execute the frozen independent T1/T2 NeSVoR route on matched maps."""
import argparse,json,shlex,subprocess
from pathlib import Path
import numpy as np
import nibabel as nib
from ismrm_v2.fit_first.fit_first import build_independent_nesvor_commands

def main():
 p=argparse.ArgumentParser();p.add_argument('inputs',type=Path);p.add_argument('output',type=Path);p.add_argument('--device',type=int,default=0);a=p.parse_args()
 wrapper=Path(__file__).resolve().parents[1]/'scripts/nesvor_v6.sh';commands=build_independent_nesvor_commands([a.inputs/name for name in ('sax','2ch','4ch')],a.output,device=a.device,nesvor_executable=wrapper)
 a.output.mkdir(parents=True,exist_ok=False)
 (a.output/'commands.json').write_text(json.dumps(commands,indent=2))
 for command,parameter,scale in zip(commands,('t1','t2'),(2500.,200.)):
  (a.output/parameter).mkdir(exist_ok=False)
  reconstruction=command.split(' && ',1)[0];subprocess.run(shlex.split(reconstruction),check=True)
  folder=a.output/parameter;image=nib.load(folder/'volume_normalized.nii.gz');data=np.asanyarray(image.dataobj).astype('f4')*scale;nib.save(nib.Nifti1Image(data,image.affine,image.header),folder/'volume_ms.nii.gz')
 print(a.output)
if __name__=='__main__':main()
