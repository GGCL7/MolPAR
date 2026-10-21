#!/usr/bin/env python3
"""Export final study abundance tables from the retained repository downloads."""
from pathlib import Path
import argparse,csv,json,zipfile

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output-dir',type=Path,required=True)
 a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 root=Path(__file__).resolve().parents[2]
 with zipfile.ZipFile(root/'data/raw/metabolomics_sources.zip') as z:
  for study in ['ST000284','ST002787']:
   records=json.loads(z.read(f'metabolomics/{study}_data.json'))
   samples=list(dict.fromkeys(s for r in records.values() for s in r['DATA']))
   dest=a.output_dir/f'{study}_abundance.csv'
   if dest.exists():raise FileExistsError(dest)
   fields=['feature_id','metabolite_id','metabolite_name','units']+samples
   with dest.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for key,r in records.items():
     w.writerow(dict(feature_id=key,metabolite_id=r.get('metabolite_id',''),metabolite_name=r.get('metabolite_name',''),units=r.get('units',''),**{s:r['DATA'].get(s,'') for s in samples}))
if __name__=='__main__':main()
