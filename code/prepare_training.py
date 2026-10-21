#!/usr/bin/env python3
"""Generate the input formats expected by the original training scripts."""
from pathlib import Path
import argparse,csv,json
from molpar_runtime import DATA,load_label_spaces

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 if a.output_dir.exists():p.error('Choose a new output directory')
 with (DATA/'molecules.csv').open(newline='') as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
 fields=[x for x in fields if x not in ['split_seed0','split_order_seed0']]
 split=[]
 for name in ['train','validation','test']:
  pairs=sorted((int(r['split_order_seed0']),i) for i,r in enumerate(rows) if r['split_seed0']==name)
  split.append([i for _,i in pairs])
 a.output_dir.mkdir(parents=True)
 with (a.output_dir/'kegg2026_hier.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 (a.output_dir/'random_seed0.txt').write_text(''.join(repr(x)+'\n' for x in split))
 for name,obj in zip(['label_space_L1.json','label_space_L2.json'],load_label_spaces()):
  (a.output_dir/name).write_text(json.dumps(obj,indent=2)+'\n')
if __name__=='__main__':main()
