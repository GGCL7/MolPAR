"""Download immutable, hashed public snapshots; never overwrite upstream data."""
from __future__ import annotations
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NOM = ROOT / 'downstream/pathway_nomination'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def session():
    s = requests.Session()
    s.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    return s

def fetch_base(out):
    out.mkdir(parents=True, exist_ok=True)
    urls = {
        'rhea_reactions.tsv': 'https://www.rhea-db.org/rhea/?query=*&columns=rhea-id,equation,chebi-id,ec,uniprot,pubmed&format=tsv',
        'kegg_compound_to_chebi.tsv': 'https://rest.kegg.jp/conv/chebi/compound',
        'kegg_enzyme_to_pathway.tsv': 'https://rest.kegg.jp/link/pathway/enzyme',
        'compound_pathways.tsv': 'https://rest.kegg.jp/link/pathway/compound',
    }
    manifest_path = out / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    s = session()
    for name, url in urls.items():
        path = out / name
        if name in manifest and path.exists() and digest(path) == manifest[name]['sha256']:
            continue
        print('Downloading', name, flush=True)
        r = s.get(url, timeout=120); r.raise_for_status()
        if '\t' not in r.text or '<html' in r.text.lower():
            raise ValueError(f'Invalid tabular response: {url}')
        path.write_text(r.text)
        manifest[name] = {'url': r.url, 'accessed_at': datetime.now(timezone.utc).isoformat(),
                          'database_release': r.headers.get('X-UniProt-Release'),
                          'version_kind': 'sha256 content snapshot; release not supplied by endpoint',
                          'sha256': digest(path), 'http_last_modified': r.headers.get('Last-Modified')}
        manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest

def fetch_uniprot(out, reaction_ids):
    """Reviewed entries with explicit Rhea cross-reference; full pagination.

    Join EC and Rhea within each catalytic-activity block, never across an
    entire multifunctional protein. Organism labels do not assert physiology.
    """
    manifest_path = out / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    s = session(); rows = []
    ids = sorted(set(reaction_ids))
    for offset in range(0, len(ids), 40):
        query = 'reviewed:true AND (' + ' OR '.join('rhea:' + x.split(':')[-1] for x in ids[offset:offset+40]) + ')'
        key = 'uniprot_' + hashlib.sha256(query.encode()).hexdigest()[:16]
        raw = out / (key + '.tsv')
        if raw.exists() and key in manifest and digest(raw) == manifest[key]['sha256']:
            frame = pd.read_csv(raw, sep='\t', dtype=str).fillna('')
        else:
            url = 'https://rest.uniprot.org/uniprotkb/search'
            params = {'query': query, 'format': 'tsv', 'fields': 'accession,ec,organism_id,organism_name,cc_catalytic_activity', 'size': 500}
            pages=[]; provenance=[]
            while url:
                r = s.get(url, params=params, timeout=90); r.raise_for_status()
                pages.append(pd.read_csv(io.StringIO(r.text), sep='\t', dtype=str).fillna(''))
                provenance.append({'url': r.url, 'release': r.headers.get('X-UniProt-Release'), 'accessed_at': datetime.now(timezone.utc).isoformat()})
                url = r.links.get('next', {}).get('url'); params=None
            frame = pd.concat(pages, ignore_index=True).drop_duplicates('Entry')
            frame.to_csv(raw, sep='\t', index=False)
            manifest[key]={'pages': provenance, 'sha256': digest(raw), 'query': query, 'reviewed_only': True, 'pagination_complete': True}
            manifest_path.write_text(json.dumps(manifest, indent=2))
        for record in frame.to_dict('records'):
            for block in record['Catalytic activity'].split('CATALYTIC ACTIVITY:')[1:]:
                rids = set(re.findall(r'Rhea:(RHEA:\d+)', block)) & set(ids)
                ecs = re.findall(r'\bEC=([0-9.\-]+)', block) or ['']
                for rid in sorted(rids):
                    for ec in ecs:
                        rows.append({'rhea_reaction_id': rid, 'ec_number': ec, 'uniprot_accession': record['Entry'],
                                     'organism_taxon_id': record['Organism (ID)'], 'organism': record['Organism'],
                                     'pubmed_ids': ';'.join(sorted(set(re.findall(r'PubMed:(\d+)', block)))),
                                     'catalytic_activity': block.strip(), 'source_snapshot': key})
        print(f'UniProt {min(offset+40,len(ids))}/{len(ids)} reactions', flush=True)
    result = pd.DataFrame(rows, columns=['rhea_reaction_id','ec_number','uniprot_accession','organism_taxon_id','organism','pubmed_ids','catalytic_activity','source_snapshot']).drop_duplicates()
    result.to_csv(out / 'uniprot_relations.csv', index=False)
    return result

if __name__ == '__main__':
    fetch_base(HERE / 'sources')
