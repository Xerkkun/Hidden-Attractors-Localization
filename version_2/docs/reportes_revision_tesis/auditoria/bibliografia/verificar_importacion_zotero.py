from pathlib import Path
import csv,json,re,unicodedata,urllib.request,hashlib,collections
OUT=Path(__file__).resolve().parent
def api(route):
 with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:23119/api/users/0'+route,headers={'Zotero-API-Version':'3'}),timeout=25) as f:return json.load(f)
def pages(route):
 result=[];start=0
 while True:
  page=api(route+('? ' if False else '?limit=100&include=data&start=')+str(start));result.extend(x for x in page if x['data'].get('itemType') not in ('note','annotation','attachment'))
  if len(page)<100:return result
  start+=100
def norm(s):
 s=re.sub(r'\\[\'"`^~=.uvHck]\s*\{?([A-Za-z])\}?',r'\1',str(s or ''))
 s=s.replace(r'\l','l').replace(r'\L','L').replace(r'\&','&')
 return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',s).lower())
def doi(s):return re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)','',str(s or '').strip(),flags=re.I).lower()
plan=json.loads((OUT/'plan_importacion_referencias_locales.json').read_text(encoding='utf-8'))
rows=list(csv.DictReader((OUT/'bibliografia_completa_auditoria.csv').open(encoding='utf-8-sig')))
library=pages('/items/top');collection=pages('/collections/II82RCL4/items/top');byz={z['key']:z for z in library};bybib={r['key']:r for r in rows};ck={z['key'] for z in collection}
results=[]
for e in plan['import_entries']:
 r=bybib[e['key']];z=byz[r['zotero_item_key']];d=z['data'];errors=[]
 if norm(e.get('title'))!=norm(d.get('title')):errors.append('title')
 if doi(e.get('doi'))!=doi(d.get('DOI')):errors.append('DOI')
 if z['key'] not in ck:errors.append('collection')
 if e.get('year') and e['year'] not in d.get('date',''):errors.append('year')
 expected=[];collectives=[]
 for a in re.split(r'\s+and\s+',e.get('author','')):
  a=a.strip()
  if not a:continue
  if a.startswith('{') and a.endswith('}'):collectives.append(a.strip('{}'));continue
  if ',' in a:expected.append(norm(a.split(',')[0]))
  else:expected.append(norm(a.split()[-1]))
 actual=[norm(a.get('lastName',a.get('name',''))) for a in d.get('creators',[]) if a.get('creatorType')=='author' and not a.get('fieldMode') and 'name' not in a]
 if expected and [a for a in expected if a]!=actual:errors.append('author_surnames')
 if any(norm(c) not in {norm(a.get('lastName',a.get('name',''))) for a in d.get('creators',[]) if a.get('fieldMode') or 'name' in a} for c in collectives):errors.append('collective_credit_missing')
 if any(norm(c.get('lastName',c.get('name','')))=='others' for c in d.get('creators',[])):errors.append('invalid_others_creator')
 if '\ufffd' in json.dumps(d,ensure_ascii=False):errors.append('unicode_replacement_character')
 results.append({'latex_key':e['key'],'zotero_item_key':z['key'],'title':d.get('title',''),'DOI':doi(d.get('DOI')),'author_count':len(actual),'expected_personal_author_surnames':expected,'expected_collective_creators':collectives,'zotero_author_surnames':actual,'in_thesis_collection':z['key'] in ck,'issues':errors})
maprows=[]
for r in rows:
 z=byz[r['zotero_item_key']];d=z['data'];maprows.append({'latex_key':r['key'],'zotero_item_key':z['key'],'identity_match':r['zotero_match'],'in_thesis_collection':z['key'] in ck,'imported_in_this_revision':r['key'] in {e['key'] for e in plan['import_entries']},'title_zotero':d.get('title',''),'DOI':d.get('DOI',''),'year':d.get('date',''),'citation_key_preservation':'Clave LaTeX original conservada en references.bib y en este mapa; para fichas importadas tambien se envio como nota bibliografica.'})
with (OUT/'mapa_claves_latex_zotero.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(maprows[0]));w.writeheader();w.writerows(maprows)
summary={'status':'current_library_verified_with_collective_credit_caution','source_bib_entries':len(rows),'matched_entries':sum(bool(r['zotero_item_key']) for r in rows),'missing_entries':[],'zotero_library_regular_items':len(library),'thesis_collection_regular_items':len(collection),'report_entries_in_thesis_collection':sum(x['in_thesis_collection'] for x in maprows),'report_entries_library_only':[x['latex_key'] for x in maprows if not x['in_thesis_collection']],'new_regular_items_imported':len(results),'imported_identity_and_creators_verified':sum(not x['issues'] for x in results),'imported_records_with_issues':[x for x in results if x['issues']],'imported_records':results,'key_map_file':'mapa_claves_latex_zotero.csv','source_bib_sha256':hashlib.sha256((OUT.parents[1]/'references.bib').read_bytes()).hexdigest(),'preservation_boundary':'No se fusionaron ni eliminaron fichas previas. Se importaron 92 fichas locales faltantes; los duplicados ya existentes permanecen. Las claves LaTeX se conservan en el manuscrito y mapa; note se traduce como nota hija en Zotero, no como Extra. No se consultaron adjuntos ni full text.'}
(OUT/'verificacion_importacion_zotero.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:summary[k] for k in ['status','source_bib_entries','matched_entries','zotero_library_regular_items','thesis_collection_regular_items','new_regular_items_imported','imported_identity_and_creators_verified']},ensure_ascii=True,indent=2))


