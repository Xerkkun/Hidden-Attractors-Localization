from pathlib import Path
import json,urllib.request,urllib.parse,urllib.error,time,concurrent.futures,difflib,re,unicodedata,csv
p=Path(__file__).resolve().parent
plan=json.loads((p/'plan_importacion_referencias_locales.json').read_text(encoding='utf-8'));cache=json.loads((p/'metadatos_crossref_verificados.json').read_text(encoding='utf-8'))
cache={k.lower():v for k,v in cache.items() if not v.get('error')}
def norm(s):
 s=re.sub(r'\\[\'"`^~=.uvHck]\s*\{?([A-Za-z])\}?',r'\1',str(s or ''));s=s.replace(r'\l','l').replace(r'\L','L').replace(r'\&','&')
 return re.sub(r'[^a-z0-9]','',unicodedata.normalize('NFKD',s).lower())
def read_json(url):
 req=urllib.request.Request(url,headers={'User-Agent':'BibliographyAudit/1.0 (local academic reference check)'})
 with urllib.request.urlopen(req,timeout=15) as f:return json.load(f)
def fetch(doi):
 if doi.lower() in cache:return cache[doi.lower()]|{'registry':'Crossref'}
 err=None
 for attempt in range(3):
  try:
   m=read_json('https://api.crossref.org/works/'+urllib.parse.quote(doi,safe=''))['message']
   return {k:m[k] for k in ['DOI','title','author','type','published','published-print','volume','issue','page','article-number','URL'] if k in m}|{'registry':'Crossref'}
  except urllib.error.HTTPError as e:
   err=str(e)
   if e.code==404:break
   if e.code==429:time.sleep(0.8*(attempt+1));continue
   break
  except Exception as e:err=str(e);break
 try:
  a=read_json('https://api.datacite.org/dois/'+urllib.parse.quote(doi,safe=''))['data']['attributes']
  return {'DOI':a['doi'],'title':[t['title'] for t in a.get('titles',[])],'author':a.get('creators',[]),'type':a.get('types'),'published':a.get('publicationYear'),'URL':a.get('url'),'registry':'DataCite'}
 except Exception as e:return {'error':str(e),'crossref_error':err}
entries=plan['import_entries'];dois=sorted({e['doi'] for e in entries if e.get('doi')})
print('DOIs to validate',len(dois),'cached',sum(d.lower() in cache for d in dois),flush=True)
registered={}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for d,m in zip(dois,pool.map(fetch,dois)):registered[d]=m
rows=[]
for e in entries:
 d=e.get('doi','');m=registered.get(d,{})
 actual=m.get('title',[''])[0] if isinstance(m.get('title'),list) and m.get('title') else ''
 score=difflib.SequenceMatcher(None,norm(e.get('title','')),norm(actual)).ratio() if actual else 0
 if not d:status='sin_DOI_metadatos_locales_ISBN'
 elif m.get('error'):status='DOI_no_verificado'
 elif norm(e.get('title',''))==norm(actual):status='DOI_titulo_coinciden'
 else:status='DOI_titulo_diferente_revisar'
 rows.append({'key':e['key'],'doi':d,'local_title':e.get('title',''),'registry_title':actual,'registry':m.get('registry',''),'status':status,'similarity':round(score,3),'metadata':m})
(p/'cotejo_registral_preimportacion.json').write_text(json.dumps({'entries':len(entries),'doi_checked':len(dois),'items':rows},ensure_ascii=False,indent=2),encoding='utf-8')
with (p/'cotejo_registral_preimportacion.csv').open('w',encoding='utf-8-sig',newline='') as f:
 cols=['key','doi','local_title','registry_title','registry','status','similarity'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:r[k] for k in cols} for r in rows)
from collections import Counter
print(json.dumps(Counter(r['status'] for r in rows)))
print(json.dumps([r for r in rows if r['status'] not in ['DOI_titulo_coinciden','sin_DOI_metadatos_locales_ISBN']],ensure_ascii=True,indent=2))
