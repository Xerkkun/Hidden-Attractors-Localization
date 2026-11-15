from pathlib import Path
import re,json,csv,collections,unicodedata,urllib.request,hashlib,shutil
BASE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
OUT.mkdir(parents=True,exist_ok=True)
# Los metadatos de lectura y busqueda se distribuyen junto a este auditor.
# No se requiere ningun directorio temporal ni se escribe fuera de BASE.
required_cache=['seleccion_bibliografica_verificada.json']
for name in required_cache:
 if not (OUT/name).exists():raise FileNotFoundError('Falta cache bibliografica distribuida: '+str(OUT/name))

def comments(s):return re.sub(r'(?<!\\)%[^\n]*',lambda m:' '*len(m[0]),s)
def skip(s,i):
 while i<len(s) and s[i].isspace():i+=1
 return i
def control(s,i):
 m=re.match(r'\\([A-Za-z@]+|.)',s[i:]);return (m[1],i+m.end()) if m else ('',i+1)
def brace(s,i,opening='{',closing='}'):
 i=skip(s,i)
 if i>=len(s) or s[i]!=opening:return '',i
 depth=1;j=i+1
 while j<len(s) and depth:
  if s[j]=='\\':_,j=control(s,j);continue
  if s[j]==opening:depth+=1
  if s[j]==closing:depth-=1
  j+=1
 return s[i+1:j-1],j

def parse_bib(path):
 s=path.read_text(encoding='utf-8-sig');entries=[];i=0
 while m:=re.search(r'@(\w+)\s*\{\s*([^,]+),',s[i:]):
  start=i+m.start();typ,key=m[1],m[2].strip();a=i+m.end();depth=1;j=a
  while j<len(s) and depth:
   if s[j]=='\\':_,j=control(s,j);continue
   if s[j]=='{':depth+=1
   elif s[j]=='}':depth-=1
   j+=1
  body=s[a:j-1];fields={};p=0
  while fm:=re.search(r'(\w+)\s*=\s*',body[p:]):
   f=fm[1].lower();a2=p+fm.end()
   if body[a2:a2+1]=='{':v,b=brace(body,a2)
   elif body[a2:a2+1]=='"':
    b=a2+1
    while b<len(body) and (body[b]!='"' or body[b-1]=='\\'):b+=1
    v=body[a2+1:b];b+=1
   else:
    b=body.find(',',a2)
    if b<0:b=len(body)
    v=body[a2:b].strip()
   fields[f]=v;p=b+1
  entries.append({'key':key,'type':typ,'line':s.count('\n',0,start)+1,**fields});i=j
 return entries

def norm(s):
 s=re.sub(r'\\[\'"`^~=.uvHck]\s*\{?([A-Za-z])\}?',r'\1',str(s or ''))
 s=s.replace(r'\l','l').replace(r'\L','L').replace(r'\&','&')
 s=unicodedata.normalize('NFKD',s).lower()
 return re.sub(r'[^a-z0-9]','',s)
def doi(s):return re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)','',str(s or '').strip(),flags=re.I).lower()

class CitationWalk:
 def __init__(self,study):
  self.study=study;self.macros={'HAFOPrintEdition':('1',0)} if study else {};self.flags={'HAFOPedagogy':study,'HAFOdigitalcontent':not study,'HAFOBasinMaterial':not study};self.scopes=[];self.cites=[];self.files=[];self.warnings=[];self.expansions=0
 def defined(self,n):return n in self.macros or n.startswith('if') and n[2:] in self.flags
 def scan_file(self,path):
  path=path.resolve()
  if not path.exists():self.warnings.append('missing_input:'+str(path));return
  self.files.append(str(path.relative_to(BASE)));s=comments(path.read_text(encoding='utf-8-sig'))
  if path.name=='reporte_maestro_atractores_ocultos.tex':
   start=s.find(r'\begin{document}')
   if start>=0:s='\n'*s.count('\n',0,start)+s[start:]
  self.scan(s,str(path.relative_to(BASE)))
 def scan(self,s,source):
  i=0;conditions=[]
  def active():return all(conditions)
  while i<len(s):
   if s[i]!='\\':i+=1;continue
   pos=i;cmd,i=control(s,i)
   if cmd=='ifdefined':
    j=skip(s,i);n,i=control(s,j);conditions.append(self.defined(n));continue
   if cmd=='ifnum':
    m=re.match(r'\s*\\(\w+)\s*([=<>])\s*(-?\d+)\s*(?:\\relax)?',s[i:])
    if not m:self.warnings.append('unknown_ifnum:'+source);conditions.append(True);continue
    i+=m.end();value=self.macros.get(m[1],('0',0))[0]
    try:v=int(value)
    except ValueError:v=0
    conditions.append({'=':v==int(m[3]),'<':v<int(m[3]),'>':v>int(m[3])}[m[2]]);continue
   if cmd.startswith('if') and cmd[2:] in self.flags:conditions.append(self.flags[cmd[2:]]);continue
   if cmd=='else':
    if not conditions:self.warnings.append('unmatched_else:'+source)
    else:conditions[-1]=not conditions[-1]
    continue
   if cmd=='fi':
    if not conditions:self.warnings.append('unmatched_fi:'+source)
    else:conditions.pop()
    continue
   if cmd in ['newcommand','providecommand','renewcommand','def']:
    i=skip(s,i)
    if s[i:i+1]=='*':i+=1;i=skip(s,i)
    if s[i:i+1]=='{':name,i=brace(s,i);name=name.lstrip('\\')
    else:name,i=control(s,i)
    argc=0;i=skip(s,i)
    if s[i:i+1]=='[':arg,i=brace(s,i,'[',']');argc=int(arg) if arg.isdigit() else 0;i=skip(s,i)
    if s[i:i+1]=='[':_,i=brace(s,i,'[',']');i=skip(s,i)
    if cmd=='def':
     while i<len(s) and s[i]!='{':i+=1
    body,i=brace(s,i)
    if active() and (cmd!='providecommand' or name not in self.macros):self.macros[name]=(body,argc)
    continue
   if not active():continue
   if cmd=='newif':
    name,i=control(s,skip(s,i));self.flags[name.removeprefix('if')]=False;continue
   if cmd.endswith('true') and cmd[:-4] in self.flags:self.flags[cmd[:-4]]=True;continue
   if cmd.endswith('false') and cmd[:-5] in self.flags:self.flags[cmd[:-5]]=False;continue
   if cmd=='begingroup':self.scopes.append((self.macros.copy(),self.flags.copy()));continue
   if cmd=='endgroup':
    if self.scopes:self.macros,self.flags=self.scopes.pop()
    continue
   if cmd in ['input','include']:
    name,i=brace(s,i);p=BASE/name
    if not p.suffix:p=p.with_suffix('.tex')
    self.scan_file(p);continue
   if cmd in ['cite','citep','citet','citealp','nocite']:
    i=skip(s,i)
    while s[i:i+1]=='[':_,i=brace(s,i,'[',']');i=skip(s,i)
    keys,i=brace(s,i)
    for key in keys.split(','):
     if key.strip():self.cites.append({'key':key.strip(),'file':source,'line':s.count('\n',0,pos)+1})
    continue
   if cmd=='begin':
    env,j=brace(s,i)
    if env in ['lstlisting','verbatim']:
     end=s.find('\\end{'+env+'}',j);i=end+len('\\end{'+env+'}') if end>=0 else len(s)
    continue
   if cmd in self.macros and cmd.startswith('HAFO'):
    body,argc=self.macros[cmd];self.expansions+=1
    if self.expansions>10000:raise RuntimeError('Macro expansion budget exceeded')
    for n in range(1,argc+1):arg,i=brace(s,i);body=body.replace('#'+str(n),arg)
    self.scan(body,source+' [macro '+cmd+']');continue
  if conditions:self.warnings.append('unclosed_condition:'+source+':'+str(len(conditions)))

walks={}
for label,study in [('monografia',False),('guia_estudio',True)]:
 w=CitationWalk(study);w.scan_file(BASE/'reporte_maestro_atractores_ocultos.tex');walks[label]=w
entries=parse_bib(BASE/'references.bib');by_key=collections.defaultdict(list)
for e in entries:by_key[e['key']].append(e)

def api(route):
 with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:23119/api/users/0'+route,headers={'Zotero-API-Version':'3'}),timeout=20) as response:return json.load(response)
collection=[];start=0
while True:
 page=api('/collections/II82RCL4/items/top?limit=100&start='+str(start)+'&include=data');collection.extend(x for x in page if x['data'].get('itemType') not in ['attachment','note','annotation'])
 if len(page)<100:break
 start+=100
library=[];start=0
while True:
 page=api('/items/top?limit=100&start='+str(start)+'&include=data');library.extend(x for x in page if x['data'].get('itemType') not in ['attachment','note','annotation'])
 if len(page)<100:break
 start+=100
collection_keys={x['key'] for x in collection}
selected=json.loads((OUT/'seleccion_bibliografica_verificada.json').read_text(encoding='utf-8'))['papers'];readings={doi(r['doi']):r for r in selected}; readings_by_key={r['key']:r for r in selected}
records=[];cited={label:{r['key'] for r in w.cites} for label,w in walks.items()};potential=collections.defaultdict(list)
for f in BASE.rglob('*.tex'):
 if 'auditoria' in f.parts:continue
 text=comments(f.read_text(encoding='utf-8-sig'))
 for m in re.finditer(r'\\cite\s*(?:\[[^]]*\]\s*)?\{([^}]+)\}',text):
  for key in m[1].split(','):potential[key.strip()].append(str(f.relative_to(BASE))+':'+str(text.count('\n',0,m.start())+1))
for key in sorted(set(by_key)|set().union(*cited.values())):
 es=by_key.get(key,[]);e=es[0] if es else {};d=doi(e.get('doi',''));t=norm(e.get('title',''))
 dm=[z for z in library if d and doi(z['data'].get('DOI',''))==d];tm=[z for z in library if t and norm(z['data'].get('title',''))==t];matches=dm or tm
 matches.sort(key=lambda z:(z['key'] not in collection_keys,norm(z['data'].get('title',''))!=t,z['key']))
 z=matches[0] if matches else None;zd=z['data'] if z else {}
 deltas=[]
 if z:
  if t!=norm(zd.get('title','')):deltas.append('titulo_diferente_revisar')
  if d and doi(zd.get('DOI',''))!=d:deltas.append('DOI_discrepante')
  if not d and zd.get('DOI'):deltas.append('DOI_ausente_en_bib')
 read=readings.get(d) or readings_by_key.get(key)
 if read:
  rs=read['reading_scope'];level='primaria_texto_selectivo' if ('Texto arXiv' in rs or 'Texto primario PDF' in rs or 'Texto editorial completo' in rs or 'PDF arXiv' in rs) else 'primaria_resumen_o_metadatos'
 else:rs='Sin lectura cientifica nueva registrada en esta revision; el cotejo con Zotero es de metadatos.';level='solo_cotejo_metadatos_Zotero' if z else 'lectura_no_verificada_en_esta_revision'
 editions=[label for label in cited if key in cited[label]]
 records.append({'key':key,'entry_count':len(es),'bib_line':e.get('line',''),'title_bib':e.get('title',''),'doi_bib':d,'year_bib':e.get('year',''),'cited_monografia':key in cited['monografia'],'cited_guia_estudio':key in cited['guia_estudio'],'edition_scope':'comun' if len(editions)==2 else editions[0] if editions else 'no_citada_en_recorrido','static_occurrences':len(potential.get(key,[])),'citation_locations':'; '.join(potential.get(key,[])),'zotero_match':'DOI' if dm else 'titulo' if tm else 'sin_correspondencia','zotero_item_key':z['key'] if z else '','zotero_in_thesis_collection':bool(z and z['key'] in collection_keys),'zotero_candidate_keys':'; '.join(q['key'] for q in matches),'title_zotero':zd.get('title',''),'doi_zotero':doi(zd.get('DOI','')),'metadata_differences':'; '.join(deltas),'reading_status_this_revision':level,'reading_scope':rs,'primary_url':read['primary_url'] if read else ''})
keydups={k:[e['line'] for e in es] for k,es in by_key.items() if len(es)>1};doigroups=collections.defaultdict(list)
for e in entries:
 if doi(e.get('doi','')):doigroups[doi(e['doi'])].append(e['key'])
doigroups={k:v for k,v in doigroups.items() if len(v)>1}
librarydois=collections.defaultdict(list)
for z in library:
 if doi(z['data'].get('DOI','')):librarydois[doi(z['data']['DOI'])].append(z['key'])
missing={label:sorted(keys-set(by_key)) for label,keys in cited.items()}
summary={'scope':'Inventario completo de references.bib; recorrido estatico reconciliado con AUX de compilacion cuando existen; cotejo actual de metadatos Zotero. No implica lectura cientifica integral.','references_bib_sha256':hashlib.sha256((BASE/'references.bib').read_bytes()).hexdigest(),'bib_entries':len(entries),'unique_bib_keys':len(by_key),'cited_keys_by_edition':{k:len(v) for k,v in cited.items()},'common_cited_keys':len(cited['monografia']&cited['guia_estudio']),'only_monografia':sorted(cited['monografia']-cited['guia_estudio']),'only_guia_estudio':sorted(cited['guia_estudio']-cited['monografia']),'missing_bib_keys':missing,'duplicate_bib_keys':keydups,'duplicate_bib_dois':doigroups,'zotero_collection_items':len(collection),'zotero_library_regular_items':len(library),'matching_zotero_collection':sum(r['zotero_in_thesis_collection'] for r in records),'matching_zotero_library_only':sum(bool(r['zotero_item_key']) and not r['zotero_in_thesis_collection'] for r in records),'not_in_zotero_library':sum(not r['zotero_item_key'] for r in records),'cited_not_in_thesis_collection':sorted(r['key'] for r in records if r['edition_scope']!='no_citada_en_recorrido' and not r['zotero_in_thesis_collection']),'cited_not_in_library':sorted(r['key'] for r in records if r['edition_scope']!='no_citada_en_recorrido' and not r['zotero_item_key']),'metadata_discrepancies':[{k:r[k] for k in ['key','title_bib','doi_bib','zotero_item_key','title_zotero','doi_zotero','metadata_differences']} for r in records if r['metadata_differences']],'reading_coverage':dict(collections.Counter(r['reading_status_this_revision'] for r in records)),'duplicate_zotero_dois_for_report_entries':{d:keys for d,keys in librarydois.items() if len(keys)>1 and any(doi(e.get('doi',''))==d for e in entries)},'walker_warnings':{label:w.warnings for label,w in walks.items()},'walked_files':{label:sorted(set(w.files)) for label,w in walks.items()},'edition_citations':{label:sorted(v) for label,v in cited.items()},'this_audit_script_zotero_write_operations':0}
summary['aux_reconciliation']={}
for label,stem in [('monografia','reporte_maestro_atractores_ocultos'),('guia_estudio','reporte_maestro_lectura_estudio')]:
 aux=BASE/'build'/(stem+'.aux')
 if aux.exists():
  auxtext=aux.read_text(encoding='utf-8-sig')
  auxkeys={k.strip() for group in re.findall(r'\\citation\{([^}]+)\}',auxtext) for k in group.split(',') if k.strip()}
  summary['aux_reconciliation'][label]={'aux_file':str(aux.relative_to(BASE)),'aux_sha256':hashlib.sha256(aux.read_bytes()).hexdigest(),'aux_unique_citations':len(auxkeys),'static_minus_aux':sorted(cited[label]-auxkeys),'aux_minus_static':sorted(auxkeys-cited[label]),'missing_from_bib':sorted(auxkeys-set(by_key)),'exact_match':auxkeys==cited[label]}
 else:summary['aux_reconciliation'][label]={'available':False,'caution':'AUX de compilacion no disponible; solo recorrido estatico.'}
summary['audited_key_rows']=len(records)
summary['bib_entries_matching_thesis_collection']=sum(r['entry_count']>0 and r['zotero_in_thesis_collection'] for r in records)
summary['bib_entries_matching_library_only']=sum(r['entry_count']>0 and bool(r['zotero_item_key']) and not r['zotero_in_thesis_collection'] for r in records)
summary['bib_entries_not_in_library']=sum(r['entry_count']>0 and not r['zotero_item_key'] for r in records)
summary['citation_keys_missing_from_bib']=sorted(set().union(*cited.values())-set(by_key))
summary['uncited_bib_keys_by_edition']={label:sorted(set(by_key)-keys) for label,keys in cited.items()}
summary['zotero_library_duplicate_doi_groups']={d:keys for d,keys in librarydois.items() if len(keys)>1}
summary['zotero_library_duplicate_doi_groups_matching_report']={d:keys for d,keys in librarydois.items() if len(keys)>1 and any(doi(e.get('doi',''))==d for e in entries)}
summary['reading_scope_caution']='No se atribuye lectura integral ni verificacion de demostraciones a entradas sin evidencia de lectura primaria selectiva. El cotejo de metadatos con Zotero no certifica exactitud cientifica editorial.'

with (OUT/'bibliografia_completa_auditoria.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
(OUT/'bibliografia_completa_resumen.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'citas_por_edicion.json').write_text(json.dumps({label:w.cites for label,w in walks.items()},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:summary[k] for k in ['bib_entries','unique_bib_keys','cited_keys_by_edition','common_cited_keys','only_monografia','only_guia_estudio','missing_bib_keys','duplicate_bib_keys','duplicate_bib_dois','matching_zotero_collection','matching_zotero_library_only','not_in_zotero_library','metadata_discrepancies','reading_coverage','walker_warnings']},ensure_ascii=True,indent=2))
