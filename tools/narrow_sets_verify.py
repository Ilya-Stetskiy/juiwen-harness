import os,sys,json,glob,re,hashlib,collections,statistics,shutil
st='/tmp/nstage'
os.environ['LAB_EXPERIMENTS']=st+'/exp'; os.environ['LAB_WORK']=st+'/work'
sys.path.insert(0,os.path.expanduser('~/harness-lab')); sys.path.insert(0,st)
import lab, stream_run as sr, yaml
EXP=st+'/exp'
names=sorted(os.path.basename(f)[:-5] for f in glob.glob(EXP+'/sets/nrw_*.yaml'))
others=sorted(os.path.basename(f)[:-5] for f in glob.glob(EXP+'/sets/*.yaml') if not os.path.basename(f).startswith('nrw_') and 'adversarial' not in f)
H=os.path.expanduser('~/harness-lab/verify/')
train=set(x.strip() for x in open('/tmp/train_ids.txt') if x.strip())
lab_ids=set(x.strip() for x in open(H+'lab_task_ids.txt') if x.strip())
stream=set(x.strip() for x in open('/tmp/stream_ids.txt') if x.strip())
G={};INS={}
def build(n,keep=False,override=None):
    s=lab.load_set(n); e={k:s[k] for k in lab.SET_KEYS if k in s}
    if override: e.update(override)
    out=lab.WORK/'v'/n
    g=lab.build_task_set(e,out)
    ins={}
    for tid in g:
        t=json.load(open(out/'tasks'/tid/'task.json'))
        ins[tid]=hashlib.md5(re.sub(r'\s+',' ',t['instruction']).encode()).hexdigest()
    shutil.rmtree(out,ignore_errors=True)
    return s,g,ins
meta={}
for n in names:
    s,g,ins=build(n); G[n]=g; INS[n]=ins; meta[n]=s
    if lab.set_hash(g)!=s['frozen']['set_hash']: print('HASH MISMATCH',n)
OG={n:set(build(n)[1]) for n in others}
bad=0
allids={}
for n in names:
    for t in G[n]:
        if t in allids: print('OVERLAP narrow',n,allids[t],t); bad+=1
        allids[t]=n
for n in names:
    ids=set(G[n]); 
    for o,og in OG.items():
        if ids&og: print('OVERLAP old',n,o,len(ids&og)); bad+=1
    if ids&train: print('TRAIN overlap',n,len(ids&train)); bad+=1
    if ids&lab_ids: print('LABIDS overlap',n,len(ids&lab_ids)); bad+=1
    for t,g in G[n].items():
        if g['expected'] is None and not g['unanswerable']: print('NO EXPECTED',t); bad+=1
# duplicate instruction text across sets (same question twice)
hs=collections.defaultdict(list)
for n in names:
    for t,h in INS[n].items(): hs[h].append((n,t))
dup=[v for v in hs.values() if len(v)>1]
print('duplicate instruction texts',len(dup),[[(a,b) for a,b in v][:3] for v in dup[:3]])
print('overlap with stream ids (informational):',{n:len(set(G[n])&stream) for n in names if set(G[n])&stream})
print('BAD',bad)
# reserve: same spec, seed+1000, excluding everything used
open(EXP+'/sets/narrow/exclude_with_narrow.txt','w').write('\n'.join(sorted(set(open(EXP+'/sets/narrow/exclude_ids.txt').read().split())|set(allids)))+'\n')
man={}
for n in names:
    s=meta[n]; nar=s['narrow']
    e_over={'seed':nar['reserve_seed'],'exclude_ids_file':'sets/narrow/exclude_with_narrow.txt','tasks':{s0:max(12,v//2) for s0,v in s['tasks'].items()}}
    q=s.get('unanswerable')
    if q: e_over['unanswerable']={k:(0 if v==0 else max(6,v//2)) for k,v in q.items()}
    res=[]
    for k in range(max(e_over['tasks'].values()),3,-1):
        try:
            e2=dict(e_over); e2['tasks']={s0:k for s0 in s['tasks']}
            if q: e2['unanswerable']={kk:(0 if vv==0 else min(vv,k//2)) for kk,vv in e_over['unanswerable'].items()}
            _,rg,_=build(n,override=e2); res=sorted(rg); break
        except Exception as ex: pass
    # later reserves of other sets must not collide
    srcs=collections.Counter(g['source'] for g in G[n].values())
    man[n]={'type':nar['type'],'subgenre':nar['subgenre'],'n_tasks':len(G[n]),'n_unanswerable':sum(1 for g in G[n].values() if g['unanswerable']),
            'sources':dict(srcs),'set_hash':s['frozen']['set_hash'],'target_cells':nar['target_cells'],'reserve_seed':nar['reserve_seed'],
            'tasks':list(G[n]),'reserve':res,'rewards':dict(collections.Counter(g['reward'] for g in G[n].values()))}
    print(n,len(G[n]),dict(srcs),'reserve',len(res))
# cross check reserves disjoint
rs=collections.Counter(t for v in man.values() for t in v['reserve']); print('reserve dups',sum(1 for c in rs.values() if c>1),'reserve in sets',len(set(rs)&set(allids)))
# cost estimate from stream rows
spec={}
fpred={}
for n in names:
    s=meta[n]; t=list(s['tasks'])[0]; conds=(s.get('select') or {}).get(t)
    for src in s['sources'][t]: fpred.setdefault(src,[]).append((n,conds))
rows=collections.defaultdict(list)
for l in open(os.path.expanduser('~/datasets/token_juiwen/stream/results.jsonl')):
    r=json.loads(l)
    if r.get('hybrid_cost',0)>0: rows[r['task_id']].append(r)
agg=collections.defaultdict(lambda: collections.defaultdict(list))
for src,preds in fpred.items():
    pre=src if src!='wikitablequestions' else 'wtq'
    with open(sr.SRC_PATH[src],encoding='utf-8') as f:
        for line in f:
            m=re.match(r'\{"task_id": "([^"]+)"',line)
            if not m or m.group(1) not in rows: continue
            r=json.loads(line)
            for n,conds in preds:
                if not lab.row_matches(r,conds): continue
                for x in rows[r['task_id']]:
                    a=agg[n]
                    a['cost'].append(x['hybrid_cost']); a['req'].append((x.get('pipeline_calls') or 0)+(x.get('agent_calls') or 0))
                    a['ok'].append(1 if x.get('correct') else 0); a['sec'].append(x.get('seconds') or 0); a['tid'].append(x['task_id'])
                    a['cost_nr'].append(x['hybrid_cost']) if x['model'].endswith('-noreason') and x['model'].startswith('qwen3.6') else None
for n in names:
    a=agg[n]; N=man[n]['n_tasks']
    if not a['cost']: man[n]['estimate']=None; print('no stream data',n); continue
    m=lambda k: statistics.fmean(a[k])
    mc=m('cost'); mr=m('req'); ms=m('sec')
    man[n]['estimate']={'stream_rows':len(a['cost']),'stream_tasks':len(set(a['tid'])),'tokens_per_task':round(mc),'tokens_per_task_noreason35':round(statistics.fmean(a['cost_nr'])) if a['cost_nr'] else None,
      'requests_per_task':round(mr,2),'seconds_per_task':round(ms,1),'stream_accuracy':round(m('ok'),2),
      'tokens_per_run':round(mc*N),'requests_per_run':round(mr*N),'minutes_per_run':round(max(mr*N/55, N*ms/5/60),1)}
    print(n,man[n]['estimate'])
json.dump(man,open('/tmp/nstage/manifest_raw.json','w'))
ss=collections.defaultdict(lambda: collections.defaultdict(list))
for tid,rs_ in rows.items():
    for x in rs_:
        ss[x['source']]['cost'].append(x['hybrid_cost']); ss[x['source']]['req'].append((x.get('pipeline_calls') or 0)+(x.get('agent_calls') or 0))
json.dump({k:{a:statistics.fmean(b) for a,b in v.items()} for k,v in ss.items()},open('/tmp/nstage/stream_src.json','w'))
