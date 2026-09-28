from collections import defaultdict

def fault_safety(result):
    if not result or result.get('episodes')!=56 or result.get('mode')!='argmax':return False
    rows=result.get('results',[])
    if len(rows)!=56 or len({r['case_id'] for r in rows})!=56:return False
    groups=defaultdict(list)
    for row in rows:groups[row['case_id'].rsplit('-',1)[0]].append(row)
    expected={'pose-noise','pose-delay','pose-mild','pose-lost','reply-lost','request-lost','channel-delay'}
    if set(groups)!=expected or any(len(v)!=8 for v in groups.values()):return False
    if any(r['reason'] in ('collision','out_of_bounds') for r in rows):return False
    for name,reason in [('pose-lost','localization_lost'),('reply-lost','command_unknown'),('request-lost','command_unknown')]:
        if any(r['reason']!=reason or r['success'] for r in groups[name]):return False
    return True
