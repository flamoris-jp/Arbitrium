"""Decision diagnostic metrics: explicit denominators and undefined support."""
from .postprocess import postprocess,LABELS
from maidionis_education.evaluation import rate

def diagnostic(rows,raws):
    if len(rows)!=len(raws):raise ValueError('complete diagnostics required')
    confusion=[[0]*3 for _ in LABELS];support=[0]*3;predsupport=[0]*3;correct=0;answers=0;uncorrect=0;un=0;brier=0.;buckets=[[] for _ in range(10)]
    for row,raw in zip(rows,raws):
        t=row['target'];p=postprocess(raw);y=p['diagnostic_label'];q=p['answerability'];predans=q>=.5
        answers+=predans==t['answerable']
        if not t['answerable']:un+=1;uncorrect+=not predans;continue
        yi=LABELS.index(t['label']);pi=LABELS.index(y);confusion[yi][pi]+=1;support[yi]+=1;predsupport[pi]+=1;correct+=yi==pi
        probs=[v['probability'] for v in p['probabilities']];brier+=sum((v-(i==yi))**2 for i,v in enumerate(probs));c=max(probs);buckets[min(9,int(c*10))].append((c,yi==pi))
    n=sum(support);f1=[]
    for i in range(3):
        denom=support[i]+predsupport[i];f1.append(2*confusion[i][i]/denom if support[i] and denom else None)
    ece=sum(len(b)/n*abs(sum(v[0] for v in b)/len(b)-sum(v[1] for v in b)/len(b)) for b in buckets if b) if n else None
    return dict(records=len(rows),answerable_support=n,decision_accuracy=correct/n if n else None,macro_f1=sum(f1)/3 if all(v is not None for v in f1) else None,
        label_support=dict(zip(LABELS,support)),label_recall={k:confusion[i][i]/support[i] if support[i] else None for i,k in enumerate(LABELS)},confusion=confusion,brier=brier/n if n else None,ece=ece,
        answerability_accuracy=answers/len(rows) if rows else None,unanswerable_support=un,unanswerable_recall=uncorrect/un if un else None,calibration='not fitted; raw diagnostic probabilities')

def reduce_predictions(predictions):
    n=sum(p['target']['answerable'] for p in predictions)
    result=[rate('decision.accuracy',sum(p['target']['answerable'] and postprocess(p['raw_output'])['diagnostic_label']==p['target']['label'] for p in predictions),n,denominator='answerable admitted samples',exclusions=len(predictions)-n),
            rate('answerability.accuracy',sum((postprocess(p['raw_output'])['answerability']>=.5)==p['target']['answerable'] for p in predictions),len(predictions))]
    for k in LABELS:
        subset=[p for p in predictions if p['target']['answerable'] and p['target']['label']==k]
        result.append(rate('recall.'+k,sum(postprocess(p['raw_output'])['diagnostic_label']==k for p in subset),len(subset),slice=k))
    subset=[p for p in predictions if not p['target']['answerable']]
    result.append(rate('unanswerable.recall',sum(postprocess(p['raw_output'])['answerability']<.5 for p in subset),len(subset),slice='unanswerable'))
    return result
