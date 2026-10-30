"""Educational bag-of-words classifier trained only on fictional utterances."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

def tokens(text): return re.findall(r"[^\W_]+",text.casefold(),flags=re.UNICODE)

class Classifier:
    def __init__(self,alpha=1.0):
        if not math.isfinite(alpha) or alpha<=0: raise ValueError('alpha must be positive')
        self.alpha=alpha; self.documents=Counter(); self.counts={}; self.vocab=set()

    def fit(self,rows):
        self.documents=Counter(); self.counts={}; self.vocab=set()
        for row in rows:
            label=row['label']; words=tokens(row['text'])
            if not label or not words: raise ValueError('each training row needs label and tokens')
            self.documents[label]+=1
            self.counts.setdefault(label,Counter()).update(words)
            self.vocab.update(words)
        if len(self.documents)<2: raise ValueError('at least two classes required')
        return self

    def predict(self,text,threshold=.65):
        if not self.vocab: raise ValueError('model is not fitted')
        if not 0<=threshold<=1: raise ValueError('threshold must be 0..1')
        words=[w for w in tokens(text) if w in self.vocab]
        if not words: return dict(label='unknown',confidence=0.0,probabilities={})
        scores={}
        total=sum(self.documents.values())
        for label in sorted(self.documents):
            counts=self.counts[label]
            denominator=sum(counts.values())+self.alpha*len(self.vocab)
            scores[label]=math.log(self.documents[label]/total)+sum(math.log((counts[w]+self.alpha)/denominator) for w in words)
        maximum=max(scores.values())
        weights={label:math.exp(value-maximum) for label,value in scores.items()}
        norm=sum(weights.values()); probs={k:v/norm for k,v in weights.items()}
        best=max(probs,key=probs.get); confidence=probs[best]
        return dict(label=best if confidence>=threshold else 'unknown',confidence=confidence,probabilities=probs)

    def save(self,path):
        if not self.vocab: raise ValueError('model is not fitted')
        Path(path).write_text(json.dumps(dict(version=1,alpha=self.alpha,documents=dict(self.documents),
            counts={k:dict(v) for k,v in self.counts.items()}),indent=2)+'\n')

    @classmethod
    def load(cls,path):
        obj=json.loads(Path(path).read_text())
        if obj.get('version')!=1: raise ValueError('unsupported model version')
        model=cls(obj['alpha']); docs=obj['documents']; counts=obj['counts']
        if len(docs)<2 or set(docs)!=set(counts): raise ValueError('invalid class set')
        if any(type(v) is not int or v<=0 for v in docs.values()): raise ValueError('invalid document counts')
        for label,words in counts.items():
            if not words or any(not isinstance(k,str) or not k or type(v) is not int or v<=0 for k,v in words.items()):
                raise ValueError('invalid token counts')
        model.documents=Counter(docs); model.counts={k:Counter(v) for k,v in counts.items()}
        model.vocab=set().union(*(set(v) for v in model.counts.values()))
        return model

def evaluate(model,rows,threshold=.65):
    if not rows: raise ValueError('evaluation set is empty')
    predictions=[dict(text=r['text'],expected=r['label'],**model.predict(r['text'],threshold)) for r in rows]
    labels=sorted(set(model.documents)|{r['label'] for r in rows})
    confusion={label:{guess:0 for guess in labels+['unknown']} for label in labels}
    for row in predictions: confusion[row['expected']][row['label']]+=1
    f1=[]
    for label in labels:
        tp=confusion[label][label]
        fp=sum(confusion[actual][label] for actual in labels if actual!=label)
        fn=sum(count for guess,count in confusion[label].items() if guess!=label)
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return dict(accuracy=sum(r['expected']==r['label'] for r in predictions)/len(rows),
                macro_f1=sum(f1)/len(f1),coverage=sum(r['label']!='unknown' for r in predictions)/len(rows),
                confusion=confusion,predictions=predictions)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path(__file__).parent/'samples/intents.json')
    parser.add_argument('--text',help='classify one utterance instead of evaluating')
    parser.add_argument('--threshold',type=float,default=.65)
    parser.add_argument('--save',type=Path)
    args=parser.parse_args(); data=json.loads(args.data.read_text())
    train_text={r['text'].casefold() for r in data['train']}
    if any(r['text'].casefold() in train_text for r in data['test']): raise ValueError('train/test text overlap')
    model=Classifier().fit(data['train'])
    if args.save: model.save(args.save)
    result=model.predict(args.text,args.threshold) if args.text is not None else evaluate(model,data['test'],args.threshold)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__': main()
