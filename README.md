# bulk-labeling-solara
A tool for bulk labeling, built in Solara!

I'm trying to rebuild my original [bulk-labeling](https://github.com/rungalileo/bulk-labeling/) app, which was Streamlit, in [Solara](https://github.com/widgetti/solara) so it can be a bit more scalable, customizable, and robust to new features!

I also want to learn how to use solara :) 


## Development
1. Setup a virtual env: `python -m venv .venv && source .venv/bin/activate`
2. Install the package: `pip install -e . && pyenv rehash`
3. Run: `solara run bulk_labeling/main.py`

Any changes you make to the app should reflect in realtime

### Note: `SentenceTransformers` doesn't play nicely with solara
If you are going to be developing, I strongly recommend commenting out
the few lines in [ml.py](utils/ml.py)
```
from sentence_transformers import SentenceTransformer
...
ENCODER = SentenceTransformer("paraphrase-MiniLM-L3-v2")
...
return ENCODER.encode(samples)
```
and uncomment
```
# return np.random.rand(len(samples), 20)
```

For some reason, on a page reload, solara breaks if these lines are running.  
It will also make prototyping faster because you won't be actually encoding strings.
