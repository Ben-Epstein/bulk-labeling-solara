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
the few lines in [ml.py](bulk_labeling/utils/ml.py):
https://github.com/Ben-Epstein/bulk-labeling-solara/blob/8281f618c33e298a0bb5de373b0087a49d58e938/bulk_labeling/utils/ml.py#L5
https://github.com/Ben-Epstein/bulk-labeling-solara/blob/8281f618c33e298a0bb5de373b0087a49d58e938/bulk_labeling/utils/ml.py#L9
https://github.com/Ben-Epstein/bulk-labeling-solara/blob/8281f618c33e298a0bb5de373b0087a49d58e938/bulk_labeling/utils/ml.py#L13

and uncomment
https://github.com/Ben-Epstein/bulk-labeling-solara/blob/8281f618c33e298a0bb5de373b0087a49d58e938/bulk_labeling/utils/ml.py#L15

For some reason, on a page reload, solara breaks if these lines are running.  
It will also make prototyping faster because you won't be actually encoding strings.
