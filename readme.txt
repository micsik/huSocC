HuSocC is a multilabel classifier for Hungarian social science interviews.
The labels will be assigned from taxonomy.tsv.
The method uses textacy with HuSpaCy and a mapping of keywords to topics (tax-kw.tsv).
To use it directly on interview text, run
    python husocc.py sample.txt

To use it with Annif, one has to install husocc-backend.py as an Annif backend, and create a new configuration
in projects.cfg like this:

[husocc]
name=HuSocC
language=hu
backend=husocc
analyzer=snowball(hungarian)
vocab=husocc-vocab

Then one can load the vocabulary and then use the other annif commands (suggest, eval, etc.):
    annif load-vocab --language=hu husocc-vocab husocc-vocab.tsv
