import pyLDAvis.gensim_models
import spacy
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from spacy.tokens import DocBin
from tqdm import tqdm

from data import read_bbc_csv

df = read_bbc_csv()

df.describe()

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("merge_noun_chunks")

# add stop words
add_stop_words = ['say', '\s', 'mr', 'Mr', 'said', 'says', 'saying', 'today', 'be', 'I']
for stopword in add_stop_words:
    lexeme = nlp.vocab[stopword]
    lexeme.is_stop = True

# We add some words to the stop word list
articles, article = [], []
doc_bin = DocBin()
dictionary = Dictionary()
corpus = []

# clean data
print('Processing articles using spaCy, removing punctuation, stop words, numbers and lemminizing')
for doc in tqdm(nlp.pipe(df.text.values, disable=["tok2vec"])):
    doc_bin.add(doc)
    article = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.like_num]
    dictionary.add_documents([article])
    corpus.append(dictionary.doc2bow(article))

print('Building LDA model')
lda_model = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)
lda_model.show_topics()

pyLDAvis.enable_notebook()
model = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
