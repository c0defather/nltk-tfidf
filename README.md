# nltk-tfidf
Simple class for calculation TF-IDF matrix for a corpus
## Example
Print cosine similarities between all pairs of documents in the corpus:
```python
from nltk import corpus
reader = CorpusReaderTFIDF(corpus=corpus.brown)
docs_cnt = len(reader.fileids)
for i in range(docs_cnt):
    for j in range(i, docs_cnt):
        f1 = reader.fileids[i]
        f2 = reader.fileids[j]
        print f1, f2, ':', reader.cosine_sim([f1, f2])
```
## CorpusReaderTFIDF class
### Constructor arguments:
* corpus (required): a corpus object
* tf (keyword): the method used to calculate term frequency. Default is raw frequency
* idf(keyword): the method used to calculate inverse document frequency. Default is inverse frequency
* stopword (keyword): if specified as “none”, then do not remove any stopwords. Otherwise this should treat as a filename where stopwords are to be read. Default is using the standard English stopwords corpus in NLTK. You should assume any word inside the stopwords file is a stopword. Otherwise you should not assume any pre-defined format of the stopword file. 
* stemmer (keyword): the stemming function to be used. Default is to use the Porter stemmer (nltk.stem.porter)
* ignorecase (keyword) if specified as “no”, then do NOT ignore case. Default is ignore case
### Methods
* tf_idf(): return a list of ALL tf-idf vector for the corpus. 
* tf_idf(fileid=fileid): return the tf-idf vector corresponding to that file.
* tf_idf(filelist=[fileid]): return a list of vectors, corresponding to the tf-idf to the list of fileid input
* tf_idf_dim(): return the list of the words corresponding to each vector of the tf-idf vector
* tf_idf_new([words]): return a vector corresponding to the tf_idf vector for the new document. 
* cosine_sim(fileid=[fileid1, fileid2]): return the cosine similarity between two documents in the corpus
* cosine_sim_new([words], fileid): return the cosine similarity between fileid and the new document specified by the [words] list.
