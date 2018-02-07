from nltk import corpus
from sys import stdout
from corpus_reader_tfidf import CorpusReaderTFIDF


# outputs tfidf matrix and cosine similarities for a given corpus
def process(corpus_reader, name):
    print name
    # print first 15 words
    for i in corpus_reader.tf_idf_dim()[:15]:
        stdout.write(i)
        stdout.write(' ')
    stdout.write('\n')
    # print tf-idf matrix for first 15 words
    for i in range(len(corpus_reader.tf_idf())):
        vector = corpus_reader.tf_idf()[i]
        stdout.write(corpus_reader.fileids[i] + ', ')
        vector = vector[:15]
        for v in vector:
            print round(v, 4),
        stdout.write('\n')
    # print cosine similarities
    docs_cnt = len(corpus_reader.fileids)
    for i in range(docs_cnt):
        for j in range(i, docs_cnt):
            f1 = corpus_reader.fileids[i]
            f2 = corpus_reader.fileids[j]
            print f1, f2, ':', round(corpus_reader.cosine_sim([f1, f2]), 4)


process(CorpusReaderTFIDF(corpus=corpus.shakespeare), "shakespeare")