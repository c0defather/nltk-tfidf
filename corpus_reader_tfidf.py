from nltk import FreqDist
from math import log, sqrt
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


class CorpusReader_TFIDF:

    # constructs a CorpusReader_TFIDF object with given attributes
    def __init__(self, corpus=None, tf="raw",
                 idf="inverse", stopword=set(stopwords.words('english')),
                 stemmer=PorterStemmer(), ignorecase=True):
        # ==== Constructor arguments ==== #
        self.corpus = corpus
        self.tf = tf
        self.idf = idf
        if stopword == "none":
            self.stopword = []
        elif isinstance(stopword, str):
            text_file = open(stopword)
            self.stopword = set(text_file.read().split(' '))
        else:
            self.stopword = stopword
        self.stemmer = stemmer
        self.ignorecase = ignorecase
        self.stopword = set([w.lower() if ignorecase else w for w in self.stopword])

        # ==== New properties of class ==== #
        self.tfidf = []  # matrix of tf-idf values
        self.tf_values = []  # matrix of tf values
        self.idf_values = []  # vector of idf_values
        self.fileids = corpus.fileids()     # list of all file ids
        self.words = []                     # list of all distinct words
        self.dictionary = dict()  # map of all distinct words their index in [words]
        self.filtered_corpus = dict()       # filtered corpus after stemming and removing stopwords

        self.preprocess()   # preprocess corpus and calculate tf-idf

    # Processes corpus and calculates tf-idf matrix.
    #   1. Filter corpus:
    #       For every word apply stemming function, stopwords check, etc. and save
    #       the results in a map, which stores pairs of <preprocessed word>,<processed word>.
    #       Use it to avoid redundant calls of stem().
    #   2. Calculate TF matrix:
    #       For every document calculate frequencies of terms and store indexes
    #       of tf values that are greater than 0 (to avoid redundant calculations in sparse matrix).
    #   3. Calculate IDF vector:
    #       Use calculated frequencies matrix, since it contains only non-zero frequencies.
    #   4. Calculate TF-IDF matrix:
    #       Iterate through cells that have non-zero TF values because of sparseness.
    def preprocess(self):
        words_cnt = 0   # total number of distinct words in the corpus
        docs_cnt = len(self.fileids)    # total number of documents in the corpus
        filtered_words = dict()     # map of <word>,<word after filtering>
        for fileid in self.fileids:
            self.filtered_corpus[fileid] = dict()

        # ==== Filter corpus ==== #
        for fileid in self.fileids:
            for word in self.corpus.words(fileid):
                pre_word = word     # word before filtering
                if pre_word in filtered_words:
                    word = filtered_words[pre_word]
                else:
                    if self.ignorecase:
                        word = word.lower()
                    word = self.stemmer.stem(word)      # apply stemming
                    filtered_words[pre_word] = word
                if word in self.stopword:       # skip a word if it is in the stopwords list
                    continue
                if word in self.filtered_corpus[fileid]:
                    self.filtered_corpus[fileid][word] += 1
                else:
                    self.filtered_corpus[fileid][word] = 1
                if word not in self.dictionary:
                    self.dictionary[word] = words_cnt       # word-index pair
                    self.words.append(word)
                    words_cnt += 1

        # ==== Calculate TF matrix ==== #
        nonzeros = []  # vectors of indexes of non zero tf values for all documents
        for fileid in self.fileids:
            tf_vector = [0] * words_cnt     # initialize tf vector with 0s
            nonzeros_cur = []   # stores indexes of non zero tf values for the document
            for word in self.filtered_corpus[fileid].keys():
                f = self.filtered_corpus[fileid][word]
                tf_vector[self.dictionary[word]] = f
                if f > 0:
                    nonzeros_cur.append(self.dictionary[word])
            nonzeros.append(nonzeros_cur)
            self.tf_values.append(tf_vector)

        # ==== Calculate IDF vector ==== #
        self.idf_values = [0] * words_cnt   # initialize idf vector with 0s
        for fileid in self.fileids:
            for word in self.filtered_corpus[fileid].keys():
                self.idf_values[self.dictionary[word]] += 1

        # ==== Calculate TF-IDF matrix ==== #
        for i in range(docs_cnt):
            vector = [0] * words_cnt
            for j in nonzeros[i]:
                tf = self.tf_values[i][j]
                idf = self.idf_values[j]
                vector[j] = self.weight(tf, idf, docs_cnt)
            self.tfidf.append(vector)
        return

    # Returns tf-idf product for a given number of total documents
    # and pairs of term frequency with inverse document frequency.
    def weight(self, tf_value, idf_value, num_docs):
        if self.tf == "log" and tf_value != 0:
            tf_value = 1 + log(tf_value, 2)
        elif self.tf == "binary":
            tf_value = 0 if tf_value == 0 else 1

        if self.idf == "probabilistic":
            if num_docs == idf_value:
                idf_value = 0
            else:
                idf_value = log((num_docs - idf_value) / float(idf_value), 2)
        elif self.idf == "smoothed":
            idf_value = log(1 + num_docs / float(idf_value), 2)
        else:
            idf_value = log(num_docs / float(idf_value), 2)
        return tf_value * idf_value

    # Return a list of ALL tf-idf vector (each vector should be a list)
    # for the corpus, ordered by the order where fields are returned
    # (the dimensions of the vector can be arbitrary,
    # but need to be consistent with all vectors).
    # If fileid is given, return the tf-idf vector corresponding to that file.
    # If filelist is given, return a list of vectors,
    # corresponding to the tf-idf to the list of fileid input.
    def tf_idf(self, fileid=None, filelist=None):
        if fileid and filelist:
            print "Pass only one argument"
        elif fileid:
            return self.tfidf[self.fileids.index(fileid)]
        elif filelist:
            result = []
            for fileid in filelist:
                result.append(self.tfidf[self.fileids.index(fileid)])
            return result
        else:
            return self.tfidf

    # return the list of the words corresponding to each vector of the tf-idf vector
    def tf_idf_dim(self):
        return self.words

    # The input should be a list of words (treated as a document).
    # Return a vector corresponding to the tf_idf vector for the new document
    # (with the same stopword, stemming, ignorecase treatment applied, if necessary).
    # You should use the idf for the original corpus to calculate the result
    # (i.e. do not treat the document as a part of the original corpus).
    def tf_idf_new(self, words):
        words_cnt = len(self.words)     # number of distinct words in corpus
        docs_cnt = len(self.fileids)        # number of documents in corpus
        # calculating tf vector
        tf_vector = [0] * words_cnt  # initialize tf vector with 0s
        freq = FreqDist(words)  # word frequencies
        for word in freq.keys():
            if self.ignorecase:
                word = word.lower()
            word = self.stemmer.stem(word)
            if word in self.stopword:
                continue
            if word in self.dictionary:
                tf_vector[self.dictionary[word]] = freq[word]

        # calculating tf-idf vector
        result = []
        for j in range(words_cnt):
                tf = tf_vector[j]
                idf = self.idf_values[j]
                result.append(self.weight(tf, idf, docs_cnt))
        return result

    # Return the cosine similarity between two documents in the corpus.
    def cosine_sim(self, fileid):
        v1 = self.tfidf[self.fileids.index(fileid[0])]
        v2 = self.tfidf[self.fileids.index(fileid[1])]
        return self.cosine_sim_vec(v1,v2)

    # The [words] is a list of words as is in the parameter of  tf_idf_new() method.
    # The fileid is the document in the corpus. The function return the cosine
    # similarity between fileid and the new document specify by the [words] list.
    # (Once again, use the idf of the original corpus).
    def cosine_sim_new(self, words, fileid):
        v1 = self.tf_idf_new(words)
        v2 = self.tfidf[self.fileids.index(fileid)]
        return self.cosine_sim_vec(v1, v2)

    # Return cosine similarity between two vectors.
    @staticmethod
    def cosine_sim_vec(v1, v2):
        sum_aa, sum_bb, sum_ab = 0, 0, 0
        for i in range(len(v1)):
            a = v1[i]
            b = v2[i]
            sum_aa += a * a
            sum_bb += b * b
            sum_ab += a * b
        return sum_ab / sqrt(sum_aa * sum_bb)
