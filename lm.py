#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import collections
from math import log
import sys

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)

class LangModel:
        
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence_to_filter_words(s)
        self.threshold()
        #print(corpus)
        self.update_corpus(corpus)
        #print(corpus)
        for s in corpus:
            self.fit_sentence(s)
        print('Vocab Size: ', len(self.vocab()))
        
    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        return pow(2.0, self.entropy(corpus))

    def entropy(self, corpus):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 1 # for EOS
            sum_logprob += self.logprob_sentence(s)
        return -(1.0/num_words)*(sum_logprob)

    def logprob_sentence(self, sentence):
        p = 0.0
        for i in xrange(len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i])
        p += self.cond_logprob('end_of_sentence', sentence)
        return p
        
    # finds the words with their counts for filtering afterwards
    def fit_sentence_to_filter_words(self, sentence): pass
    # for words with count < threshold_frequency
    # replace their count by incrementing count of UNK
    def threshold(self): pass
    # removes the filtered word from the corpus 
    # replaces them with UNKs to update corpus
    def update_corpus(self, corpus): pass   
    # finds the words, word pairs and triplets with their counts
    def fit_sentence(self): pass
    # return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous): pass
    # required, the list of words the language model suports 
    # including EOS and UNK
    def vocab(self): pass
    # prints model state
    def printModel(self): pass
    

class Unigram(LangModel):
    
    def __init__(self, backoff = 0.000001):
        self.threshold_frequency = 4
        self.deleted_words = [] 
        self.before_filter = dict()
        self.total_number_of_words = 0
        self.count1 = dict()
        self.lbackoff = log(backoff, 2)
        self.alpha = 0.9
        
    def inc_word_before_filter(self, w):
        if w in self.before_filter:
            self.before_filter[w] += 1.0
        else:
            self.before_filter[w] = 1.0
        
    def fit_sentence_to_filter_words(self, sentence):
        for w in sentence:
            self.inc_word_before_filter(w.lower()) 
        self.inc_word_before_filter('end_of_sentence')
        
    def threshold(self):
        self.before_filter['UNK'] = 0.0
        for word in self.before_filter:
            if (self.before_filter[word] <= self.threshold_frequency):
                self.before_filter['UNK'] += self.before_filter[word]
                self.deleted_words.append(word.lower())
    
    def update_corpus(self, corpus):
        i = 0
        self.number_of_sentences = len(corpus)
        for sentence in corpus:
            if i % 10000 == 0:
                print('Unigram Model: Sentence ' + str(i) + ' of ' + str(len(corpus)) + ' sentences')
            j = 0;
            for word in sentence:
                if word.lower() in self.deleted_words:
                    corpus[i][j] = 'unk'
                j += 1
            i += 1
        
    def inc_word(self, w):
        if w in self.count1:
            self.count1[w] += 1.0
        else:
            self.count1[w] = 1.0
            
    def fit_sentence(self, sentence):
        for w in sentence:
            self.inc_word(w.lower()) 
        self.inc_word('end_of_sentence')
        self.total_number_of_words += len(sentence) + 1

    def cond_logprob(self, word, previous):
        if word not in self.count1:
            word = 'unk'
        return log(self.count1[word] + self.alpha, 2) - log(self.total_number_of_words + self.alpha * len(self.vocab()), 2)

    def vocab(self):
        return self.count1.keys()

    def printModel(self):
        print(self.count1)
        

class Bigram(LangModel):
    
    def __init__(self, backoff = 0.000001):
        #modify here
        self.threshold_frequency = 4
        self.deleted_words = [] 
        self.before_filter = dict()
        self.number_of_sentences = 0
        self.count1 = dict() 
        self.count2 = dict() 
        self.lbackoff = log(backoff, 2)
        self.alpha = 0.9

    def inc_word_before_filter(self, w):
        if w in self.before_filter:
            self.before_filter[w] += 1.0
        else:
            self.before_filter[w] = 1.0
    
    def fit_sentence_to_filter_words(self, sentence):
        for w in sentence:
            self.inc_word_before_filter(w.lower()) # language model is case insensitive, 'The' is the same as 'the'
        self.inc_word_before_filter('end_of_sentence')
    
    def threshold(self):
        self.before_filter['UNK'] = 0.0
        for word in self.before_filter:
            if (self.before_filter[word] <= self.threshold_frequency):
                self.before_filter['UNK'] += self.before_filter[word]
                self.deleted_words.append(word.lower())
    
    def update_corpus(self, corpus):
        i = 0
        self.number_of_sentences = len(corpus)
        for sentence in corpus:
            if i % 10000 == 0:
                print('Bigram Model: Sentence ' + str(i) + ' of ' + str(len(corpus)) + ' sentences')
            j = 0;
            for word in sentence:
                if word.lower() in self.deleted_words:
                    corpus[i][j] = 'unk'
                j += 1
            i += 1
            
    def inc_2word(self, w):
        if w in self.count2:
            self.count2[w] += 1.0
        else:
            self.count2[w] = 1.0
            
    def inc_word(self, w):
        if w in self.count1:
            self.count1[w] += 1.0
        else:
            self.count1[w] = 1.0

    def fit_sentence(self, sentence):
        i = 0
        for w in sentence:
            if i == 0:
                self.inc_2word('* ' + w.lower()) 
            else:
                self.inc_2word(sentence[i-1].lower() + ' ' + w.lower())
            i += 1
        self.inc_2word(sentence[i-1].lower() + ' end_of_sentence')

        # self.inc_word('*')
        for w in sentence:
            self.inc_word(w.lower()) 
        self.inc_word('end_of_sentence')
        
    def cond_logprob(self, word, previous):
        word = word.lower()
        if word not in self.count1:
            word = 'unk'
            
        if len(previous) >= 1:
            prev = previous[len(previous) - 1]
            prev = prev.lower
            if prev not in self.count1:
                prev = 'unk'
        else:
            prev = '*'
        
        pair = str(prev) + ' ' + str(word)
        if pair in self.count2 and prev in self.count1:
            nume = self.count2[pair]
            deno = self.count1[prev]
            return log(nume, 2) - log(deno, 2)
        else:
            if prev in self.count1:
                deno = self.count1[prev]
                return log(self.alpha, 2) - log(deno + self.alpha * len(self.vocab()), 2)
            else:
                return log(self.alpha, 2) - log(self.alpha * len(self.vocab()), 2)
            
    def vocab(self):
        return self.count1.keys()
    
    def printModel(self):
        print(self.count1)
        print(self.count2)
        
        
class Trigram(LangModel):
    
    def __init__(self, backoff = 0.000001):
        #modify here
        self.threshold_frequency = 4 # filter to replace words with count < threshold by UNK
        self.deleted_words = [] # to keep track of words with count < threshold_frequency
        self.before_filter = dict() # word:count before filtering
        self.number_of_sentences = 0
        self.count1 = dict() # word:count after filtering
        self.count2 = dict() # word pair:count
        self.count3 = dict() # word triplet:count
        self.lbackoff = log(backoff, 2)
        self.alpha = 0.9 # Laplace smoothing parameter
    
    # keeps a count of the words seen in the original corpus        
    def inc_word_before_filter(self, w):
        if w in self.before_filter:
            self.before_filter[w] += 1.0
        else:
            self.before_filter[w] = 1.0
    
    # finds the words with their counts in the original corpus, before filtering
    def fit_sentence_to_filter_words(self, sentence):
        for w in sentence:
            self.inc_word_before_filter(w.lower()) # language model is case insensitive, 'The' is the same as 'the'
        self.inc_word_before_filter('end_of_sentence')
    
    # deletes words with count < threshold_frequency 
    # inserts UNK in the dictionary with its count
    # keeps track of the deleted words
    def threshold(self):
        #insert UNK into the dictionary
        self.before_filter['UNK'] = 0.0
        for word in self.before_filter:
            if (self.before_filter[word] <= self.threshold_frequency):
                # increment count of UNK by the count of the filtered word
                self.before_filter['UNK'] += self.before_filter[word]
                # maintain a list of the filtered words
                self.deleted_words.append(word.lower())
    
    # updates the corpus with UNK
    def update_corpus(self, corpus):
        i = 0
        self.number_of_sentences = len(corpus)
        for sentence in corpus:
            if i % 10000 == 0:
                print('Trigram Model: Sentence ' + str(i) + ' of ' + str(len(corpus)) + ' sentences')
            j = 0;
            for word in sentence:
                # if the word is a filtered word i.e. count < threhold frequency
                if word.lower() in self.deleted_words:
                    # place unk wherever the filtered word is found in the corpus and update the corpus
                    corpus[i][j] = 'unk'
                j += 1
            i += 1
        
    # keeps a count of word triplets
    def inc_3word(self, w):
        if w in self.count3:
            self.count3[w] += 1.0
        else:
            self.count3[w] = 1.0
    
    # keeps a count of word pairs       
    def inc_2word(self, w):
        if w in self.count2:
            self.count2[w] += 1.0
        else:
            self.count2[w] = 1.0
    
    # keeps a count of words in the updated corpus       
    def inc_word(self, w):
        if w in self.count1:
            self.count1[w] += 1.0
        else:
            self.count1[w] = 1.0

    #count every word triplets, pairs and single words that appear in the updated corpus
    def fit_sentence(self, sentence):
        # count word triplets
        i = 0
        for w in sentence:
            # for first word, prepend two *'s
            if i == 0:
                self.inc_3word('* * ' + w.lower()) # language model is case insensitive, 'The' is the same as 'the'
            # for the second word, prepend a *
            elif i == 1:
                self.inc_3word('* ' + sentence[i-1].lower() + ' ' + w.lower())
            else:
                self.inc_3word(sentence[i-2].lower() + ' ' + sentence[i-1].lower() + ' ' + w.lower())
            i += 1
        # append 'end of sentence'
        self.inc_3word(sentence[i-2].lower() + ' ' + sentence[i-1].lower() + ' end_of_sentence')
        
        # count word pairs
        i = 0
        self.inc_2word('* *')
        for w in sentence:
            # for the first word, prepend a *
            if i == 0:
                self.inc_2word('* ' + w.lower()) # language model is case insensitive, 'The' is the same as 'the'
            else:
                self.inc_2word(sentence[i-1].lower() + ' ' + w.lower())
            i += 1
        # append 'end of sentence' 
        self.inc_2word(sentence[i-1].lower() + ' end_of_sentence')
        
        # count words
        # self.inc_word('*')
        for w in sentence:
            self.inc_word(w.lower()) # language model is case insensitive, 'The' is the same as 'the'
        self.inc_word('end_of_sentence')
    
    # computes and returns the log probabilities for word triplets
    def cond_logprob(self, word, previous):
        word = word.lower()
        # if word not seen in training data, replace with unk
        if word not in self.count1:
            word = 'unk'
            
        # prepare the bigram upon which the trigram will be conditioned on
        if len(previous) >= 2:
            prev = previous[len(previous) - 2 : len(previous)]
            a = prev[0].lower()
            b = prev[1].lower()
            if a not in self.count1:
                a = 'unk'
            if b not in self.count1:
                b = 'unk'
            prev = a + ' ' + b
        elif len(previous) == 1:
            prev = previous[len(previous) - 1]
            a = prev.lower()
            if a not in self.count1:
                a = 'unk'
            prev = '* ' + a 
        else:
            prev = '* *'
        
        # prepare the trigram
        pair = str(prev) + ' ' + str(word)
        
        # if the trigram and the bigram are observed during training
        # return log count(u, v, w) - log count(u, v)
        if pair in self.count3 and prev in self.count2:
            nume = self.count3[pair] # count of trigram 
            deno = self.count2[prev] # count of bigram
            return log(nume, 2) - log(deno, 2)
        # else do Laplace smoothing
        else:
            # if the bigram is observed during training
            # return log(alpha) - log(count(u, v) + alpha*vocab_size)
            if prev in self.count2:
                deno = self.count2[prev]
                return log(self.alpha, 2) - log(deno + self.alpha * len(self.vocab()), 2)
            # if neither trigram nor bigram is seen
            # return log(alpha) - log(alpha*vocab_size)
            else:
                return log(self.alpha, 2) - log(self.alpha * len(self.vocab()), 2)
        
    # return the set of unique words in the updated corpus
    def vocab(self):
        return self.count1.keys()
    
    # prints the word, word pair and triplet dictionary
    def printModel(self):
        print(self.count1)
        print(self.count2)
        print(self.count3)
