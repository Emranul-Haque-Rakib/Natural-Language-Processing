import nltk
import re
import heapq
import numpy as np
paragraph="""A sub-committee of the AL is now working in full swing to complete a draft of the manifesto following directives of party chief and Prime Minister Sheikh Hasina, according to two of its senior leaders.
The BNP also has assigned a group of experts to draft its manifesto though the party has yet to formally announce if it will contest the upcoming polls, said sources.
In the draft, the AL sub-committee is highlighting major development projects the government has implemented since 2009. The manifesto will list some ongoing mega projects to persuade people to vote for the party again for “continuation of development”. 
Considering the number of young voters, the party seeks to target them with some slogans like "Power of the Youth is the Prosperity of Bangladesh" and "New Bangladesh for the Young Generation", said the AL sources.
"""

dataset=nltk.sent_tokenize(paragraph)
for i in range(len(dataset)):
    dataset[i]=dataset[i].lower()
    dataset[i]=re.sub(r'\W',' ',dataset[i])
    dataset[i]=re.sub(r'S+',' ',dataset[i]) 
    
##creating the histogram

word2count={}
for data in dataset:
    words=nltk.word_tokenize(data)
    for word in words:
        if word not in word2count.keys():
            word2count[word]=1
        else:
            word2count[word]+=1
            
##finding most frequent words
frequent_words=heapq.nlargest(10,word2count,key=word2count.get)
print(                    "printing........IDF"              )
print("--------------------------------------------------------")
print("")
##Inverse document frequency
## IDF= Number of documents(sentences)/number of documents containing the word
word_idfs={}
for word in frequent_words: ##will go through frequent words
        doc_count=0
        for data in dataset: ##will go through every sentence
            if word in nltk.word_tokenize(data): ##finding the word in words of that sentences
                doc_count+=1
            
        word_idfs[word]=np.log(len(dataset)/doc_count)  ## caluclating the IDF

print(word_idfs)
print("")

##Term frequencyt Matrix
## TF=Number of occurences of a word in a document/number of words in that document

print(                        "printing.....TF"        )
print("-------------------------------------------------")
print("")

tf_matrix={}
for word in frequent_words:
    doc_tf=[]
    for data in dataset:   ## will go through every sentence
        frequency=0
        for w in nltk.word_tokenize(data):  ## will go through every word in the sentence
            if w==word:     ## finding the frequent word in the words of that sentence
                frequency+=1 ## storing the frequency of a word in a sentence
                
        tf_word=frequency/len(nltk.word_tokenize(data))  ## calculating the TF for one sentence(will calculate one by one)
        doc_tf.append(tf_word) ## appending the result for every sentences

    tf_matrix[word]=doc_tf

print(tf_matrix)
print(" ")

##TF-IDF MATRIX
print("PRINTING THE TF-TDF MATRIX")
print("--------------------------")
print("")
tfidf_matrix=[]
for word in tf_matrix.keys(): ## taking the value of those word
    tfidf=[]
    for value in tf_matrix[word]: ##Going through each value
        score=value*word_idfs[word] ## multiplying tf values with IDF
        tfidf.append(score)
        
    tfidf_matrix.append(tfidf)
    
print(tfidf)
        
         
x=np.asarray(tfidf_matrix)
print(x)

print(" ")

x=np.transpose(x)
print(x)



