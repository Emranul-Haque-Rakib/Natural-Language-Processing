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

##Inverse document frequency
word_idfs={}
for word in frequent_words:
        doc_count=0
        for data in dataset:
            if word in nltk.word_tokenize(data):
                doc_count+=1
            
        word_idfs[word]=np.log(len(dataset)/doc_count)  ## caluclating the IDF

print(word_idfs)

"""   -------------------------------------------OUTPUT----------------------------------------
    {'the': 0.0, 'of': 0.22314355131420976,
    'to': 0.22314355131420976, 'party': 0.22314355131420976, 
    'a': 0.91629073187415511, 'al': 0.51082562376599072, 
    'is': 0.51082562376599072, 'draft': 0.51082562376599072, 
    'manifesto': 0.51082562376599072, 'has': 0.91629073187415511} 
    
    """
