import nltk
import re
import heapq
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
            
##finding most 10 frequent words
frequent_words=heapq.nlargest(10,word2count,key=word2count.get)
print(frequent_words)

##Creating Matrix
##IN here there is a list in a list the second list vecctor will contain binary result for one sentence
## and the list X will caontain the whole vector lint in it
x=[]
for data in dataset:
    vector=[]
    for word in frequent_words:
        if word in nltk.word_tokenize(data):
            vector.append(1)
        else:
            vector.append(0)
    x.append(vector)
    
print(x)

            
--------------------------------------------------OUTPUT------------------------------------------------------------------------------

['the', 'of', 'to', 'party', 'a', 'al', 'is', 'draft', 'manifesto', 'has']  ## Most 10 frequent sentences


              [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0],        ## MAtrix represantation of  most 10 frequent words in each sentences
               [1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
               [1, 0, 0, 0, 0, 1, 1, 1, 0, 1],
               [1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
               [1, 1, 1, 1, 0, 1, 1, 0, 0, 0]]
               
