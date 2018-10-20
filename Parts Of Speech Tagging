import nltk
 
paragraph="""Domestic confined any but son bachelor advanced remember. How proceed offered her offence shy forming.
Returned peculiar pleasant but appetite differed she. Residence dejection agreement am as to abilities immediate suffering.
Ye am depending propriety sweetness distrusts belonging collected. Smiling mention he in thought equally musical.
Wisdom new and valley answer. Contented it so is discourse recommend. Man its upon him call mile. An pasture he himself
believe ferrars besides cottage. """

words=nltk.word_tokenize(paragraph)
tagged_words=nltk.pos_tag(words)   ##tokenize pargraph to words

words_tags=[]                     ## made a list
for tw in tagged_words:
    words_tags.append(tw[0]+" "+tw[1])  ## inserting  words and their corresponding POS  append 
                                        ##index which is word and 1 index which is POS
 
    
print(words_tags)



------------------------------------------------------------------OUTPUT--------------------------------------------------------------------


['Domestic JJ', 'confined VBD', 'any DT', 'but CC', 'son NN', 'bachelor NN', 'advanced VBD', 'remember NN', '. 
.', 'How WRB', 'proceed JJ', 'offered VBD', 'her PRP$', 'offence NN', 'shy NN', 'forming NN', '. .', 'Returned VBN',
'peculiar JJ', 'pleasant NN', 'but CC', 'appetite NN', 'differed VBD', 'she PRP', '. .', 'Residence NNP', 'dejection NN',
'agreement NN', 'am VBP', 'as IN', 'to TO', 'abilities NNS', 'immediate JJ', 'suffering NN', '. .', 'Ye NNP', 'am VBP', 
'depending VBG', 'propriety NN', 'sweetness NN', 'distrusts NNS', 'belonging VBG', 'collected VBN', '. .', 'Smiling VBG',
'mention NN', 'he PRP', 'in IN', 'thought VBN', 'equally RB', 'musical JJ', '. .', 'Wisdom NNP', 'new JJ', 'and CC', 'valley JJ',
'answer NN', '. .', 'Contented VBD', 'it PRP', 'so RB', 'is VBZ', 'discourse JJ', 'recommend NN', '. .', 'Man CC', 'its PRP$',
'upon IN', 'him PRP', 'call VB', 'mile NN', '. .', 'An DT', 'pasture NN', 'he PRP', 'himself PRP', 'believe VBP', 'ferrars NNS',
'besides IN', 'cottage NN', '. .']
