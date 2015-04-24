te(sentence):
    count = 0
    newword = " "
    while count < len(sentence):
          if "!" in sentence[count] or "'" in sentence[count] or "?" in sentence[count]:
              count = count + 1
              continue
          else:
              newword = newword + sentence[count]
              count = count + 1
    print newword


unpunctuate("Hey there! How's it going?")

--

from collections import Counter


counts = Counter()

def get_bag_of_words_for_single_document(sentence):
    counts.update(word.strip('.,?!"\'').lower() for word in sentence.split())
    print counts
    
get_bag_of_words_for_single_document("John also likes to watch football games to.")
print counts.elements()

--

import collections

counts = collections.Counter()
newdict = []

sentenceList = ["John likes to watch movies. Mary likes movies too.",
    "John also likes to watch football games.",] 

def get_bag_of_words(sentences):
    for sentence in sentences:
        counts.update(word.strip('.,?!"\'').lower() for word in sentence.split())
    print counts


def removeduplicates(seq): 
   checked = []
   for word in seq:
       if word not in checked:
           checked.append(word)
   return checked

def turn_words_into_indices(collectcount):
    print removeduplicates(sort(list(collectcount.elements())))

    
get_bag_of_words(sentenceList)

turn_words_into_indices(counts)

--

string2 = "hey there you are here."

string2.append("blah")

print string2

--

import re

wordlist = ["also", "football", "games", "John", "likes", "Mary", "movies", "to", "too", "watch"]

dict1 = {}
madelist = []

for i in range(len(wordlist)):
    dict1[wordlist[i].lower()] = 0
    

    
    
    
def vectorize(document):
    altdocument = prepString(document)
    print altdocument
    for m in wordlist:
        for phrase in re.finditer(" "+m+" ", altdocument):
            dict1[m] = dict1[m] + 1
    for key in sorted(dict1.iterkeys()):
        madelist.append(dict1[key])
    print madelist
    
def prepString(samplestr):
    newword = " "
    newstring = " "
    for word in samplestr:
        newstring = newstring + word.strip('.,?!"\'').lower()
    return newstring
    
    
vectorize("The sun also football rises like watch football? Hey Mary! in football movies ")


