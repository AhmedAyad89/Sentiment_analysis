import json
from utils import  *
import spacy

#data = parse_tokenized_comments_json('DataSet/organic/tokenized/en.json', sentence_level=False)
seeds={'good', 'great', 'excellent', 'amazing', 'bad', 'terrible', 'poor', 'cheap', 'expensive',\
	 'fast', 'reliable', 'heavy', 'love', 'hate', 'like', 'dislike', 'enjoy'}

data = json.load(open("DataSet/organic/tokenized/en.json"))
comments=[]
nlp = spacy.load('en_core_web_lg')

spacy_comments=[]
for line in data[:20]:
	for comment in line['comments']:
		comments.append(comment['comment_text'])
		spacy_comments.append(nlp(comment['comment_text']))
print(len(comments))

for token in spacy_comments[0]:
	print(line)
	print('\n---')


