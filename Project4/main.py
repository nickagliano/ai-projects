import stemmer as stemmer
import re # for regex
import csv

infile = open('Project4_sentences/stop_words.txt', 'r')

# build regex for stop words
stop_word_regex = ''
firstLineFlag = 1;
while 1:
	line = infile.readline()
	if line == '':
		break
	if (firstLineFlag):
		stop_word_regex += "\\b" + line.strip() + "\\b"
		firstLineFlag = 0
	else:
		stop_word_regex += "|\\b" + line.strip() + "\\b"
infile.close()

infile = open('Project4_sentences/sentences.txt', 'r')
outfile = open('Project4_sentences/clean_sentences.txt', 'w')
while 1:
	line = infile.readline()
	if line == '':
		break
	temp = re.sub('[^a-z\s]', ' ' , line.lower()) # remove numbers, symbols, special chars
	temp = re.sub(stop_word_regex, '' , temp.lower()) # remove stop words
	outfile.write(temp)
infile.close()
outfile.close()


# # stem sentences
s = stemmer.PorterStemmer();

stemmedSentences = s.nick('Project4_sentences/clean_sentences.txt')

tokenizedSentences = []
for x in stemmedSentences:
	tokens = re.split("\s+", x.strip()) # split into words
	tokenizedSentences.append(tokens)


globalWords = []
globalOccurances = []

for sentence in tokenizedSentences:
	wordInSentence = []
	wordOccurancesInSentence = []
	for word in sentence:
		if word in wordInSentence: # word is already in local sentence list
			index = wordInSentence.index(word)
			wordOccurancesInSentence[index] += 1
		else:	# word needs to be added to local sentence list
			wordInSentence.append(word)
			wordOccurancesInSentence.append(1)

		if word in globalWords: # word is already in global word list
			index = globalWords.index(word)
			globalOccurances[index] += 1
		else: # word needs to be added to global word list
			globalWords.append(word)
			globalOccurances.append(1)

# print(sentence)
# print(wordInSentence)
# print(wordOccurancesInSentence)
# print(globalOccurances)

i = 0
# for (word, count) in zip(globalWords, globalOccurances):
# while i < len(globalWords):
# 	if globalOccurances[i] < 4:
# 		globalWords[i] = None
# 		globalOccurances[i] = 0
# 	# print(str(globalWords[i]) + ' ' + str(globalOccurances[i]))
# 	i+=1

# globalWords = list(filter(lambda a: a != None, globalWords))
# globalOccurances = list(filter(lambda a: a != 0, globalOccurances))
# print(globalWords)
# print(globalOccurances)

# print(len(globalWords))


with open('Project4_data/feature_vector.csv', mode='w') as file:
	writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	writer.writerow(globalWords)
	# for sentence in tokenizedSentences:
	# 	wordInSentence = []
	# 	wordOccurancesInSentence = []
	# 	for word in sentence:
	# 		if word in wordInSentence: # word is already in local sentence list
	# 			index = wordInSentence.index(word)
	# 			wordOccurancesInSentence[index] += 1
	# 		else:	# word needs to be added to local sentence list
	# 			wordInSentence.append(word)
	# 			wordOccurancesInSentence.append(1)
	# 	newRow = [0] * len(globalWords)
	# 	for (word, count) in zip(wordInSentence, wordOccurancesInSentence):
	# 		if word in globalWords:
	# 			index = globalWords.index(word)
	# 			newRow[index] = count
	# 	writer.writerow(newRow)
