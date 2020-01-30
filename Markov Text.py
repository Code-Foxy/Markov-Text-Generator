import string
import random
import math

def tokenize(text):
    
    newStr = ''

    #for the text, we will surround any characters which are defined as punctuation in spaces
    for char in text:
        
        if char in string.punctuation:
            char = ' ' + char + ' '
            
        newStr += char

    #we split this new string to optain our list of tokens
    return newStr.split()

def ngrams(n, tokens):
    
    listOut = []
    
    #add <END> to the end of the tokens list
    tokens.append('<END>')

    #create a list of n-1 <START>s and extend the tokens list over it
    paddedTokens = ['<START>'] * (n-1)

    paddedTokens.extend(tokens)
    tokenCounter = 0

    #generate our ngrams for the given set and return
    while tokenCounter <= len(paddedTokens)-1:
        
        if paddedTokens[tokenCounter] != '<START>':
            listOut.append( ( tuple(paddedTokens[tokenCounter - (n-1):tokenCounter]) , paddedTokens[tokenCounter] ) )
            
        tokenCounter += 1

    return listOut
    

class NgramModel(object):

    def __init__(self, n):
        self.n = n
        self.contextCount = {}
        self.pairCount = {}
        self.contextToTokens = {}

    def update(self, sentence):

        #tokenize the sentence for counting
        toCount = ngrams( self.n, tokenize(sentence) )

        #get and store the counts of the context and the context-token pairing for probability computation
        #store these in dictionaries for fast lookup
        for context, token in toCount:
            
            if context in self.contextCount:
                self.contextCount[context] += 1
            else:
                self.contextCount[context] = 1

            if ( '(' + str(context) + ',' + token + ')' ) in self.pairCount:
                self.pairCount['(' + str(context) + ',' + token + ')'] += 1
            else:
                self.pairCount['(' + str(context) + ',' + token + ')'] = 1

            if context in self.contextToTokens:
                self.contextToTokens[context].append(token)
            else:
                self.contextToTokens[context] = [token]

    def prob(self, context, token):

        #if the pairing exists, return the probabily
        if ('(' + str(context) + ',' + token + ')') in self.pairCount:
            return self.pairCount['(' + str(context) + ',' + token + ')']/self.contextCount[context]
        else: return 0.0

    def random_token(self, context):

        rand = random.random()
        possibleTokens = []
        tokenCounter = 0

        #find all the tokens associated with the given context and list them
        if context in self.contextToTokens:
            possibleTokens = list(set(self.contextToTokens[context]))

        #sort the list
        possibleTokens.sort()

        #set the initial sums (this covers the case where the first token would be selected)
        firstSum = 0
        secondSum = self.prob(context, possibleTokens[tokenCounter])

        #compare our probability sums to the random value
        while tokenCounter < len(possibleTokens):
            
            if firstSum <= rand and rand < secondSum:
                return possibleTokens[tokenCounter]

            tokenCounter += 1
            #firstSum is now the second sum
            firstSum = secondSum
            #secondSum is incremented by the next probability value
            secondSum += self.prob(context, possibleTokens[tokenCounter])
            

    def random_text(self, token_count):
        
        stringOut = []
        #generate our context for the first token
        context = ('<START>',) * (self.n - 1)
        
        for i in range(token_count):
            #pull a token from random
            token = self.random_token(context)
            stringOut.append(token)

            #if n = 1, then the context = () so don't change anything
            if token != ('<END>') and self.n != 1:
                context = context[1:] + (token,)
            #if <END> was found, we will reset our context to the starting value
            else:
                context = ('<START>',) * (self.n - 1)

        #join our list with spaces to give the string output
        return ' '.join(stringOut)

    def perplexity(self, sentence):

        denom = 0
        tokens = tokenize(sentence)
        nGrams = ngrams(self.n, tokens)

        #in log-space, the product in the denominator becomes a sum
        for context, token in nGrams:
            denom += math.log(self.prob(context, token))

        #take the fraction out of log-sapce and compute the m root of it
        return (1/math.exp(denom)) ** ((1.)/(len(tokens)))
        

def create_ngram_model(n, path):

    #we are just opening the file and reading in each line
    file = open(path, "r")
    m = NgramModel(n)
    
    for line in file:
        m.update(line)

    return m
