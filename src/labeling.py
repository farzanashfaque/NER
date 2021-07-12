from tqdm import tqdm
import en_core_web_sm
import spacy
from spacy.matcher import PhraseMatcher
from spacy.lang.en import English


class Label:
    
    def __init__(self, data, entity_dict):
        self.data = data
        self.entity_dict = entity_dict
        self.nlp = en_core_web_sm.load(disable = ['ner', 'tagger', 'lemmatizer', 'textcat'])
        self.nlp2 = English()

    def get_labels(self):
        nlp = self.nlp2
        matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
        labels = []
        phrase_list = list(self.entity_dict.keys())
        patterns = [nlp.make_doc(phrase) for phrase in phrase_list]
        matcher.add('Phrase', None, *patterns)

        for sentence in tqdm(self.data, total=len(self.data)):
            sent_labels = ['O']*len(sentence)
            assert(len(sentence)==len(sent_labels))

            sentence_joined = ' '.join(sentence)
            doc = nlp(sentence_joined)
            matches = matcher(doc)

            for _, start, end in matches:
                try:
                    span = doc[start:end]
                    sent_labels[start:end] = [self.entity_dict[span.text]]*(end-start)
                except:
                    pass

            #assert(len(sentence)==len(sent_labels))
            labels.append(sent_labels)
        return labels

    @staticmethod
    def find_sub_list(sl,l):
        results=[]
        sll=len(sl)
        for ind in (i for i,e in enumerate(l) if e==sl[0]):
            if l[ind:ind+sll]==sl:
                results.append((ind,ind+sll))

        return results

    def label_data(self):
        nlp = self.nlp2
        labels = []
        for l in tqdm(self.data, total=len(self.data)):
            label = ['O']*len(l)
            for key in list(self.entity_dict.keys()):
                sl = [token.text for token in nlp(key)]
                result = self.find_sub_list(sl, l)
                for start, end in result:
                    label[start:end] = [self.entity_dict[key]]*(end-start)
            labels.append(label)
        return labels
    
    def dependency_parse(self):
        compound = []
        for sentence in tqdm(self.data, total=len(self.data)):
            row = []
            sentence = ' '.join(sentence)
            parsed = self.nlp(sentence)
            i = 0
            while i<len(parsed):
                token = parsed[i]
                dep = token.dep_
                if dep == 'compound':
                    num = token.head.i - token.i + 1
                    row.extend([1]*num)
                    i += num
                else:
                    row.append(0)
                    i += 1
            compound.append(row)
        return compound
    
    def relabel(self, labels, dependency):
        for i in tqdm(range(len(labels))):
            row = labels[i]
            dep_row = dependency[i]
            label_changes_sent = []
            flag = 0
            start = -1
            end = -1
            label = 'O'
            for j in range(len(row)):
                if dep_row[j] == 0:
                    if flag == 1:
                        end = j
                        flag = 0
                else:
                    if flag == 0:
                        start = j
                        flag = 1
                        if row[j] != 'O':
                            label = row[j]
                    else:
                        if j == len(row)-1:
                            end = j+1
                            flag = 0
                        if row[j] != 'O':
                            label = row[j]
                if start != end and flag == 0:
                    label_changes_sent.append((start, end, label))
                    label = 'O'
                    start = -1
                    end = -1
            for item in label_changes_sent:
                start = item[0]
                end = item[1]
                label = item[2]
                labels[i][start:end] = [label]*(end-start)
        
        return labels


    
            


        

        

        
        



