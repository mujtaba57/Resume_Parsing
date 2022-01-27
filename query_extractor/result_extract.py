import ast
import html
import re
import sys
import warnings
import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords
if not sys.warnoptions:
    warnings.simplefilter("ignore")
stopwords = stopwords.words('english')


class PreprocessData:

    def __init__(self) -> None:
        self._nlp = spacy.load('en_core_web_sm')
        self.df = ''
        self.key = None
        self.val = None
        self.final_result = None

    @staticmethod
    def sentence_finder(para_data):
        word = ''
        regular_exp = r"[^.][a-z]"
        for sentence in para_data.split(" "):
            if re.findall(regular_exp, sentence):
                word += sentence.replace(".", "@") + " "
            else:
                word += sentence + " "
        return word

    def clean_dataframe(self, paragraph_text, key):
        self.df = pd.DataFrame(paragraph_text, columns=[key])
        self.df['filtered'] = self.df[key].apply(
            lambda x: " ".join(re.sub(r'(http|https)://[a-zA-Z0-9./-]+', '', x) for x in x.split()))
        self.df['filtered'] = self.df['filtered'].apply(lambda x: " ".join(html.unescape(x) for x in x.split()))
        self.df['filtered'] = self.df['filtered'].apply(lambda x: " ".join(x.lower() for x in x.split()))
        self.df['filtered'] = self.df['filtered'].str.replace(r'[^\w\s]', '')
        self.df['filtered'] = self.df['filtered'].apply(
            lambda x: " ".join(re.sub(r'\d+', '', x) for x in x.split()))

        self.df['filtered'] = self.df['filtered'].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))

    def lematize_text(self, text) -> list:
        """
        This function generate tokens and return their root words
        against each sentence
        :return: lemm_token_list
        """
        lemm_token_list = []
        doc = self._nlp(text)
        for token in doc:
            lemm_token_list.append(token.lemma_)
        return lemm_token_list

    def lematization(self) -> list:
        """
        This function calls the lemmatization function that generate tokens and return their root words
        against each sentence and save them into dataframe
        :return: None
        """
        lema = []
        for text in self.df['filtered']:
            lema.append(self.lematize_text(text))
        self.df['lemmatize'] = lema
        return self.df


class QueryMatcher:
    """
    A class takes document dataframe and query dataframe and return the related sentence to query
    in dictionary format.
    ...
    Attributes
    ----------
    :param doc_frame : dataframe
        clean dataframe of document object
    :param query_frame : dataframe
        clean dataframe of query object
    """

    def __init__(self):
        self.df = None
        self.query_result = []
        self.query_format = None
        self._nlp = spacy.load('en_core_web_sm')
        self.file = open("skill_tagger.txt")
        self.taggers = ast.literal_eval(self.file.read())

    def pos_tagger(self):
        whole_skill_list = []
        for sentence in self.df['lemmatize']:
            regexp_tagger = nltk.RegexpTagger(self.taggers)
            tagger_result = regexp_tagger.tag(sentence)
            skill_list = []
            for item in tagger_result:
                if item[1] == 'UNK':
                    pass
                else:
                    skill_list.append(item[0])
            whole_skill_list.append(skill_list.copy())
        self.df['skill'] = whole_skill_list

    def query_pos_tagger(self, token_text):
        skill_list = []
        regexp_tagger = nltk.RegexpTagger(self.taggers)
        tagger_result = regexp_tagger.tag(token_text)
        for item in tagger_result:
            if item[1] == 'UNK':
                pass
            else:
                skill_list.append(item)
        return skill_list

    def calculate_percentile(self, query_skill_li):
        per = 10 / len(query_skill_li)
        percentile_list_inner, percentile_main = [], []
        for skill_list in self.df['skill']:
            sum_index = 0
            for word in query_skill_li:
                if word in skill_list:
                    sum_index = sum_index + per
            percentile_list_inner.append(sum_index)
        self.df['percentage'] = percentile_list_inner


class DictParser(PreprocessData, QueryMatcher):

    def __init__(self):
        self.queries = None
        self._dictionary_obj = None
        self._query_dataframe = None
        self._doc_dataframe = None
        PreprocessData.__init__(self)
        QueryMatcher.__init__(self)

    def input_data(self, _dict):
        """
        This function set the value to paragraph dictionary and query dictionary.
        :return: None
        """
        self._dictionary_obj = _dict.copy()
        self.queries = self._dictionary_obj.pop('requests')

    def response_result(self, key, query_skill_li):
        docs = []
        per = 10 / len(query_skill_li)
        for index, per_word in enumerate(self.df['percentage']):
            if per_word >= per:
                docs.append(self.df[key][index])
        return docs

    def result_extractor(self) -> list:
        requestObj, queries_list = {}, []
        for i in range(len(self.queries)):
            queryObj = {}
            for key, val in self.queries[i].items():
                if key != "queries":
                    requestObj[key] = val
                else:
                    for j in range(len(self.queries[i][key])):
                        combine_answer_list = []
                        for key_in, val_in in self.queries[i][key][j].items():
                            queryObj[key_in] = val_in
                        text = " ".join(
                            x.lower() for x in self.queries[i][key][j]['query'].split() if x not in stopwords)
                        token_list = self.lematize_text(text)
                        skill_tuple = self.query_pos_tagger(token_list)
                        query_skill_li = [x[0] for x in skill_tuple]
                        for doc_key, doc_val in self._dictionary_obj.items():
                            listOfSentence = self.sentence_finder(doc_val)
                            self.clean_dataframe(listOfSentence.split("@"), doc_key)
                            self.lematization()
                            self.pos_tagger()
                            self.calculate_percentile(query_skill_li)
                            answer_list = self.response_result(doc_key, query_skill_li)
                            if answer_list:
                                combine_answer_list.append(answer_list)
                        queryObj["result"] = combine_answer_list
                        queries_list.append(queryObj.copy())
                    requestObj[key] = queries_list
        return requestObj

