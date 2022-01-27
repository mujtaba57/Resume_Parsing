""""
    This program takes query and paragraph as input and clean the paragraph data using NLP-pipleline
    and return the output sentences of paragraph which are more similar to query.
"""
import ast
import html
import json
import os
import re
import sys
import warnings

import pandas as pd
import spacy
from nltk.corpus import stopwords
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from collections import OrderedDict

class PreprocessData:
    def __init__(self) -> None:
        self._nlp = spacy.load('en_core_web_sm')
        self.df = ''
        self.key = None
        self.val = None
        self.final_result = None

    def clean_dataframe(process_func):
        """
        This function loads all the sentence and clean them according to NLP-media pipleline.
        ......
        Attributes
        -----------------
        :param process_func: Decorator function take function as a parameter.
        :return inner_function: Return inner function to the decorate function
        """

        def inner_function(self):
            self.df = pd.DataFrame(self.data)
            self.df['filtered'] = self.df[self.key].apply(
                lambda x: " ".join(re.sub(r'(http|https)://[a-zA-Z0-9./-]+', '', x) for x in x.split()))
            self.df['filtered'] = self.df['filtered'].apply(lambda x: " ".join(html.unescape(x) for x in x.split()))
            self.df['filtered'] = self.df['filtered'].apply(lambda x: " ".join(x.lower() for x in x.split()))
            self.df['filtered'] = self.df['filtered'].str.replace(r'[^\w\s]', '')
            self.df['filtered'] = self.df['filtered'].apply(
                lambda x: " ".join(re.sub(r'\d+', '', x) for x in x.split()))
            stop = stopwords.words('english')
            self.df['filtered'] = self.df['filtered'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
            process_func(self)

        return inner_function

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

    @clean_dataframe
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
        self.query_result = []
        self.query_format = None

    def match_text(self, doc_frame, query_frame) -> dict:
        """
        This function match the query's tokenize words with the tokens of
        paragraph's sentence after processing and save them into dataframe
        :return: self.df
        """
        query_nmber = 1
        _query_weightage = ''
        for query_list in query_frame['lemmatize']:
            per = 1 / len(query_list)
            weight = []
            for index, word in enumerate(doc_frame['lemmatize']):
                sum = 0
                for query_word in query_list:
                    if query_word in word:
                        sum += per
                weight.append(sum)
            self.df[f'query_{query_nmber}'] = weight
            query_nmber += 1
        return self.df

    @staticmethod
    def experience_result(exp_data):
        string_data = ''.join(exp_data)
        file_name = "experience.json"
        if os.path.exists(file_name):
            file_open = open(file_name, "r")
            file_exist_data = file_open.read()
            file_open.close()
            if string_data in file_exist_data:
                pass
            else:
                f = open(file_name, "a")
                f.write(string_data + "\n")
                f.close()
        else:
            f = open(file_name, "w")
            f.write(string_data + "\n")
            f.close()

    @staticmethod
    def type_finder(query, word):
        query_type = [query[i]['type'] for i in range(len(query)) if word in query[i]['query']]
        return " ".join(query_type)

    def compare_sentence(self, query, dataframe, query_dataframe) -> list:
        """
            This function compare the query's words with
            paragraph's sentence and return it in the form of dictionary
            :return: query_result
            """
        main_result, query_count = {}, 0
        self.query_format = query
        query_df = dataframe.copy(deep=True)
        columns = [keys for keys, val in dataframe.items() if not re.search("query_*", keys)]
        query_df.drop(columns, inplace=True, axis=1)
        for query_index in query_df:
            sentence_above_thresh = []
            for index, value in enumerate(query_df[query_index]):
                val = float(value)
                if val >= 0.1:
                    for keys in dataframe.keys():
                        if keys in ["filtered", "lemmatize"] or re.search("query_*", keys):
                            pass
                        else:
                            sentence_above_thresh.append(dataframe[keys][index])

                main_result = {
                    "query": query_dataframe['queries'][query_count],
                    "result": sentence_above_thresh,
                }
            query_count += 1
            self.query_result.append(main_result)
        return self.query_result


class DictParser(PreprocessData, QueryMatcher):
    """
    DictParser class inherit from two classes. This class read the input dictionary format
    and break it into two dataframe of paragraph's and query's dataframe. After processing
    it convert into the dictionary result format.
    ...
    Attributes
    ----------
    :param dict_obj : dictionary input
    """

    def __init__(self) -> None:
        self.data = None
        self._dictionary_obj = None
        self._query_dataframe = None
        self._doc_dataframe = None
        self.queries = None

        PreprocessData.__init__(self)
        QueryMatcher.__init__(self)

    @staticmethod
    def remove_dot_query(data):
        new_Data, query_list_obj, queries_list = {}, {}, []
        new_Data['summary'] = data['summary']
        new_Data['experience'] = data['experience']
        new_Data['education'] = data['education']
        new_Data['certification'] = data['certification']
        for item in data['requests']:
            query_list = []
            for i in range(len(item['queries'])):
                if "." in item['queries'][i]['query']:
                    query_list.append(item['queries'][i]['query'].replace(".", ""))
                else:
                    query_list.append(item['queries'][i]['query'])
            query_list_obj['type'] = item['type']
            query_list_obj['queries'] = query_list
            queries_list.append(query_list_obj.copy())
        new_Data['requests'] = queries_list
        return new_Data

    def input_data(self, _dict):
        """
        This function set the value to paragraph dictionary and query dictionary.
        :return: None
        """
        new_dict = self.remove_dot_query(_dict)
        self._dictionary_obj = new_dict.copy()
        self.queries = self._dictionary_obj.pop('requests')

    @staticmethod
    def query_split(query) -> list:
        query_list = []
        for i in range(len(query)):
            for queries in query[i]['queries']:
                if "." in queries:
                    dot_replace = queries.replace(".", "")
                    query_list.append(dot_replace)
                else:
                    query_list.append(queries)
        return query_list

    def parse_query(self) -> None:
        """
        This function clean query's dataframe after convert dict_query
        into dataframe.
        :return: None
        """
        emb_query = self.query_split(self.queries)
        self.key = 'queries'
        self.data = {
            self.key: '.'.join([str(str_sent) for str_sent in emb_query]).split(".")
        }
        self.lematization()
        self._query_dataframe = self.df

    def parse_data(self) -> list:
        """
        This function clean paragraph's dataframe after convert dict document
        into dataframe.
        :return: None
        """
        result_list = None
        for key, val in self._dictionary_obj.items():
            self.key, self.val = key, val
            self.data = {
                self.key: self.val.split(".")
            }
            self.lematization()
            self._doc_dataframe = self.df
            weightage_df = self.match_text(self._doc_dataframe, self._query_dataframe)
            result_list = self.compare_sentence(self.queries, weightage_df, self._query_dataframe)
        return result_list

    @staticmethod
    def concatenate_obj(json_result):
        global result_data
        result_list = ast.literal_eval(json_result)
        groups = {}
        for result_data in result_list:
            if result_data['query'] not in groups:
                groups[result_data['query']] = {'result': result_data['result']}
            else:
                groups[result_data['query']]['result'] += result_data['result']
        groups[result_data['query']]['result'] = set(groups[result_data['query']]['result'])
        return [{**{'query': k}, **v} for k, v in groups.items()]

    def reformat_obj(self, response_result):
        data_response = []
        for i in range(len(self.query_format)):
            response = {
                'type': self.query_format[i]['type']
            }
            data_response.append(response)
        for i in range(len(self.query_format)):
            queries_obj_list = []
            if self.query_format[i]['type'] == data_response[i]['type']:
                for j in range(len(response_result)):
                    if response_result[j]['query'] in self.query_format[i]['queries']:
                        response_result[j]['result'] = set(response_result[j]['result'])
                        queries_obj_list.append(response_result[j])
                data_response[i]['queries'] = queries_obj_list
        return data_response

    def indexing(self, data_response):
        count = 0
        new_data_dict, data_list, new_query_obj, new_query_list = {}, [], {}, []
        for i in range(len(data_response)):
            data_response[i]['sortOrder'] = count
            for j in range(len(data_response[i]['queries'])):
                count += 1
                data_response[i]['queries'][j]['sortOrder'] = count
            count += 1
        for i in range(len(data_response)):
            new_query_list = []
            new_data_dict['type'] = data_response[i]['type']
            new_data_dict['sortOrder'] = data_response[i]['sortOrder']
            for j in range(len(data_response[i]['queries'])):
                new_query_obj['query'] = data_response[i]['queries'][j]['query']
                new_query_obj['sortOrder'] = data_response[i]['queries'][j]['sortOrder']
                new_query_obj['result'] = data_response[i]['queries'][j]['result']
                new_query_list.append(new_query_obj.copy())
            new_data_dict['queries'] = new_query_list
            data_list.append(new_data_dict.copy())
        return data_list


