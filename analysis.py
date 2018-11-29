#!/usr/bin/env python
import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import datetime
import seaborn as sns

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def compile_data():
    data_dirs = os.listdir("LucidData")
    comp_data_list = {}
    for dir in data_dirs:
        comp_data_list[dir] = []
        # for file in dataFiles:
        #     print ("\t" + file)
        data_files = os.listdir("LucidData/" + dir)
        with open("LucidData/" + dir + "/" + "SurveyData.csv", 'r') as input:
            input_reader = csv.DictReader(input, delimiter=',')
            row1 = next(input_reader)

            data_list = []
            dict_template = {}
            for item in row1:
                dict_template[item] = ""

            for row in input_reader:
                temp_dict = dict_template.copy()
                for item in row:
                    temp_dict[item] = row[item]
                data_list.append(temp_dict)

            # for dict in data_list:
            #     print(dict)

        comp_data_list[dir].append(data_list)

    return comp_data_list


def better_sentiment(resp_list):
    def analyzer(sentences):
        sid = SentimentIntensityAnalyzer()
        scores = {"compound": 0, "neg": 0, "neu": 0, "pos": 0}
        n = len(sentences)
        for sentence in sentences:
            ss = sid.polarity_scores(sentence)
            for k in sorted(ss):
                scores[k] += ss[k]

        for score in scores:
            scores[score] /= n

        return scores

    resp_pre = []
    resp_post = []
    resp_complete = []

    for resp in resp_list:
        resp_pre.append(resp["In a few sentences, how do you feel right now?"])
        resp_post.append(resp['In a few sentences, how are you feeling right now?'])
        resp_complete.append(resp["In a few sentences, how do you feel right now?"])
        resp_complete.append(resp['In a few sentences, how are you feeling right now?'])

    analysis_pre = analyzer(resp_pre)
    analysis_post = analyzer(resp_post)
    analysis_complete = analyzer(resp_complete)
    print("\tSentiment Analysis Complete")
    print("\t\tCompound: " + str(analysis_complete['compound']))
    print("\t\tNegative: " + str(analysis_complete['neg']))
    print("\t\tNeutral: " + str(analysis_complete['neu']))
    print("\t\tPositive: " + str(analysis_complete['pos']))

    print("\tSentiment Analysis Improvement Pre/Post")
    print("\t\tCompound: " + str(analysis_post['compound'] - analysis_pre['compound']))
    print("\t\tNegative: " + str(analysis_post['neg'] - analysis_pre['neg']))
    print("\t\tNeutral: " + str(analysis_post['neu'] - analysis_pre['neu']))
    print("\t\tPositive: " + str(analysis_post['pos'] - analysis_pre['pos']))


def better_acceptance(resp_list):
    resp_pre = 0
    resp_post = 0
    resp_length = 0
    for resp in resp_list:
        # print(resp)
        try:
            resp_pre += int(resp['How culturally acceptable\xa0is it to seek rest during a work day?'])
            resp_post += int(resp['How culturally acceptable would it be to use LUCID at work for the purpose of resting?'])
        except:
            if resp['How culturally acceptable\xa0is it to seek rest during a work day?'] == "Very Appropriate":
                resp_pre += 1
            elif resp['How culturally acceptable\xa0is it to seek rest during a work day?'] == "Somewhat Appropriate":
                resp_pre += 2
            elif resp['How culturally acceptable\xa0is it to seek rest during a work day?'] == "Neutral":
                resp_pre += 3
            elif resp['How culturally acceptable\xa0is it to seek rest during a work day?'] == "Somwhat Inappropriate":
                resp_pre += 4
            elif resp['How culturally acceptable\xa0is it to seek rest during a work day?'] == "Very Inappropriate":
                resp_pre += 5
            try:
                if resp['How culturally acceptable would it be to use LUCID at work for the purpose of rejuvenation?'] == "Very Appropriate":
                    resp_post += 1
                elif resp['How culturally acceptable would it be to use LUCID at work for the purpose of rejuvenation?'] == "Somewhat Appropriate":
                    resp_post += 2
                elif resp['How culturally acceptable would it be to use LUCID at work for the purpose of rejuvenation?'] == "Neutral":
                    resp_post += 3
                elif resp['How culturally acceptable would it be to use LUCID at work for the purpose of rejuvenation?'] == "Somwhat Inappropriate":
                    resp_post += 4
                elif resp['How culturally acceptable would it be to use LUCID at work for the purpose of rejuvenation?'] == "Very Inappropriate":
                    resp_post += 5
            except:
                if resp['How culturally acceptable would it be to use LUCID at work for the purpose of resting?'] == "Very Appropriate":
                    resp_post += 1
                elif resp['How culturally acceptable would it be to use LUCID at work for the purpose of resting?'] == "Somewhat Appropriate":
                    resp_post += 2
                elif resp['How culturally acceptable would it be to use LUCID at work for the purpose of resting?'] == "Neutral":
                    resp_post += 3
                elif resp['How culturally acceptable would it be to use LUCID at work for the purpose of resting?'] == "Somwhat Inappropriate":
                    resp_post += 4
                elif resp['How culturally acceptable would it be to use LUCID at work for the purpose of resting?'] == "Very Inappropriate":
                    resp_post += 5



    print("\tAcceptance Level Improvement: " + str(resp_pre/len(resp_list) - resp_post/len(resp_list)))


def correlation(resp_list):
    resp_pre_rest = 0
    resp_post_rest = 0
    resp_perceived_length = 0
    resp_rating = 0
    resp_recommendation = 0
    resp_accepted_length = 0
    for resp in resp_list:
        # print(resp)
        try:
            resp_pre_rest += int(resp['How rested do you feel today?'])
            resp_post_rest += int(resp['How rested do you feel compared to when you arrived?'])
            resp_perceived_length += int(resp['How long did your rest feel like?'])

        except:
            resp_pre_rest += int(resp['How rejuvenated do you feel today?'])
            resp_post_rest += int(resp['How rejuvenated do you feel compared to when you arrived?'])
            resp_perceived_length += int(resp['How long did your rejuvenation feel like?'])

        resp_rating += int(resp['How would you rate your LUCID experience?'])
        try:
            resp_accepted_length += int(resp['How many minutes would you be willing to rest during a work day?'])
        except:
            resp_accepted_length += int(resp['How many minutes would you be willing to rejuvenate during a work day?'])

        if resp['How likely are you to recommend the LUCID experience to a co-worker?'] == 'Very likely':
            resp_recommendation += 5
        elif resp['How likely are you to recommend the LUCID experience to a co-worker?'] == 'Likely':
            resp_recommendation += 4
        elif resp['How likely are you to recommend the LUCID experience to a co-worker?'] == 'Neither likely nor unlikely':
            resp_recommendation += 3
        elif resp['How likely are you to recommend the LUCID experience to a co-worker?'] == 'Unlikely':
            resp_recommendation += 2
        elif resp['How likely are you to recommend the LUCID experience to a co-worker?'] == 'Very unlikely':
            resp_recommendation += 1


    print("\tRest Level Improvement: " + str(resp_post_rest / len(resp_list) - resp_pre_rest / len(resp_list)))
    print("\tPerceived Length: " + str(resp_perceived_length / len(resp_list)))
    print("\tLucid Rating: " + str(resp_rating/len(resp_list)))
    print("\tLucid Recommendation: " + str(resp_recommendation/len(resp_list)))
    print("\tAccepted Rest Length: " + str(resp_accepted_length/len(resp_list)))


def hr_trends(resp_list, title):
    hr_ids = []
    for resp in resp_list:
        temp_id = resp['Subject ID'].lower()
        if len(temp_id) < 3:
            temp_id = temp_id[0] + "0" + temp_id[1]
        hr_ids.append(temp_id + '.csv')

    complete_hr_dict = {}
    data_dirs = os.listdir("LucidData")
    for dir in data_dirs:
        # print(dir)
        data_files = os.listdir("LucidData/" + dir)
        for file in data_files:
            if file in hr_ids:
                # print("\t" + file)
                with open("LucidData/" + dir + "/" + file, 'r') as input:
                    input_reader = csv.reader(input, delimiter=',')

                    for row in input_reader:
                        if row[0] != "":
                            t = row[0]
                            h, m, s = re.split(':', t)
                            ts = datetime.timedelta(hours=int(h), minutes=int(m), seconds=int(s))
                            if ts <= datetime.timedelta(hours=0, minutes=3, seconds=30):
                                try:
                                    complete_hr_dict[ts].append(int(row[1]))
                                except:
                                    complete_hr_dict[ts] = [int(row[1]), ]
    df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in complete_hr_dict.items() ]))

    df = df.mask(df.sub(df.mean()).div(df.std()).abs().gt(3))
    print(df)

    df.mean().plot(label='mean')
    plt.title(title)
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('HR', fontsize=15)
    plt.show()

def better_analysis(data_lists):
    complete_dict = {
        'complete':[],
        'empath': [],
        'brody': [],
        'v1': [],
        'v2': [],
        'no_audio':[],
        'male': [], #2
        'female' : [],#1
        'age1824' : [], #2
        'age2534' : [], #3
        'age3544' : [], #4
        'age4554' : [], #5
        'age5564' : [], #6
        'meeting' : [], #1
        'home' : [], #2
        'call' : [], #3
        'meal' : [], #4
        'working' : [], #5
        'other' : [], #6
    }

    complete_df = []

    for dir in data_lists:
        for resp in data_lists[dir][0]:
            # print(resp)
            complete_dict['complete'].append(resp)
            if dir == "LUCID_Room_1_V1" or dir == "LUCID_Room_1_V2":
                complete_dict['empath'].append(resp)
            else:
                complete_dict['brody'].append(resp)

            if dir == "LUCID_Room_1_V1" or dir == "LUCID_Room_2_V1":
                complete_dict['v1'].append(resp)
            else:
                complete_dict['v2'].append(resp)

            if resp['Subject ID'] == 'D6' or resp['Subject ID'] == 'D7' or resp['Subject ID'] == 'D8':
                complete_dict['no_audio'].append(resp)

            if resp['With what gender do you identify?'] == "Male" or resp['With what gender do you identify?'] == '2':
                complete_dict['male'].append(resp)
            else:
                complete_dict['female'].append(resp)

            if resp["Age"] == "18-24" or resp["Age"] == '2':
                complete_dict['age1824'].append(resp)
            elif resp["Age"] == "25-34" or resp["Age"] == '3':
                complete_dict['age2534'].append(resp)
            elif resp["Age"] == "35-44" or resp["Age"] == '4':
                complete_dict['age3544'].append(resp)
            elif resp["Age"] == "45-54" or resp["Age"] == '5':
                complete_dict['age4554'].append(resp)
            elif resp["Age"] == "55-64" or resp["Age"] == '6':
                complete_dict['age5564'].append(resp)

            if resp['Where you are currently coming\xa0from?'] == 'From a meeting' or resp['Where you are currently coming\xa0from?'] == '1':
                complete_dict['meeting'].append(resp)
            elif resp['Where you are currently coming\xa0from?'] == 'From home' or resp["Where you are currently coming\xa0from?"] == '2':
                complete_dict['home'].append(resp)
            elif resp['Where you are currently coming\xa0from?'] == 'From a phone call' or resp["Where you are currently coming\xa0from?"] == '3':
                complete_dict['call'].append(resp)
            elif resp['Where you are currently coming\xa0from?'] == 'From a meal (Break/Lunch/Snack)' or resp["Where you are currently coming\xa0from?"] == '4':
                complete_dict['meal'].append(resp)
            elif resp['Where you are currently coming\xa0from?'] == 'Working on my own' or resp["Where you are currently coming\xa0from?"] == '5':
                complete_dict['working'].append(resp)
            elif resp['Where you are currently coming\xa0from?'] == 'Other (please specify)' or resp["Where you are currently coming\xa0from?"] == '0':
                complete_dict['other'].append(resp)

    for i in complete_dict:
        print(i + " : N=" + str(len(complete_dict[i])))
        correlation(complete_dict[i])
        better_acceptance(complete_dict[i])
        better_sentiment(complete_dict[i])
        complete_df.append(hr_trends(complete_dict[i], i))


if __name__ == '__main__':
    nltk.download('vader_lexicon')
    data_lists = compile_data()
    better_analysis(data_lists)