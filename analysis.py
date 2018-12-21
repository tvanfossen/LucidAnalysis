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
        if 'SurveyData.csv' in data_files:
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

    analysis_dict = {'sentiment_pre_neg': analysis_pre['neg'],'sentiment_pre_neu': analysis_pre['neu'],
                     'sentiment_pre_pos': analysis_pre['pos'],'sentiment_pre_compound': analysis_pre['compound'],
                     'sentiment_post_neg': analysis_post['neg'], 'sentiment_post_neu': analysis_post['neu'],
                     'sentiment_post_pos': analysis_post['pos'], 'sentiment_post_compound': analysis_post['compound'],
                     'sentiment_complete_neg': analysis_complete['neg'], 'sentiment_complete_neu': analysis_complete['neu'],
                     'sentiment_complete_pos': analysis_complete['pos'], 'sentiment_complete_compound': analysis_complete['compound'],
                     'sentiment_improv_neg': analysis_post['neg'] - analysis_pre['neg'],
                     'sentiment_improv_neu': analysis_post['neu'] - analysis_pre['neu'],
                     'sentiment_improv_pos': analysis_post['pos'] - analysis_pre['pos'],
                     'sentiment_improv_compound': analysis_post['compound'] - analysis_pre['compound'],

                     }


    return analysis_dict

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

    temp_dict = {"acceptance_pre": resp_pre/len(resp_list),
                 "acceptance_post": resp_post/len(resp_list),
                 "acceptance_improv": resp_pre/len(resp_list) - resp_post/len(resp_list)}
    return temp_dict

def correlation(resp_list, df):
    resp_pre_rest = 0
    resp_post_rest = 0
    resp_perceived_length = 0
    resp_rating = 0
    resp_recommendation = 0
    resp_accepted_length = 0
    caffeine_level = 0

    for resp in resp_list:
        # print(resp['How many servings of caffeine have you had today?'])
        try:
            resp_pre_rest += int(resp['How rested do you feel today?'])
            resp_post_rest += int(resp['How rested do you feel compared to when you arrived?'])
            resp_perceived_length += int(resp['How long did your rest feel like?'])

        except:
            resp_pre_rest += int(resp['How rejuvenated do you feel today?'])
            resp_post_rest += int(resp['How rejuvenated do you feel compared to when you arrived?'])
            resp_perceived_length += int(resp['How long did your rejuvenation feel like?'])

        if resp['How many servings of caffeine have you had today?'] == '4 or more':
            caffeine_level += 4
        elif resp['How many servings of caffeine have you had today?'] == "":
            caffeine_level += 0
        else:
            caffeine_level += int(resp['How many servings of caffeine have you had today?'])

        resp_rating += int(resp['How would you rate your LUCID experience?'])
        try:
            resp_accepted_length += int(resp['How many minutes would you be willing to rest during a work day?'])
        except:
            resp_accepted_length += int(resp['How many minutes would you be willing to rejuvenate during a work day?'])

        if (resp['How likely are you to recommend the LUCID experience to a co-worker?'] == 'Very likely' or
            resp['How likely are you to recommend the LUCID experience to a co-worker?'] == "1"):
            resp_recommendation += 5
        elif (resp['How likely are you to recommend the LUCID experience to a co-worker?'] == 'Likely' or
              resp['How likely are you to recommend the LUCID experience to a co-worker?'] == "2"):
            resp_recommendation += 4
        elif (resp['How likely are you to recommend the LUCID experience to a co-worker?'] == 'Neither likely nor unlikely' or
              resp['How likely are you to recommend the LUCID experience to a co-worker?'] == "3"):
            resp_recommendation += 3
        elif (resp['How likely are you to recommend the LUCID experience to a co-worker?'] == 'Unlikely' or
              resp['How likely are you to recommend the LUCID experience to a co-worker?'] == "4"):
            resp_recommendation += 2
        elif (resp['How likely are you to recommend the LUCID experience to a co-worker?'] == 'Very unlikely' or
              resp['How likely are you to recommend the LUCID experience to a co-worker?'] == "5"):
            resp_recommendation += 1


    average_bpm_60 = 0
    average_bpm_30 = 0
    average_bpm_0 = 0

    average_start_60 = 0
    average_start_30 = 0
    average_start_0 = 0

    average_end = 0


    for j in range(0, len(df[datetime.timedelta(hours=int(0), minutes=int(3), seconds=int(30))])):

        average_start_60 += df[datetime.timedelta(hours=int(0), minutes=int(0), seconds=int(60))][j]
        average_start_30 += df[datetime.timedelta(hours=int(0), minutes=int(0), seconds=int(30))][j]
        average_start_0 += df[datetime.timedelta(hours=int(0), minutes=int(0), seconds=int(1))][j]

        average_end += df[datetime.timedelta(hours=int(0), minutes=int(3), seconds=int(30))][j]

        average_bpm_60 += (df[datetime.timedelta(hours=int(0), minutes=int(3), seconds=int(30))][j] -
                                df[datetime.timedelta(hours=int(0), minutes=int(0), seconds=int(60))][j])
        average_bpm_30 += (df[datetime.timedelta(hours=int(0), minutes=int(3), seconds=int(30))][j] -
                               df[datetime.timedelta(hours=int(0), minutes=int(0), seconds=int(30))][j])
        average_bpm_0 += (df[datetime.timedelta(hours=int(0), minutes=int(3), seconds=int(30))][j] -
                           df[datetime.timedelta(hours=int(0), minutes=int(0), seconds=int(1))][j])


    print("\tRest Level Improvement: " + str(resp_post_rest / len(resp_list) - resp_pre_rest / len(resp_list)))
    print("\tPerceived Length: " + str(resp_perceived_length / len(resp_list)))
    print("\tLucid Rating: " + str(resp_rating/len(resp_list)))
    print("\tLucid Recommendation: " + str(resp_recommendation/len(resp_list)))
    print("\tAccepted Rest Length: " + str(resp_accepted_length/len(resp_list)))

    temp_dict = {"rest_level_pre": resp_pre_rest / len(resp_list), 'rest_level_post': resp_post_rest / len(resp_list),
                 "rest_level_improv":resp_post_rest / len(resp_list) - resp_pre_rest / len(resp_list),
                 "perceived_length": resp_perceived_length / len(resp_list),
                 "lucid_rating": resp_rating/len(resp_list), "lucid_recommendation":resp_recommendation/len(resp_list),
                 "accepted_rest_length":resp_accepted_length/len(resp_list),
                 "average_bpm_start_60": average_start_60/len(df[datetime.timedelta(hours=int(0), minutes=int(3), seconds=int(30))]),
                 "average_bpm_start_30": average_start_30 / len(
                     df[datetime.timedelta(hours=int(0), minutes=int(3), seconds=int(30))]),
                 "average_bpm_start_00": average_start_0 / len(
                     df[datetime.timedelta(hours=int(0), minutes=int(3), seconds=int(30))]),
                 "average_bpm_end": average_end / len(
                     df[datetime.timedelta(hours=int(0), minutes=int(3), seconds=int(30))]),

                 "bpm_change_60": average_bpm_60/len(df[datetime.timedelta(hours=int(0), minutes=int(3), seconds=int(30))]),
                 "bpm_change_30": average_bpm_30/len(df[datetime.timedelta(hours=int(0), minutes=int(3), seconds=int(30))]),
                 "bpm_change_0": average_bpm_0/len(df[datetime.timedelta(hours=int(0), minutes=int(3), seconds=int(30))]),
                 "caffeine_intake": caffeine_level/len(resp_list)
                 }

    return temp_dict

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
    # print(df)

    # df.mean().plot(label='mean')
    # plt.title(title)
    # plt.xlabel('Time', fontsize=15)
    # plt.ylabel('HR', fontsize=15)
    # plt.show()

    return complete_hr_dict

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
        'am':[],
        'pm':[],
        'tuesday':[],
        'wednesday':[],
        'tuesdayam':[],
        'tuesdaypm':[],
        'wednesdayam':[],
        'wednesdaypm':[],
        'caffeine_0': [],
        'caffeine_1': [],
        'caffeine_2': [],
        'caffeine_3': [],
        'caffeine_4': [],
        'rating_1': [],
        'rating_2': [],
        'rating_3': [],
        'rating_4': [],
        'rating_5': [],
        'recommend_very_unlikely': [],
        'recommend_unlikely': [],
        'recommend_neutral': [],
        'recommend_likely': [],
        'recommend_very_likely': [],

    }

    complete_df = []
    day_ids = {'\ufeffTuesday AM': [], 'Tuesday PM':[], 'Wed AM': [], 'Wed PM': []}



    with open("LucidData/am&pm/am&pm.csv", 'r') as input:
        input_reader = csv.DictReader(input, delimiter=',')
        for row in input_reader:
            for entry in row:
                if row[entry] != "":
                    day_ids[entry].append(row[entry])

    for dir in data_lists:
        if dir != 'am&pm':
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

                if resp['How many servings of caffeine have you had today?'] == '4 or more':
                    complete_dict['caffeine_4'].append(resp)
                elif resp['How many servings of caffeine have you had today?'] == '3':
                    complete_dict['caffeine_3'].append(resp)
                elif resp['How many servings of caffeine have you had today?'] == '2':
                    complete_dict['caffeine_2'].append(resp)
                elif resp['How many servings of caffeine have you had today?'] == '1':
                    complete_dict['caffeine_1'].append(resp)
                elif resp['How many servings of caffeine have you had today?'] == '0':
                    complete_dict['caffeine_0'].append(resp)

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

                if resp['Subject ID'] in day_ids['\ufeffTuesday AM']:
                    complete_dict['tuesdayam'].append(resp)
                    complete_dict['tuesday'].append(resp)
                    complete_dict['am'].append(resp)
                elif resp['Subject ID'] in day_ids['Tuesday PM']:
                    complete_dict['tuesdaypm'].append(resp)
                    complete_dict['tuesday'].append(resp)
                    complete_dict['pm'].append(resp)
                elif resp['Subject ID'] in day_ids['Wed AM']:
                    complete_dict['wednesdayam'].append(resp)
                    complete_dict['wednesday'].append(resp)
                    complete_dict['am'].append(resp)
                elif resp['Subject ID'] in day_ids['Wed PM']:
                    complete_dict['wednesdaypm'].append(resp)
                    complete_dict['wednesday'].append(resp)
                    complete_dict['pm'].append(resp)

                if (resp['How likely are you to recommend the LUCID experience to a co-worker?'] == 'Very likely' or
                        resp['How likely are you to recommend the LUCID experience to a co-worker?'] == "1"):
                    complete_dict['recommend_very_likely'].append(resp)
                elif (resp['How likely are you to recommend the LUCID experience to a co-worker?'] == 'Likely' or
                      resp['How likely are you to recommend the LUCID experience to a co-worker?'] == "2"):
                    complete_dict['recommend_likely'].append(resp)
                elif (resp[
                          'How likely are you to recommend the LUCID experience to a co-worker?'] == 'Neither likely nor unlikely' or
                      resp['How likely are you to recommend the LUCID experience to a co-worker?'] == "3"):
                    complete_dict['recommend_neutral'].append(resp)
                elif (resp['How likely are you to recommend the LUCID experience to a co-worker?'] == 'Unlikely' or
                      resp['How likely are you to recommend the LUCID experience to a co-worker?'] == "4"):
                    complete_dict['recommend_unlikely'].append(resp)
                elif (resp['How likely are you to recommend the LUCID experience to a co-worker?'] == 'Very unlikely' or
                      resp['How likely are you to recommend the LUCID experience to a co-worker?'] == "5"):
                    complete_dict['recommend_very_unlikely'].append(resp)


                if int(resp['How would you rate your LUCID experience?']) == 5:
                    complete_dict['rating_5'].append(resp)
                elif int(resp['How would you rate your LUCID experience?']) == 4:
                    complete_dict['rating_4'].append(resp)
                elif int(resp['How would you rate your LUCID experience?']) == 3:
                    complete_dict['rating_3'].append(resp)
                elif int(resp['How would you rate your LUCID experience?']) == 2:
                    complete_dict['rating_2'].append(resp)
                elif int(resp['How would you rate your LUCID experience?']) == 1:
                    complete_dict['rating_1'].append(resp)
    complete_dict_by_sort = {}

    for i in complete_dict:
        print(i + " : N=" + str(len(complete_dict[i])))
        if len(complete_dict[i]) > 0:
            complete_dict_by_sort[i] = {}

            complete_dict_by_sort[i].update(correlation(complete_dict[i], hr_trends(complete_dict[i], i)))
            complete_dict_by_sort[i].update(better_acceptance(complete_dict[i]))
            complete_dict_by_sort[i].update(better_sentiment(complete_dict[i]))
            complete_dict_by_sort[i].update({'sort_rule': i})
            complete_dict_by_sort[i].update({"N": len(complete_dict[i])})

    with open('output.csv', 'w', newline='') as csvfile:
        fieldnames = []
        for i in complete_dict_by_sort['complete']:
            fieldnames.append(i)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in complete_dict_by_sort:
            writer.writerow(complete_dict_by_sort[i])


if __name__ == '__main__':
    # nltk.download('vader_lexicon')
    data_lists = compile_data()
    better_analysis(data_lists)