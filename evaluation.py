import pandas as pd
from sentence_transformers import SentenceTransformer, util
import inflect
from collections import Counter
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import numpy as np
import ast
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics.cluster import contingency_matrix
import string
import difflib
import shutil
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# Initialize the sentence transformer model for contextual matching
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to compute contextual similarity
def contextual_match(gt, pred, threshold=0.8):
    gt_embedding = model.encode(gt, convert_to_tensor=True)
    pred_embedding = model.encode(pred, convert_to_tensor=True)
    similarity = util.cos_sim(gt_embedding, pred_embedding).item()
    return similarity >= threshold


def getGT():
    data = pd.read_csv("data_mturk/filtered_mturk.csv")
    gt = data.groupby("Input.image_id")["Answer.original_ans"].apply(list).to_dict()
    qsn_type = data.groupby("Input.image_id")["Input.qsn_type"].first().to_dict()

    return gt, qsn_type


def majority_vote(answers):
    # return max(set(answers), key=answers.count)

    normalized_answers = [ans.strip().lower() for ans in answers]
    counter = Counter(normalized_answers)

    max_count = max(counter.values())
    majority = [ans for ans, count in counter.items() if count == max_count]

    return majority[0]


p = inflect.engine()


def areEquivalent(gt, pred):
    if isinstance(gt, str) and gt.isdigit():
        gt = int(gt)
    if isinstance(pred, str) and pred.isdigit():
        pred = int(pred)

    if isinstance(gt, int):
        gt = p.number_to_words(gt)
    if isinstance(pred, int):
        pred = p.number_to_words(pred)

    return pred.strip().lower().replace(".", "") == gt.strip().lower().replace(".", "")


def exactMatching(data, gt, qtype_dict):
    contextual_matches = 0
    total_matches_yes_no = 0
    total_yes_no = 0
    total_other = 0
    total_num, total_match_num = 0, 0
    total_matches_other = 0

    qtype = [
        "how many",
        "how many people are",
        "how many animals are",
        "how many windows are",
        "how many pillows",
        "how many trees are",
        "how many pictures",
        "how many plants are",
        "how many birds are",
        "how many clouds are",
        "how many bushes",
    ]

    for index, row in data.iterrows():
        # print(index)
        imageid = row["image_id"]
        gt_ans = gt[imageid]
        pred_ans = row["ans_claude_modques"]
        majority_ans = majority_vote(gt_ans)  # Get majority from ground truth
        majority_ans = " ".join(majority_ans.split()).lower().replace(".", "")
        # majority_ans = re.sub(r"[^\w\s]", "", majority_ans.strip().lower())
        # pred_ans = re.sub(r"[^\w\s]", "", pred_ans.strip().lower())
        print(imageid)
        pred_ans = " ".join(pred_ans.split()).lower().replace(".", "")

        try:
            if majority_ans in ["yes", "no"]:
                # Exact match for yes/no answers
                # majority_ans = majority_vote(gt_ans)  # Get majority from ground truth
                total_yes_no += 1
                # print(pred_ans, majority_ans)
                if pred_ans == majority_ans:
                    # print("****")
                    # if pred_ans.strip().lower().replace(
                    #     ".", ""
                    # ) == majority_ans.strip().lower().replace(".", ""):
                    total_matches_yes_no += 1

            elif qtype_dict[imageid] in qtype:
                # majority_ans = majority_vote(gt_ans)  # Get majority from ground truth
                res = areEquivalent(majority_ans, pred_ans)
                total_num += 1
                if res:
                    total_match_num += 1

            else:
                # continue
                # Contextual match for other answers
                total_other += 1
                if contextual_match(
                    majority_ans.strip().lower().replace(".", ""),
                    pred_ans.strip().lower().replace(".", ""),
                ):
                    contextual_matches += 1
                if pred_ans.strip().lower().replace(
                    ".", ""
                ) == majority_ans.strip().lower().replace(".", ""):
                    total_matches_other += 1
        except Exception as e:
            print(e)
    # Calculate accuracy
    yes_no_accuracy = total_matches_yes_no / total_yes_no if total_yes_no > 0 else 0
    print("total yes/no no, total match", total_yes_no, total_matches_yes_no)
    print("binary class accuracy", yes_no_accuracy)
    num_accuracy = total_match_num / total_num if total_num > 0 else 0
    print("number accuracy:", num_accuracy)
    other_exact_accuracy = total_matches_other / total_other if total_other > 0 else 0
    other_context_accuracy = contextual_matches / total_other if total_other > 0 else 0
    print("other exact accuracy:", other_exact_accuracy)
    print("other contexual accuracy:", other_context_accuracy)
    # # contextual_accuracy = contextual_matches / total_other if total_other > 0 else 0


def blue(data, gt_ans, model):
    bleu_scores = []
    smoothie = SmoothingFunction().method1

    for index, row in data.iterrows():
        image_id = row["image_id"]
        pred_ans_modques = str(row[model]).strip().lower()
        references = gt_ans[image_id]  # assumed to be a list of strings
        print('***', pred_ans_modques, references)
        ref_tokens = [ref.strip().lower().split() for ref in references]
        # Tokenize predicted answer
        pred_tokens = pred_ans_modques.split()
        # Compute BLEU score
        bleu_score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
        bleu_scores.append(bleu_score)

    # data["bleu_score"] = bleu_scores
    return bleu_scores

def starts_with_any(text, prefixes):
    if not isinstance(text, str):
        return False
    return any(text.startswith(prefix) for prefix in prefixes)

def yesNoQuestions(data, model):
    qtype_yn = ["is", "are", "does", "did", "will", "do", "can"] 
    qtype_num = ["how"]
    yes_no_vector = []
    gt, qtype_dict = getGT()    
    for _, row in data.iterrows():
        imageid = row["image_id"]
        gt_ans = gt[imageid]
        pred_ans = row[model]
        if pd.isna(pred_ans):
            continue
        pred_ans = " ".join(pred_ans.split()).lower().replace(".", "")

        try:
            if starts_with_any(qtype_dict[imageid], qtype_yn):
                majority_ans = majority_vote(gt_ans)  # Get majority from ground truth
                majority_ans = " ".join(majority_ans.split()).lower().replace(".", "")
                if pred_ans == majority_ans:
                    yes_no_vector.append(1)
                else:
                    yes_no_vector.append(0)

            if starts_with_any(qtype_dict[imageid], qtype_num):
                majority_ans = majority_vote(gt_ans)  # Get majority from ground truth
                majority_ans = " ".join(majority_ans.split()).lower().replace(".", "")
                # print(majority_ans, pred_ans)
                res = areEquivalent(majority_ans, pred_ans)
                if res:
                    yes_no_vector.append(1)
                else:
                    yes_no_vector.append(0)
        except Exception as e:
            print(e)
    return yes_no_vector #num_vector

def openEndQuesMatching(data, model):
    qsnType = ['what', 'why', 'who', 'where', 'which']
    gt, qtype_dict = getGT()
    yes_no_vector = []
    for _, row in data.iterrows():
        imageid = row["image_id"]
        gt_ans = gt[imageid]
        pred_ans = row[model]
        if pd.isna(pred_ans):
            continue
        pred_ans = " ".join(pred_ans.split()).lower().replace(".", "")

        try:
            if starts_with_any(qtype_dict[imageid], qsnType):
                gt_ans = [ans.strip().lower().replace(".", "") for ans in gt_ans]
                # if pred_ans.lower() in ["unknown", "none", "unclear"] or re.search(r"\bno\b", pred_ans.lower()):
                if pred_ans in gt_ans:
                    yes_no_vector.append(1)
                else:
                    yes_no_vector.append(0)
        except Exception as e:
            print(e)

    return yes_no_vector

def yesNoQuestionsTrain(data, train_data, vlm):
    qtype_yn = ["is", "are", "does", "did", "will", "do", "can"] 
    qtype_num = ["how"]
    yes_no_vector = []
    for v in vlm:
        for _, row in data.iterrows():
            # question_id_unique = str(row["question_id_unique"])
            # imageid = row["image_id"]
            pred_ans = row[v]
            question_id = row["question_id_unique"]#question_id_unique[:-1]
            qtype_dict = data[data["question_id"]== int(question_id)]["q_type"]
            
            qtype_dict = qtype_dict.iloc[0].lower()
            if pd.isna(pred_ans):
                continue
            pred_ans = " ".join(pred_ans.split()).lower().replace(".", "")
            try:
                if starts_with_any(qtype_dict, qtype_yn):
                    gt_ans = train_data[train_data["question_id"] == int(question_id)]["answers"].values[0]
                    gt_ans = ast.literal_eval(gt_ans)
                    gt_ans = [ans['answer'] for ans in gt_ans]
                    majority_ans = majority_vote(gt_ans)  # Get majority from ground truth
                    # majority_ans = " ".join(majority_ans.split()).lower().replace(".", "")
                    if pred_ans == majority_ans:
                        yes_no_vector.append(1)
                    else:
                        yes_no_vector.append(0)
                if starts_with_any(qtype_dict, qtype_num):
                    gt_ans = train_data[train_data["question_id"] == int(question_id)]["answers"].values[0]
                    gt_ans = ast.literal_eval(gt_ans)
                    gt_ans = [ans['answer'] for ans in gt_ans]
                    majority_ans = majority_vote(gt_ans)  # Get majority from ground truth
                    # majority_ans = " ".join(majority_ans.split()).lower().replace(".", "")
                    # print(majority_ans, pred_ans)
                    res = areEquivalent(majority_ans, pred_ans)
                    if res:
                        yes_no_vector.append(1)
                    else:
                        yes_no_vector.append(0)
            except Exception as e:
                print(e)
        print("accuracy for", v, sum(yes_no_vector)/len(yes_no_vector))
    return yes_no_vector #num_vector

def openEndQuesMatchingTrain(data, train_data, vlm):
    qsnType = ['what', 'why', 'who', 'where', 'which']
    yes_no_vector = []
    # vlm = ["ans_gpt4o_modques", "ans_gemini_modques", "ans_claude_modques", "ans_llava_modques"]
    # vlm = ["ans_gpt4o_question", "ans_gemini_question", "ans_claude_modques", "ans_llava_question"]
    for v in vlm:
        for _, row in data.iterrows():
            # imageid = row["image_id"]
            question_id_unique = str(row["question_id_unique"])
            pred_ans = row[v]
            question_id = question_id_unique[:-1]
            qtype_dict = data[data["question_id_unique"]== int(question_id_unique)]["q_type"]
            qtype_dict = qtype_dict.iloc[0].lower()
            if pd.isna(pred_ans):
                continue
            pred_ans = " ".join(pred_ans.split()).lower().replace(".", "")
            try:
                if starts_with_any(qtype_dict, qsnType):
                    gt_ans = train_data[train_data["question_id"] == int(question_id)]["answers"].values[0]
                    gt_ans = ast.literal_eval(gt_ans)
                    gt_ans = [ans['answer'] for ans in gt_ans]
                        # if pred_ans.lower() in ["unknown", "none", "unclear"] or re.search(r"\bno\b", pred_ans.lower()):
                    if pred_ans in gt_ans:
                    # if gt_ans.count(pred_ans) >= 3:
                        yes_no_vector.append(1)
                    else:
                        yes_no_vector.append(0)
            except Exception as e:
                print(e)
        print("accuracy for", v, sum(yes_no_vector)/len(yes_no_vector))
    return yes_no_vector


def normalize_series(series):
    return series.astype(str).str.lower().str.strip().str.rstrip('.')


def contingency_table():
    data1 = pd.read_csv("data_for_mturk/amt_res_effort_majority.csv", encoding='latin1')
    data2 = pd.read_csv("data_for_mturk/amt_res_betterans_majority.csv", encoding='latin1')

    W, G = [], []
    for index, row in data2.iterrows():
        image_id = row["Input.image_id"]
        if row["Answer.better_answer.answer1"] == True:
            W.append(0)
            if data1[data1['Input.image_id'] == image_id]["Answer.q2_more_time.no"].bool() == True:
                G.append(0)
            if data1[data1['Input.image_id'] == image_id]["Answer.q2_more_time.yes"].bool() == True:
                G.append(1)
        if row["Answer.better_answer.answer2"] == True:
            W.append(1)
            if data1[data1['Input.image_id'] == image_id]["Answer.q2_more_time.no"].bool() == True:
                G.append(0)
            if data1[data1['Input.image_id'] == image_id]["Answer.q2_more_time.yes"].bool() == True:
                G.append(1)

    cont_table = pd.crosstab(pd.Series(W, name='W'), pd.Series(G, name='G'))

    return cont_table

def macnemars_test(V1, V2):
    C = contingency_matrix(V1,V2)
    print("Contingency Matrix:\n", C)
    result = mcnemar(C, exact=False, correction=False)
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))

def man_whitney_u_test():

    data1 = pd.read_csv("data_for_mturk/amt_res_effort_majority.csv", encoding='latin1')
    data2 = pd.read_csv("data_for_mturk/amt_res_betterans_majority.csv", encoding='latin1')
    times = data1["WorkTimeInSeconds"].astype(float)

    W, G = [], []
    for index, row in data2.iterrows():
        image_id = row["Input.image_id"]
        if row["Answer.better_answer.answer1"] == True:
            W.append(0)
            G.append(float(data1[data1['Input.image_id'] == image_id]["WorkTimeInSeconds"]))
        if row["Answer.better_answer.answer2"] == True:
            W.append(1)
            G.append(float(data1[data1['Input.image_id'] == image_id]["WorkTimeInSeconds"]))

    group0 = [G[i] for i in range(len(G)) if W[i] == 0]
    group1 = [G[i] for i in range(len(G)) if W[i] == 1]

    stat, p_value = mannwhitneyu(group0, group1, alternative='two-sided')

    print("Mann-Whitney U statistic:", stat)
    print("P-value:", p_value)

def filterColorModifiers(df):
    colors = [
        "red", "blue", "green", "yellow", "pink", "purple", "orange", "brown", "black", "white", "gray", "grey", "maroon", "beige", "gold", "silver", "teal", "turquoise"
        ]

    # Filter rows where any color is present in Modified_question (case-insensitive)
    mask = df["Mod_qns_Llava"].str.lower().apply(lambda x: any(color in x for color in colors))
    filtered_df = df[mask]

    # Save or use filtered_df as needed
    # filtered_df.to_csv("data_from_vlms/gpt4o_visMod_diffAns_colors.csv", index=False)
    return filtered_df

def accuracyTestDiffAns(data):
    # gt_ans, qtype_dict = getGT()
    # vlmlist = ["ans_gpt4o_modques", "ans_gemini_modques", "ans_claude_modques", "ans_llava_modques"]
    # vlmlist = ["ans_gpt4o_origques", "ans_gemini_origques", "ans_claude_origques", "ans_llava_origques"]
    vlmlist = ["Answer.answer_q2"]
    for vlm in vlmlist:
        vec1 = yesNoQuestions(data, vlm)
        vec2 = openEndQuesMatching(data, vlm)
        # print("accuracy yes/no", (len(vec1) - sum(vec1))/len(vec1))
        print("accuracy yes/no", 1- (sum(vec1))/len(vec1))
        print("accuracy open ended", 1 - sum(vec2)/len(vec2))
        print(len(vec1), len(vec2))

def accuracyTestSameAns(data):
    # gt_ans, qtype_dict = getGT()
    # vlmlist = ["answer_gpt4o_question", "ans_gemini_question"]
    # vlmlist = ["ans_gpt4o_origques", "ans_gemini_origques", "ans_claude_origques"]
    vlmlist = ["ans_gpt4o_modques", "ans_gemini_modques", "ans_claude_modques", "ans_llava_modques"]
    for vlm in vlmlist:
        vec1 = yesNoQuestions(data, vlm)
        vec2 = openEndQuesMatching(data, vlm)
        print("accuracy yes/no", sum(vec1)/len(vec1))
        print("accuracy open ended", sum(vec2)/len(vec2))
        print(sum(vec2), len(vec2))
    

def statistical_test(data):
    # read Williams data
    # data =  pd.read_csv("data/vqa_modifier_2024_same.csv", encoding='latin1', keep_default_na=False)
    data_train_gt = pd.read_csv("data/v2_mscoco_train2014_annotations.csv", encoding='latin1')    
    vlm_list = ["ans_GPT4o_modques", "ans_Gemini_modques", "ans_Claude_modques", "ans_Llava_modques"]
    # vlm_list = ["GPT4o_ans_open_ended", "Gemini_ans_open_ended", "Llava_ans_open_ended"]
    vlm_list1 = ["ans_gpt4o_modques", "ans_gemini_modques", "ans_claude_modques", "ans_llava_modques"]
    # vlm_list1 = ["ans_gpt4o_question", "ans_gemini_question", "ans_llava_question"]
    data_for_original = pd.read_csv("data/vqa_modifier_2024_same.csv", encoding='latin1', keep_default_na=False)

    #get the accuracy for each vlm for original questions --exact match 
    for j, vlm in enumerate(vlm_list):
        match_count = 0
        vector1,vector2 = [],[]
        for i, row in data.iterrows():
            imageid = row["image_id"]
            qsnId = data_for_original[data_for_original["image_id"]==imageid]["question_id"].values[0]
            gt_ans = data_train_gt[data_train_gt["question_id"]== qsnId]["answers"]
            gt_ans_str = gt_ans.iloc[0]
            # qsnId = row["question_id"]
            gt_ans = data_train_gt[data_train_gt["question_id"]== qsnId]["answers"]
            gt_ans_str = gt_ans.iloc[0]
            dict_list = ast.literal_eval(gt_ans_str)
            answers = [d['answer'] for d in dict_list]
            majority_ans = majority_vote(answers)
            vlm_ans = row[vlm]
            # vlm_ans = data[data["question_id"] == qsnId][vlm].values[0]
            vlm_ans = vlm_ans.strip().lower().replace(".", "")

            # vlm_ans1 = row[vlm_list1[j]]
            vlm_ans1 = data_for_original[data_for_original["question_id"] == qsnId][vlm_list1[j]].values[0]
            # print("vlm_ans", (vlm,vlm_ans), "vlm_ans1", (vlm_list1[j],vlm_ans1), qsnId)
            # exit()
            vlm_ans1 = vlm_ans1.strip().lower().replace(".", "")
            if vlm_ans == majority_ans:
                match_count += 1
                vector1.append(1)
            else:
                vector1.append(0)
            if vlm_ans1 == majority_ans:
                vector2.append(1)
            else:
                vector2.append(0)
        print("_________________________________")
        print(f"Mcnemars Test: {vlm},{vlm_list1[j]}")
        macnemars_test(vector1, vector2)
    exit()

def filter_open_ended_questions(data):
    qsnType = ['what', 'why', 'who', 'where', 'which']
    _, qtype_dict = getGT()
    # Create a boolean mask for open-ended questions
    mask = data["image_id"].apply(lambda imgid: starts_with_any(qtype_dict[imgid], qsnType))
    return data[mask].copy()


def filterDifferentVLMResponses(df1, col1, df2, col2):
    """
    Filters the DataFrame to keep only rows where the answers to
    modified and original questions are different (case-insensitive, ignores periods and whitespace).
    Handles missing values gracefully.
    Adds a column 'origques_diff' with the original answers that differ.
    """
    origques= df1[col1].fillna("").str.lower().str.strip().str.replace(".", "", regex=False)
    modques = df2[col2].fillna("").str.lower().str.strip().str.replace(".", "", regex=False)
    mask = modques != origques
    filtered_df = df2[mask].copy()
    filtered_df[col1] = origques[mask].values
    return filtered_df

def filterDifferentVLMResponsesExp2(df1, col1, df2, col2):
    """
    Filters the DataFrame to keep only rows where the Gemini model's answers to
    modified and original questions are different (case-insensitive, ignores periods and whitespace).
    Handles missing values gracefully.
    """

    modques = df1[col1].fillna("").str.lower().str.strip().str.replace(".", "", regex=False)
    origques = df2[col2].fillna("").str.lower().str.strip().str.replace(".", "", regex=False)
    filtered_df = df2[modques != origques]
    return filtered_df

def getOpenEndedQuesCount(data):
    qsnType = ['what', 'why', 'who', 'where', 'which']
    _, qtype_dict = getGT()
    openended_count = origTrue = modTrue = 0
    for _, row in data.iterrows():
        imageid = row["image_id"]
        try:
            if starts_with_any(qtype_dict[imageid], qsnType):
                    openended_count += 1
                    if data[data["image_id"]== imageid]["Answer.better_answer.answer1"].any():
                        origTrue += 1
                    if data[data["image_id"]== imageid]["Answer.better_answer.answer2"].any():
                        modTrue += 1
                    if data[data["image_id"]== imageid]["Answer.better_answer.neither"].any():
                        origTrue += 1
                        modTrue += 1
        except Exception as e:
            print(e)

    return openended_count, origTrue, modTrue

def amazonTurkDataAnalysis():
    data = pd.read_csv("data_from_vlms/gpt4o_visMod_sameAns_corrected_full.csv", encoding='latin1', keep_default_na=False)
    openended_total, _, _ = getOpenEndedQuesCount(data)
    vlm = ["gpt4o", "gemini"] #, "claude", "llava"]
    for v in vlm:
        # df = pd.read_csv("data_for_mturk/gpt4ogen/amt_results_{}_majority.csv".format(v), encoding='latin1')
        df = pd.read_csv("data_for_mturk/exp2_amtresponse/{}_majority.csv".format(v), encoding='latin1')
        openended_count, origTrue, modTrue = getOpenEndedQuesCount(df)    
        print("Open ended questions count:", openended_count, origTrue, modTrue)
        percentage_mod_worse = (origTrue/ openended_total)* 100
        print(f"Percentage of modified questions that are worse than original: {percentage_mod_worse:.2f}%")
    exit()
    # get the majority vote for the better answer for all three columns, can do this using chatgpt 
    print(sum(df["Answer.better_answer.answer1"]))
    print(sum(df["Answer.better_answer.answer2"]))
    print(sum(df["Answer.better_answer.neither"]))
    print(len(df))
    # prep_data_for_mturk()
    # contingency

def accuracyDiffTest():
    #read test data
    data1 = pd.read_csv("data_from_vlms/gpt4o_visMod_sameAns_corrected_full.csv", encoding='latin1')
    data2 = pd.read_csv("data_from_vlms/llava_visMod_sameAns_corrected_full.csv", encoding='latin1')
    
    vlm_orig = ["ans_gpt4o_origques", "ans_gemini_origques", "ans_claude_origques", "ans_llava_origques"]
    vlm_mod = ["ans_gpt4o_modques", "ans_gemini_modques", "ans_claude_modques", "ans_llava_modques"]
    
    for i, v in enumerate(vlm_orig):
        # acc_orig = yesNoQuestions(data1, v)
        # acc_mod = yesNoQuestions(data2, vlm_mod[i])
        
        acc_orig = openEndQuesMatching(data1, v)
        acc_mod = openEndQuesMatching(data2, vlm_mod[i])

        orig_mean =  sum(acc_orig)/len(acc_orig)
        mod_mean =  sum(acc_mod)/len(acc_mod)

        print("Original questions yes/no accuracy:", orig_mean)
        print("Modified questions yes/no accuracy:", mod_mean)

        print("% change in accuracy:", ((orig_mean - mod_mean) / orig_mean)*100)

def accuracyDiffTrain():
    df = pd.read_csv("data/vqa_modifier_2024_same.csv", encoding='latin1', keep_default_na=False)
    train_data = pd.read_csv("data/v2_mscoco_train2014_annotations.csv", encoding='latin1')
    data = filterColorModifiers(df)
    data = pd.read_csv("data_from_vlms/Gemma_responses_vqa2024_origqns_130sample.csv", encoding='latin1')
    
    
    acc_orig = openEndQuesMatching(data1, "ans_gpt4o_modques_corr")
    acc_mod = openEndQuesMatching(data2, "ans_gemini_modques_corr")

    acc_orig = openEndQuesMatchingTrain(df, train_data, ["ans_gpt4o_question"])
    acc_mod = openEndQuesMatchingTrain(df, train_data, ["ans_gpt4o_modques"])


    orig_mean =  sum(acc_orig)/len(acc_orig)
    mod_mean =  sum(acc_mod)/len(acc_mod)

    print("Original questions yes/no accuracy:", orig_mean)
    print("Modified questions yes/no accuracy:", mod_mean)

    print("% change in accuracy:", ((orig_mean - mod_mean) / orig_mean)*100)

def dataForMcnemarsTest():
    # read test data
    vlm_original = ["ans_gpt4o_origques", "ans_gemini_origques", "ans_claude_origques", "ans_llava_origques"]
    vlm_modified = ["ans_gpt4o_modques", "ans_gemini_modques", "ans_claude_modques", "ans_llava_modques"]
    data = pd.read_csv("data_from_vlms/gpt4o_visMod_sameAns_corrected_full.csv", encoding='latin1', keep_default_na=False)
    data1 = pd.read_csv("data_from_vlms/llava_visMod_sameAns_corrected_full.csv", encoding='latin1', keep_default_na=False)
    
    for i in range(len(vlm_original)):
        # make a vector of yes/no questions from original
        yn_original = yesNoQuestions(data, vlm_original[i])
        # make a vector of yes/no questions from modified
        yn_modified = yesNoQuestions(data1, vlm_modified[i])
        print(vlm_original[i])
        macnemars_test(yn_original, yn_modified)

def filterQsnWrtQsnType():
    """   
    data1 ---->> original questions
    data2 ---->> modified questions
    1. get the question type for each question in data1 and data2
    2. if data1 and data2 has same questiontype then keep the data else drop the data
    3. return the filtered data1 and data2
    """
    data = pd.read_csv("data_from_vlms/train_vlm_gen_yesno/Qns_yn_Mod_Llava.csv", encoding='latin1', keep_default_na=False)
    count = 0
    for i, row in data.iterrows():
        orig_qsn = row["question"]
        mod_qsn = row["Mod_qns_Llava"]
        qsntype = row["q_type"]
        if orig_qsn.startswith(qsntype) and mod_qsn.startswith(qsntype):
            count += 1
        else:
            print(row["image_id"])
    print("Total questions:", len(data))
    print("Filtered questions:", count)
    print("Percentage of questions that are filtered:", (count/len(data))*100)
    # save the filtered data
    data_filtered = data[data["question"].str.startswith(tuple(qsntype)) & data["Mod_qns_Llava"].str.startswith(tuple(qsntype))]
    data_filtered.to_csv("data_from_vlms/train_vlm_gen_yesno/Qns_yn_Mod_Llava.csv", encoding='latin1', index=False)
    # print("Filtered data saved to Qns_yn_Mod_GPT4o_filtered.csv")

def replaceModifierColor(orig, mod):
    # Remove punctuation and lowercase for comparison
    orig_words = orig.lower().translate(str.maketrans('', '', string.punctuation)).split()
    mod_words = mod.split()
    matcher = difflib.SequenceMatcher(None, orig_words, [w.lower().strip(string.punctuation) for w in mod_words])
    new_mod_words = []
    opcodes = matcher.get_opcodes()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            new_mod_words.extend(mod_words[j1:j2])
        else:
            # Color all extra words in modified question
            new_mod_words.append(f'<span style="color:red">{" ".join(mod_words[j1:j2])}</span>')
    return ' '.join(new_mod_words)

def replace_extra_with_input(orig, mod):
    # Remove punctuation and lowercase for comparison
    orig_words = orig.lower().translate(str.maketrans('', '', string.punctuation)).split()
    mod_words = mod.split()
    matcher = difflib.SequenceMatcher(None, orig_words, [w.lower().strip(string.punctuation) for w in mod_words])
    new_mod_words = []
    opcodes = matcher.get_opcodes()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            new_mod_words.extend(mod_words[j1:j2])
        else:
            # Insert a single input box for the whole sequence of extra words
            input_html = '<input type="text" name="mod_replacement" placeholder="..." style="width: 80px; display:inline;" />'
            new_mod_words.append(input_html)
    return ' '.join(new_mod_words)


def accuracy_selected_yesno_train():
    data_train_gt = pd.read_csv("data/v2_mscoco_train2014_annotations.csv", encoding='latin1')    
    # vlm_list = ["ans_gpt4o_question", "ans_gemini_question", "ans_claude_question", "ans_llava_question"]
    vlm_list = ["ans_GPT4o_modques", "ans_Gemini_modques", "ans_Claude_modques", "ans_Llava_modques"]
    data = pd.read_csv("data_from_vlms/train_vlm_gen_yesno/resp_Qns_yn_Mod_Llava_500.csv", encoding='latin1', keep_default_na=False)
    full_data = pd.read_csv("data/vqa_modifier_2024_same.csv", encoding='latin1', keep_default_na=False)

    #get the accuracy for each vlm for sent questions --exact match 
    for vlm in vlm_list:
        yes_no_vector = []
        for i, row in data.iterrows():
            imageid = row["image_id"]
            # pred_ans = full_data[full_data["image_id"]==imageid][vlm].values[0]
            pred_ans = row[vlm]
            pred_ans = pred_ans.strip().lower().replace(".", "")
            qsnId = full_data[full_data["image_id"]==imageid]["question_id"].values[0]
            gt_ans = data_train_gt[data_train_gt["question_id"]== qsnId]["answers"]
            gt_ans_str = gt_ans.iloc[0]
            dict_list = ast.literal_eval(gt_ans_str)
            answers = [d['answer'] for d in dict_list]
            majority_ans = majority_vote(answers)  # Get majority from ground truth
            res = areEquivalent(majority_ans, pred_ans)
            if res:
                yes_no_vector.append(1)
            else:
                yes_no_vector.append(0)
        print("accuracy for", vlm, sum(yes_no_vector)/len(yes_no_vector))

def normalize(text):
    if pd.isna(text):
        return ""
    return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

def exp1HumanEval():
    data = pd.read_csv("data_from_vlms/llava_visMod_sameAns_corrected_full.csv", encoding='latin1', keep_default_na=False)
    fd1 = filter_open_ended_questions(data)
    data_original = pd.read_csv("data_from_vlms/gpt4o_visMod_sameAns_corrected_full.csv", encoding='latin1', keep_default_na=False)
    fd2 = filter_open_ended_questions(data_original)
    fil_data = filterDifferentVLMResponses(fd2, "ans_gpt4o_origques", fd1, "ans_gpt4o_modques")
    fil_data.to_csv("data_for_mturk/llavagen/visMod_gpt4o_diff_responses.csv", encoding='latin1', index=False)

def exp2HumanEval():
    # for train data, human modified and ai modified questions
    data = pd.read_csv("data_from_vlms/train_vlm_gen_openended/gpt4o_train_130_openended.csv", encoding='latin1', keep_default_na=False)
    human_gen_res = "ans_llava_modques_sent"
    ai_gen_res = "ans_llava_modques"
    # data from human gen and ai that get different answers
    fd1 = filterDifferentVLMResponses(data, human_gen_res, data, ai_gen_res)
    
    # check if sentences are exactly same after normalization
    # fd1['is_equal'] = fd1.apply(lambda row: normalize(row['sent']) == normalize(row['GPT4o_response_open_ended']), axis=1)
    # print(fd1[fd1['is_equal']])
    # 2. all the ai modified questions,     print(combined)

    data_gemini = pd.read_csv("data_from_vlms/train_vlm_gen_openended/gemini_train_130_openended.csv", encoding='latin1', keep_default_na=False)
    # gemini_gen_res = "ans_gemini_modques"
    fd2 = filterDifferentVLMResponses(data, human_gen_res, data_gemini, ai_gen_res)
    combined = pd.concat([fd1, fd2], ignore_index=True)
    
    data_claude = pd.read_csv("data_from_vlms/train_vlm_gen_openended/claude_train_130_openended.csv", encoding='latin1', keep_default_na=False)
    # claude_gen_res = "ans_gemini_modques"
    fd2 = filterDifferentVLMResponses(data, human_gen_res, data_claude, ai_gen_res)
    combined = pd.concat([combined, fd2], ignore_index=True)

    data_llava = pd.read_csv("data_from_vlms/train_vlm_gen_openended/llava_train_130_openended.csv", encoding='latin1', keep_default_na=False)
    # llava_gen_res = "ans_gemini_modques"
    fd2 = filterDifferentVLMResponses(data, human_gen_res, data_llava, ai_gen_res)
    combined = pd.concat([combined, fd2], ignore_index=True)
    
    
    combined = combined.drop_duplicates(subset='question_id_unique', keep='first')
    # for each question_id_unique get the gpt4o response for sent from first file, add them in the combined and remove the unwanted columns 
    # Create a mapping from question_id_unique to ans_GPT4o_modques_sent in data
    sent_map = data.set_index('question_id_unique')[human_gen_res].to_dict()

    # Fill NaN values in combined['ans_GPT4o_modques_sent'] using the mapping
    combined[human_gen_res] = combined.apply(
        lambda row: sent_map.get(row['question_id_unique'], row[human_gen_res])
        if pd.isna(row[human_gen_res]) else row[human_gen_res],
        axis=1
    )

    sent_map = data.set_index('question_id_unique')['image_id'].to_dict()

    # Fill NaN values in combined['ans_GPT4o_modques_sent'] using the mapping
    combined['image_id'] = combined.apply(
        lambda row: sent_map.get(row['question_id_unique'], row['image_id'])
        if pd.isna(row['image_id']) else row['image_id'],
        axis=1
    )

    sent_map = data.set_index('question_id_unique')['question'].to_dict()

    # Fill NaN values in combined['ans_GPT4o_modques_sent'] using the mapping
    combined['question'] = combined.apply(
        lambda row: sent_map.get(row['question_id_unique'], row['question'])
        if pd.isna(row['question']) else row['question'],
        axis=1
    )
    sent_map = data.set_index('question_id_unique')['sent'].to_dict()

    # Fill NaN values in combined['ans_GPT4o_modques_sent'] using the mapping
    combined['sent'] = combined.apply(
        lambda row: sent_map.get(row['question_id_unique'], row['sent'])
        if pd.isna(row['sent']) else row['sent'],
        axis=1
    )

    columns_to_keep = [
    "question_id_unique",
    "image_id",
    "question",
    "sent",
    "Modified_question",
    human_gen_res,
    ai_gen_res,
    ]
    combined = combined[columns_to_keep]
    data_for_mturk = imageUrl(combined)
    return data_for_mturk


def filterDifferentVLMResponsesTrainYesNo(df1, col1, df2, col2, key="image_id"):
    # Merge the two DataFrames on the key
    merged = pd.merge(df1[[key, col1]], df2[[key, col2]], on=key, suffixes=('_df1', '_df2'))
    # Normalize both columns
    modques = merged[col1].fillna("").str.lower().str.strip().str.replace(".", "", regex=False)
    origques = merged[col2].fillna("").str.lower().str.strip().str.replace(".", "", regex=False)
    # Filter rows where responses are different
    filtered_df = merged[modques != origques].copy()

    sent_map = df1.set_index(key)['sent'].to_dict()
    filtered_df.loc[:, 'sent'] = filtered_df[key].map(sent_map)
    sent_map = df2.set_index(key)['Mod_qns_GPT4o'].to_dict()
    filtered_df.loc[:, 'Mod_qns_GPT4o'] = filtered_df[key].map(sent_map)
    return filtered_df


def get_human_questions_matching_gt(human_gen, gt_col="ans_gpt4o_modques", question_col="question", image_id_col="image_id"):
    """
    Returns a list of human-generated questions whose answers match the majority ground truth answer.
    """
    data_train_gt = pd.read_csv("data/v2_mscoco_train2014_annotations.csv", encoding='latin1')    
    matching_row = []
    for _, row in human_gen.iterrows():
        image_id = row[image_id_col]
        human_ans = str(row[gt_col]).strip().lower().replace(".", "")
        qsnId = human_gen[human_gen["image_id"]==image_id]["question_id"].values[0]
        gt_ans = data_train_gt[data_train_gt["question_id"]== qsnId]["answers"]
        gt_ans_str = gt_ans.iloc[0]
        dict_list = ast.literal_eval(gt_ans_str)
        answers = [d['answer'] for d in dict_list]
        majority_ans = majority_vote(answers)  # Get majority from ground truth
        if human_ans == majority_ans:
            matching_row.append(row)
    return pd.DataFrame(matching_row)

def openEndQuesMatchingHumanEval1(data, qsnType_):
    # qsnType = ['what', 'why', 'who', 'where', 'which']
    qsnType = [qsnType_]
    gt, qtype_dict = getGT()
    vector = []
    for _, row in data.iterrows():
        imageid = row["image_id"]
        # gt_ans = gt[imageid]
        # pred_ans = row[model]
        # if pd.isna(pred_ans):
            # continue
        # pred_ans = " ".join(pred_ans.split()).lower().replace(".", "")

        try:
            if starts_with_any(qtype_dict[imageid], qsnType):
                # gt_ans = [ans.strip().lower().replace(".", "") for ans in gt_ans]
                # if pred_ans.lower() in ["unknown", "none", "unclear"] or re.search(r"\bno\b", pred_ans.lower()):
                # if pred_ans in gt_ans:
                    # yes_no_vector.append(1)
                # else:
                    # yes_no_vector.append(0)
                vector.append(row)
                
        except Exception as e:
            print(e)

    return pd.DataFrame(vector)

def dataForMcnemarsTrain():
    # read test data
    vlm_original = ["ans_gpt4o_origques", "ans_gemini_origques", "ans_claude_origques", "ans_llava_origques"]
    vlm_modified = ["ans_gpt4o_modques", "ans_gemini_modques", "ans_claude_modques", "ans_llava_modques"]
    data = pd.read_csv("data_from_vlms/gpt4o_visMod_sameAns_corrected_full.csv", encoding='latin1', keep_default_na=False)
    data1 = pd.read_csv("data_from_vlms/llava_visMod_sameAns_corrected_full.csv", encoding='latin1', keep_default_na=False)
    
    for i in range(len(vlm_original)):
        # make a vector of yes/no questions from original
        yn_original = yesNoQuestions(data, vlm_original[i])
        # make a vector of yes/no questions from modified
        yn_modified = yesNoQuestions(data1, vlm_modified[i])
        print(vlm_original[i])
        macnemars_test(yn_original, yn_modified)
    
def accuracy_selected_yesno_train():
    data_train_gt = pd.read_csv("data/v2_mscoco_train2014_annotations.csv", encoding='latin1')    
    # vlm_list = ["ans_gpt4o_question", "ans_gemini_question", "ans_claude_question", "ans_llava_question"]
    vlm_list = ["ans_GPT4o_modques", "ans_Gemini_modques", "ans_Claude_modques", "ans_Llava_modques"]
    data = pd.read_csv("data_from_vlms/train_vlm_gen_yesno/resp_Qns_yn_Mod_Llava_500.csv", encoding='latin1', keep_default_na=False)
    full_data = pd.read_csv("data/vqa_modifier_2024_same.csv", encoding='latin1', keep_default_na=False)

    #get the accuracy for each vlm for sent questions --exact match 
    for vlm in vlm_list:
        yes_no_vector = []
        for i, row in data.iterrows():
            imageid = row["image_id"]
            # pred_ans = full_data[full_data["image_id"]==imageid][vlm].values[0]
            pred_ans = row[vlm]
            pred_ans = pred_ans.strip().lower().replace(".", "")
            qsnId = full_data[full_data["image_id"]==imageid]["question_id"].values[0]
            gt_ans = data_train_gt[data_train_gt["question_id"]== qsnId]["answers"]
            gt_ans_str = gt_ans.iloc[0]
            dict_list = ast.literal_eval(gt_ans_str)
            answers = [d['answer'] for d in dict_list]
            majority_ans = majority_vote(answers)  # Get majority from ground truth
            res = areEquivalent(majority_ans, pred_ans)
            if res:
                yes_no_vector.append(1)
            else:
                yes_no_vector.append(0)
        print("accuracy for", vlm, sum(yes_no_vector)/len(yes_no_vector))


def main():
    train_data = pd.read_csv("data/v2_mscoco_train2014_annotations.csv", encoding='latin1')
    data_human = pd.read_csv("/home/monika/Downloads/humangen_full_with_qid_by_sent_with_qtype_allcols.csv", encoding='latin1', keep_default_na=False)
    data_ai = pd.read_csv("data_for_mturk/human_ai_time_exp/gpt4o_gen.csv", encoding='latin1', keep_default_na=False)

    vec_human = yesNoQuestionsTrain(data_human, train_data, ["Input.ans_gpt4o_modques"])

    # data = pd.read_csv("data_from_vlms/test_vlm_gen_diff_ans_random_modifier/random_mod_100_first_row_per_image.csv")
    # accuracyTestDiffAns(data)

    data = pd.read_csv("data_from_vlms/gpt4o_visMod_sameAns_corrected_full.csv", encoding='latin1', keep_default_na=False)
    # data = filterColorModifiers(data)
    accuracyTestDiffAns(data)

    # # read the yes/no question generated by human
    # #  read gpt4o  questions
    # # make a list of all the human generated questions whose answers are same as GT answers
    ## for these list of questions, get the gpt4o generated questions
    human_gen = pd.read_csv("data/vqa_modifier_2024_same.csv", encoding='latin1', keep_default_na=False) 
    ai_gen = pd.read_csv("data_from_vlms/train_vlm_gen_yesno/resp_Qns_yn_Mod_GPT4o_500.csv", encoding='latin1', keep_default_na=False)
    matching_df = get_human_questions_matching_gt(human_gen)
    # for all the image ids in matching df get the gpt4o generated questions
    # Merge matching_df and ai_gen on image_id
    merged = pd.merge(
        matching_df,
        ai_gen,
        on="image_id",
        suffixes=('_human', '_gpt4o')
    )

    # Select columns you want to keep (example: question and answer columns from both)
    columns_to_keep = [
        "image_id",
        "image_url",
        "sent",         # human_gen question column
        "ans_gpt4o_modques",# human_gen answer column
        "Mod_qns_GPT4o",          # ai_gen question column
        "ans_GPT4o_modques"       # ai_gen answer column
    ]
    # Adjust column names above to match your actual DataFrame columns
    result = merged[columns_to_keep]

    # Save to CSV
    result.to_csv("common_imageid_questions_answers.csv", index=False, encoding='latin1')
    
    exit()
    # human_gen_res = "ans_gpt4o_modques"
    # ai_gen_res = "ans_GPT4o_modques"
    # # data from human gen and ai that get different answers
    # fd1 = filterDifferentVLMResponsesTrainYesNo(human_gen, human_gen_res, ai_gen, ai_gen_res, key = "image_id")
    # fd1 = imageUrl(fd1)
    
    # fd1.to_csv("data_from_vlms/train_vlm_gen_yesno/gpt4o_diff_response.csv", encoding='latin1', index=False)
    # exit()


    df = pd.read_csv("data_for_mturk/gpt4ogen/amt_results_gemini_majority.csv", encoding='latin1')
    qsnType = ['what', 'why', 'who', 'where', 'which']
    for i in qsnType:
        data = openEndQuesMatchingHumanEval1(df, i)
        print((sum(data["Answer.better_answer.answer1"]) + sum(data["Answer.better_answer.neither"]))/ len(data))
        print(len(data))
    exit()
    # amazonTurkDataAnalysis()
    data = exp1HumanEval()
    
    exit()
    # statistical_test(data)
    data = pd.read_csv("data_from_vlms/test_vlm_gen_diff_ans/gpt4o_diffAns.csv", encoding='latin1', keep_default_na=False)
    # data = filterColorModifiers(data)
    accuracyTestDiffAns(data)
    exit()
    # openEndQuesMatchingTrain(df, train_data)
    # yesNoQuestionsTrain(df, train_data)
    

if __name__ == "__main__":
    main()
