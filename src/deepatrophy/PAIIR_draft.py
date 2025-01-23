import os
import random
import shutil
import numpy as np
import pandas as pd
import csv
import scipy
import scipy.linalg
import pathlib
import argparse
import h5py
from sklearn.linear_model import LinearRegression
from datetime import datetime
from sklearn.metrics import confusion_matrix
# from scipy.io import savemat
# from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

workdir='/data/mengjin/DeepAtrophyPackage/DeepAtrophy/PAIIR'
prefix='resnet50_2020-07-08_12-39'

train_pair_spreadsheet='/data/mengjin/DeepAtrophyPackage/DeepAtrophy/data/resnet50_2020-07-08_12-39train_0135_train_pair_update.csv'
test_pair_spreadsheet='/data/mengjin/DeepAtrophyPackage/DeepAtrophy/data/resnet50_2020-07-08_12-39train_0135_test_pair_update.csv'
test_double_pair_spreadsheet='/data/mengjin/DeepAtrophyPackage/DeepAtrophy/data/resnet50_2020-07-08_12-39train_0135_test_modified.csv'

min_date = 180
max_date = 400

# Read and organize the data

# Read CSV files
train_pair = pd.read_csv(train_pair_spreadsheet)
train_pair = train_pair.rename(columns={
    'bl_time1': 'bl_time',
    'fu_time1': 'fu_time',
    'date_diff1': 'date_diff_true',
    'label_date_diff1': 'label_date_diff',
    'pred_date_diff1': 'pred_date_diff'
})

# Subset the data for stage == 0
train_spreadsheet0 = train_pair[train_pair['stage'] == 0]

# Load the test pair CSV
test_pair = pd.read_csv(test_pair_spreadsheet)

# Rename columns in the test pair
test_pair = test_pair.rename(columns={
    'bl_time1': 'bl_time',
    'fu_time1': 'fu_time',
    'date_diff1': 'date_diff_true',
    'label_date_diff1': 'label_date_diff',
    'pred_date_diff1': 'pred_date_diff'
})

# Fit the linear model
X_train = train_spreadsheet0[['score0', 'score1', 'score2', 'score3', 'score4']]
y_train = train_spreadsheet0['date_diff_true']
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test pair
X_test = test_pair[['score0', 'score1', 'score2', 'score3', 'score4']]
test_pair['CNN_date_diff'] = model.predict(X_test)

# Convert date columns to datetime format
test_pair['bl_time'] = pd.to_datetime(test_pair['bl_time']) # , format='%Y-%m-%d'
test_pair['fu_time'] = pd.to_datetime(test_pair['fu_time'])
train_pair['bl_time'] = pd.to_datetime(train_pair['bl_time'])
train_pair['fu_time'] = pd.to_datetime(train_pair['fu_time'])

# Now test_spreadsheet and train_spreadsheet have the date columns in proper format and test_spreadsheet has predictions.
test_double_pair = pd.read_csv(test_double_pair_spreadsheet)

# print column names
column_names = test_double_pair.columns
print("column names of double pair spreadsheet", column_names)


def run():


    # STO accuracy
    STO_accuracy()

    # RISI accuracy
    RISI_accuracy()

    # obtain a single measurement for each subject (predicted interscan interval, and PAIIR)
    obtain_PAIIR()

    # plot PAIIR for each stage, not corrected for age
    plot_PAIIR()

    # print("finished runnning PAIIRLauncher.run")
    

def STO_accuracy():

    # Assuming common_subj_1 is already a pandas DataFrame

    column_names = test_pair.columns
    print(column_names)

    # Mutate to create CNN_pred_date_diff
    test_pair['CNN_pred_date_diff'] = test_pair['CNN_date_diff'].apply(lambda x: 1 if x > 0 else 0)

    # Convert columns to categorical (factor equivalent in R)
    test_pair['CNN_pred_date_diff'] = test_pair['CNN_pred_date_diff'].astype(int).astype('category')
    test_pair['label_date_diff'] = test_pair['label_date_diff'].astype(int).astype('category')

    # Accuracy for CNN by group (stage)
    cnn_accuracy_by_group = test_pair.groupby('stage').apply(
        lambda group: (group['label_date_diff'] == group['CNN_pred_date_diff']).sum() / len(group)
    ).reset_index(name='accuracy')

    # Overall accuracy for CNN
    cnn_overall_accuracy = (test_pair['label_date_diff'] == test_pair['CNN_pred_date_diff']).mean()

    # Print results
    print("CNN accuracy by group:")
    print(cnn_accuracy_by_group)

    print("\nOverall CNN accuracy:", cnn_overall_accuracy)

    return


column_names = test_double_pair.columns
print("column names of single pair spreadsheet", column_names)

def RISI_accuracy():

    global test_double_pair, test_pair

    test_pair = test_pair.rename(columns={
        'bl_time': 'bl_time1',
        'fu_time': 'fu_time1'
    })

    test_double_pair['side'] = test_double_pair['side'].astype(str)
    test_pair['side'] = test_pair['side'].astype(str)

    test_double_pair['stage'] = test_double_pair['stage'].astype(int)
    test_pair['stage'] = test_pair['stage'].astype(int)

    test_double_pair['bl_time1'] = pd.to_datetime(test_double_pair['bl_time1'], format="%m/%d/%y %H:%M")
    test_double_pair['fu_time1'] = pd.to_datetime(test_double_pair['fu_time1'], format="%m/%d/%y %H:%M")

    test_double_pair['bl_time2'] = pd.to_datetime(test_double_pair['bl_time2'], format="%m/%d/%y %H:%M")
    test_double_pair['fu_time2'] = pd.to_datetime(test_double_pair['fu_time2'], format="%m/%d/%y %H:%M")

    test_pair['bl_time1'] = pd.to_datetime(test_pair['bl_time1'], format="%m/%d/%y %H:%M")
    test_pair['fu_time1'] = pd.to_datetime(test_pair['fu_time1'], format="%m/%d/%y %H:%M")


    test_double_pair1 = test_double_pair.merge(test_pair, on=["subjectID", "side", "stage", "bl_time1", "fu_time1"]) \
                                .rename(columns={'CNN_date_diff': 'CNN_date_diff1'})
    
    test_pair = test_pair.rename(columns={
        'bl_time1': 'bl_time2',
        'fu_time1': 'fu_time2'
    })

    test_double_pair2 = test_double_pair1.merge(test_pair, on=["subjectID", "side", "stage", "bl_time2", "fu_time2"]) \
                            .rename(columns={'CNN_date_diff': 'CNN_date_diff2'}) \
                            .assign(CNN_date_diff_ratio=lambda df: df.apply(
                                lambda row: 1 if abs(row['CNN_date_diff2']) > abs(row['CNN_date_diff1']) else 0, axis=1)) \
                            .assign(label_time_interval_binary=lambda df: df.apply(
                                lambda row: 1 if row['label_time_interval'] >= 2  else 0, axis=1))

    test_pair = test_pair.rename(columns={
        'bl_time2': 'bl_time',
        'fu_time2': 'fu_time'
    })

    # Group by 'stage' and calculate accuracy for CNN by group
    cnn_accuracy_by_group = test_double_pair2.groupby('stage').apply(
        lambda group: (group['label_time_interval_binary'] == group['CNN_date_diff_ratio']).sum() / len(group)
    ).reset_index(name='accuracy')

    # Overall accuracy for CNN
    cnn_overall_accuracy = (test_double_pair2['label_time_interval_binary'] == test_double_pair2['CNN_date_diff_ratio']).mean()

    # Convert label_time_interval and CNN_date_diff_ratio to categorical
    test_double_pair2['label_time_interval_binary'] = test_double_pair2['label_time_interval_binary'].astype('category')
    test_double_pair2['CNN_date_diff_ratio'] = test_double_pair2['CNN_date_diff_ratio'].astype('category')

    # Confusion matrix for CNN
    cnn_confusion_matrix = confusion_matrix(test_double_pair2['label_time_interval_binary'], test_double_pair2['CNN_date_diff_ratio'])

    # Print results
    print("CNN accuracy by group:")
    print(cnn_accuracy_by_group)

    print("\nOverall CNN accuracy:", cnn_overall_accuracy)

    print("\nCNN Confusion Matrix:")
    print(cnn_confusion_matrix)

    print("\n manual calculation of confusion matrix")

    test_double_pair2.to_csv(workdir + '/test_double_pair2.csv', index=False)

    # Iterate through each unique stage
    for stage in test_double_pair2['stage'].unique():
        # Filter DataFrame for the current stage
        test_double_pair_temp_stage = test_double_pair2[test_double_pair2['stage'] == stage]

        count1 = test_double_pair_temp_stage[(test_double_pair_temp_stage['label_time_interval_binary'] == 0) & (test_double_pair_temp_stage['CNN_date_diff_ratio'] == 0)].shape[0]
        count2 = test_double_pair_temp_stage[(test_double_pair_temp_stage['label_time_interval_binary'] == 0) & (test_double_pair_temp_stage['CNN_date_diff_ratio'] == 1)].shape[0]
        count3 = test_double_pair_temp_stage[(test_double_pair_temp_stage['label_time_interval_binary'] == 1) & (test_double_pair_temp_stage['CNN_date_diff_ratio'] == 0)].shape[0]
        count4 = test_double_pair_temp_stage[(test_double_pair_temp_stage['label_time_interval_binary'] == 1) & (test_double_pair_temp_stage['CNN_date_diff_ratio'] == 1)].shape[0]

        print("confusion matrix for stage ", stage, " is:")

        print("count1", count1)
        print("count2", count2)
        print("count3", count3)
        print("count4", count4)

    return

def obtain_PAIIR():

    common_subj_2 = test_pair.copy()

    # save csv file
    common_subj_2.to_csv(workdir + '/common_subj_2.csv', index=False)

    # Update columns conditionally based on the value of date_diff_true
    common_subj_2["bl_time"] = common_subj_2.apply(
        lambda row: row["bl_time"] if row["date_diff_true"] > 0 else row["fu_time"], axis=1
    )
    common_subj_2["fu_time"] = common_subj_2.apply(
        lambda row: row["fu_time"] if row["date_diff_true"] > 0 else row["bl_time"], axis=1
    )
    common_subj_2["CNN_date_diff"] = common_subj_2.apply(
        lambda row: row["CNN_date_diff"] if row["date_diff_true"] > 0 else -row["CNN_date_diff"], axis=1
    )
    common_subj_2["date_diff_true"] = common_subj_2.apply(
        lambda row: row["date_diff_true"] if row["date_diff_true"] > 0 else -row["date_diff_true"], axis=1
    )

    # Step 3: Aggregating by specific columns and computing the mean
    common_subj_3 = common_subj_2.groupby(["subjectID", "stage", "bl_time", "fu_time"]).agg({
        "CNN_date_diff": "mean",
        "date_diff_true": "mean",
    }).reset_index()

    # Final result
    common_subj = common_subj_3

    obtain_PAIIR_per_subject(common_subj)

    return


def obtain_PAIIR_per_subject(common_subj):
    
    common_subj_zoom_in = common_subj[(common_subj['date_diff_true'] >= min_date) & (common_subj['date_diff_true'] < max_date)]

    common_subj_zoom_in['CNN_date_diff_ratio'] = common_subj_zoom_in['CNN_date_diff'] / common_subj_zoom_in['date_diff_true']


    # Calculate weighted sum for the numerator
    weighted_sum_nume_CNN = common_subj_zoom_in.groupby("subjectID").apply(
        lambda x: (x['CNN_date_diff_ratio'] * x['date_diff_true'] * 365).sum()
    ).reset_index(name="CNN_weighted_sum_nume")

    # Calculate weighted sum for the denominator
    weighted_sum_deno = common_subj_zoom_in.groupby("subjectID").apply(
        lambda x: (x['date_diff_true'] * x['date_diff_true']).sum()
    ).reset_index(name="weighted_sum_deno")

    global weighted_atrophy

    # Join the weighted sums and add distinct fields for each subjectID
    weighted_atrophy = (
        weighted_sum_nume_CNN
        .merge(weighted_sum_deno, on="subjectID")
        .merge(common_subj_zoom_in.drop_duplicates("subjectID"), on="subjectID")
    )

    # Calculate the DA_Atrophy_raw column
    weighted_atrophy['DA_Atrophy'] = (
        weighted_atrophy['CNN_weighted_sum_nume'] / weighted_atrophy['weighted_sum_deno']
    )

    # Calculate the PAIIR column
    print("weighted_atrophy.head()")
    print(weighted_atrophy.head())

    weighted_atrophy.to_csv(workdir + '/weighted_atrophy.csv', index=False)

    return

def plot_PAIIR():
    
    # Grouping by stage and calculating the mean of weighted_atrophy

    stage_labels = {0: "A- CU", 1: "A+ CU", 3: "A+ eMCI", 5: "A+ lMCI"}
    stage_order = ["A- CU", "A+ CU", "A+ eMCI", "A+ lMCI"]

    weighted_atrophy['stage'] = weighted_atrophy['stage'].map(stage_labels)
    # mean_values = weighted_atrophy.groupby('stage')['DA_Atrophy'].mean().reset_index()
    # mean_std = weighted_atrophy.groupby('stage')['DA_Atrophy'].agg(['mean', 'std']).reindex(stage_order).reset_index()
    
    mean_se = weighted_atrophy.groupby('stage')['DA_Atrophy'].agg(
        mean='mean', 
        se=lambda x: np.std(x, ddof=1) / np.sqrt(len(x))  # Standard error
    ).reindex(stage_order).reset_index()

    print("mean_se", mean_se)

    # Create a bar plot
    plt.figure(figsize=(8, 6))
    sns.set_palette("Paired")  # Brighter color palette

    # bar_plot = sns.barplot(x='stage', y='DA_Atrophy', data=mean_values, order=stage_order, ci=None)
    # sns.barplot(x=mean_std['stage'], y=mean_std['mean'], yerr=mean_std['std'], capsize=0.1)
    bar_plot = sns.barplot(x='stage', y='mean', data=mean_se, order=stage_order, ci=None)


    plt.errorbar(x=range(len(mean_se)), y=mean_se['mean'], yerr=mean_se['se'], fmt='none', c='black', capsize=5)

    # # Perform t-tests comparing each stage with stage "0"
    # p_values = {}
    # for stage in mean_values['stage']:
    #     if stage != 'A- CU':  # Compare only with stage 0
    #         stage_0_values = weighted_atrophy[weighted_atrophy['stage'] == 'A- CU']['DA_Atrophy']
    #         stage_values = weighted_atrophy[weighted_atrophy['stage'] == stage]['DA_Atrophy']
    #         t_stat, p_value = stats.ttest_ind(stage_0_values, stage_values, equal_var=False)  # Welch's t-test
    #         p_values[stage] = p_value

    # print("p_values", p_values)

    # Perform t-tests for each group against "A- CU" and display p-values
    base_group = weighted_atrophy[weighted_atrophy['stage'] == "A- CU"]['DA_Atrophy']
    for idx, stage in enumerate(stage_order[1:]):  # Skip the base group itself
        comparison_group = weighted_atrophy[weighted_atrophy['stage'] == stage]['DA_Atrophy']
        _, p_value = stats.ttest_ind(base_group, comparison_group)
        
        # Format p-value in scientific notation
        p_text = f"p = {p_value:.1e}"
        bar_plot.text(idx + 1, mean_se['mean'][idx + 1] + mean_se['se'][idx + 1] * 1.1, p_text, ha='center', color='black')

    # Add titles and labels
    plt.title('Mean Weighted PAIIR by Diagnosis (Comparing with A- CU)')
    plt.xlabel('Diagnosis')
    plt.ylabel('Weighted PAIIR (mean ± SE)')
    plt.xticks(ticks=range(len(stage_order)), labels=stage_order, rotation=45)
    # plt.ylim(0, max(mean_values['DA_Atrophy']) + 0.1)  # Adjust y limit for p-value display

    # Show the plot
    plt.show()

   
run()