import pandas as pd
import csv

"""
positive
    /m/01b82r -- sawing
    /m/01j4z9 -- chainsaw
    /m/0_ksk -- power tool

negative
    /m/06_y0by -- environmental
    /m/020bb7 -- birdy stuff
    /m/028v0c -- silence
"""

negative_labels = ['m06_y0by', 'm020bb7', 'm028v0c']

csv_list = ["dataset/google_audio_dataset/unbalanced_1_modified.csv","dataset/google_audio_dataset/unbalanced_2_modified.csv", "dataset/google_audio_dataset/unbalanced_3_modified.csv", "dataset/google_audio_dataset/unbalanced_4_modified.csv", "dataset/google_audio_dataset/unbalanced_5_modified.csv", "dataset/google_audio_dataset/eval_segments_modified.csv", "dataset/google_audio_dataset/balanced_train_modified.csv"]
# csv_list = ['dataset/google_audio_dataset/balanced_train_modified.csv']


def read_csv(csv_path):
    return pd.read_csv(csv_path, low_memory=False)

def fetch_req_tag(df, tag):
    df1 = df[df['positive_labels'].str.contains(tag)]
    return df1

def write_csv(df, filename):
    df.to_csv(filename, index=False)

def concat_dfs(df, df1):
    return pd.concat([df, df1], axis=0)



# if __name__ == '__main__':
#     for i, dataset_path in enumerate(csv_list):
#         df = read_csv(dataset_path)
#         new_file_path = "dataset/google_audio_dataset/main/negative/" + str(i) + '.csv'

#         with open(new_file_path, 'w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(['# YTID', 'start_seconds', 'end_seconds', 'positive_labels'])

#         for tag in negative_labels:
#             df_req_labels = fetch_req_tag(df, tag)
#             df_merged = concat_dfs(read_csv(new_file_path), df_req_labels)
#             write_csv(df_merged, new_file_path)

# import os
# directory = "dataset/google_audio_dataset/main/negative/"

# df_list = []

# for filename in os.listdir(directory):
#     if filename.endswith('.csv'):
#         file_path = os.path.join(directory, filename)
#         df = pd.read_csv(file_path)
#         df_list.append(df)

# # Concatenate all data frames into a single data frame
# concatenated_df = pd.concat(df_list, axis=0)

# # Write the concatenated data frame to a CSV file
# output_file = 'dataset/google_audio_dataset/main/negative/metadata.csv'
# concatenated_df.to_csv(output_file, index=False)

    
df = read_csv('dataset/google_audio_dataset/main/negative/metadata.csv')
df = df.drop_duplicates(df)
write_csv(df, "dataset/google_audio_dataset/main/negative/metadata_drop.csv")





# df3, df4 = pd.read_csv('google_audio_dataset1.csv'), pd.read_csv('google_audio_dataset2.csv')

# # # Merge the DataFrames
# merged_data = pd.concat([df5, df6, df7], axis=1)

# # # Write the merged data to a new CSV file
# merged_data.to_csv("dataset/google_audio_dataset/google_audio_dataset.csv", index=False)

# merged_data.to_csv('dataset/google_audio_dataset/google_audio_dataset2.csv')
# print(df.head)

# df_positive_labels = df1.drop(df1[~df1[' num_positive_labels=52882'].isin('m01b82r')].index)
# df_dropped = df1[~df1[' num_positive_labels=52882'].str.contains('m01b82r'))]



# df_pos_label2 = df1[df1['m04rlf m09x0r m0ytgt'].str.contains("m01j4z9")]
# df_pos_label3 = df1[df1['m04rlf m09x0r m0ytgt'].str.contains("m0_ksk")]
# df_pos_label1 = df1[df1['m04rlf m09x0r m0ytgt'].str.contains("m01b82r")]

# df_pos_labels = pd.concat([df_pos_label1, df_pos_label2, df_pos_label3])

# df_pos_labels = df_pos_labels.drop_duplicates()

# df_pos_labels.to_csv('dataset/google_audio_dataset/google_audio_dataset7.csv')
# print(pd.read_csv('dataset/google_audio_dataset/google_audio_dataset7.csv'))


# df1 = pd.read_csv("dataset/google_audio_dataset/google_audio_dataset1.csv", low_memory=False)
# df2 = pd.read_csv("dataset/google_audio_dataset/google_audio_dataset2.csv", low_memory=False)
# df3 = pd.read_csv("dataset/google_audio_dataset/google_audio_dataset3.csv", low_memory=False)
# df4 = pd.read_csv("dataset/google_audio_dataset/google_audio_dataset4.csv", low_memory=False)
# df5 = pd.read_csv("dataset/google_audio_dataset/google_audio_dataset5.csv", low_memory=False)
# df6 = pd.read_csv("dataset/google_audio_dataset/google_audio_dataset6.csv", low_memory=False)
# df7 = pd.read_csv("dataset/google_audio_dataset/google_audio_dataset7.csv", low_memory=False)


# df_pos_labels = pd.concat([df1, df2], axis=1)
# df_pos_labels = df_pos_labels.drop_duplicates()
# df_pos_labels.to_csv('dataset/google_audio_dataset/main/positive/metadata.csv')
