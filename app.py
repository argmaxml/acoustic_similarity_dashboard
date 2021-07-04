import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import os
import re
from datetime import date
from pathlib import Path
from multiapp import MultiApp
import base64
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_squared_error
from io import StringIO

sns.set()
# load annotators data
annotation_data_dir = Path(os.getcwd()).absolute() / 'ground_truth'
annotation_files = [f for f in annotation_data_dir.rglob('*.csv')]
dict_annotations_dfs = {f.parts[-1].split('.')[0]: pd.read_csv(f) for f in annotation_files}

# load songid's genres
df_genres = pd.read_csv('song_genres.csv')
df_genres.set_index('SONGID', inplace=True)

# load embedding data
data_dir = annotation_data_dir.parent / 'embeddings'
data_files = [f for f in data_dir.rglob('*.parquet')]
dict_data_embeddings = {f.parts[-1].split('.')[0]: pd.read_parquet(f) for f in data_files}
for k in dict_data_embeddings.keys():
    dict_data_embeddings[k].index = dict_data_embeddings[k].index.astype(int)

# shown tables
df_summary_table = pd.read_csv(annotation_data_dir.parent / 'logs' / 'results_log.csv')


######################################################################################
##########################  work functions   #########################################
######################################################################################


def calc_similarity(embeddings, annotations, reference, uploaded_file=False):
    """
    input:
            embeddings: dict - contains data frames of all existing model embeddings
            annotations: dict - contains data frames of all existing ground-truth annotations
            reference: str - dictionary key picked in dashboard from available annotations as current ground truth
            uploaded_file : bool  - whether the new annotations are added from the "upload csv" page
                                  (not from existing annotations dictionary)

    """
    # list of songs with annotations to be evaluated
    cols = ['song1', 'song2', 'similarity score']
    if uploaded_file:
        annotations.columns = cols
        annotations = {None: annotations}
    eval_song_ids = list(set.union(set(annotations[reference].iloc[:, 0].tolist()),
                                   set(annotations[reference].iloc[:, 1].tolist())))
    annotations[reference].columns = cols
    df_report = annotations[reference].copy()

    for model_key in embeddings.keys():
        # each model_key is a model who's embedding we are using
        reference_ids_in_embedding = embeddings[model_key][
            embeddings[model_key].index.isin(eval_song_ids)].index.rename(None)
        sim_matrix = cosine_similarity(embeddings[model_key].loc[reference_ids_in_embedding])
        df = pd.DataFrame(sim_matrix, index=reference_ids_in_embedding, columns=reference_ids_in_embedding)
        df = df.where(np.tril(sim_matrix, -1).astype(np.bool))
        df = df.stack().reset_index()
        df.columns = cols
        arr = np.empty((df_report.shape[0], 1))
        arr[:] = np.NaN
        for row in annotations[reference].index:
            try:
                arr[row] = df.query(
                    f"(song1=={annotations[reference].loc[row, 'song1']} & "
                    f"song2=={annotations[reference].loc[row, 'song2']}) |"
                    f" (song1=={annotations[reference].loc[row, 'song2']} & "
                    f"song2=={annotations[reference].loc[row, 'song1']})")['similarity score']
            except ValueError:
                continue
        df_report[model_key] = arr
        df_report[f"{model_key}-difference"] = df_report[model_key] - df_report['similarity score']
    return df_report


def dense_report(model):
    """
    Input:
            model: string - the name of the picked model
    """
    df = pd.DataFrame(date.today(), columns=['Date'], index=[model])
    for annotation in dict_annotations_dfs.keys():
        temp_dict = {model: dict_data_embeddings[model]}
        temp_df = calc_similarity(temp_dict, dict_annotations_dfs, annotation, False)
        temp_df.dropna(inplace=True)
        df[f'{annotation}-annotation-MSE score'] = mean_squared_error(temp_df['similarity score'],
                                                                      temp_df[f'{model}-difference'])
    return df
    # """
    # Adds row entry to the model result log (global setting of model evaluation)
    # input :
    #         df: DataFrame - dataframe containing song pairs, annotated similarity and each model score
    #                         and difference between model score and similarity score
    #         reference: str - Which annotation is being used as reference.
    #
    # """
    # df_summarise = pd.DataFrame.from_dict({'Date': [date.today().isoformat()], 'Reference': [reference]})
    # cols = [col for col in df.drop(["song1", "song2"], axis=1).columns if "difference" not in col]
    # for i in range(1, len(cols)):
    #     df_slice = pd.DataFrame([df[cols[0]], df[cols[i]]]).T
    #     df_slice = df_slice.dropna(axis=0)
    #     df_summarise[f'{cols[i]} MSE score'] = round(
    #         mean_squared_error(df_slice.iloc[:, 0], df_slice.iloc[:, 1], squared=True), 3)
    # return df_summarise


def get_similarity(vect, matrix, metric):
    if metric == 'cosine similarity':
        return pd.DataFrame(data=cosine_similarity(vect, matrix).T, index=matrix.index,
                            columns=['similarity score']).sort_values('similarity score', ascending=False)
    return pd.DataFrame(data=pairwise_distances(vect, matrix, metric=metric).T, index=matrix.index,
                        columns=['similarity score']).sort_values('similarity score', ascending=False)


def get_k_similar(song_id, embeddings_dic, k, similarity_metric):
    """
    input:
            song_id :           int - Song id whose neighbours we wish to get
            embeddings_dic :   dict - Contains data frames of all existing model embeddings
            k:                  int - The number of neighbours desired
            similarity_metric : str - The metric how the distance between the different songs is to be measured.
    output:
            df:           DataFrame - The table displayed in the "offline model" page (k nearest neighbours per model)
    """
    cols = ['Song ID', f'{k} most similar songs', "Similarity metric", 'Model']
    df = pd.DataFrame(columns=cols)
    for model_embedding in embeddings_dic.keys():
        df_similiarty = get_similarity(embeddings_dic[model_embedding].loc[song_id].to_numpy().reshape(1, -1),
                                       embeddings_dic[model_embedding].drop(song_id, axis=0), similarity_metric)
        df = df.append(pd.DataFrame(np.array(
            [song_id, ", ".join([str(i) for i in df_similiarty.head(k).index.to_list()]), similarity_metric,
             model_embedding]).reshape(1, -1), columns=cols))
    return df


def to_excel(df):
    """
    used by get_table_download_link function
    """
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1', float_format="%.2f")
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Your_File.xlsx">Download Excel file</a>'  # decode b'abc' => abc


def get_single_song_table(df_report, song_id):
    df_deep_dive = pd.DataFrame(columns=['song1', 'song2', 'similarity score'])
    df_deep_dive = df_deep_dive.append(df_report.query(f"song1=={song_id}"))
    df_switch = df_report.query(f"song2=={song_id}")
    org_col = df_switch.columns.to_list()
    col_order = df_switch.columns.to_list()
    ind1 = col_order.index('song1')
    ind2 = col_order.index('song2')
    col_order[ind2], col_order[ind1] = col_order[ind1], col_order[ind2]
    df_switch = df_switch[col_order]
    df_switch.columns = org_col
    return df_deep_dive.append(df_switch)


def get_song_genres(song_ids):
    df_genre_analysis = pd.DataFrame(np.array(song_ids), columns=['song_id'])
    df_genre_analysis = df_genre_analysis.join(df_genres, on='song_id')
    df_genre_analysis = df_genre_analysis[['song_id', 'NAME']]
    df_genre_analysis['NAME'] = df_genre_analysis['NAME'].str[1:-1].str.split(', ')
    exploded = df_genre_analysis['NAME'].explode()
    df_genre_analysis = df_genre_analysis[['song_id']].join(pd.crosstab(exploded.index, exploded))
    df_genre_analysis.set_index('song_id', inplace=True)
    col_ls = [col[1:-1] for col in df_genre_analysis.columns]
    df_genre_analysis.columns = col_ls
    df_genre_analysis['No Genre'] = df_genre_analysis.isna().sum(axis=1).astype(bool).astype(float).values
    df_genre_analysis.fillna(0, inplace=True)
    return df_genre_analysis

@st.cache()
def get_barplot(song_ids, df_report, metric="MSE"):
    df_genre_analysis = get_song_genres(song_ids)
    # genre_mask = df_genre_analysis.astype(bool)
    df_summed_diff = pd.DataFrame()
    df_average_diff = pd.DataFrame()
    index_song_n_list = []
    models_columns = df_report.columns[df_report.columns.str.contains("difference")].to_list() + (
        df_report.columns[df_report.columns.str.contains("song")].to_list())
    for model in df_report.drop(columns=(["similarity score"] + models_columns)).columns.to_list():
        genre_count = pd.DataFrame(data=np.zeros((1, df_genre_analysis.shape[1])), columns=df_genre_analysis.columns)
        df_model_genre_summary = pd.DataFrame(data=np.zeros((1, df_genre_analysis.shape[1])),
                                              columns=df_genre_analysis.columns)
        for genre in df_genre_analysis.columns:
            songs_col1 = [song for song in df_report['song1'] if
                          song in df_genre_analysis[df_genre_analysis[genre] == 1].index]
            songs_col2 = [song for song in df_report['song2'] if
                          song in df_genre_analysis[df_genre_analysis[genre] == 1].index]
            genre_songs = df_report.query(f"song1=={songs_col1} | song2=={songs_col2}").dropna(axis=0)
            if metric == "Abs Summed Difference":
                df_model_genre_summary[genre] = genre_songs[f"{model}-difference"].abs().sum(axis=0)
            elif metric == "Summed Difference":
                df_model_genre_summary[genre] = genre_songs[f"{model}-difference"].sum(axis=0)
            elif metric == "MSE":
                try:
                    df_model_genre_summary[genre] = mean_squared_error(genre_songs["similarity score"], genre_songs[model], squared=True)
                except:
                    df_model_genre_summary.drop(genre,axis=1, inplace=True)
            # get n of sons in genre
            if genre not in df_model_genre_summary.columns:
                genre_count.drop(genre,axis=1, inplace=True)
                continue
            genre_count[genre] = len(set.union(set(genre_songs['song1']), set(genre_songs['song2'])))
        df_model_genre_summary.index = [f"{model} {metric}"]
        genre_count = df_model_genre_summary.values / genre_count.values
        genre_count = pd.DataFrame(genre_count, columns=df_model_genre_summary.columns)
        genre_count.index = [f"{model} {metric}"]
        df_summed_diff = pd.concat([df_summed_diff, df_model_genre_summary], axis=0)
        df_average_diff = pd.concat([df_average_diff, genre_count], axis=0)
    df_summed_diff.reset_index(inplace=True)
    df_average_diff.reset_index(inplace=True)
    df_summed_diff.drop(df_average_diff.columns[df_average_diff.isna().loc[0]], axis=1, inplace=True)
    tidy1 = df_summed_diff.melt(id_vars='index').rename(columns=str.title)
    tidy2 = df_average_diff.melt(id_vars='index').rename(columns=str.title)
    return tidy1, tidy2


###################################################################################################
#####################################Streamlit pages start here####################################
###################################################################################################

def offline_trial():
    """
    Displays streamlit offline models k nearest neighbours
    """
    st.title("Offline trial")
    st.subheader("Pick a song, and see what the top k nearest neighbours each model would predict")
    # dict_data_embeddings
    col1, col2, col3 = st.beta_columns(3)
    song_id = col1.selectbox('Pick song id', (sorted(list(set(dict_data_embeddings['one-hot'].index.tolist())))))
    similar_k = col2.number_input('Pick the number of similar songs:', min_value=0, step=1, value=5, )
    similarity_metric = col3.selectbox('Pick similarity metric', ['cosine similarity', 'hamming', 'l2', 'l1'])
    df = get_k_similar(song_id, dict_data_embeddings, similar_k, similarity_metric)
    st.table(df)


def model_eval():
    """
    Displays streamlit model evaluation page
    """
    # display on page
    st.title("Model evaluation page")
    st.write("This is the page with the data")
    model_option = st.selectbox('Pick model for evaluation', (list(dict_data_embeddings.keys())))
    df_temp = dense_report(model_option)
    # df_summary = df_summary_table.append(df_temp)
    # df_summary = df_summary.drop_duplicates()
    # df_summary = df_summary.dropna(axis=0)

    mode = st.radio("Set presentation scale", ('Models overview', 'Song pairs'))
    if mode == 'Models overview':
        table_or_plot = st.radio("Set presentation layout", ('Table layout', 'Barplot layout'))
        if table_or_plot == 'Table layout':
            st.dataframe(df_temp)
            st.markdown(get_table_download_link(df_temp), unsafe_allow_html=True)
        else:
            col1, col2 = st.beta_columns(2)
            ground_truth = col1.selectbox('Pick ground truth for evaluation', (list(dict_annotations_dfs.keys())))
            metric_option = col2.selectbox('Pick evaluation metric', ["Abs Summed Difference", "Summed Difference", "MSE"])
            df_report = calc_similarity(dict_data_embeddings, dict_annotations_dfs, ground_truth)
            song_ids = sorted(list(set.union(set(df_report.iloc[:, 0].tolist()), set(df_report.iloc[:, 1].tolist()))))
            # plot

            tidy1, tidy2 = get_barplot(song_ids, df_report, metric_option)
            char1 = alt.Chart(tidy1).mark_bar().encode(
                x=alt.X('Value', title="Difference from annotation"),
                y=alt.Y('Index', title=" "),
                color='Index',
                row='Variable'
            ).properties(title=f"{metric_option} between models and ground truth")

            char2 = alt.Chart(tidy2).mark_bar().encode(
                x=alt.X('Value', title="Difference from annotation"),
                y=alt.Y('Index', title=" "),
                color='Index',
                row='Variable'
            ).properties(title=f"Average {metric_option} between models and ground truth")
            with st.beta_expander("Metric explanation"):
                st.write("Summed Difference: The sum of the 'model_name-difference' column by genre.")
                st.write(
                    "Abs Summed Difference: The absoloute value sum of the 'model_name-difference' column by genre.")
                st.write(
                    "MSE: The Mean Square Error between the annotation song pair score, and the model predicted score")
            st.altair_chart(char1| char2)


    else: # Song pairs
        option = st.selectbox('Pick ground truth for evaluation', (list(dict_annotations_dfs.keys())))
        df_report = calc_similarity(dict_data_embeddings, dict_annotations_dfs, option)
        song_ids = sorted(list(set.union(set(df_report.iloc[:, 0].tolist()), set(df_report.iloc[:, 1].tolist()))))
        st.dataframe(df_report)
        st.markdown(get_table_download_link(df_report), unsafe_allow_html=True)
        col1, col2 = st.beta_columns(2)
        df_song_genres = get_song_genres(song_ids)
        genre_id = col1.selectbox('Pick specific song genre', df_song_genres.columns.tolist())
        try:
            song_id = col2.selectbox('Pick specific song id from the above pairs',
                                     (df_song_genres[df_song_genres[genre_id] == 1].index.to_list()))
            df_deep_dive = get_single_song_table(df_report, song_id)
            st.dataframe(df_deep_dive)
            with st.beta_expander("Show Table Averages"):
                st.write(
                    df_deep_dive[df_deep_dive.columns[df_deep_dive.columns.str.endswith("difference")]].mean(axis=0))
            st.markdown(get_table_download_link(df_deep_dive), unsafe_allow_html=True)
        except:
            song_id = col2.selectbox('Selected annotation has no songs of this genre', ([]))


def upload_page():
    st.title("Data upload")
    st.subheader(
        "Upload a csv file with song id pairs in two columns, with an optional third column of labels of these pairs.")
    df = pd.DataFrame()
    uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False)
    if uploaded_file:
        data = StringIO(uploaded_file.read().decode('utf8'))
        data = pd.read_csv(data)
        try:
            st.write("filename:", uploaded_file.name)
            df = calc_similarity(dict_data_embeddings, data, None, True)
            st.dataframe(df)
        except:
            st.write("Upload file to display")

    if df.empty:
        st.write('How the csv file should look like: ')
        st.dataframe(pd.DataFrame(np.array([[1, 2, 0.5], [1, 3, 0.2], [2, 3, 0.8]]),
                                  columns=[['song id 1', 'song id 2', 'similarity']]))
    else:
        st.markdown(get_table_download_link(df), unsafe_allow_html=True)


app = MultiApp()
app.add_app("Offline model song similarity test", offline_trial)
app.add_app("Detailed model evaluation", model_eval)
app.add_app("Upload CSV ", upload_page)

app.run()
