# Import dependencies
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.patches import Rectangle, Patch
from sklearn.manifold import TSNE
import argparse
from random import randint
import pickle
from visualization_utils.color_utils import get_colors
from visualization_utils.dynamic_graph_utils import get_graph

def show_tsne(x, how_to_calc, color_paired=False):
    color = []
    # n = len(x.columns)
    n = 12
    for i in range(n):
        color.append('#%06X' % randint(0, 0xFFFFFF))
    color_pairs = {
        "ja":color[0],
        "ko":color[0],
        "ta":color[1],
        "te":color[1],
        "hu":color[2],
        "fi":color[2],
        "zh":color[3],
        "my":color[3],
        "ar":color[4],
        "he":color[4],
        "cy":color[5],
        "ga":color[5],
        "hi":color[6],
        "ne":color[6],
        "en":color[7],
        "de":color[7],
        "hy":color[8],
        "el":color[8],
        "fr":color[9],
        "pms":color[9],
        "ru":color[10],
        "sv":color[10],
        "other":color[11]
    }
    tsne = TSNE(n_components=2, verbose=1, n_iter=300)
    #['rows', 'cols', 'concat']
    how_to_calc = set(how_to_calc)
    if how_to_calc == set(['rows']):
        to_calc = x.to_numpy()
    elif how_to_calc == set(['cols']):
        to_calc = x.to_numpy().T
    elif how_to_calc == set(['rows', 'cols']):
        to_calc = np.concatenate((x.to_numpy(),x.to_numpy().T)).T
    tsne_results = tsne.fit_transform(to_calc)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    codes = x.columns
    for i in range(len(x)):  # plot each point + it's index as text above
        if color_paired:
            ax.scatter(tsne_results[i, 0], tsne_results[i, 1],
                       color=color_pairs[codes[i]] if codes[i] in color_pairs else color_pairs['other'], alpha=0.8)
            ax.text(tsne_results[i, 0], tsne_results[i, 1], codes[i], zorder=1, color='k')
        elif not color_paired:
            ax.scatter(tsne_results[i, 0], tsne_results[i, 1],
                       color=color_pairs['other'], alpha=0.8)
            ax.text(tsne_results[i, 0], tsne_results[i, 1], codes[i], zorder=1, color='k')
    return fig


def init_data(df_path, feat_idx2val_path, lang_feat_df_path, lang_list):
    lang_list.sort()
    feat_idx2value = pickle.load(open(feat_idx2val_path, 'rb'))
    feat_value2idx = {}
    for feat in feat_idx2value.keys():
        feat_value2idx[feat] = dict(zip(feat_idx2value[feat].values(), feat_idx2value[feat].keys()))
    st.set_page_config(layout="wide")
    st.title('Network Graph Visualization of Lang-Lang Interactions', )

    full_df = pd.read_csv(df_path,index_col=0)
    feat_df = pd.read_csv(lang_feat_df_path, index_col=0)
    feat_df = feat_df.loc[lang_list]
    __feats = []
    for feat in feat_df.columns:
        if len(set(feat_df[feat])) <= 3:
            continue
        else:
            __feats.append(feat)
    feat_df = feat_df[__feats]
    for feat in feat_df.columns:
        if len(set(feat_df[feat])) <= 2:
            feat_df = feat_df[[x for x in feat_df.columns if x != feat]]
    wiki2name = {'english': 'en', 'german': 'de', 'french': 'fr', 'russian': 'ru', 'spanish': 'es', 'japanese': 'ja',
                 'italian': 'it', 'chinese': 'zh', 'polish': 'po', 'dutch': 'nl', 'swedish': 'se', 'arabic': 'ar',
                 'catalan': 'ca',
                 'hungarian': 'hu', 'czech': 'cz', 'persian': 'fa', 'vietnamese': 'vt', 'korean': 'ko', 'finnish': 'fi',
                 'indonesian': 'id', 'hebrew': 'he', 'turkish': 'tu', 'romanian': 'ro', 'greek': 'el', 'armenian': 'hy',
                 'danish': 'da', 'thai': 'th', 'slovak': 'sv', 'tamil': 'ta', 'hindi': 'hi', 'urdu': 'ur',
                 'telugu': 'te',
                 'afrikaans': 'af', 'burmese': 'my', 'icelandic': 'ic', 'nepali': 'ne', 'swahili': 'sw',
                 'malagasy': 'ma', 'haitian': 'ha', 'yoruba': 'yo', 'piedmontese': 'pms', 'yiddish': 'yi',
                 'amharic': 'am', 'ladino': 'lad', 'welsh': 'cy', 'irish': 'ga', }
    name2wiki = dict(zip(wiki2name.values(), wiki2name.keys()))

    return feat_df, feat_idx2value, feat_value2idx, full_df, lang_list, name2wiki


def main(args):
    df_path = args.df_path
    idx2val_path=args.idx2val_df_path
    lang_feat_df_path=args.lang_feat_df_path
    COLORS=get_colors()
    lang_list=args.lang_list

    # Read dataset

    feat_df, feat_idx2value, feat_value2idx, full_df, lang_list, name2wiki = init_data(df_path=df_path, feat_idx2val_path=idx2val_path, lang_feat_df_path=lang_feat_df_path, lang_list=lang_list)

    c1, c2 = st.columns((1, 2))

    # Define selection options and sort alphabetically

    # Implement multiselect dropdown menu for option selection
    selected_langs = c1.multiselect('Select lang(s) to visualize', lang_list, default=lang_list, format_func=lambda x:name2wiki[x])
    # reset_langs = c1.button('Reset to all langauges')
    selected_feats = c1.selectbox('Select a feature to filter by', ['No feature']+sorted(feat_df.columns), format_func=lambda x:" ".join(x.split("_")))


    # after you choose features (combobox) - open sub options for the given features (to show or not too show)
    # Set info message on initial site load
    if len(selected_langs) == 0:
        c1.text('Please choose at least 1 language to get started')
    # Create network graph when user selects >= 1 item
    else:
        filtered_df = full_df.copy()
        feat = selected_feats
        val_colors_dict = {}
        collapse = False
        if feat != 'No feature':
            collapse = c1.checkbox('Collapse feature')
            just_cols = filtered_df[[x for x in feat_df.index[list(feat_df[feat] != -1)] if x in selected_langs]]
            filtered_df = just_cols.loc[[x for x in feat_df.index[list(feat_df[feat] != -1)] if x in selected_langs]]
            feat_vals=list(set(feat_idx2value[feat].values()))
            selected_feature_values = c1.multiselect('Select feature values to visualize', feat_vals, default=feat_vals)
            selected_feat_indices = [feat_value2idx[feat][val] for val in selected_feature_values]
            hex_colors, rgb_colors = COLORS[len(feat_vals)+1]
            val_colors_dict = dict(zip(feat_vals+[-1], hex_colors))
            val_colors_dict_rgb = dict(zip(feat_vals+[-1], rgb_colors))
            just_cols = filtered_df[[x for x in feat_df.index[list(feat_df[feat].isin(selected_feat_indices))] if x in selected_langs]]
            filtered_df = just_cols.loc[[x for x in feat_df.index[list(feat_df[feat].isin(selected_feat_indices))] if x in selected_langs]]
            if collapse:
                new_selected_feat_values = []
                feat_lang_dict = {}
                for feat_idx, feat_val in zip(selected_feat_indices, selected_feature_values):
                    feat_langs = [x for x in feat_df.index[list(feat_df[feat] == feat_idx)] if x in selected_langs]
                    if len(set(feat_langs)) > 1:
                        feat_lang_dict[feat_idx] = set(feat_langs)
                        new_selected_feat_values.append(feat_val)
                new_df = np.ones(shape=(len(feat_lang_dict),len(feat_lang_dict)))
                for i,feat_val_1 in enumerate(feat_lang_dict):
                    for j,feat_val_2 in enumerate(feat_lang_dict):
                        if i!=j:
                            new_df[i][j] = ((filtered_df.loc[feat_lang_dict[feat_val_1]])[feat_lang_dict[feat_val_2]].sum()/len(feat_lang_dict[feat_val_1])).sum()/len(feat_lang_dict[feat_val_2])
                selected_langs = new_selected_feat_values
                selected_feature_values = new_selected_feat_values
                selected_feat_indices = [feat_value2idx[feat][val] for val in selected_feature_values]
                filtered_df=pd.DataFrame(new_df, index=selected_feature_values, columns=selected_feature_values)
        else:
            filtered_df = filtered_df[selected_langs]
            filtered_df = filtered_df.loc[selected_langs]
        vis = c2.radio(
            "Choose visualization",
            ('Heatmap','Dynamic graph'))
        option = 'None'
        if vis == 'Heatmap':
            with c2:
                codes = filtered_df.columns
                if feat != 'No feature' and not collapse:
                    ordered_feats = feat_df.loc[codes][feat].sort_values()
                    new_lang_order = ordered_feats.index
                    # new_lang_order = new_lang_order[new_lang_order!=-1]
                    filtered_df = filtered_df[new_lang_order]
                    filtered_df = filtered_df.loc[new_lang_order]
                else:
                    # give option to sort
                    option = c1.selectbox(
                        'Sort table',
                        ['None','Row score','Column score', 'Cluster']+selected_langs)
                    if option == 'Row score':
                        ordered_feats = filtered_df.sum(axis=1).sort_values(ascending=False).index
                        filtered_df = filtered_df[ordered_feats]
                        filtered_df = filtered_df.loc[ordered_feats]
                    elif option == 'Column score':
                        ordered_feats = filtered_df.sum(axis=0).sort_values(ascending=False).index
                        filtered_df = filtered_df[ordered_feats]
                        filtered_df = filtered_df.loc[ordered_feats]
                    elif option in selected_langs:
                        how = c1.selectbox(
                            ' according to',
                            ['Contribution to','Contribution from'])
                        if how == 'Contribution to':
                            indices = np.argsort(filtered_df[filtered_df.index == option].to_numpy()[0]*-1)
                            filtered_df = filtered_df.iloc[:,indices]
                        else:
                            indices = np.argsort(filtered_df[option]*-1)
                            filtered_df = filtered_df.iloc[indices]
                normalized = filtered_df
                # if not collapse:

                normalized -= 1

                if feat == 'No feature' and option == 'Cluster':
                    cluster_args = c1.multiselect('Select what to cluster', ['rows', 'cols'], default=['rows', 'cols'])
                    row_cluster, col_cluster = False, False
                    if 'rows' in cluster_args:
                        row_cluster = True
                    if 'cols' in cluster_args:
                        col_cluster = True
                    fig = sb.clustermap(normalized,  linewidths=0.2, annot=True, fmt='.3f', cmap="afmhot", row_cluster=row_cluster, col_cluster=col_cluster)
                    c2.pyplot(fig)
                else:
                    fig = plt.figure(figsize=(15,15))
                    contributors = np.sum(normalized, axis=1) # sum across rows
                    contributed = np.sum(normalized, axis=0) # sum across cols
                    shot_types = False
                    if option == 'None':
                        shot_types = c1.checkbox("Show types graph")
                    ax1 = plt.subplot2grid((20,20), (0,0), colspan=19, rowspan=19)
                    ax2 = plt.subplot2grid((20,20), (19,0), colspan=19, rowspan=1)
                    ax3 = plt.subplot2grid((20,20), (0,19), colspan=1, rowspan=19)

                    normalized = normalized * 100
                    x = sb.heatmap(normalized, ax=ax1, annot=True, cmap="afmhot", linecolor='white', linewidths=0.5, cbar=False,  fmt='.0f', annot_kws={"fontsize":18})
                    x.set_xticklabels((pd.DataFrame(contributed).T).columns, size=20)
                    x.set_yticklabels((pd.DataFrame(contributors).T).columns, size=20)
                    ax1.xaxis.tick_top()
                    ax1.set_xticklabels(normalized.columns,rotation=40)
                    ax1.set_yticklabels(normalized.index,rotation=40)
                    if feat != 'No feature' and not collapse:
                        patch_locations=[0]+np.where([False]+[ordered_feats[i]!=ordered_feats[i-1] for i in range(1,len(ordered_feats))])[0].tolist()+[len(ordered_feats)]
                        _,patch_colors_rgb = COLORS[len(patch_locations)-1]
                        for i in range(len(patch_locations)-1):
                            ax1.add_patch(Rectangle((patch_locations[i], patch_locations[i]), patch_locations[i+1]-patch_locations[i], patch_locations[i+1]-patch_locations[i], fill=False, edgecolor=patch_colors_rgb[i], lw=5))


                    sb.heatmap((pd.DataFrame(contributed*100)).transpose(), ax=ax2,  annot=True, cmap="afmhot", cbar=False, xticklabels=False, yticklabels=False, fmt='.0f', annot_kws={"fontsize":16})
                    sb.heatmap(pd.DataFrame(contributors*100), ax=ax3,  annot=True, cmap="afmhot", cbar=False, xticklabels=False, yticklabels=False, fmt='.0f', annot_kws={"fontsize":16})
                    sb.set(font_scale=0.8)

                    c2.pyplot(fig)

                    if shot_types:
                        assert len(contributed) == len(contributors)
                        fig = plt.figure(figsize=(15,15))
                        ax = fig.add_subplot(111)
                        for i in range(len(contributed)):  # plot each point + it's index as text above
                            if contributed[i] > 0 and contributors[i]>0: # A
                                ax.scatter(contributed[i], contributors[i],
                                           color='orange', alpha=0.8, s=600)
                                ax.text(contributed[i], contributors[i], normalized.columns[i], zorder=1, color='k', fontsize=43)
                            elif contributed[i] > 0 and contributors[i]<0: # AB
                                ax.scatter(contributed[i], contributors[i],
                                           color='blue', alpha=0.8, s=600)
                                ax.text(contributed[i], contributors[i], normalized.columns[i], zorder=1, color='k', fontsize=43)
                            elif contributed[i] < 0 and contributors[i]>0: # O
                                ax.scatter(contributed[i], contributors[i],
                                           color='red', alpha=0.8, s=600)
                                ax.text(contributed[i], contributors[i], normalized.columns[i], zorder=1, color='k', fontsize=43)
                            elif contributed[i] < 0 and contributors[i]<0:
                                ax.scatter(contributed[i], contributors[i],
                                           color='cyan', alpha=0.8, s=600)
                                ax.text(contributed[i], contributors[i], normalized.columns[i], zorder=1, color='k', fontsize=43)
                            else:
                                ax.scatter(contributed[i], contributors[i],
                                           color='black', alpha=0.8, s=600)
                                ax.text(contributed[i], contributors[i], normalized.columns[i], zorder=1, color='k', fontsize=43)


                        ax.axvline(x=0, linewidth = 3, color = "black", linestyle="--")
                        ax.axhline(y=0, linewidth = 3, color = "black", linestyle="--")
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_facecolor("white")


                        ax.set_xlabel('recipient',fontweight ='bold', fontsize=46)
                        ax.set_ylabel('donor',fontweight ='bold', fontsize=46)
                        c1.pyplot(fig)

                if feat != 'No feature' and not collapse:
                    leg_fig = plt.figure(figsize=(5,5))
                    # Create a color palette
                    palette = dict(zip([feat_idx2value[feat][ordered_feats[loc]] for loc in patch_locations[:-1]], patch_colors_rgb))
                    # Create legend handles manually
                    handles = [Patch(color=palette[x], label=x) for x in palette.keys()]
                    # Create legend
                    ax_colors = plt.subplot2grid((20,20), (0,0), colspan=19, rowspan=19)

                    ax_colors.legend(handles=handles)
                    # Get current axes object and turn off axis
                    ax_colors.set_axis_off()
                    c1.pyplot(leg_fig)

        elif vis == 'Dynamic graph':
            with c2:
                mrr_tr = c1.slider("min transfer score", min_value=full_df.min().min(), max_value=full_df.max().max(), value=(1.0,float(full_df.max().max())), step=0.01) # TODO - set to min, max in data
                edge_weight_amplifier = c1.slider("edge amplifier", min_value=1, max_value=25, value=2, step=1)
                # Code for filtering dataframe and generating network
                drug_net = get_graph(edge_weight_amplifier, feat, feat_df, feat_idx2value, filtered_df, mrr_tr,
                                     name2wiki, selected_langs, val_colors_dict)
                if feat != 'No feature' and not collapse:
                    leg_fig = plt.figure(figsize=(5,5))
                    # Create a color palette
                    palette = {key:val for key,val in zip(val_colors_dict_rgb.keys(),val_colors_dict_rgb.values()) if key!=-1}
                    # Create legend handles manually
                    handles = [Patch(color=palette[x], label=x) for x in palette.keys()]
                    # Create legend
                    ax_colors = plt.subplot2grid((20,20), (0,0), colspan=19, rowspan=19)

                    ax_colors.legend(handles=handles)
                    # Get current axes object and turn off axis
                    ax_colors.set_axis_off()
                    c1.pyplot(leg_fig)
                # Save and read graph as HTML file (on Streamlit Sharing)
                try:
                    path = 'visualization_tool/visualization_utils'
                    drug_net.save_graph(f'{path}/pyvis_graph.html')
                    HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')
                # Save and read graph as HTML file (locally)
                except:
                    path = 'visualization_tool/visualization_utils'
                    drug_net.save_graph(f'{path}/pyvis_graph.html')
                    HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

                    # Load HTML file in HTML component for display on Streamlit page
                components.html(HtmlFile.read(), height=720, width=1250, scrolling=True)
        if not (feat != 'No feature' and not collapse) and option == 'None':
            if c1.checkbox('Show TSNE on chosen languges'):
                hot_to_tsne = c1.multiselect('What to cast? (select both to concat)', ['rows', 'cols'], default=['rows'])
                if collapse:
                    fig = show_tsne(filtered_df, hot_to_tsne, color_paired=False)
                else:
                    fig = show_tsne(filtered_df, hot_to_tsne,color_paired=True)
                c1.pyplot(fig)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--df_path', type=str, default="data/3_epochs_ft_normalized_0shot_False_mapped_False.csv")
    parser.add_argument('-i', '--idx2val_df_path', type=str, default='data/featidx2value.pkl')
    parser.add_argument('-l', '--lang_feat_df_path', type=str, default="data/lang_features_df.csv")
    parser.add_argument('--lang_list', nargs="+", default="ja ko ta te hu fi zh my ar he cy ga hi ne en de hy el fr pms ru sv".split())
    args = parser.parse_args()
    main(args)
