import datetime
from PIL import Image
import os
import json
import pandas as pd
import numpy as np

import plotly
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
pio.templates.default = "plotly_white"



def clean(data):
    data_df = pd.DataFrame.from_records(data)

    data_df = pd.concat([
        data_df.drop(['data'],axis=1),
        pd.json_normalize(data_df['data'], sep='_')
        ], axis=1)

    data_df['time'] = data_df['time'].astype(int).apply(lambda x : datetime.date.fromtimestamp(x))


    data_df.drop(['type', 'session'], axis=1, inplace=True)

    return data_df

def process_logs(logs):
    logs = [json.loads(l) for l in logs]

    sessions_params_df = clean([l for l in logs if l['type']=="session"])
    logs_train_df = clean([l for l in logs if l['type']=="train"])
    logs_valid_df = clean([l for l in logs if l['type']=="val"])

    id_vars = ['time', 'elapsed', 'epoch', 'iter']
    logs_valid_df = pd.melt(
        logs_valid_df,
        id_vars=id_vars
    )

    logs_valid_df['variable'] = logs_valid_df['variable'].str.split('_')
    logs_valid_df[['type', 'thr']] = pd.DataFrame(logs_valid_df['variable'].tolist(), index= logs_valid_df.index)
    logs_valid_df = logs_valid_df.pivot_table(columns=['thr'], index = id_vars+['type'], values = ['value'])

    return sessions_params_df, logs_train_df, logs_valid_df

def plot_training_logs(logs_train_df):

    losses = [l for l in logs_train_df.columns if 'loss' in l]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for l in losses:
        fig.add_trace(
            go.Scatter(x=logs_train_df.iter, y=logs_train_df[l], name =l),
            secondary_y=False)

    fig.add_trace(
        go.Scatter(x=logs_train_df.iter, y=logs_train_df.lr, name='lr', marker_color='grey'),
        secondary_y=True)

    # Set x-axis title
    fig.update_xaxes(title_text="Iteration")

    # Set y-axes titles
    fig.update_yaxes(title_text="Loss", secondary_y=False)
    fig.update_yaxes(title_text="Learning rate", secondary_y=True)

    return fig


def get_image_with_inference(
    img_name , path_imgs, path_masks, path_results,
    horizontal = True, resize_factor=1, step=20,
    img_extention = '.jpg', seg_extention = '.png'):


    img = Image.open(os.path.join(path_imgs, img_name + img_extention))
    img_mask = Image.open(os.path.join(path_masks, img_name + img_extention))
    img_seg = Image.open(os.path.join(path_results, img_name + seg_extention))

    if horizontal:

        new_image = Image.new('RGB',(3*img.size[0]+step*2, img.size[1]), (250,250,250))

        new_image.paste(img,(0,0))
        new_image.paste(img_mask,(img.size[0]+step, 0))
        new_image.paste(img_seg,(((img.size[0]+step)*2,0)))

    else:

        new_image = Image.new('RGB',(img.size[0], 3*img.size[1]+step*2), (250,250,250))

        new_image.paste(img,(0,0))
        new_image.paste(img_mask,(0,img.size[1]+step))
        new_image.paste(img_seg,((0,(img.size[1]+step)*2)))


    if resize_factor<1:
        new_size = (int(new_image.size[0]*resize_factor),int(new_image.size[1]*resize_factor))
        new_image = new_image.resize(new_size)

    return new_image