import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import base64
import cv2
from flask import Flask
from flask_socketio import SocketIO
import logging
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import math
import sys
from skimage import io, color, filters
import math
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go




logging.basicConfig(level=logging.INFO)

server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
socketio = SocketIO(app.server)

TEMP_VIDEO_PATH = 'temp_uploaded_video.mp4'
video_cap = None
video_playing = False
total_frames = 0

# ... [rest of the imports]

uciqe_values = []  # This list will store the UCIQE values over time

# ... [rest of the imports]

app.layout = html.Div([
    
    # Title
    html.H1("LIACi Image Quality Indicator", style={'textAlign': 'center', 'marginBottom': '50px'}),
    
    # Row for upload button, start button, and dropdown
    html.Div([
        dcc.Upload(
            id='upload-video',
            children=html.Button('Upload Video'),
            multiple=False,
            accept='video/*'
        ),
        html.Button("Start Video", id="btn-start"),
        dcc.Dropdown(
            id='frame-dropdown',
            options=[
                {'label': 'Every Frame', 'value': 1},
                {'label': 'Every 5th Frame', 'value': 5},
                {'label': 'Every 10th Frame', 'value': 10},
                {'label': 'Every 30th Frame', 'value': 30},
                {'label': 'Every 60th Frame', 'value': 60},
                {'label': 'Every 100th Frame', 'value': 100},
                # ... Add more options as needed
            ],
            value=1,
            clearable=False,
            style={'width': '40%'}
        )
    ], style={'display': 'flex', 'justify-content': 'space-between', 'marginBottom': '20px'}),
    
    # Video display and graph in the same row
    html.Div([
        html.Div([
            html.Img(id="video-frame", src=""),
        ], style={'display': 'inline-block', 'width': '48%', 'verticalAlign': 'top'}),
        
        html.Div([
            dcc.Graph(id="uciqe-plot", figure={'data': [], 'layout': go.Layout(title="Real-time UCIQE Value")}),
        ], style={'display': 'inline-block', 'width': '48%', 'verticalAlign': 'top'}),
    ], style={'marginTop': '20px'}),
    
    dcc.Interval(id="video-update", interval=100, n_intervals=0),
    html.Script(src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"),
    html.Script("""
        let socket = io.connect('http://localhost:8050');
        socket.on('connect', function() {
            console.log("Socket connected!");
        });
        socket.on('update_frame', function(data) {
            document.getElementById('video-frame').src = data.image;
        });
    """),
])



@socketio.on("connect")
def handle_connect():
    print("Client connected")

@app.callback(
    Output('btn-start', 'n_clicks'),
    [Input('upload-video', 'contents'),
    Input('btn-start', 'n_clicks')])
def upload_video(contents, n_clicks):
    global video_cap, video_playing, total_frames

    if n_clicks:
        video_playing = True
    else:
        video_playing = False

    if contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        with open(TEMP_VIDEO_PATH, 'wb') as f:
            f.write(decoded)

        video_cap = cv2.VideoCapture(TEMP_VIDEO_PATH)
        # get the number of frames in the video
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not video_cap.isOpened():
            os.remove(TEMP_VIDEO_PATH)
            raise Exception("Could not open video from uploaded content")
        else:
            logging.info("Uploaded video opened successfully")

    return n_clicks


@app.callback(
    [Output("video-frame", "src"), Output("uciqe-plot", "figure")],  # This line remains the same for two outputs
    [Input("video-update", "n_intervals"), Input('frame-dropdown', 'value')]
)
def update_frame(n_intervals, frame_dropdown_value):
    global video_cap, video_playing

    uciqe_threshold_low = 20
    uciqe_threshold_high = 27
    count = 0

    if video_playing and video_cap:
        ret, frame = video_cap.read()

        
        # Instead of resetting the video, stop playing it after the last frame
        if not ret:
            video_playing = False
            video_cap.release()
            os.remove(TEMP_VIDEO_PATH)
            # clear the figure
            figure = {
                'data': [],  # Ordering matters! Plot UCIQE last.
                'layout': go.Layout(title="Real-time UCIQE Value")
            }
            logging.info("Video released and temporary file removed")
            raise PreventUpdate 

        if ret:
            # process only every 5th frame
            if count % frame_dropdown_value != 0:
                count += 1
                return dash.no_update, dash.no_update
            # resize the frame to 
            frame = cv2.resize(frame, (640, 480))
            frame, uciqe = nmetrics(frame, uciqe_threshold_low, uciqe_threshold_high)

            # Append the UCIQE value to our list
            uciqe_values.append(uciqe)

            x_range = list(range(len(uciqe_values)))

            # Create filled areas for color-coded thresholds
            # red area is the area below the lower threshold
            # Green area above the upper threshold
            green_area = {
                'type': 'scatter',
                'x': x_range,
                'y': [max(uciqe_values) + 0.1] * len(uciqe_values),  # To ensure the fill goes beyond the highest UCIQE value
                'fill': 'tonexty',
                'fillcolor': 'rgba(255, 255, 0, 0.3)',
                'line': {'color': 'green'},
                'mode': 'none'
            }

            # Yellow area between the thresholds
            yellow_area = {
                'type': 'scatter',
                'x': x_range,
                'y': [uciqe_threshold_high] * len(uciqe_values),
                'fill': 'tonexty',
                'y0': [uciqe_threshold_low] * len(uciqe_values),  # start of the fill area
                'fillcolor': 'rgba(0, 255, 0, 0.3)',
                'line': {'color': 'yellow'},
                'mode': 'none'
            }

            # Red area below the lower threshold
            red_area = {
                'type': 'scatter',
                'x': x_range,
                'y': [uciqe_threshold_low] * len(uciqe_values),
                'fill': 'tozeroy',
                'fillcolor': 'rgba(255, 0, 0, 0.3)',
                'line': {'color': 'red'},
                'mode': 'none'
            }

            uciqe_plot = {
                'type': 'scatter',
                'x': x_range,
                'y': uciqe_values,
                'line': {'color': 'blue'},  # UCIQE plot line color
                'mode': 'lines'
            }

            figure = {
                'data': [green_area, yellow_area, red_area, uciqe_plot],  # Ordering matters! Plot UCIQE last.
                'layout': go.Layout(title="Real-time UCIQE Value")
            }

            _, buffer = cv2.imencode('.jpg', frame)
            frame_encoded = base64.b64encode(buffer).decode('utf-8')
            count += 1
            
            return f"data:image/jpeg;base64,{frame_encoded}", figure

    raise PreventUpdate

'''
Metrics for unferwater image quality evaluation.

Author: Xuelei Chen 
Email: chenxuelei@hotmail.com

Usage:
python evaluate.py RESULT_PATH
'''
# UCIQE
c1 = 0.4680
c2 = 0.2745
c3 = 0.2576
def nmetrics(a, uciqe_threshold_low , uciqe_threshold_high):

    # This remains the same
    lab = color.rgb2lab(a)
    l = lab[:,:,0]

    # Use np.linalg.norm to compute chroma, this remains the same
    chroma = np.linalg.norm(lab[:,:,1:], axis=-1)

    # Compute sc using np.std, this remains the same
    sc = np.std(chroma)

    # Compute top with direct integer arithmetic for rounding effect, this remains the same
    top = (l.shape[0] * l.shape[1] + 50) // 100

    # Sorting and computing conl, this remains the same
    sl = np.sort(l, axis=None)
    isl = sl[::-1]
    conl = np.mean(isl[:top])-np.mean(sl[:top])

    # 3rd term: Optimizing the saturation computation using array operations
    chroma_mask = chroma != 0
    l_mask = l != 0
    combined_mask = chroma_mask & l_mask

    satur_array = np.zeros_like(chroma)
    satur_array[combined_mask] = chroma[combined_mask] / l[combined_mask]

    us = np.mean(satur_array)

    uciqe = c1 * sc + c2 * conl + c3 * us

    # Determine the color based on the metrics
    if uciqe > uciqe_threshold_high:
        indicator_color = (0, 255, 0)  # Green
        #color = (0, 255, 0)
    elif uciqe > uciqe_threshold_low:
        indicator_color = (0, 255, 255)  # Yellow
        #color = (255, 255, 0)
    else:
        indicator_color = (0, 0, 255)  # Red
        #color = (255, 0, 0)

    # Draw a circle indicator on the top-left corner of the image
    cv2.circle(a, (50, 50), 30, indicator_color, -1)
    # add the numarical values next to the circle indicator
    cv2.putText(a, "UCIQE: {:.2f}".format(uciqe), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, indicator_color, 2)
    #print(" uciqe: ", uciqe)
    return a, uciqe

def eme(ch,blocksize=8):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    eme = 0
    w = 2. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb,ylb:yrb]

            blockmin = float(np.min(block))
            blockmax = float(np.max(block))

            # # old version
            # if blockmin == 0.0: eme += 0
            # elif blockmax == 0.0: eme += 0
            # else: eme += w * math.log(blockmax / blockmin)

            # new version
            if blockmin == 0: blockmin+=1
            if blockmax == 0: blockmax+=1
            eme += w * math.log(blockmax / blockmin)
    return eme

def plipsum(i,j,gamma=1026):
    return i + j - i * j / gamma

def plipsub(i,j,k=1026):
    return k * (i - j) / (k - j)

def plipmult(c,j,gamma=1026):
    return gamma - gamma * (1 - j / gamma)**c

def logamee(ch,blocksize=16):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    s = 0
    w = 1. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb,ylb:yrb]
            blockmin = float(np.min(block))
            blockmax = float(np.max(block))

            top = plipsub(blockmax,blockmin)
            bottom = plipsum(blockmax,blockmin)
            if bottom == 0.0:
                m = 0.0
            else:
                m = top/bottom
            if m ==0.:
                s+=0
            else:
                s += (m) * np.log(m)

    return plipmult(w,s)

if __name__ == '__main__':
    socketio.run(app.server, debug=True, port=8050)
