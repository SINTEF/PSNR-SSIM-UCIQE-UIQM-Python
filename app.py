import dash
from dash import dcc, html
import dash_uploader as du
import os
import cv2
from flask import Flask, send_from_directory, send_file
import shutil
from nevaluate import nmetrics, display_indicator
from dash.dependencies import Input, Output


# Initialize the Dash app
# Use Dash's default stylesheet for a cleaner look.
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

UPLOAD_FOLDER_ROOT = os.path.abspath('tmp_uploaded_files/')
if not os.path.exists(UPLOAD_FOLDER_ROOT):
    os.makedirs(UPLOAD_FOLDER_ROOT)

PROCESSED_FOLDER_ROOT = os.path.abspath('tmp_processed_files/')
if not os.path.exists(PROCESSED_FOLDER_ROOT):
    os.makedirs(PROCESSED_FOLDER_ROOT)

du.configure_upload(app, UPLOAD_FOLDER_ROOT)
#du.configure_upload(app, PROCESSED_FOLDER_ROOT)

@server.route('/video/<path:filename>')
def serve_video(filename):
    filepath = os.path.join(UPLOAD_FOLDER_ROOT, filename)
    print(f"Attempting to serve: {filepath}")  # Log the path
    if os.path.exists(filepath):
        print(f"Found file at: {filepath}")  # Log the path
        return send_file(filepath, mimetype='video/mp4')
    else:
        return f"File {filepath} not found!", 404
    
@server.route('/processed_video/<path:filename>')
def serve_processed_video(filename):
    filepath = os.path.join(PROCESSED_FOLDER_ROOT, filename)
    print(f"Attempting to serve: {filepath}")  # Log the path
    if os.path.exists(filepath):
        print(f"Found file at: {filepath}")
        return send_file(filepath, mimetype='video/mp4')
    else:
        return f"Processed video {filepath} not found!", 404



app.layout = html.Div([
    html.H2("Image Quality Processor", style={'textAlign': 'center'}),
        
    # Upload Component
    html.Div([
        du.Upload(
            id='upload-video',
            text='Drag and Drop or Select Video',
            filetypes=['mp4', 'avi'],
            max_file_size=1024*7  # 7 GB
        )
    ], className='six columns', style={'padding': '20px'}),
    # Video Output
    html.Div([
        html.Div(id='output-data', style={'textAlign': 'center'}),
    ], className='twelve columns', style={'marginTop': '20px', 'padding': '20px'}),
    
    html.Div(id='upload-warning',  style={'color': 'red', 'textAlign': 'center', 'marginTop': '20px'}),
    # Button
    html.Div([
        html.Button('Load Processed Video', id='load-video-btn', n_clicks=0, style={'width': '100%'}),
    ], className='six columns', style={'padding': '20px'}),
        
])
    
# ... [rest of the code]

@app.callback(
    dash.dependencies.Output('output-data', 'children'),
    [dash.dependencies.Input('upload-video', 'isCompleted'),
     dash.dependencies.Input('upload-video', 'fileNames'),
    dash.dependencies.Input('load-video-btn', 'n_clicks')]
)

def get_a_list(iscompleted, filenames, n_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "Upload a video first."
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == "load-video-btn" and n_clicks > 0:
        video_path = find_processed_video(PROCESSED_FOLDER_ROOT)
        print(f"DEBUG: video_path: {video_path}")

        if not video_path:
            return "Processed video not found."

        return html.Video(
            controls=True,
            children=[html.Source(src=f'/processed_video/{video_path}', type='video/mp4')],
            style={'width': '80%'}
        )


    if button_id == "upload-video" and iscompleted:
        video_path = find_uploaded_video(UPLOAD_FOLDER_ROOT)
        absolute_video_path = os.path.join(UPLOAD_FOLDER_ROOT, video_path)

        if not video_path:
            return "Error locating the uploaded video."

        processed_video_path = os.path.join(PROCESSED_FOLDER_ROOT, os.path.basename(absolute_video_path))
        print(f"DEBUG: processed_video_path: {processed_video_path}")
        #  Video processing logic here
        cap = cv2.VideoCapture(absolute_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))

        result = cv2.VideoWriter(processed_video_path, codec, fps, (width, height))
        if not result.isOpened():
            print("Error: Couldn't create the output video.")
            return
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0
        processing_msg = "Processing video..."
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            print(f"Processing frame {count}/{frame_count}")
            uiqm, uciqe = nmetrics(frame)
            frame = display_indicator(frame, uiqm, uciqe)
            result.write(frame)
        cap.release()
        result.release()
        cv2.destroyAllWindows()
        processing_msg = ""

        #return html.Video(
        #    controls=True,
        #    children=[html.Source(src=f'/video/{video_path}', type='video/mp4')],
        #    style={'width': '80%'}
        #)

        return "Video processed. Click 'Load Processed Video' button to view.", processing_msg

    return "Upload a video to proceed."

def find_processed_video(main_directory):
    for dirpath, dirnames, filenames in os.walk(main_directory):
        for file in filenames:
            if file.endswith(('.avi', '.mp4')):
                relative_path = os.path.relpath(os.path.join(dirpath, file), main_directory)
                print(f"DEBUG: find_processed_video returns {relative_path}")
                return relative_path
    return None

def find_uploaded_video(main_directory):
    for dirpath, dirnames, filenames in os.walk(main_directory):
        for file in filenames:
            if file.endswith(('.avi', '.mp4')):
                relative_path = os.path.relpath(os.path.join(dirpath, file), main_directory)
                print(f"DEBUG: find_uploaded_video returns {relative_path}")
                return relative_path
    return None




if __name__ == '__main__':
    app.run_server(debug=True)
