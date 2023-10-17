## Dash Demonstrator to visualize image quality using UCIQE metric

## Instructions

To get started, first clone this repo:


```
git clone git@gitlab.sintef.no:Iroshani.Jayawardene/image_quality_demonstrator.git
cd image_quality_demonstrator

```


Create and activate a python virtual environment:
```
python -m venv dash_env
source dash_env/bin/activate # for Windows, use venv\Scripts\activate.bat
```

Install all the requirements:

```
pip install -r requirements.txt
```

You can now run the app:
```
python app.py
```

and visit http://127.0.0.1:8050/.


## Run object detection on videos

1. Upload the video ( Example videos are inside data folder)
2. Select the every xth frame to be processed from the drop down list.
3. Click Start video
4. Video will be displayed with the image quality metric UCIQE

UCIQE value has the range 0 to infinity.
For the  videos provided by VUVI, UCIQE value is desplayed as below
if UCIQE < 20 : color = red
if 20 < UCIQE < 27 : color = yellow
if UCIQE > 27 : color = green

reference : https://github.com/xueleichen/PSNR-SSIM-UCIQE-UIQM-Python


