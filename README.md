# ml_tr_dia_demo
Demo using ML tools for transcription and diarization.

## Prerequisites

### Python
This requires Python > 3.9 (tested on Python 3.10.12)

### ToC
 The following packages need to have the user conditions agreed to:
 + https://huggingface.co/pyannote/segmentation-3.0
 + https://huggingface.co/pyannote/speaker-diarization-3.1

### Hugging Face Access Token
You will need to generate a Hugging Face Access token here: https://huggingface.co/login?next=%2Fsettings%2Ftokens. 
> Note: you will need to create an account to do this.

Once the token is acquired, this can either be hard coded into the code (not recommended) or read in to the code in various ways (in this example it is an environment variable I have created). 

## Installing and running the demo

### Using Python and `pip`
There are two methods of using this repo:

1. Clone the repo down,
    ```
    git clone https://github.com/hahnx117/ml_tr_dia_demo.git
    ```

2. Download the zip file. 
    1. Click on the green "Code" button
    2. "Download zip"
    3. Extract the zip to a directory of your choice.

Once the repo is on your machine, you can use the command line to install a virtual environment and install the requirements,

```
python3 -m venv .

source bin/activate

python3 -m pip install -r requirements.txt
```

