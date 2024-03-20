from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import whisper
from pprint import pprint
import torch
import os

WHISPER_HF_TOKEN = os.getenv("WHISPER_HF_TOKEN")
TEST_FILE = "data/clip.mp3"

def find_speaker(search_tuple, diarization_result):
    #, 
    """
    Take in diarization result and return speaker block.
    return speaker id.
    """

    avg_time = (search_tuple[0] + search_tuple[1]) / 2.0

    time_correction = None

    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        if time_correction == None:
            time_correction = turn.start
        if turn.start <= avg_time + time_correction <= turn.end:
            return speaker



## First create the transcript
model = whisper.load_model("tiny.en")
result = model.transcribe(TEST_FILE)

## Now create the diarization
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=WHISPER_HF_TOKEN)


# Comment out the CPU line and use GPU if you have CUDA installed and available
# pipeline.to(torch.device("cuda"))
pipeline.to(torch.device("cpu"))

with ProgressHook() as hook:
    diarization = pipeline(TEST_FILE, hook=hook)


## Now go through line by line and assign a speaker
for i in result['segments']:
    time_tuple = (i['start'], i['end'])
    active_speaker = find_speaker(time_tuple, diarization)
    print(f"{name_speaker(active_speaker)}: {i['text']}")
    #print(f"{i['text']}")