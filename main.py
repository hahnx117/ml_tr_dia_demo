from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import whisper
from pprint import pprint
import torch
import os
from sys import exit

WHISPER_HF_TOKEN = os.getenv("WHISPER_HF_TOKEN")
TEST_FILE = "data/clip.mp3"

SPEAKER_00 = "SPEAKER_00"
SPEAKER_01 = "SPEAKER_01"
#SPEAKER_02 = "SPEAKER_02"
#SPEAKER_03 = "SPEAKER_03"

def name_speaker(speaker_id):
    if speaker_id == "SPEAKER_00":
        return SPEAKER_00
    elif speaker_id == "SPEAKER_01":
        return SPEAKER_01
    #elif speaker_id == "SPEAKER_02":
    #    return SPEAKER_02
    #elif speaker_id == "SPEAKER_03":
    #    return SPEAKER_03
    else:
        return "UNKNOWN"

def find_speaker(search_tuple, diarization_result):
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

def transcript_demo(transcription):
    """Demo transcriptions and segmentation."""
    ## Print the transcrption itself
    print("Transcription:")
    print(transcription['text'])
    print("\n\n")

    ## Show segmentation of transcription result
    print("Transcription segmentation:")
    print(transcription['segments'][:3])
    print("\n\n")

def diarization_demo(diarization):
    """Demo diarization."""
    ## Print the diarization
    print("Diarization:")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
    print("\n\n")

def combo_demo(transcription, diarization):
    """Demo the combination of the two."""
    ## Now go through line by line and assign a speaker
    for i in transcription['segments']:
        time_tuple = (i['start'], i['end'])
        active_speaker = find_speaker(time_tuple, diarization)
        print(f"{name_speaker(active_speaker)}: {i['text']}")
        #print(f"{i['text']}")

if __name__ == "__main__":
    ## First create the transcript
    model = whisper.load_model("tiny.en")
    try:
        result = model.transcribe(TEST_FILE)
    except FileNotFoundError:
        print(f"Error: The file {TEST_FILE} was not found.")
        exit(1)
    except Exception as e:
        print(f"Error: An unexpected error occurred while reading the file {TEST_FILE}: {str(e)}")
        exit(1)

    ## Now create the diarization
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=WHISPER_HF_TOKEN)


    ## Comment out the "mps" and "cpu" line and use gpu if you have CUDA installed and available.
    ## Comment out the "cpu" and "gpu" lines if you have a Mac M1/M2/M3
    # pipeline.to(torch.device("cuda")) # Use this line if you have an Nvidia GPU
    # pipeline.to(torch.device("mps")) # Use this line if you have a Mac M1/M2/M3
    pipeline.to(torch.device("cpu")) # Use this line if you only have a CPU

    with ProgressHook() as hook:
        diarization = pipeline(TEST_FILE, hook=hook)

    transcript_demo(result)
    diarization_demo(diarization)
    combo_demo(result, diarization)