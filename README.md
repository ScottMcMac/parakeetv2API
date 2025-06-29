# parakeetv2API

This project will serve parakeet-tdt-0.6b-v2 as an OpenAI API compatable transcription endpoint (e.g. http://0.0.0.0:8005/v1/audio/transcriptions). It will use Python and FastAPI to create the API. Eventually the goal is to make it easy to deploy via LXC and Docker. For now we are just deploying via python. 

Users will be able to connect to the API, submit an audio file, and recieve a transcription. 

We need two python packages to run this code `nemo_toolkit["asr"]` and `cuda-python>=12.3`

I set it up as follows:

```
conda create -n nemo python=3.11
conda activate nemo
pip install -U nemo_toolkit["asr"]
pip install cuda-python>=12.3
```

Steps the server will take when initialized

To prepare to recieve, convert and transcribe audio, the following should be run on startup:

```
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
output = asr_model.transcribe(['./tests/audio_files/test_wav_16000Hz_mono.wav'])
print(output[0].text)
```





Steps the server will take in response to a request:

- Check if the inputs are valid as specified below and return informative errors if now. 
- Check if submitted file is valid type: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm
- Check if submitted file is a type that works with parakeet-tdt-0.6b-v2: 16kHz, mono, .wav or .flac
    - If not, use ffmpeg to convert to 16kHz, mono, .wav
- Return the appropriate response (described below)





The above should return `The quick-brown fox jumped over the lazy dog.`



Need to ensure model starts loaded and stays loaded and is always ready for fast transcription:



Use python 3.11 environment.

may need `sudo apt install build-essential cmake libprotoc-dev protobuf-compiler ffmpeg` before setting up environment

"The quick brown fox jumped over the lazy dog." case insensitive and ignore punctuation.

Need to make sure the model stays loaded at all times. 


# Handling Requests

The OpenAI API specification allows for several items in the request body when making requests to `https://api.openai.com/v1/audio/transcriptions`. Here is how our server will handle each of them when requests are made to `serveraddress:port/v1/audio/transcriptions`.:

## file 

file
Required

The audio file object (not file name) to transcribe, in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.

## model

string
Required

Always will use the same model, but Included to match OpenAI API spec. ID of the model to use. The options are gpt-4o-transcribe, gpt-4o-mini-transcribe, parakeet-tdt-0.6b-v2 and whisper-1. Any option will use the parakeet-tdt-0.6b-v2 model backend. 

## chunking_strategy

"auto" or object
Optional

Ignored if provided.

## include[]

array
Optional

Additional information to include in the transcription response. For now, we'll ignore and provide the normal transcription

## language

string
Optional

The language of the input audio in ISO-639-1. The model only supports English (en). So, provide an informative error if provided and not `en`.

## prompt

string
Optional

Ignored

## response_format

string
Optional
Defaults to json

The format of the output, in one of these options: json, text, srt, verbose_json, or vtt.

## stream

boolean or null
Optional
Defaults to false

Streaming is not supported. This will be ignored if the model is set to  whisper-1 or parakeet-tdt-0.6b-v2 and will be ignored if set to true. If the model is set to gpt-4o-transcribe or gpt-4o-mini-transcribe and stream is set to true, provide an informative error.

## temperature

number
Optional

Ignored

## timestamp_granularities[]

array
Optional
Defaults to segment

The timestamp granularities to populate for this transcription. response_format must be set verbose_json to use timestamp granularities. Either or both of these options are supported: word, or segment. Note: There is no additional latency for segment timestamps, but generating word timestamps incurs additional latency.

# Returns

The OpenAI API specification can provide three types of returns when requests are made to `https://api.openai.com/v1/audio/transcriptions`, a transcription object, a verbose transcription object or a stream of transcript events. This project will only support a transcription object. 

# Example

Given the above, if our server recieves the following request

```
curl serveraddress:port/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F file="@/path/to/file/audio.mp3" \
  -F model="gpt-4o-transcribe"
```

it should respond with something like 

```
{
  "text": "The quick-brown fox jumped over the lazy dog.",
  "usage": {
    "type": "tokens",
    "input_tokens": 1,
    "input_token_details": {
      "text_tokens": 0,
      "audio_tokens": 1
    },
    "output_tokens": 1,
    "total_tokens": 2
  }
}
```

Since we don't actually care about tracking usage by token, but want to match the format of what the OpenAI API would provide. the server will just always say that there was 1 (audio) input token and 1 output token, for 2 "total_tokens".