# parakeetv2API

This project will serve parakeet-tdt-0.6b-v2 via an OpenAI API compatable transcription endpoint (e.g. http://0.0.0.0:8011/v1/audio/transcriptions). It will use Python and FastAPI to create the API. Eventually the goal is to make it easy to deploy via LXC and Docker. For now, we are just creating the Python code to serve the API. 

Users will be able to connect to the API, submit an audio file, and recieve a transcription. 

## Environment Setup

We need two python packages to run this code `nemo_toolkit["asr"]` and `cuda-python>=12.3`

I set up a conda environment as follows:

```
sudo apt install build-essential cmake libprotoc-dev protobuf-compiler ffmpeg
conda create -n nemo python=3.11
conda activate nemo
pip install -U nemo_toolkit["asr"]
pip install cuda-python>=12.3
```

**This is where I've stopped. More packages may be needed as we develop the project.** For now, we have the nemo conda environment available and no code written.

## How the server will work

This code must be run on a machine with an NVIDIA GPU to run the model. 

The user setting up the server should be able to specify the following:

- Whether to listen just locally, or also externally, e.g. just on `localhost` or `0.0.0.0` too.
- Which GPU to use if more than one are available (e.g. 0-3 if 4 NVIDIA GPUs are present)
- The port on which to listen (default to `8011`)


### Steps the server will take when initialized

To prepare to recieve and transcribe audio, something like the following should be run on startup:

```
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2") # Ensure the program can inference the model without loading it.
output = asr_model.transcribe(['./tests/audio_files/test_wav_16000Hz_mono.wav']) # Ensure it is ready for 16-bit 16000Hz mono wav files
output = asr_model.transcribe(['./tests/audio_files/test_wav_16000Hz_mono_32bit.wav']) # Ensure model is ready for 16000Hz mono wav files in other than 16-bit
output = asr_model.transcribe(['./tests/audio_files/test_flac_8000Hz_mono.flac']) # Ensure model is ready for 16000Hz mono wav files in other than 16-bit
```

The model should stay loaded to be ready for fast responses.

### Steps the server will take in response to a request

- Check if the inputs are valid as specified below and return informative errors if not. 
    - For checking valid file types first check if the extension is one of the valid inputs: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm. Then, check if they contain valid audio data recognized by ffmpeg.
- Check if submitted file is a type that works with parakeet-tdt-0.6b-v2: Specifically, the audio must be 16kHz, mono channel, and a .wav or .flac format.
    - If not, use ffmpeg to convert to 16kHz, mono, .wav
- Return the appropriate response (described below)
- Clean up (e.g. delete any cached files)

# Requests

The OpenAI API specification allows for several items in the request body when making requests to `https://api.openai.com/v1/audio/transcriptions`. Below is how our server will need each of them when requests are made to `serveraddress:port/v1/audio/transcriptions`.:

## file 

file
Required

The audio file object (not file name) to transcribe, in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.

## model

string
Optional

Included to match the OpenAI API spec. ID of the model to use. Expected options are gpt-4o-transcribe, gpt-4o-mini-transcribe, parakeet-tdt-0.6b-v2 and whisper-1, but any option will actually use the parakeet-tdt-0.6b-v2 model backend. 

## chunking_strategy

"auto" or object
Optional

Ignored if provided.

## include[]

array
Optional

Additional information to include in the transcription response. We'll ignore and provide the normal transcription.

## language

string
Optional

The language of the input audio in ISO-639-1. The model only supports English (en). So, provide an informative error if provided and not `en`.

## prompt

string
Optional

Ignored if provided.

## response_format

string
Optional
Defaults to json

The format of the output, in one of these options: json, text, srt, verbose_json, or vtt. Our server only supports json. So, provide an informative error if provided and not `json`.

## stream

boolean or null
Optional
Defaults to false

Streaming is not supported. For compatability, this will be ignored if the model is set to  whisper-1 or parakeet-tdt-0.6b-v2. If the model is set to gpt-4o-transcribe or gpt-4o-mini-transcribe and stream is set to true, provide an informative error.

## temperature

number
Optional

Ignored if provided.

## timestamp_granularities[]

array
Optional 

Not supported here. Return an informative error if included.

# Returns

The OpenAI API specification can provide three types of returns when requests are made to `https://api.openai.com/v1/audio/transcriptions`, a transcription object, a verbose transcription object or a stream of transcript events. This project will only support a transcription object. 

## Example

Below is an example request to the API (Note, our server will not require an API key. So, it will ignore `$OPENAI_API_KEY` if included. It will allow it for compatability.)

```
curl serveraddress:port/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F file="@/path/to/file/audio.mp3" \
  -F model="gpt-4o-transcribe"
```

The server should respond in the following JSON format:

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

Since we don't actually care about tracking usage by token, but want to match the format of what the OpenAI API would provide, the server will just always say that there was 1 (audio) input token and 1 output token, for 2 "total_tokens". So, server responses will only vary from the above by the text included in the text field, depending on the transcription output by the model.

# Model Endpoints

The OpenAI API specification allows querying model information via the API. 

## All Models

Users should be able to get a list of available models like they would via the OpenAI API. So, e.g.

```
curl serveraddress:port/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

(again, we can ignore $OPENAI_API_KEY) should yield:

```
{
  "object": "list",
  "data": [
    {
      "id": "gpt-4o-transcribe",
      "object": "model",
      "created": 1744718400,
      "owned_by": "parakeet-tdt-0.6b-v2-released-by-nvidia-with-cc-by-40-license"
    },
    {
      "id": "gpt-4o-mini-transcribe",
      "object": "model",
      "created": 1744718400,
      "owned_by": "parakeet-tdt-0.6b-v2-released-by-nvidia-with-cc-by-40-license",
    },
    {
      "id": "parakeet-tdt-0.6b-v2",
      "object": "model",
      "created": 1744718400,
      "owned_by": "parakeet-tdt-0.6b-v2-released-by-nvidia-with-cc-by-40-license"
    },
    {
      "id": "whisper-1",
      "object": "model",
      "created": 1744718400,
      "owned_by": "parakeet-tdt-0.6b-v2-released-by-nvidia-with-cc-by-40-license"
    },
  ],
  "object": "list"
}
```

## Single Model

Users should be able query information about individual models via the OpenAI API. So, e.g.


```
curl serveraddress:port/v1/models/parakeet-tdt-0.6b-v2 \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

should yield:

```
{
    "id": "parakeet-tdt-0.6b-v2",
    "object": "model",
    "created": 1744718400,
    "owned_by": "parakeet-tdt-0.6b-v2-released-by-nvidia-with-cc-by-40-license"
}
```

and similarly for the other three supported model names.

## Coding practices

The code should make use of tests for all features to ensure everything works as expected. To support this `./tests/audio_files/` contains many audio_files that should all be tested against to ensure the software works as expected across various audio inputs. Some may need to be converted by ffmpeg. Once converted (if necessary), parakeet-tdt-0.6b-v2 should transcribe all the files to `The quick brown fox jumped over the lazy dog.` The tests should pass regardless of capitalization and punctuation, e.g. `the quick-brown fox jumped over the lazy Dog!` should also pass, but `The quick brown fax jumped over the lazy dog?` sould not. `./tests/non_audio_files/` contains two files that should trigger errors, but not crash the software. MisleadingEncoding.mp3 is not actually an mp3 file, and WrongType.txt is a text file. If submitted they server should send errors.

Remember SOLID, KISS, and YAGNI!
