# Talk To Me
Talk To Me is a Spring Boot application that takes a text and creates an audio file.
It's using the [ONNX Runtime](https://onnxruntime.ai/) and a text-to-speech model and provides a very simple REST API.

The project is a proof of concept and in early development. It's main goal is to try out the ONNX Runtime and how to implement support for TTS engines.
It's not performance-optimized, but rather focused on getting a working prototype up and running for experiments.

## Features
- Text-to-speech conversion using ONNX Runtime with GPU (CUDA) and CPU support. GPU is only supported for Windows x64 and Linux x64. See [documentation](https://onnxruntime.ai/docs/get-started/with-java.html)
- Base structure to support multiple TTS Engines. Currently only Pocket TTS is implemented
- Most application settings are configurable. See [application.properties](src/main/resources/application.properties) for details
- Generated Audio files are saved at local storage (default path: `storage/`)


## Getting started 
Currently only [Pocket TTS](https://huggingface.co/kyutai/pocket-tts) with [this ONNX models](https://huggingface.co/KevinAHM/pocket-tts-onnx) is supported.

### Required files
- ONNX files: INT8 or FP32 versions of `flow_lm_flow` and `flow_lm_main`. The `mimi_decoder.onnx`, `mimi_encoder.onnx` and `text_conditioner.onnx`
- `tokenizer.model` file for the SentencePiece tokenizer
- WAV file for the voice cloning. This should be a WAV file with a sample rate of 24000 Hz, mono channel, and 16-bit depth and a duration of around 10 seconds. In my tests the `reference_sample.wav` (16000 Hz, mono) provided in the HuggingFace repository wasn't ideal. So I recommend using a custom file with the specifications. 

### Configuration and running
- Settings like address/port, basic auth, model path, model filenames or GPU support can be configured in [application.properties](src/main/resources/application.properties)
- Execute `./gradlew bootRun` to start the application. The API will be available at `http://127.0.0.1:8080/api/tts`
- The included [Bruno collection](bruno/bruno.json) has example API calls. See the [Bruno documentation](https://www.usebruno.com/) how to install it.

## License
The project is licensed by the [Apache 2 license](LICENSE).