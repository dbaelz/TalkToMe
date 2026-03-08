# Talk To Me
Talk To Me is a Spring Boot application that takes a text and creates an audio file from it using [ONNX Runtime](https://onnxruntime.ai/) and a text-to-speech model.
It provides a very simple REST API for clients to submit text (with an optional config and engine).
Then it generated the audio file, which can be downloaded via another endpoint. The generated audio files are stored locally on the server.

The project is a proof of concept and is currently in early development. Therefore, the main goal is to try out the ONNX Runtimeand how to implement support for (different) TTS engines.
It's not performance-optimized, but rather focused on getting a working prototype up and running for experiments.

## Features
- Text-to-speech conversion using ONNX Runtime (work-in-progress) with GPU (CUDA) and CPU support. GPU is only supported for Windows x64 and Linux x64. See [documentation](https://onnxruntime.ai/docs/get-started/with-java.html)
- Simple job queue and basic support for multiple providers (currently only Pocket TTS implemented)
- Waveform sampling and helper utilities for audio processing
- Local storage of generated audio (`storage/`)

### Models/Engines
Currently only [Pocket TTS](https://huggingface.co/kyutai/pocket-tts) with exported [ONNX models](https://huggingface.co/KevinAHM/pocket-tts-onnx) is supported.
The models can be placed anywhere on the filesystem, but the default path is `models/pocket-tts/` (configurable via `application.properties`).

### Configuration
Configuration with [application.properties](src/main/resources/application.properties).


### Development
For development the project includes a [bruno collection](bruno/bruno.json) with the basic API calls. Install [Bruno](https://www.usebruno.com/) to use it.


## License
The project is licensed by the [Apache 2 license](LICENSE).