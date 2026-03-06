package de.dbaelz.ttm.tts

import org.springframework.stereotype.Component

@Component
class OnnxWrapper {
    fun synthesize(tokenIds: IntArray): ByteArray {
        return ByteArray(0) // stub: integrate ONNX Runtime and model
    }
}

