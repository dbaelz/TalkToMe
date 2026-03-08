package de.dbaelz.ttm.audio

import org.springframework.stereotype.Component
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

@Component
class WaveformSampler {
    fun fromFloatArray(audio: FloatArray, sampleRate: Int = 24000): ByteArray {
        val byteRate = sampleRate * CHANNELS * BITS_PER_SAMPLE / 8
        val dataBytesSize = audio.size * Short.SIZE_BYTES

        val headerBytes = header(
            sampleRate = sampleRate,
            byteRate = byteRate,
            dataBytesSize = dataBytesSize
        )

        val buffer =
            ByteBuffer.allocate(headerBytes.size + dataBytesSize).order(ByteOrder.LITTLE_ENDIAN)
        buffer.put(headerBytes)

        for (f in audio) {
            val amplitude = when {
                f.isNaN() -> 0f
                f > 1f -> 1f
                f < -1f -> -1f
                else -> f
            }
            val s = (amplitude * 32767.0f).roundToInt().coerceIn(-32768, 32767).toShort()
            buffer.putShort(s)
        }

        return buffer.array()
    }

    private fun header(
        sampleRate: Int,
        byteRate: Int,
        dataBytesSize: Int
    ): ByteArray = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).apply {
        put("RIFF".toByteArray())
        putInt(36 + dataBytesSize)
        put("WAVE".toByteArray())
        put("fmt ".toByteArray())
        putInt(16)
        putShort(1.toShort())
        putShort(CHANNELS.toShort())
        putInt(sampleRate)
        putInt(byteRate)
        putShort((CHANNELS * BITS_PER_SAMPLE / 8).toShort())
        putShort(BITS_PER_SAMPLE.toShort())
        put("data".toByteArray())
        putInt(dataBytesSize)
    }.array()


    private companion object {
        const val CHANNELS = 1
        const val BITS_PER_SAMPLE = 16
    }
}
