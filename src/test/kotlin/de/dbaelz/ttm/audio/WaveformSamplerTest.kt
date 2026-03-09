package de.dbaelz.ttm.audio

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import kotlin.math.roundToInt

class WaveformSamplerTest {
    @Test
    fun `empty input produces valid 44-byte wav header`() {
        val sampler = WaveformSampler()

        val result = sampler.fromFloatArray(floatArrayOf())

        assertEquals(HEADER_SIZE, result.size)
        assertEquals(RIFF, String(result, 0, 4))
        assertEquals(36, readResultRange(result, 4))
        assertEquals(WAVE, String(result, 8, 4))
        assertEquals(65537, readResultRange(result, 20))
        assertEquals(0, readResultRange(result, 40))
    }

    @Test
    fun `input produces valid 44-byte wav header`() {
        val sampler = WaveformSampler()

        val audio = floatArrayOf(1f, 0.5f, -0.5f)

        val result = sampler.fromFloatArray(audio)

        val dataBytes = audio.size * Short.SIZE_BYTES
        val expectedSize = HEADER_SIZE + dataBytes

        assertEquals(expectedSize, result.size)
        assertEquals(RIFF, String(result, 0, 4))
        assertEquals(36 + dataBytes, readResultRange(result, 4))
        assertEquals(WAVE, String(result, 8, 4))
        assertEquals(65537, readResultRange(result, 20))
        assertEquals(dataBytes, readResultRange(result, 40))
    }

    @Test
    fun `sample conversion matches expected 16bit values`() {
        val sampler = WaveformSampler()
        val samples = floatArrayOf(0f, 1f, -1f, 0.5f, -0.5f)
        val bytes = sampler.fromFloatArray(samples, sampleRate = 8000)

        assertTrue(bytes.size >= 44 + samples.size * 2)

        val expected = IntArray(samples.size) { i ->
            (samples[i] * 32767.0f).roundToInt().coerceIn(-32768, 32767)
        }

        for (i in samples.indices) {
            val pos = 44 + i * 2
            val actual = readResultContent(bytes, pos)
            assertEquals(expected[i], actual)
        }
    }

    private fun readResultRange(bytes: ByteArray, offset: Int): Int {
        val end = minOf(bytes.size, offset + 4)
        var value = 0
        var shift = 0
        for (i in offset until end) {
            value = value or ((bytes[i].toInt() and 0xFF) shl shift)
            shift += 8
        }
        return value
    }

    private fun readResultContent(bytes: ByteArray, offset: Int): Int {
        val combined = readResultRange(bytes, offset) and 0xFFFF
        return combined.toShort().toInt()
    }

    private companion object {
        const val HEADER_SIZE = 44
        const val RIFF = "RIFF"
        const val WAVE = "WAVE"
    }
}
