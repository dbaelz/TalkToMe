package de.dbaelz.ttm.audio

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

class WaveformSamplerTest {
    @Test
    fun `empty input produces valid 44-byte wav header`() {
        val sampler = WaveformSampler()
        val result = sampler.fromFloatArray(arrayListOf())
        assertEquals(44, result.size)
        val buf = ByteBuffer.wrap(result).order(ByteOrder.LITTLE_ENDIAN)
        val riff = ByteArray(4)
        buf.get(riff)
        assertEquals("RIFF", String(riff))
        val chunkSize = buf.int
        assertEquals(36, chunkSize)
        val wave = ByteArray(4)
        buf.get(wave)
        assertEquals("WAVE", String(wave))
        buf.position(20)
        val byteRate = buf.int

        assertEquals(65537, byteRate)
        buf.position(40)
        val dataSize = buf.int
        assertEquals(0, dataSize)
    }

    @Test
    fun `sample conversion matches expected 16bit values`() {
        val sampler = WaveformSampler()
        val samples = arrayListOf(0f, 1f, -1f, 0.5f, -0.5f)
        val bytes = sampler.fromFloatArray(samples, sampleRate = 8000)

        assertTrue(bytes.size >= 44 + samples.size * 2)

        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        buf.position(44)

        fun readShortAt(pos: Int): Short {
            return ByteBuffer.wrap(bytes, pos, 2).order(ByteOrder.LITTLE_ENDIAN).short
        }

        val expected = IntArray(samples.size) { i ->
            val s = (samples[i] * 32767.0f).roundToInt().coerceIn(-32768, 32767)
            s
        }

        for (i in samples.indices) {
            val pos = 44 + i * 2
            val actual = readShortAt(pos).toInt()
            assertEquals(expected[i], actual)
        }
    }
}
