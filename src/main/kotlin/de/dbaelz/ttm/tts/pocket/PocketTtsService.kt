package de.dbaelz.ttm.tts.pocket

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import de.dbaelz.ttm.model.JobStatus
import de.dbaelz.ttm.model.TtsJob
import de.dbaelz.ttm.onnx.OnnxWrapper
import de.dbaelz.ttm.repository.JobRepository
import de.dbaelz.ttm.service.LocalStorageService
import de.dbaelz.ttm.service.TtsService
import org.springframework.beans.factory.annotation.Value
import org.springframework.stereotype.Service
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Paths
import java.util.*
import java.util.concurrent.Executors
import javax.sound.sampled.AudioFormat
import javax.sound.sampled.AudioInputStream
import javax.sound.sampled.AudioSystem
import kotlin.io.path.absolutePathString
import kotlin.math.abs

@Service
class PocketTtsService(
    @Value("\${tts.models.pocket-tts}") private val modelsPath: String,
    @Value("\${tts.models.pocket-tts.encoder}") private val encoder: String,
    @Value("\${tts.models.pocket-tts.text_conditioner}") private val textConditioner: String,
    @Value("\${tts.models.pocket-tts.flow_lm_main}") private val lmMain: String,
    @Value("\${tts.models.pocket-tts.flow_lm_flow}") private val lmFlow: String,
    @Value("\${tts.models.pocket-tts.decoder}") private val decoder: String,
    @Value("\${tts.models.pocket-tts.voice}") private val voice: String,
    private val tokenizer: SentencePieceTokenizer,
    private val onnx: OnnxWrapper,
    private val storage: LocalStorageService,
    private val repo: JobRepository
) : TtsService {
    init {
        onnx.loadModelFiles(
            modelsPath = modelsPath,
            modelFiles = listOf(encoder, textConditioner, lmMain, lmFlow, decoder)
        )
    }

    private val executor = Executors.newFixedThreadPool(2)

    override fun generate(text: String): TtsJob {
        val id = UUID.randomUUID().toString()
        val job = TtsJob(id = id, text = text)
        repo.save(job)

        executor.submit {
            process(job)
        }

        return job
    }

    private fun process(job: TtsJob) {
        try {
            job.status = JobStatus.RUNNING
            repo.save(job)

            val audio = generateAudio(job.text)

            val fileId = storage.save(audio)
            job.fileId = fileId
            job.status = JobStatus.DONE
            repo.save(job)
        } catch (e: Exception) {
            job.status = JobStatus.FAILED
            repo.save(job)
        }
    }

    private fun generateAudio(text: String): ByteArray {
        val session = onnx.getSession()
        val textConditionerModel = session[textConditioner]
        val lmMainModel = session[lmMain]
        val lmFlowModel = session[lmFlow]
        val decoderModel = session[decoder]
        val encoderModel = session[encoder]

        if (textConditionerModel == null || lmMainModel == null
            || lmFlowModel == null || decoderModel == null || encoderModel == null
        ) {
            throw IllegalStateException("One or more ONNX models could not be loaded")
        }

        val voiceEmbeddings = voiceEmbeddings(encoderModel)
        val tokens = tokenizer.tokenize(text)




        return ByteArray(0) // Placeholder for generated audio data
    }

    private fun voiceEmbeddings(encoder: OrtSession): Array<FloatArray> {
        val voicePath = Paths.get(modelsPath).resolve(voice).absolutePathString()
        val file = File(voicePath)
        if (!file.exists()) {
            throw IllegalArgumentException("Voice file not found: $voicePath")
        }

        val originalAis: AudioInputStream = AudioSystem.getAudioInputStream(file)
        val originalFormat: AudioFormat = originalAis.format

        val targetSampleRate = 24000f

        // Try converting to PCM_SIGNED, 16-bit, mono, little-endian
        val targetFormat = AudioFormat(
            AudioFormat.Encoding.PCM_SIGNED,
            originalFormat.sampleRate,
            16,
            1,
            2,
            originalFormat.sampleRate,
            false
        )

        val pcmFloats: FloatArray = try {
            val convertedAis = AudioSystem.getAudioInputStream(targetFormat, originalAis)
            val raw = convertedAis.readBytes()
            convertedAis.close()
            originalAis.close()

            val bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
            val sb = bb.asShortBuffer()
            val n = sb.remaining()
            val floats = FloatArray(n)
            for (i in 0 until n) {
                floats[i] = sb.get(i) / 32768.0f
            }
            floats
        } catch (e: Exception) {
            throw e
        }

        // Build input tensor with shape [1, 1, length]
        val env = onnx.getEnvironment()
        val shape = longArrayOf(1L, 1L, pcmFloats.size.toLong())
        val data = FloatArray(pcmFloats.size)
        System.arraycopy(pcmFloats, 0, data, 0, pcmFloats.size)

        val tensor = OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(data), shape)

        val inputs = mapOf("audio" to tensor)
        val results = encoder.run(inputs)

        // Convert output to Array<FloatArray> representing [numFrames][embedDim]
        val embeddings: Array<FloatArray> = when (val outVal = results[0].value) {
            is Array<*> -> {
                val first = outVal[0]
                if (first is Array<*>) {
                    val inner = first
                    val n = inner.size
                    val d0 = inner[0]
                    if (d0 is FloatArray) {
                        val dim = d0.size
                        val resultArr = Array(n) { FloatArray(dim) }
                        for (i in 0 until n) {
                            val row = inner[i] as FloatArray
                            System.arraycopy(row, 0, resultArr[i], 0, dim)
                        }
                        resultArr
                    } else if (d0 is Array<*>) {
                        Array(0) { FloatArray(0) }
                    } else {
                        Array(0) { FloatArray(0) }
                    }
                } else if (first is FloatArray) {
                    val row = first
                    arrayOf(row)
                } else {
                    Array(0) { FloatArray(0) }
                }
            }

            is java.nio.FloatBuffer -> {
                val buf = outVal
                buf.rewind()
                val total = buf.remaining()
                val dims = total / 1024
                val out = Array(dims) { FloatArray(1024) }
                for (i in 0 until dims) {
                    buf.get(out[i])
                }
                out
            }

            else -> Array(0) { FloatArray(0) }
        }

        results.close()
        tensor.close()

        return embeddings
    }


    override fun getJob(id: String): TtsJob? = repo.findById(id)
    override fun getFile(id: String): ByteArray? = storage.load(id)
}
