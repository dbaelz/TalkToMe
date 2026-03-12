package de.dbaelz.ttm.tts.chatterbox

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.TensorInfo
import de.dbaelz.ttm.audio.WaveformSampler
import de.dbaelz.ttm.model.TtsEngine
import de.dbaelz.ttm.model.TtsJob
import de.dbaelz.ttm.onnx.OnnxWrapper
import de.dbaelz.ttm.tts.ChatterboxConfig
import de.dbaelz.ttm.tts.TtsExecutor
import org.springframework.beans.factory.annotation.Value
import org.springframework.stereotype.Service
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.nio.file.Paths
import javax.sound.sampled.AudioFormat
import javax.sound.sampled.AudioInputStream
import javax.sound.sampled.AudioSystem

@Service
class ChatterboxTtsExecutor(
    @Value("\${tts.models.chatterbox:models/chatterbox}") private val modelsPath: String,
    @Value("\${tts.models.chatterbox.speech_encoder:speech_encoder.onnx}") private val speechEncoder: String,
    @Value("\${tts.models.chatterbox.embed_tokens:embed_tokens.onnx}") private val embedTokens: String,
    @Value("\${tts.models.chatterbox.language_model:language_model.onnx}") private val languageModel: String,
    @Value("\${tts.models.chatterbox.conditional_decoder:conditional_decoder.onnx}") private val conditionalDecoder: String,
    @Value("\${tts.models.chatterbox.voice:voice.wav}") private val voice: String,
    private val tokenizer: ChatterboxTokenizer,
    private val onnx: OnnxWrapper,
    private val waveformSampler: WaveformSampler
) : TtsExecutor {

    override fun invoke(job: TtsJob): ByteArray {
        if (job.config !is ChatterboxConfig) throw IllegalArgumentException("Invalid config type for ChatterboxExecutor")

        onnx.loadModelFiles(
            engine = TtsEngine.CHATTERBOX,
            modelsPath = modelsPath,
            modelFiles = listOf(speechEncoder, embedTokens, languageModel, conditionalDecoder)
        )
        return generateAudio(job.text, job.config)
    }

    private fun generateAudio(text: String, config: ChatterboxConfig): ByteArray {
        val speechEncoderModel = onnx.getModel(TtsEngine.CHATTERBOX, speechEncoder)
        val embedTokensModel = onnx.getModel(TtsEngine.CHATTERBOX, embedTokens)
        val languageModelModel = onnx.getModel(TtsEngine.CHATTERBOX, languageModel)
        val conditionalDecoderModel = onnx.getModel(TtsEngine.CHATTERBOX, conditionalDecoder)

        val env = onnx.getEnvironment()
        val inputIds = tokenizer.tokenize(text)
        if (inputIds.isEmpty()) throw IllegalArgumentException("Text tokenization produced no tokens")

        val embedInputs = LinkedHashMap<String, OnnxTensor>()
        val generateTokens = ArrayList<Long>()
        generateTokens.add(START_SPEECH_TOKEN)

        var attentionMask: LongArray? = null
        var pastKeyValues: LinkedHashMap<String, OnnxTensor>? = null

        var promptTokenTensor: OnnxTensor? = null
        var refXVector: OnnxTensor? = null
        var promptFeat: OnnxTensor? = null
        var speechTokensTensor: OnnxTensor? = null

        try {
            embedInputs["input_ids"] =
                createLongTensor(env, inputIds, longArrayOf(1L, inputIds.size.toLong()))
            embedInputs["position_ids"] = createInitialPositionIds(env, inputIds)
            embedInputs["exaggeration"] =
                OnnxTensor.createTensor(
                    env,
                    FloatBuffer.wrap(floatArrayOf(config.exaggeration)),
                    longArrayOf(1L)
                )

            val voiceAudio = loadVoiceAudio()
            val audioTensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(voiceAudio),
                longArrayOf(1L, voiceAudio.size.toLong())
            )
            val speechRes = speechEncoderModel.run(mapOf("audio_values" to audioTensor))
            val condEmb = try {
                val cond = cloneTensor(speechRes[0] as OnnxTensor)
                promptTokenTensor = cloneTensor(speechRes[1] as OnnxTensor)
                refXVector = cloneTensor(speechRes[2] as OnnxTensor)
                promptFeat = cloneTensor(speechRes[3] as OnnxTensor)
                cond
            } finally {
                speechRes.close()
                audioTensor.close()
            }

            val maxNewTokens = 256
            for (i in 0 until maxNewTokens) {
                val embedRes = embedTokensModel.run(embedInputs)
                val tokenEmbeds = try {
                    cloneTensor(embedRes[0] as OnnxTensor)
                } finally {
                    embedRes.close()
                }

                val lmEmbeds = if (i == 0) {
                    val joined = concatenateEmbeds(env, condEmb, tokenEmbeds)
                    tokenEmbeds.close()
                    joined
                } else {
                    tokenEmbeds
                }

                if (i == 0) {
                    val seqLen = lmEmbeds.info.shape[1].toInt()
                    attentionMask = LongArray(seqLen) { 1L }
                    pastKeyValues = createInitialPastKeyValues()
                }

                val mask = attentionMask ?: throw IllegalStateException("Missing attention mask")
                val kv = pastKeyValues ?: throw IllegalStateException("Missing kv cache")

                val attentionMaskTensor =
                    createLongTensor(env, mask, longArrayOf(1L, mask.size.toLong()))
                val lmInputs = LinkedHashMap<String, OnnxTensor>()
                lmInputs["inputs_embeds"] = lmEmbeds
                lmInputs["attention_mask"] = attentionMaskTensor
                for ((name, tensor) in kv) lmInputs[name] = tensor

                val lmRes = languageModelModel.run(lmInputs)
                try {
                    val logits = extractLastLogits(lmRes[0] as OnnxTensor)
                    val next = applyRepetitionPenaltyAndArgmax(generateTokens, logits)
                    generateTokens.add(next)

                    val updated = LinkedHashMap<String, OnnxTensor>()
                    var outIdx = 1
                    for ((name, old) in kv) {
                        val newTensor = cloneTensor(lmRes[outIdx] as OnnxTensor)
                        old.close()
                        updated[name] = newTensor
                        outIdx += 1
                    }
                    pastKeyValues = updated

                    if (next == STOP_SPEECH_TOKEN) break

                    embedInputs.remove("input_ids")?.close()
                    embedInputs.remove("position_ids")?.close()
                    embedInputs["input_ids"] =
                        createLongTensor(env, longArrayOf(next), longArrayOf(1L, 1L))
                    embedInputs["position_ids"] =
                        createLongTensor(env, longArrayOf((i + 1).toLong()), longArrayOf(1L, 1L))
                    attentionMask = appendOne(mask)
                } finally {
                    lmRes.close()
                    lmEmbeds.close()
                    attentionMaskTensor.close()
                }
            }
            condEmb.close()

            val promptToken = extractLongArray(promptTokenTensor)
            val body =
                if (generateTokens.size > 2) generateTokens.subList(1, generateTokens.size - 1)
                    .toLongArray() else LongArray(0)
            val speechTokens = concatLongArrays(promptToken, body)
            speechTokensTensor =
                createLongTensor(env, speechTokens, longArrayOf(1L, speechTokens.size.toLong()))

            val condInputs = mapOf(
                "speech_tokens" to speechTokensTensor,
                "speaker_embeddings" to refXVector,
                "speaker_features" to promptFeat
            )
            val condRes = conditionalDecoderModel.run(condInputs)
            val wav = try {
                extractFloatArray(condRes[0] as OnnxTensor)
            } finally {
                condRes.close()
            }
            return waveformSampler.fromFloatArray(wav, S3GEN_SR)
        } finally {
            for (v in embedInputs.values) v.close()
            promptTokenTensor?.close()
            refXVector?.close()
            promptFeat?.close()
            speechTokensTensor?.close()
            pastKeyValues?.values?.forEach { it.close() }
        }
    }

    private fun createInitialPastKeyValues(): LinkedHashMap<String, OnnxTensor> {
        val env = onnx.getEnvironment()
        val out = LinkedHashMap<String, OnnxTensor>()
        for (layer in 0 until NUM_HIDDEN_LAYERS) {
            for (kv in listOf("key", "value")) {
                val name = "past_key_values.$layer.$kv"
                out[name] = OnnxTensor.createTensor(
                    env,
                    FloatBuffer.wrap(FloatArray(0)),
                    longArrayOf(BATCH_SIZE, NUM_KEY_VALUE_HEADS, 0L, HEAD_DIM)
                )
            }
        }
        return out
    }

    private fun createInitialPositionIds(
        env: ai.onnxruntime.OrtEnvironment,
        inputIds: LongArray
    ): OnnxTensor {
        val ids = LongArray(inputIds.size)
        for (i in inputIds.indices) {
            ids[i] = if (inputIds[i] >= START_SPEECH_TOKEN) 0L else i.toLong() - 1L
        }
        return createLongTensor(env, ids, longArrayOf(1L, ids.size.toLong()))
    }

    private fun appendOne(src: LongArray): LongArray {
        val out = LongArray(src.size + 1)
        System.arraycopy(src, 0, out, 0, src.size)
        out[out.size - 1] = 1L
        return out
    }

    private fun createLongTensor(
        env: ai.onnxruntime.OrtEnvironment,
        data: LongArray,
        shape: LongArray
    ): OnnxTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(data), shape)

    private fun cloneTensor(tensor: OnnxTensor): OnnxTensor {
        val env = onnx.getEnvironment()
        val shape = tensor.info.shape
        val info = tensor.info as TensorInfo
        return when (info.type) {
            OnnxJavaType.INT64 -> {
                val arr = extractLongArray(tensor)
                OnnxTensor.createTensor(env, LongBuffer.wrap(arr), shape)
            }

            else -> {
                val arr = extractFloatArray(tensor)
                OnnxTensor.createTensor(env, FloatBuffer.wrap(arr), shape)
            }
        }
    }

    private fun extractLastLogits(logitsTensor: OnnxTensor): FloatArray {
        val shape = logitsTensor.info.shape
        val vocabSize = shape.last().toInt()
        val values = extractFloatArray(logitsTensor)
        return values.copyOfRange(values.size - vocabSize, values.size)
    }

    private fun extractFloatArray(tensor: OnnxTensor): FloatArray {
        return when (val v = tensor.value) {
            is FloatArray -> v
            is FloatBuffer -> {
                v.rewind()
                val out = FloatArray(v.remaining())
                v.get(out)
                out
            }

            is Array<*> -> {
                val out = ArrayList<Float>()
                flattenToFloat(v, out)
                out.toFloatArray()
            }

            else -> FloatArray(0)
        }
    }

    private fun extractLongArray(tensor: OnnxTensor): LongArray {
        return when (val v = tensor.value) {
            is LongArray -> v
            is LongBuffer -> {
                v.rewind()
                val out = LongArray(v.remaining())
                v.get(out)
                out
            }

            is Array<*> -> {
                val out = ArrayList<Long>()
                flattenToLong(v, out)
                out.toLongArray()
            }

            else -> LongArray(0)
        }
    }

    private fun flattenToFloat(value: Any?, out: MutableList<Float>) {
        when (value) {
            is Number -> out.add(value.toFloat())
            is FloatArray -> for (x in value) out.add(x)
            is DoubleArray -> for (x in value) out.add(x.toFloat())
            is IntArray -> for (x in value) out.add(x.toFloat())
            is LongArray -> for (x in value) out.add(x.toFloat())
            is Array<*> -> for (x in value) flattenToFloat(x, out)
        }
    }

    private fun flattenToLong(value: Any?, out: MutableList<Long>) {
        when (value) {
            is Number -> out.add(value.toLong())
            is LongArray -> for (x in value) out.add(x)
            is IntArray -> for (x in value) out.add(x.toLong())
            is Array<*> -> for (x in value) flattenToLong(x, out)
        }
    }

    private fun concatenateEmbeds(
        env: ai.onnxruntime.OrtEnvironment,
        left: OnnxTensor,
        right: OnnxTensor
    ): OnnxTensor {
        val a = extractFloatArray(left)
        val b = extractFloatArray(right)
        val hidden = left.info.shape[2].toInt()
        val leftSeq = left.info.shape[1].toInt()
        val rightSeq = right.info.shape[1].toInt()
        val out = FloatArray(a.size + b.size)
        System.arraycopy(a, 0, out, 0, a.size)
        System.arraycopy(b, 0, out, a.size, b.size)
        return OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(out),
            longArrayOf(1L, (leftSeq + rightSeq).toLong(), hidden.toLong())
        )
    }

    private fun concatLongArrays(a: LongArray, b: LongArray): LongArray {
        val out = LongArray(a.size + b.size)
        System.arraycopy(a, 0, out, 0, a.size)
        System.arraycopy(b, 0, out, a.size, b.size)
        return out
    }

    private fun applyRepetitionPenaltyAndArgmax(
        generated: List<Long>,
        logits: FloatArray
    ): Long {
        val adjusted = logits.copyOf()
        for (token in generated) {
            val idx = token.toInt()
            if (idx < 0 || idx >= adjusted.size) continue
            val score = logits[idx]
            adjusted[idx] =
                if (score < 0f) score * REPETITION_PENALTY else score / REPETITION_PENALTY
        }
        var maxIdx = 0
        var maxVal = Float.NEGATIVE_INFINITY
        for (i in adjusted.indices) {
            if (adjusted[i] > maxVal) {
                maxVal = adjusted[i]
                maxIdx = i
            }
        }
        return maxIdx.toLong()
    }

    private fun loadVoiceAudio(): FloatArray {
        val speechPath = Paths.get(modelsPath).resolve(voice).toAbsolutePath().toString()
        val voiceFile = File(speechPath)
        if (!voiceFile.exists()) throw IllegalArgumentException("Voice file not found: $speechPath")
        return readAndResampleAudioToMono(voiceFile)
    }

    private fun readAndResampleAudioToMono(file: File): FloatArray {
        val ais0: AudioInputStream = AudioSystem.getAudioInputStream(file)
        val baseFormat = ais0.format
        val pcmFormat = AudioFormat(
            AudioFormat.Encoding.PCM_SIGNED,
            baseFormat.sampleRate,
            16,
            baseFormat.channels,
            baseFormat.channels * 2,
            baseFormat.sampleRate,
            false
        )
        val ais: AudioInputStream = AudioSystem.getAudioInputStream(pcmFormat, ais0)
        val bytes = ais.readAllBytes()
        ais.close()

        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val shortBuf = bb.asShortBuffer()
        val totalSamples = shortBuf.remaining()
        val channels = pcmFormat.channels
        val frames = if (channels > 0) totalSamples / channels else 0

        val tmp = ShortArray(totalSamples)
        shortBuf.get(tmp)
        val mono = FloatArray(frames)

        var idx = 0
        for (f in 0 until frames) {
            var sum = 0f
            for (c in 0 until channels) {
                sum += tmp[idx++].toInt() / 32768.0f
            }
            mono[f] = sum / channels
        }

        val srcRate = pcmFormat.sampleRate.toInt()
        if (srcRate == S3GEN_SR) return mono

        val newLen = (mono.size.toLong() * S3GEN_SR / srcRate).toInt()
        if (newLen <= 0) return FloatArray(0)

        val out = FloatArray(newLen)
        for (i in 0 until newLen) {
            val pos = i.toDouble() * mono.size / newLen
            val i0 = pos.toInt().coerceIn(0, mono.size - 1)
            val i1 = (i0 + 1).coerceIn(0, mono.size - 1)
            val t = pos - i0
            out[i] = (1.0 - t).toFloat() * mono[i0] + t.toFloat() * mono[i1]
        }
        return out
    }

    companion object {
        private const val S3GEN_SR = 24000
        private const val START_SPEECH_TOKEN = 6561L
        private const val STOP_SPEECH_TOKEN = 6562L
        private const val REPETITION_PENALTY = 1.2f
        private const val BATCH_SIZE = 1L
        private const val NUM_HIDDEN_LAYERS = 30
        private const val NUM_KEY_VALUE_HEADS = 16L
        private const val HEAD_DIM = 64L
    }
}
