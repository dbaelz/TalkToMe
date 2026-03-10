package de.dbaelz.ttm.tts.pocket

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import de.dbaelz.ttm.audio.WaveformSampler
import de.dbaelz.ttm.model.TtsEngine
import de.dbaelz.ttm.model.TtsJob
import de.dbaelz.ttm.onnx.OnnxWrapper
import de.dbaelz.ttm.tts.TtsConfig
import de.dbaelz.ttm.tts.TtsExecutor
import org.springframework.beans.factory.annotation.Value
import org.springframework.stereotype.Service
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.LongBuffer
import java.nio.file.Paths
import java.util.*
import kotlin.io.path.absolutePathString
import kotlin.math.min
import kotlin.math.sqrt

@Service
class PocketTtsExecutor(
    @Value($$"${tts.models.pocket-tts:models/pocket-tts}") private val modelsPath: String,
    @Value($$"${tts.models.pocket-tts.encoder:mimi_encoder.onnx}") private val encoder: String,
    @Value($$"${tts.models.pocket-tts.text_conditioner:text_conditioner.onnx}") private val textConditioner: String,
    @Value($$"${tts.models.pocket-tts.flow_lm_main:flow_lm_main_int8.onnx}") private val lmMain: String,
    @Value($$"${tts.models.pocket-tts.flow_lm_flow:flow_lm_flow_int8.onnx}") private val lmFlow: String,
    @Value($$"${tts.models.pocket-tts.decoder:mimi_decoder_int8.onnx}") private val decoder: String,
    @Value($$"${tts.models.pocket-tts.voice:voice.wav}") private val voice: String,
    private val tokenizer: SentencePieceTokenizer,
    private val onnx: OnnxWrapper,
    private val waveformSampler: WaveformSampler
) : TtsExecutor {
    override fun invoke(job: TtsJob): ByteArray {
        onnx.loadModelFiles(
            engine = TtsEngine.POCKET,
            modelsPath = modelsPath,
            modelFiles = listOf(encoder, textConditioner, lmMain, lmFlow, decoder)
        )

        return generateAudio(job.text, job.config)
    }

    private fun generateAudio(text: String, config: TtsConfig): ByteArray {
        val textConditionerModel = onnx.getModel(TtsEngine.POCKET, textConditioner)
        val lmMainModel = onnx.getModel(TtsEngine.POCKET, lmMain)
        val lmFlowModel = onnx.getModel(TtsEngine.POCKET, lmFlow)
        val decoderModel = onnx.getModel(TtsEngine.POCKET, decoder)
        val encoderModel = onnx.getModel(TtsEngine.POCKET, encoder)

        val voiceEmbeddings = computeVoiceEmbeddings(encoderModel)
        if (voiceEmbeddings.isEmpty()) throw IllegalArgumentException("Voice embedding produced no data")

        val tokenIds = tokenizer.tokenize(text)
        if (tokenIds.isEmpty()) throw IllegalArgumentException("Text tokenization produced no tokens")

        val env = onnx.getEnvironment()

        // Prepare tensors used repeatedly
        val emptySeq = createEmptyFloatTensor(env, longArrayOf(1L, 0L, FRAME_SIZE))
        val emptyText = createEmptyFloatTensor(env, longArrayOf(1L, 0L, 1024L))

        val numEmbeddings = voiceEmbeddings.size / VOICE_EMBEDDING_SIZE
        val embeddingSize =
            if (voiceEmbeddings.isNotEmpty()) voiceEmbeddings.size / numEmbeddings else 0
        val voiceTensor = createFloatTensorFrom1D(
            env = env,
            flatData = voiceEmbeddings,
            numEmbeddings = numEmbeddings,
            embeddingSize = embeddingSize
        )

        // Init states (create according to model's declared inputs)
        val flowState = initStateFromSession(lmMainModel)
        val decoderState = initStateFromSession(decoderModel)

        try {
            // Voice conditioning
            val voiceInputs = buildInputsForSession(
                lmMainModel,
                flowState,
                mapOf(KEY_SEQUENCE to emptySeq, KEY_TEXT_EMBEDDINGS to voiceTensor)
            )
            var res = lmMainModel.run(voiceInputs)
            updateStateFromResults(flowState, res, lmMainModel)
            res.close()

            // Text conditioning
            val tokenTensor = createLongTensor(env, tokenIds)
            val textRes = textConditionerModel.run(mapOf(KEY_TOKEN_IDS to tokenTensor))
            val textEmb = textRes[0] as OnnxTensor

            // Ensure shape is [1, 1, 1024] or [1, N, 1024]
            // If necessary, wrap/reshape by recreating tensor with same data
            val textEmbTensor = normalizeTextEmb(env, textEmb)
            textRes.close()
            tokenTensor.close()

            // Run flow main with empty seq + text embeddings to update state
            val mainInputs = buildInputsForSession(
                lmMainModel,
                flowState,
                mapOf(KEY_SEQUENCE to emptySeq, KEY_TEXT_EMBEDDINGS to textEmbTensor)
            )
            res = lmMainModel.run(mainInputs)
            updateStateFromResults(flowState, res, lmMainModel)
            res.close()
            textEmbTensor.close()

            // Autoregressive generation
            var currLatent = createNaNTensor(env, longArrayOf(1L, 1L, FRAME_SIZE))
            val generatedLatents = ArrayList<FloatArray>()
            var eosStep: Int? = null

            // Seed and std for latent noise
            val rng = if (config.seed > 0) Random(config.seed.toLong()) else Random()
            val randomStd = if (config.temperature > 0) {
                sqrt(config.temperature.toDouble()).toFloat()
            } else 0f

            for (step in 0 until MAX_FRAMES) {
                val arInputs = buildInputsForSession(
                    lmMainModel,
                    flowState,
                    mapOf(KEY_SEQUENCE to currLatent, KEY_TEXT_EMBEDDINGS to emptyText)
                )
                val arRes = lmMainModel.run(arInputs)

                val conditioningTensor = arRes[0] as OnnxTensor
                val eosTensor = arRes[1] as OnnxTensor

                val conditioning = extractFloatArrayFromTensor(conditioningTensor)
                val eosVal = extractScalarFloat(eosTensor)

                updateStateFromResults(flowState, arRes, lmMainModel)
                arRes.close()

                if (eosStep == null && eosVal > -4.0f) eosStep = step
                if (eosStep != null && step >= eosStep + FRAMES_AFTER_EOS) break

                // Calculate latent noise
                val x = if (randomStd > 0f) {
                    FloatArray(32) { (rng.nextGaussian() * randomStd).toFloat() }
                } else {
                    FloatArray(32) { 0f }
                }

                val dt = 1.0f / config.steps
                for (j in 0 until config.steps) {
                    val s = floatArrayOf(j.toFloat() / config.steps)
                    val t = floatArrayOf((j + 1).toFloat() / config.steps)
                    val conditioningTensor = createFloatTensorFrom1D(env, conditioning)
                    val simpleStepTensor = createFloatTensorFrom1D(env, s, longArrayOf(1L, 1L))
                    val nextStepTensor = createFloatTensorFrom1D(env, t, longArrayOf(1L, 1L))
                    val latentTensor = createFloatTensorFrom1D(env, x, longArrayOf(1L, FRAME_SIZE))

                    val flowInputs = HashMap<String, OnnxTensor>()
                    flowInputs[KEY_CONDITIONING_TENSOR] = conditioningTensor
                    flowInputs[KEY_SIMPLE_STEP_TENSOR] = simpleStepTensor
                    flowInputs[KEY_NEXT_STEP_TENSOR] = nextStepTensor
                    flowInputs[KEY_LATENT_TENSOR] = latentTensor

                    val flowRes = lmFlowModel.run(flowInputs)
                    val vTensor = flowRes[0] as OnnxTensor
                    val v = extractFloatArrayFromTensor(vTensor)

                    for (k in x.indices) x[k] += v[k] * dt

                    flowRes.close()
                    conditioningTensor.close()
                    simpleStepTensor.close()
                    nextStepTensor.close()
                    latentTensor.close()
                }

                generatedLatents.add(x.copyOf())

                currLatent.close()
                currLatent = createFloatTensorFrom2DForLatent(env, x)
            }

            // Decode latents in chunks
            var audioBuf = FloatArray(kotlin.math.max(1024, generatedLatents.size * 256))
            var audioPos = 0
            var idx = 0
            while (idx < generatedLatents.size) {
                val end = min(idx + DECODE_CHUNK_SIZE, generatedLatents.size)
                val chunk = generatedLatents.subList(idx, end)
                val latentTensor = createLatentsTensor(env, chunk)

                val decoderInputs =
                    buildInputsForSession(
                        decoderModel,
                        decoderState,
                        mapOf("latent" to latentTensor)
                    )
                val decRes = decoderModel.run(decoderInputs)

                val audioTensor = decRes[0] as OnnxTensor
                val audioOut = extractFloatArrayFromTensor(audioTensor)
                val outLen = audioOut.size
                if (audioPos + outLen > audioBuf.size) {
                    var newCap = kotlin.math.max(audioBuf.size * 2, audioPos + outLen)
                    val newBuf = FloatArray(newCap)
                    System.arraycopy(audioBuf, 0, newBuf, 0, audioPos)
                    audioBuf = newBuf
                }
                System.arraycopy(audioOut, 0, audioBuf, audioPos, outLen)
                audioPos += outLen

                updateStateFromResults(decoderState, decRes, decoderModel)

                decRes.close()
                latentTensor.close()
                idx = end
            }
            val finalAudio = if (audioPos == audioBuf.size) audioBuf else audioBuf.copyOf(audioPos)

            return waveformSampler.fromFloatArray(finalAudio, 24000)

        } finally {
            emptySeq.close()
            emptyText.close()
            voiceTensor.close()

            // close all state tensors
            for (tensor in flowState.values) tensor.close()
            for (tensor in decoderState.values) tensor.close()
        }
    }

    private fun createLatentsTensor(
        env: ai.onnxruntime.OrtEnvironment,
        chunk: List<FloatArray>
    ): OnnxTensor {
        val frames = chunk.size
        val flat = FloatArray(frames * 32)
        for (i in 0 until frames) System.arraycopy(chunk[i], 0, flat, i * 32, 32)
        return OnnxTensor.createTensor(
            env,
            java.nio.FloatBuffer.wrap(flat),
            longArrayOf(1L, frames.toLong(), 32L)
        )
    }

    private fun createFloatTensorFrom1D(
        env: ai.onnxruntime.OrtEnvironment,
        flatData: FloatArray,
        numEmbeddings: Int,
        embeddingSize: Int
    ): OnnxTensor {
        return OnnxTensor.createTensor(
            env,
            java.nio.FloatBuffer.wrap(flatData),
            longArrayOf(1L, numEmbeddings.toLong(), embeddingSize.toLong())
        )
    }

    private fun createFloatTensorFrom2DForLatent(
        env: ai.onnxruntime.OrtEnvironment,
        data: FloatArray
    ): OnnxTensor {
        return OnnxTensor.createTensor(
            env,
            java.nio.FloatBuffer.wrap(data),
            longArrayOf(1L, 1L, data.size.toLong())
        )
    }

    private fun createFloatTensorFrom1D(
        env: ai.onnxruntime.OrtEnvironment,
        data: FloatArray,
        shape: LongArray? = null
    ): OnnxTensor {
        val s = shape ?: longArrayOf(1L, data.size.toLong())
        return OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(data), s)
    }

    private fun createEmptyFloatTensor(
        env: ai.onnxruntime.OrtEnvironment,
        shape: LongArray
    ): OnnxTensor {
        val total = shape.fold(1L) { acc, v -> if (v <= 0L) 0L else acc * v }
        val buf = FloatArray(total.toInt())
        return OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(buf), shape)
    }

    private fun createNaNTensor(env: ai.onnxruntime.OrtEnvironment, shape: LongArray): OnnxTensor {
        val total = shape.fold(1L) { acc, v -> if (v <= 0L) 0L else acc * v }
        val buf = FloatArray(total.toInt()) { Float.NaN }
        return OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(buf), shape)
    }

    private fun createLongTensor(
        env: ai.onnxruntime.OrtEnvironment,
        tokens: List<Int>
    ): OnnxTensor {
        val arr = LongArray(tokens.size)
        for (i in tokens.indices) arr[i] = tokens[i].toLong()
        return OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(arr),
            longArrayOf(1L, tokens.size.toLong())
        )
    }

    private fun createEmptyBoolTensor(): OnnxTensor {
        return OnnxTensor.createTensor(
            onnx.getEnvironment(),
            EMPTY_BOOL_BUFFER.duplicate(),
            longArrayOf(1L),
            OnnxJavaType.BOOL
        )
    }

    private fun createEmptyLongTensor(
        env: ai.onnxruntime.OrtEnvironment,
        shape: LongArray
    ): OnnxTensor {
        val total = shape.fold(1L) { acc, v -> if (v <= 0L) 0L else acc * v }
        val buf = LongArray(total.toInt())
        return OnnxTensor.createTensor(env, LongBuffer.wrap(buf), shape)
    }

    private fun extractScalarFloat(tensor: OnnxTensor): Float {
        return when (val v = tensor.value) {
            is FloatArray -> v[0]
            is java.nio.FloatBuffer -> {
                v.rewind(); v.get()
            }

            is Array<*> -> {
                when (val first = v[0]) {
                    is FloatArray -> first[0]
                    is Array<*> -> {
                        val inner = first[0]
                        inner as? Float ?: (inner as Number).toFloat()
                    }

                    else -> (first as Number).toFloat()
                }
            }

            else -> 0f
        }
    }

    private fun extractFloatArrayFromTensor(tensor: OnnxTensor): FloatArray {
        return when (val v = tensor.value) {
            is java.nio.FloatBuffer -> {
                v.rewind()
                val out = FloatArray(v.remaining())
                v.get(out)
                out
            }

            is FloatArray -> v

            is Array<*> -> {
                fun countElems(input: Any?): Int {
                    return when (input) {
                        is Float -> 1
                        is Number -> 1
                        is FloatArray -> input.size
                        is Array<*> -> {
                            var sum = 0
                            for (it in input) sum += countElems(it)
                            sum
                        }

                        else -> 0
                    }
                }

                val total = countElems(v)
                if (total == 0) return FloatArray(0)

                val out = FloatArray(total)
                val idx = intArrayOf(0)

                fun fill(input: Any?) {
                    when (input) {
                        is Float -> out[idx[0]++] = input
                        is Number -> out[idx[0]++] = input.toFloat()
                        is FloatArray -> {
                            System.arraycopy(input, 0, out, idx[0], input.size)
                            idx[0] += input.size
                        }

                        is Array<*> -> input.forEach { fill(it) }
                    }
                }

                v.forEach { fill(it) }
                out
            }

            else -> FloatArray(0)
        }
    }

    private fun normalizeTextEmb(
        env: ai.onnxruntime.OrtEnvironment,
        tensor: OnnxTensor
    ): OnnxTensor {
        // If tensor is FloatBuffer or FloatArray or nested array, create a 3D tensor if needed
        val flat = extractFloatArrayFromTensor(tensor)
        val shape = tensor.info.shape

        // Determine dims: if shape length is 2 -> [1, dim] -> convert to [1,1,dim]
        return if (shape.size == 2) {
            val dim = shape[1].toInt()
            OnnxTensor.createTensor(
                env,
                java.nio.FloatBuffer.wrap(flat),
                longArrayOf(1L, 1L, dim.toLong())
            )
        } else {
            // reuse by creating new tensor with same shape and data
            val newShape = shape
            OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(flat), newShape)
        }
    }

    private fun initStateFromSession(session: OrtSession): MutableMap<String, OnnxTensor> {
        val state = mutableMapOf<String, OnnxTensor>()
        val env = onnx.getEnvironment()

        for ((name, node) in session.inputInfo) {
            if (!name.startsWith(STATE_PREFIX)) continue
            val infoAny = node.info
            val shape = (infoAny as? ai.onnxruntime.TensorInfo)?.shape ?: longArrayOf(0L)
            val infoStr = infoAny.toString().lowercase()

            val tensor = when {
                infoStr.contains(TYPE_BOOL) -> createEmptyBoolTensor()
                infoStr.contains(TYPE_INT64) -> createEmptyLongTensor(env, shape)
                else -> createEmptyFloatTensor(env, shape)
            }
            state[name] = tensor
        }

        return state
    }

    private fun updateStateFromResults(
        state: MutableMap<String, OnnxTensor>,
        res: OrtSession.Result,
        session: OrtSession
    ) {
        val outputs = session.outputInfo.keys.toList()
        for (i in outputs.indices) {
            val name = outputs[i]
            if (!name.startsWith(OUT_STATE_PREFIX)) continue
            val stateKey = "state_${name.removePrefix(OUT_STATE_PREFIX)}"
            val valObj = res[i]
            if (valObj is OnnxTensor) {
                val v = valObj.value
                val shape = valObj.info.shape
                // create tensor matching the result's value type
                val newTensor = when (v) {
                    is FloatArray -> OnnxTensor.createTensor(
                        onnx.getEnvironment(),
                        java.nio.FloatBuffer.wrap(v),
                        shape
                    )

                    is java.nio.FloatBuffer -> {
                        v.rewind()
                        val a = FloatArray(v.remaining()); v.get(a); OnnxTensor.createTensor(
                            onnx.getEnvironment(),
                            java.nio.FloatBuffer.wrap(a),
                            shape
                        )
                    }

                    is LongArray -> OnnxTensor.createTensor(
                        onnx.getEnvironment(),
                        LongBuffer.wrap(v),
                        shape
                    )

                    is LongBuffer -> {
                        v.rewind()
                        val a = LongArray(v.remaining()); v.get(a); OnnxTensor.createTensor(
                            onnx.getEnvironment(),
                            LongBuffer.wrap(a),
                            shape
                        )
                    }

                    is BooleanArray -> {
                        val a = ByteArray(v.size)
                        for (ii in v.indices) a[ii] = if (v[ii]) 1 else 0
                        val bb = ByteBuffer.wrap(a)
                        OnnxTensor.createTensor(onnx.getEnvironment(), bb, shape, OnnxJavaType.BOOL)
                    }

                    is Array<*> -> {
                        // try float nested
                        val flat = extractFloatArrayFromTensor(valObj)
                        OnnxTensor.createTensor(
                            onnx.getEnvironment(),
                            java.nio.FloatBuffer.wrap(flat),
                            shape
                        )
                    }

                    else -> {
                        // fallback to float
                        val flat = extractFloatArrayFromTensor(valObj)
                        OnnxTensor.createTensor(
                            onnx.getEnvironment(),
                            java.nio.FloatBuffer.wrap(flat),
                            shape
                        )
                    }
                }
                state[stateKey]?.close()
                state[stateKey] = newTensor
            }
        }
    }

    private fun buildInputsForSession(
        session: OrtSession,
        state: Map<String, OnnxTensor>,
        overrides: Map<String, OnnxTensor>
    ): MutableMap<String, OnnxTensor> {
        val inputs = LinkedHashMap<String, OnnxTensor>()

        for ((name, node) in session.inputInfo) {
            if (overrides.containsKey(name)) {
                inputs[name] =
                    convertTensorToDeclared(node.info.toString().lowercase(), overrides[name]!!)
                continue
            }

            if (state.containsKey(name)) {
                inputs[name] =
                    convertTensorToDeclared(node.info.toString().lowercase(), state[name]!!)
                continue
            }
        }

        return inputs
    }

    fun convertTensorToDeclared(declaredInfo: String, tensor: OnnxTensor): OnnxTensor {
        val env = onnx.getEnvironment()

        if (declaredInfo.contains(TYPE_BOOL) && !(tensor.value is ByteBuffer || tensor.value is ByteArray)) {
            val shape = tensor.info.shape
            return when (val v = tensor.value) {
                is LongArray -> {
                    val b = ByteArray(v.size) { i -> if (v[i] != 0L) 1 else 0 }
                    OnnxTensor.createTensor(env, ByteBuffer.wrap(b), shape, OnnxJavaType.BOOL)
                }

                is LongBuffer -> {
                    v.rewind()
                    val b = ByteArray(v.remaining()).also {
                        for (i in it.indices) it[i] = if (v.get() != 0L) 1 else 0
                    }
                    OnnxTensor.createTensor(env, ByteBuffer.wrap(b), shape, OnnxJavaType.BOOL)
                }

                else -> tensor
            }
        }
        if (declaredInfo.contains(TYPE_INT64) && !(tensor.value is LongBuffer || tensor.value is LongArray)) {
            val shape = tensor.info.shape
            return when (val v = tensor.value) {
                is ByteBuffer -> {
                    v.rewind()
                    val b = LongArray(v.remaining()).also {
                        for (i in it.indices) it[i] = if (v.get(i) != 0.toByte()) 1L else 0L
                    }
                    OnnxTensor.createTensor(env, LongBuffer.wrap(b), shape)
                }

                is ByteArray -> {
                    val a = LongArray(v.size).also {
                        for (i in it.indices) it[i] = if (v[i] != 0.toByte()) 1L else 0L
                    }
                    OnnxTensor.createTensor(env, LongBuffer.wrap(a), shape)
                }

                else -> tensor
            }
        }
        return tensor
    }

    private fun computeVoiceEmbeddings(encoder: OrtSession): FloatArray {
        val voicePath = Paths.get(modelsPath).resolve(voice).absolutePathString()
        val file = File(voicePath)

        if (!file.exists()) throw IllegalArgumentException("Voice file not found: $voicePath")

        val originalAis = javax.sound.sampled.AudioSystem.getAudioInputStream(file)
        val originalFormat = originalAis.format

        val targetFormat = javax.sound.sampled.AudioFormat(
            javax.sound.sampled.AudioFormat.Encoding.PCM_SIGNED,
            originalFormat.sampleRate,
            16,
            1,
            2,
            originalFormat.sampleRate,
            false
        )

        val pcmFloats: FloatArray = try {
            val convertedAis =
                javax.sound.sampled.AudioSystem.getAudioInputStream(targetFormat, originalAis)
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

        val env = onnx.getEnvironment()
        val shape = longArrayOf(1L, 1L, pcmFloats.size.toLong())
        val data = FloatArray(pcmFloats.size)
        System.arraycopy(pcmFloats, 0, data, 0, pcmFloats.size)

        val tensor = OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(data), shape)

        val inputs = mapOf(KEY_AUDIO to tensor)
        val results = encoder.run(inputs)

        val embeddings: FloatArray = when (val outVal = results[0].value) {
            is Array<*> -> {
                when (val first = outVal[0]) {
                    is Array<*> -> {
                        val inner = first
                        val n = inner.size
                        if (n == 0) return FloatArray(0)

                        FloatArray(n * VOICE_EMBEDDING_SIZE).apply {
                            for (i in inner.indices) {
                                System.arraycopy(inner[i], 0, this, i * 1024, 1024)
                            }
                        }
                    }

                    is FloatArray -> first
                    else -> FloatArray(0)
                }
            }

            is java.nio.FloatBuffer -> {
                val buf = outVal
                buf.rewind()
                val total = buf.remaining()
                if (total % VOICE_EMBEDDING_SIZE != 0) return FloatArray(0)

                FloatArray(total).also { array ->
                    buf.get(array)
                }
            }

            else -> FloatArray(0)
        }

        results.close()
        tensor.close()

        return embeddings
    }

    private companion object {
        const val KEY_AUDIO = "audio"
        const val KEY_SEQUENCE = "sequence"
        const val KEY_TOKEN_IDS = "token_ids"
        const val KEY_TEXT_EMBEDDINGS = "text_embeddings"
        const val KEY_CONDITIONING_TENSOR = "c"
        const val KEY_SIMPLE_STEP_TENSOR = "s"
        const val KEY_NEXT_STEP_TENSOR = "t"
        const val KEY_LATENT_TENSOR = "x"

        const val TYPE_BOOL = "bool"
        const val TYPE_INT64 = "int64"

        const val STATE_PREFIX = "state_"
        const val OUT_STATE_PREFIX = "out_state_"

        const val VOICE_EMBEDDING_SIZE = 1024
        const val FRAME_SIZE = 32L
        const val MAX_FRAMES = 500
        const val DECODE_CHUNK_SIZE = 15
        const val FRAMES_AFTER_EOS = 3

        private val EMPTY_BOOL_BUFFER = ByteBuffer.wrap(byteArrayOf(0)).asReadOnlyBuffer()
    }
}
