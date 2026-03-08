package de.dbaelz.ttm.tts.pocket

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import de.dbaelz.ttm.audio.WaveformSampler
import de.dbaelz.ttm.model.TtsJob
import de.dbaelz.ttm.model.TtsProvider
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
    @Value($$"${tts.models.pocket-tts}") private val modelsPath: String,
    @Value($$"${tts.models.pocket-tts.encoder}") private val encoder: String,
    @Value($$"${tts.models.pocket-tts.text_conditioner}") private val textConditioner: String,
    @Value($$"${tts.models.pocket-tts.flow_lm_main}") private val lmMain: String,
    @Value($$"${tts.models.pocket-tts.flow_lm_flow}") private val lmFlow: String,
    @Value($$"${tts.models.pocket-tts.decoder}") private val decoder: String,
    @Value($$"${tts.models.pocket-tts.voice}") private val voice: String,
    private val tokenizer: SentencePieceTokenizer,
    private val onnx: OnnxWrapper,
    private val waveformSampler: WaveformSampler
) : TtsExecutor {
    override fun invoke(job: TtsJob): ByteArray {
        onnx.loadModelFiles(
            provider = TtsProvider.POCKET,
            modelsPath = modelsPath,
            modelFiles = listOf(encoder, textConditioner, lmMain, lmFlow, decoder)
        )

        return generateAudio(job.text, job.config)
    }

    private fun generateAudio(text: String, config: TtsConfig): ByteArray {
        val textConditionerModel = onnx.getModel(TtsProvider.POCKET, textConditioner)
        val lmMainModel = onnx.getModel(TtsProvider.POCKET, lmMain)
        val lmFlowModel = onnx.getModel(TtsProvider.POCKET, lmFlow)
        val decoderModel = onnx.getModel(TtsProvider.POCKET, decoder)
        val encoderModel = onnx.getModel(TtsProvider.POCKET, encoder)

        val voiceEmbeddings = computeVoiceEmbeddings(encoderModel)
        if (voiceEmbeddings.isEmpty()) throw IllegalArgumentException("Voice embedding produced no data")

        val tokenIds = tokenizer.tokenize(text)
        if (tokenIds.isEmpty()) throw IllegalArgumentException("Text tokenization produced no tokens")

        val env = onnx.getEnvironment()

        // Prepare tensors used repeatedly
        val emptySeq = createEmptyFloatTensor(env, longArrayOf(1L, 0L, FRAME_SIZE))
        val emptyText = createEmptyFloatTensor(env, longArrayOf(1L, 0L, 1024L))
        val voiceTensor = createFloatTensorFrom2D(env, voiceEmbeddings)

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

                val conditioning =
                    extractFloatArrayFromTensor(conditioningTensor) // shape [1,1,condDim] or [1,condDim]
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
            val audioFloatsList = ArrayList<Float>()
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

                // audio out
                val audioTensor = decRes[0] as OnnxTensor
                val audioOut = extractFloatArrayFromTensor(audioTensor)
                audioFloatsList.addAll(audioOut.toList())

                updateStateFromResults(decoderState, decRes, decoderModel)

                decRes.close()
                latentTensor.close()
                idx = end
            }

            return waveformSampler.fromFloatArray(audioFloatsList, 24000)
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

    private fun createFloatTensorFrom2D(
        env: ai.onnxruntime.OrtEnvironment,
        data: Array<FloatArray>
    ): OnnxTensor {
        val n = data.size
        val d = if (n > 0) data[0].size else 0
        val flat = FloatArray(n * d)
        for (i in 0 until n) System.arraycopy(data[i], 0, flat, i * d, d)
        return OnnxTensor.createTensor(
            env,
            java.nio.FloatBuffer.wrap(flat),
            longArrayOf(1L, n.toLong(), d.toLong())
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
        when (val v = tensor.value) {
            is java.nio.FloatBuffer -> {
                v.rewind()
                val a = FloatArray(v.remaining()); v.get(a); return a
            }

            is Array<*> -> {
                // try to flatten nested arrays to 1D
                val list = ArrayList<Float>()
                fun recurse(o: Any?) {
                    when (o) {
                        is Float -> list.add(o)
                        is Number -> list.add(o.toFloat())
                        is FloatArray -> for (x in o) list.add(x)
                        is Array<*> -> for (x in o) recurse(x)
                    }
                }
                recurse(v)
                return list.toFloatArray()
            }

            is FloatArray -> return v
            else -> return FloatArray(0)
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

    private fun computeVoiceEmbeddings(encoder: OrtSession): Array<FloatArray> {
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

        val inputs = mapOf("audio" to tensor)
        val results = encoder.run(inputs)

        val embeddings: Array<FloatArray> = when (val outVal = results[0].value) {
            is Array<*> -> {
                when (val first = outVal[0]) {
                    is Array<*> -> {
                        val inner = first as Array<FloatArray>
                        val n = inner.size
                        if (n == 0) return Array(0) { FloatArray(0) }

                        Array(n) { inner[it].copyOf() } // Use copyOf for array duplication
                    }
                    is FloatArray -> arrayOf(first)
                    else -> Array(0) { FloatArray(0) }
                }
            }
            is java.nio.FloatBuffer -> {
                val buf = outVal
                buf.rewind()
                val total = buf.remaining()
                val dims = total / 1024
                if (total % 1024 != 0) return Array(0) { FloatArray(0) }

                Array(dims) { FloatArray(1024).also { buf.get(it) } }
            }
            else -> Array(0) { FloatArray(0) }
        }

        results.close()
        tensor.close()

        return embeddings
    }

    private companion object {
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

        const val FRAME_SIZE = 32L
        const val MAX_FRAMES = 500
        const val DECODE_CHUNK_SIZE = 12
        const val FRAMES_AFTER_EOS = 3

        private val EMPTY_BOOL_BUFFER = ByteBuffer.wrap(byteArrayOf(0)).asReadOnlyBuffer()
    }
}
