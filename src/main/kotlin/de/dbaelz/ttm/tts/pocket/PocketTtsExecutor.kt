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

        val frameSize = 32
        val framesAfterEos = 3
        val maxFrames = 500
        val decodeChunkSize = 12

        // Prepare tensors used repeatedly
        val emptySeq = createEmptyFloatTensor(env, longArrayOf(1L, 0L, frameSize.toLong()))
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
                mapOf("sequence" to emptySeq, "text_embeddings" to voiceTensor)
            )
            var res = lmMainModel.run(voiceInputs)
            updateStateFromResults(flowState, res, lmMainModel)
            res.close()

            // Text conditioning
            val tokenTensor = createLongTensor(env, tokenIds)
            val textRes = textConditionerModel.run(mapOf("token_ids" to tokenTensor))
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
                mapOf("sequence" to emptySeq, "text_embeddings" to textEmbTensor)
            )
            res = lmMainModel.run(mainInputs)
            updateStateFromResults(flowState, res, lmMainModel)
            res.close()
            textEmbTensor.close()

            // Autoregressive generation
            var currLatent = createNaNTensor(env, longArrayOf(1L, 1L, frameSize.toLong()))
            val generatedLatents = ArrayList<FloatArray>()
            var eosStep: Int? = null

            for (step in 0 until maxFrames) {
                val arInputs = buildInputsForSession(
                    lmMainModel,
                    flowState,
                    mapOf("sequence" to currLatent, "text_embeddings" to emptyText)
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
                if (eosStep != null && step >= eosStep + framesAfterEos) break

                // Flow matching
                val rng = if (config.seed > 0) Random(config.seed.toLong()) else Random()
                val std =
                    if (config.temperature > 0) sqrt(config.temperature.toDouble()).toFloat() else 0f
                val x = if (std > 0f) {
                    FloatArray(32) { (rng.nextGaussian() * std).toFloat() }
                } else {
                    FloatArray(32) { 0f }
                }

                val dt = 1.0f / config.diffusionSteps
                for (j in 0 until config.diffusionSteps) {
                    val s = floatArrayOf(j.toFloat() / config.diffusionSteps)
                    val t = floatArrayOf((j + 1).toFloat() / config.diffusionSteps)
                    val cTensor = createFloatTensorFrom1D(env, conditioning)
                    val sTensor = createFloatTensorFrom1D(env, s, longArrayOf(1L, 1L))
                    val tTensor = createFloatTensorFrom1D(env, t, longArrayOf(1L, 1L))
                    val xTensor =
                        createFloatTensorFrom1D(env, x, longArrayOf(1L, frameSize.toLong()))

                    val flowInputs = HashMap<String, OnnxTensor>()
                    flowInputs["c"] = cTensor
                    flowInputs["s"] = sTensor
                    flowInputs["t"] = tTensor
                    flowInputs["x"] = xTensor

                    val flowRes = lmFlowModel.run(flowInputs)
                    val vTensor = flowRes[0] as OnnxTensor
                    val v = extractFloatArrayFromTensor1D(vTensor)

                    for (k in x.indices) x[k] += v[k] * dt

                    flowRes.close()
                    cTensor.close()
                    sTensor.close()
                    tTensor.close()
                    xTensor.close()
                }

                // Add latent
                generatedLatents.add(x.copyOf())

                // prepare currLatent for next AR step
                currLatent.close()
                currLatent = createFloatTensorFrom2DForLatent(env, x)
            }

            // Decode latents in chunks
            val audioFloatsList = ArrayList<Float>()
            var idx = 0
            while (idx < generatedLatents.size) {
                val end = min(idx + decodeChunkSize, generatedLatents.size)
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
                val audioOut = extractFloatArrayFromTensor1D(audioTensor)
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
            for (v in flowState.values) v.close()
            for (v in decoderState.values) v.close()
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
            java.nio.LongBuffer.wrap(arr),
            longArrayOf(1L, tokens.size.toLong())
        )
    }

    private fun createBoolTensor(shape: LongArray = longArrayOf(1L)): OnnxTensor {
        val b = ByteArray(shape.fold(1L) { acc, v -> if (v <= 0L) acc else acc * v }.toInt())
        if (b.isNotEmpty()) b[0] = 0
        val bb = ByteBuffer.wrap(b)
        return OnnxTensor.createTensor(onnx.getEnvironment(), bb, shape, OnnxJavaType.BOOL)
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

    private fun extractFloatArrayFromTensor1D(tensor: OnnxTensor): FloatArray {
        return extractFloatArrayFromTensor(tensor)
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
            if (!name.startsWith("state_")) continue
            val infoAny = node.info
            val shape = try {
                (infoAny as? ai.onnxruntime.TensorInfo)?.shape ?: longArrayOf(0L)
            } catch (_: Exception) {
                longArrayOf(0L)
            }
            val infoStr = infoAny.toString().lowercase()
            val tensor = when {
                infoStr.contains("bool") -> createBoolTensor(shape)
                infoStr.contains("int64") -> createEmptyLongTensor(env, shape)
                else -> createEmptyFloatTensor(env, shape)
            }
            state[name] = tensor
        }

        return state
    }

    private fun createEmptyLongTensor(
        env: ai.onnxruntime.OrtEnvironment,
        shape: LongArray
    ): OnnxTensor {
        val total = shape.fold(1L) { acc, v -> if (v <= 0L) 0L else acc * v }
        val buf = LongArray(total.toInt())
        return OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(buf), shape)
    }

    private fun updateStateFromResults(
        state: MutableMap<String, OnnxTensor>,
        res: OrtSession.Result,
        session: OrtSession
    ) {
        val outputs = session.outputInfo.keys.toList()
        for (i in outputs.indices) {
            val name = outputs[i]
            if (!name.startsWith("out_state_")) continue
            val stateKey = "state_${name.removePrefix("out_state_")}"
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
                        java.nio.LongBuffer.wrap(v),
                        shape
                    )

                    is java.nio.LongBuffer -> {
                        v.rewind()
                        val a = LongArray(v.remaining()); v.get(a); OnnxTensor.createTensor(
                            onnx.getEnvironment(),
                            java.nio.LongBuffer.wrap(a),
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

        if (declaredInfo.contains("bool") && !(tensor.value is ByteBuffer || tensor.value is ByteArray)) {
            val shape = tensor.info.shape
            return when (val v = tensor.value) {
                is LongArray -> {
                    val b = ByteArray(v.size); for (i in v.indices) b[i] =
                        if (v[i] != 0L) 1 else 0
                    OnnxTensor.createTensor(
                        env,
                        ByteBuffer.wrap(b),
                        shape,
                        OnnxJavaType.BOOL
                    )
                }

                is java.nio.LongBuffer -> {
                    v.rewind()
                    val a = LongArray(v.remaining()); v.get(a)
                    val b = ByteArray(a.size); for (i in a.indices) b[i] =
                        if (a[i] != 0L) 1 else 0
                    OnnxTensor.createTensor(
                        env,
                        ByteBuffer.wrap(b),
                        shape,
                        OnnxJavaType.BOOL
                    )
                }

                else -> tensor
            }
        }
        if (declaredInfo.contains("int64") && !(tensor.value is java.nio.LongBuffer || tensor.value is LongArray)) {
            val shape = tensor.info.shape
            return when (val v = tensor.value) {
                is ByteBuffer -> {
                    v.rewind()
                    val n = v.remaining()
                    val a = LongArray(n); for (i in 0 until n) a[i] =
                        if (v.get(i) != 0.toByte()) 1L else 0L
                    OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(a), shape)
                }

                is ByteArray -> {
                    val a = LongArray(v.size); for (i in v.indices) a[i] =
                        if (v[i] != 0.toByte()) 1L else 0L
                    OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(a), shape)
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
                        val inner = first
                        val n = inner.size
                        when (val d0 = inner[0]) {
                            is FloatArray -> {
                                val dim = d0.size
                                val resultArr = Array(n) { FloatArray(dim) }
                                for (i in 0 until n) {
                                    val row = inner[i] as FloatArray
                                    System.arraycopy(row, 0, resultArr[i], 0, dim)
                                }
                                resultArr
                            }

                            is Array<*> -> {
                                Array(0) { FloatArray(0) }
                            }

                            else -> {
                                Array(0) { FloatArray(0) }
                            }
                        }
                    }

                    is FloatArray -> {
                        val row = first
                        arrayOf(row)
                    }

                    else -> {
                        Array(0) { FloatArray(0) }
                    }
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
}
