package de.dbaelz.ttm.tts.pocket

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import de.dbaelz.ttm.audio.WaveformSampler
import de.dbaelz.ttm.model.JobStatus
import de.dbaelz.ttm.model.TtsJob
import de.dbaelz.ttm.onnx.OnnxWrapper
import de.dbaelz.ttm.repository.JobRepository
import de.dbaelz.ttm.service.LocalStorageService
import de.dbaelz.ttm.service.TtsService
import org.springframework.beans.factory.annotation.Value
import org.springframework.stereotype.Service
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Paths
import java.util.*
import java.util.concurrent.Executors
import kotlin.io.path.absolutePathString
import kotlin.math.min
import kotlin.math.sqrt

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
    private val waveformSampler: WaveformSampler,
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

        val voiceEmbeddings = computeVoiceEmbeddings(encoderModel)
        val tokenIds = tokenizer.tokenize(text)
        if (tokenIds.isEmpty()) throw IllegalArgumentException("Text tokenization produced no tokens")

        val env = onnx.getEnvironment()

        val textConditioner = textConditionerModel
        val flowMain = lmMainModel
        val flowFlow = lmFlowModel
        val mimi = decoderModel

        val frameSize = 32
        val diffusionSteps = 10
        val temperature = 0.3f
        val framesAfterEos = 3
        val maxFrames = 500
        val decodeChunkSize = 12

        // Prepare tensors used repeatedly
        val emptySeq = createEmptyFloatTensor(env, longArrayOf(1L, 0L, frameSize.toLong()))
        val emptyText = createEmptyFloatTensor(env, longArrayOf(1L, 0L, 1024L))
        val voiceTensor = createFloatTensorFrom2D(env, voiceEmbeddings)

        // Init states (create according to model's declared inputs)
        val flowState = initStateFromSession(flowMain)
        val mimiState = initStateFromSession(mimi)

        try {
            // Voice conditioning
            val voiceInputs = buildInputsForSession(flowMain, flowState, mapOf("sequence" to emptySeq, "text_embeddings" to voiceTensor))
            var res = flowMain.run(voiceInputs)
            updateStateFromResults(flowState, res, flowMain)
            res.close()

            // Text conditioning
            val tokenTensor = createLongTensor(env, tokenIds)
            val textRes = textConditioner.run(mapOf("token_ids" to tokenTensor))
            val textEmb = textRes[0] as OnnxTensor
            // Ensure shape is [1, 1, 1024] or [1, N, 1024]
            // If necessary, wrap/reshape by recreating tensor with same data
            val textEmbTensor = normalizeTextEmb(env, textEmb)
            textRes.close()
            tokenTensor.close()

            // Run flow main with empty seq + text embeddings to update state
            val mainInputs = buildInputsForSession(flowMain, flowState, mapOf("sequence" to emptySeq, "text_embeddings" to textEmbTensor))
            res = flowMain.run(mainInputs)
            updateStateFromResults(flowState, res, flowMain)
            res.close()

            // Autoregressive generation
            var currLatent = createNaNTensor(env, longArrayOf(1L, 1L, frameSize.toLong()))
            val generatedLatents = ArrayList<FloatArray>()
            var eosStep: Int? = null

            val rng = Random()

            for (step in 0 until maxFrames) {
                val arInputs = buildInputsForSession(flowMain, flowState, mapOf("sequence" to currLatent, "text_embeddings" to emptyText))
                val arRes = flowMain.run(arInputs)

                val conditioningTensor = arRes[0] as OnnxTensor
                val eosTensor = arRes[1] as OnnxTensor

                val conditioning = extractFloatArrayFromTensor(conditioningTensor) // shape [1,1,condDim] or [1,condDim]
                val eosVal = extractScalarFloat(eosTensor)

                updateStateFromResults(flowState, arRes, flowMain)
                conditioningTensor.close()
                eosTensor.close()
                arRes.close()

                if (eosStep == null && eosVal > -4.0f) eosStep = step
                if (eosStep != null && step >= eosStep + framesAfterEos) break

                // Flow matching
                val x = FloatArray(frameSize) { 0f }
                val std = if (temperature > 0) sqrt(temperature.toDouble()).toFloat() else 0f
                if (std > 0f) {
                    for (i in x.indices) x[i] = (rng.nextGaussian() * std).toFloat()
                }

                val dt = 1.0f / diffusionSteps
                for (j in 0 until diffusionSteps) {
                    val s = floatArrayOf(j.toFloat() / diffusionSteps)
                    val t = floatArrayOf((j + 1).toFloat() / diffusionSteps)
                    val cTensor = createFloatTensorFrom1D(env, conditioning)
                    val sTensor = createFloatTensorFrom1D(env, s, longArrayOf(1L, 1L))
                    val tTensor = createFloatTensorFrom1D(env, t, longArrayOf(1L, 1L))
                    val xTensor = createFloatTensorFrom1D(env, x, longArrayOf(1L, frameSize.toLong()))

                    val flowInputs = HashMap<String, OnnxTensor>()
                    flowInputs["c"] = cTensor
                    flowInputs["s"] = sTensor
                    flowInputs["t"] = tTensor
                    flowInputs["x"] = xTensor

                    val flowRes = flowFlow.run(flowInputs)
                    val vTensor = flowRes[0] as OnnxTensor
                    val v = extractFloatArrayFromTensor1D(vTensor)

                    for (k in x.indices) x[k] += v[k] * dt

                    vTensor.close()
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

                val decoderInputs = buildInputsForSession(mimi, mimiState, mapOf("latent" to latentTensor))
                val decRes = mimi.run(decoderInputs)

                // audio out
                val audioTensor = decRes[0] as OnnxTensor
                val audioOut = extractFloatArrayFromTensor1D(audioTensor)
                audioFloatsList.addAll(audioOut.toList())

                // update mimi state
                updateStateFromResults(mimiState, decRes, mimi)

                audioTensor.close()
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
            for (v in mimiState.values) v.close()
        }
    }

    private fun createLatentsTensor(env: ai.onnxruntime.OrtEnvironment, chunk: List<FloatArray>): OnnxTensor {
        val frames = chunk.size
        val flat = FloatArray(frames * 32)
        for (i in 0 until frames) System.arraycopy(chunk[i], 0, flat, i * 32, 32)
        return OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(flat), longArrayOf(1L, frames.toLong(), 32L))
    }

    private fun createFloatTensorFrom2D(env: ai.onnxruntime.OrtEnvironment, data: Array<FloatArray>): OnnxTensor {
        val n = data.size
        val d = if (n > 0) data[0].size else 0
        val flat = FloatArray(n * d)
        for (i in 0 until n) System.arraycopy(data[i], 0, flat, i * d, d)
        return OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(flat), longArrayOf(1L, n.toLong(), d.toLong()))
    }

    private fun createFloatTensorFrom2DForLatent(env: ai.onnxruntime.OrtEnvironment, data: FloatArray): OnnxTensor {
        return OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(data), longArrayOf(1L, 1L, data.size.toLong()))
    }

    private fun createFloatTensorFrom1D(env: ai.onnxruntime.OrtEnvironment, data: FloatArray, shape: LongArray? = null): OnnxTensor {
        val s = shape ?: longArrayOf(1L, data.size.toLong())
        return OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(data), s)
    }

    private fun createEmptyFloatTensor(env: ai.onnxruntime.OrtEnvironment, shape: LongArray): OnnxTensor {
        val total = shape.fold(1L) { acc, v -> if (v <= 0L) 0L else acc * v }
        val buf = FloatArray(total.toInt())
        return OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(buf), shape)
    }

    private fun createNaNTensor(env: ai.onnxruntime.OrtEnvironment, shape: LongArray): OnnxTensor {
        val total = shape.fold(1L) { acc, v -> if (v <= 0L) 0L else acc * v }
        val buf = FloatArray(total.toInt()) { Float.NaN }
        return OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(buf), shape)
    }

    private fun createLongTensor(env: ai.onnxruntime.OrtEnvironment, values: LongArray): OnnxTensor {
        return OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(values), longArrayOf(values.size.toLong()))
    }

    private fun createLongTensor(env: ai.onnxruntime.OrtEnvironment, tokens: List<Int>): OnnxTensor {
        val arr = LongArray(tokens.size)
        for (i in tokens.indices) arr[i] = tokens[i].toLong()
        return OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(arr), longArrayOf(1L, tokens.size.toLong()))
    }

    private fun createBoolTensor(value: Boolean, shape: LongArray = longArrayOf(1L)): OnnxTensor {
        val b = ByteArray(shape.fold(1L) { acc, v -> if (v <= 0L) acc else acc * v }.toInt())
        if (b.isNotEmpty()) b[0] = if (value) 1 else 0
        val bb = ByteBuffer.wrap(b)
        return OnnxTensor.createTensor(onnx.getEnvironment(), bb, shape, OnnxJavaType.BOOL)
    }

    private fun extractScalarFloat(tensor: OnnxTensor): Float {
        val v = tensor.value
        return when (v) {
            is FloatArray -> v[0]
            is java.nio.FloatBuffer -> { v.rewind(); v.get() }
            is Array<*> -> {
                val first = v[0]
                when (first) {
                    is FloatArray -> first[0]
                    is Array<*> -> {
                        val inner = first[0]
                        if (inner is Float) inner else (inner as Number).toFloat()
                    }
                    else -> (first as Number).toFloat()
                }
            }
            else -> 0f
        }
    }

    private fun extractFloatArrayFromTensor(tensor: OnnxTensor): FloatArray {
        val v = tensor.value
        when (v) {
            is java.nio.FloatBuffer -> {
                v.rewind(); val a = FloatArray(v.remaining()); v.get(a); return a
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

    private fun normalizeTextEmb(env: ai.onnxruntime.OrtEnvironment, tensor: OnnxTensor): OnnxTensor {
        val v = tensor.value
        // If tensor is FloatBuffer or FloatArray or nested array, create a 3D tensor if needed
        val flat = extractFloatArrayFromTensor(tensor)
        val shape = tensor.info.shape
        // Determine dims: if shape length is 2 -> [1, dim] -> convert to [1,1,dim]
        return if (shape.size == 2) {
            val dim = shape[1].toInt()
            OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(flat), longArrayOf(1L, 1L, dim.toLong()))
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
            val shape = try { (infoAny as? ai.onnxruntime.TensorInfo)?.shape ?: longArrayOf(0L) } catch (e: Exception) { longArrayOf(0L) }
            val infoStr = infoAny.toString().lowercase()
            val tensor = when {
                infoStr.contains("bool") -> createBoolTensor(false, shape)
                infoStr.contains("int64") -> createEmptyLongTensor(env, shape)
                else -> createEmptyFloatTensor(env, shape)
            }
            state[name] = tensor
        }
        return state
    }

    private fun createEmptyLongTensor(env: ai.onnxruntime.OrtEnvironment, shape: LongArray): OnnxTensor {
        val total = shape.fold(1L) { acc, v -> if (v <= 0L) 0L else acc * v }
        val buf = LongArray(total.toInt())
        return OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(buf), shape)
    }

    private fun initFlowState(): MutableMap<String, OnnxTensor> {
        val state = mutableMapOf<String, OnnxTensor>()
        val kvShape = longArrayOf(2L, 1L, 1000L, 16L, 64L)
        for (i in 0..15 step 3) {
            state["state_$i"] = createEmptyFloatTensor(onnx.getEnvironment(), kvShape)
            state["state_${i + 1}"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(0L))
            state["state_${i + 2}"] = createLongTensor(onnx.getEnvironment(), longArrayOf(0L))
        }
        return state
    }

    private fun initMimiState(): MutableMap<String, OnnxTensor> {
        val state = mutableMapOf<String, OnnxTensor>()
        state["state_0"] = createBoolTensor(false)
        state["state_1"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 512L, 6L))
        state["state_2"] = createBoolTensor(false)
        state["state_3"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 64L, 2L))
        state["state_4"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 256L, 6L))
        state["state_5"] = createBoolTensor(false)
        state["state_6"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 256L, 2L))
        state["state_7"] = createBoolTensor(false)
        state["state_8"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 128L, 0L))
        state["state_9"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 128L, 5L))
        state["state_10"] = createBoolTensor(false)
        state["state_11"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 128L, 2L))
        state["state_12"] = createBoolTensor(false)
        state["state_13"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 64L, 0L))
        state["state_14"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 64L, 4L))
        state["state_15"] = createBoolTensor(false)
        state["state_16"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 64L, 2L))
        state["state_17"] = createBoolTensor(false)
        state["state_18"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 32L, 0L))
        state["state_19"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(2L, 1L, 8L, 1000L, 64L))
        state["state_20"] = createLongTensor(onnx.getEnvironment(), longArrayOf(0L))
        state["state_21"] = createLongTensor(onnx.getEnvironment(), longArrayOf(0L))
        state["state_22"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(2L, 1L, 8L, 1000L, 64L))
        state["state_23"] = createLongTensor(onnx.getEnvironment(), longArrayOf(0L))
        state["state_24"] = createLongTensor(onnx.getEnvironment(), longArrayOf(0L))
        state["state_25"] = createBoolTensor(false)
        state["state_26"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 512L, 16L))
        state["state_27"] = createBoolTensor(false)
        state["state_28"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 1L, 6L))
        state["state_29"] = createBoolTensor(false)
        state["state_30"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 64L, 2L))
        state["state_31"] = createBoolTensor(false)
        state["state_32"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 32L, 0L))
        state["state_33"] = createBoolTensor(false)
        state["state_34"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 512L, 2L))
        state["state_35"] = createBoolTensor(false)
        state["state_36"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 64L, 4L))
        state["state_37"] = createBoolTensor(false)
        state["state_38"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 128L, 2L))
        state["state_39"] = createBoolTensor(false)
        state["state_40"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 64L, 0L))
        state["state_41"] = createBoolTensor(false)
        state["state_42"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 128L, 5L))
        state["state_43"] = createBoolTensor(false)
        state["state_44"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 256L, 2L))
        state["state_45"] = createBoolTensor(false)
        state["state_46"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 128L, 0L))
        state["state_47"] = createBoolTensor(false)
        state["state_48"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 256L, 6L))
        state["state_49"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(2L, 1L, 8L, 1000L, 64L))
        state["state_50"] = createLongTensor(onnx.getEnvironment(), longArrayOf(0L))
        state["state_51"] = createLongTensor(onnx.getEnvironment(), longArrayOf(0L))
        state["state_52"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(2L, 1L, 8L, 1000L, 64L))
        state["state_53"] = createLongTensor(onnx.getEnvironment(), longArrayOf(0L))
        state["state_54"] = createLongTensor(onnx.getEnvironment(), longArrayOf(0L))
        state["state_55"] = createEmptyFloatTensor(onnx.getEnvironment(), longArrayOf(1L, 512L, 16L))
        return state
    }

    private fun updateStateFromResults(state: MutableMap<String, OnnxTensor>, res: OrtSession.Result, session: OrtSession) {
        val outputs = session.outputInfo.keys.toList()
        for (i in outputs.indices) {
            val name = outputs[i]
            if (!name.startsWith("out_state_")) continue
            val stateKey = "state_${name.removePrefix("out_state_") }"
            val valObj = res[i]
            if (valObj is OnnxTensor) {
                val v = valObj.value
                val shape = valObj.info.shape
                // create tensor matching the result's value type
                val newTensor = when (v) {
                    is FloatArray -> OnnxTensor.createTensor(onnx.getEnvironment(), java.nio.FloatBuffer.wrap(v), shape)
                    is java.nio.FloatBuffer -> {
                        v.rewind(); val a = FloatArray(v.remaining()); v.get(a); OnnxTensor.createTensor(onnx.getEnvironment(), java.nio.FloatBuffer.wrap(a), shape)
                    }
                    is LongArray -> OnnxTensor.createTensor(onnx.getEnvironment(), java.nio.LongBuffer.wrap(v), shape)
                    is java.nio.LongBuffer -> { v.rewind(); val a = LongArray(v.remaining()); v.get(a); OnnxTensor.createTensor(onnx.getEnvironment(), java.nio.LongBuffer.wrap(a), shape) }
                    is BooleanArray -> {
                        val a = ByteArray(v.size)
                        for (ii in v.indices) a[ii] = if (v[ii]) 1 else 0
                        val bb = java.nio.ByteBuffer.wrap(a)
                        OnnxTensor.createTensor(onnx.getEnvironment(), bb, shape, OnnxJavaType.BOOL)
                    }
                    is Array<*> -> {
                        // try float nested
                        val flat = extractFloatArrayFromTensor(valObj as OnnxTensor)
                        OnnxTensor.createTensor(onnx.getEnvironment(), java.nio.FloatBuffer.wrap(flat), shape)
                    }
                    else -> {
                        // fallback to float
                        val flat = extractFloatArrayFromTensor(valObj as OnnxTensor)
                        OnnxTensor.createTensor(onnx.getEnvironment(), java.nio.FloatBuffer.wrap(flat), shape)
                    }
                }
                state[stateKey]?.close()
                state[stateKey] = newTensor
            }
        }
    }

    private fun buildInputsForSession(session: OrtSession, state: Map<String, OnnxTensor>, overrides: Map<String, OnnxTensor>): MutableMap<String, OnnxTensor> {
         val inputs = LinkedHashMap<String, OnnxTensor>()
         val env = onnx.getEnvironment()
         fun convertTensorToDeclared(declaredInfo: String, tensor: OnnxTensor): OnnxTensor {
             if (declaredInfo.contains("bool") && !(tensor.value is ByteBuffer || tensor.value is ByteArray)) {
                 val shape = tensor.info.shape
                 val v = tensor.value
                 return when (v) {
                     is LongArray -> {
                         val b = ByteArray(v.size); for (i in v.indices) b[i] = if (v[i] != 0L) 1 else 0
                         OnnxTensor.createTensor(env, java.nio.ByteBuffer.wrap(b), shape, OnnxJavaType.BOOL)
                     }
                     is java.nio.LongBuffer -> {
                         v.rewind(); val a = LongArray(v.remaining()); v.get(a)
                         val b = ByteArray(a.size); for (i in a.indices) b[i] = if (a[i] != 0L) 1 else 0
                         OnnxTensor.createTensor(env, java.nio.ByteBuffer.wrap(b), shape, OnnxJavaType.BOOL)
                     }
                     else -> tensor
                 }
             }
             if (declaredInfo.contains("int64") && !(tensor.value is java.nio.LongBuffer || tensor.value is LongArray)) {
                 val shape = tensor.info.shape
                 val v = tensor.value
                 return when (v) {
                     is java.nio.ByteBuffer -> {
                         v.rewind(); val n = v.remaining(); val a = LongArray(n); for (i in 0 until n) a[i] = if (v.get(i) != 0.toByte()) 1L else 0L
                         OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(a), shape)
                     }
                     is ByteArray -> {
                         val a = LongArray(v.size); for (i in v.indices) a[i] = if (v[i] != 0.toByte()) 1L else 0L
                         OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(a), shape)
                     }
                     else -> tensor
                 }
             }
             return tensor
         }
         for ((name, node) in session.inputInfo) {
             if (overrides.containsKey(name)) {
                 val declaredInfo = node.info.toString().lowercase()
                 inputs[name] = convertTensorToDeclared(declaredInfo, overrides[name]!!)
                 continue
             }
             if (state.containsKey(name)) {
                 val st = state[name]!!
                 // ensure the state's tensor matches expected element type by comparing declared info strings
                 val declaredInfo = node.info.toString().lowercase()
                 val storedInfo = st.info.toString().lowercase()
                 if (declaredInfo.contains("bool") && !(st.value is java.nio.ByteBuffer || st.value is ByteArray)) {
                     // convert stored int64 -> bool byte tensor
                     val shape = st.info.shape
                     val stVal = st.value
                     val newTensor = when (stVal) {
                         is LongArray -> {
                             val b = ByteArray(stVal.size); for (i in stVal.indices) b[i] = if (stVal[i] != 0L) 1 else 0
                             OnnxTensor.createTensor(env, java.nio.ByteBuffer.wrap(b), shape, OnnxJavaType.BOOL)
                         }
                         is java.nio.LongBuffer -> {
                             stVal.rewind(); val a = LongArray(stVal.remaining()); stVal.get(a)
                             val b = ByteArray(a.size); for (i in a.indices) b[i] = if (a[i] != 0L) 1 else 0
                             OnnxTensor.createTensor(env, java.nio.ByteBuffer.wrap(b), shape, OnnxJavaType.BOOL)
                         }
                         else -> st
                     }
                     inputs[name] = newTensor
                 } else if (declaredInfo.contains("int64") && !(st.value is java.nio.LongBuffer || st.value is LongArray)) {
                     // convert stored bool/byte -> int64
                     val shape = st.info.shape
                     val stVal = st.value
                     val newTensor = when (stVal) {
                         is java.nio.ByteBuffer -> {
                             stVal.rewind(); val n = stVal.remaining(); val a = LongArray(n); for (i in 0 until n) a[i] = if (stVal.get(i) != 0.toByte()) 1L else 0L
                             OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(a), shape)
                         }
                         is ByteArray -> {
                             val a = LongArray(stVal.size); for (i in stVal.indices) a[i] = if (stVal[i] != 0.toByte()) 1L else 0L
                             OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(a), shape)
                         }
                         else -> st
                     }
                     inputs[name] = newTensor
                 } else {
                     inputs[name] = st
                 }
                  continue
             }

             val infoAny = node.info
             val shape = try { (infoAny as? ai.onnxruntime.TensorInfo)?.shape ?: longArrayOf(0L) } catch (e: Exception) { longArrayOf(0L) }
             val infoStr = infoAny.toString().lowercase()
             val tensor = when {
                 infoStr.contains("bool") -> createBoolTensor(false)
                 infoStr.contains("int64") -> createLongTensor(env, longArrayOf(0L))
                 else -> createEmptyFloatTensor(env, shape)
             }
             inputs[name] = tensor
         }
        // For decoder session, ensure declared bool inputs are actual boolean tensors (byte buffers)
        try {
            val maybeDecoder = onnx.getSession()[decoder]
            if (session === maybeDecoder) {
                for ((name, node) in session.inputInfo) {
                    val declaredInfo = node.info.toString().lowercase()
                    if (!declaredInfo.contains("bool")) continue
                    val tensor = inputs[name] ?: continue
                    val v = tensor.value
                    if (v is LongArray) {
                        val arr = ByteArray(v.size)
                        for (i in v.indices) arr[i] = if (v[i] != 0L) 1 else 0
                        inputs[name] = OnnxTensor.createTensor(env, java.nio.ByteBuffer.wrap(arr), tensor.info.shape, OnnxJavaType.BOOL)
                    } else if (v is java.nio.LongBuffer) {
                        v.rewind()
                        val a = LongArray(v.remaining())
                        v.get(a)
                        val arr = ByteArray(a.size)
                        for (i in a.indices) arr[i] = if (a[i] != 0L) 1 else 0
                        inputs[name] = OnnxTensor.createTensor(env, java.nio.ByteBuffer.wrap(arr), tensor.info.shape, OnnxJavaType.BOOL)
                    }
                }
                println("[PocketTtsService] Decoder inputs mapping:")
                for ((k, v) in inputs) {
                    val declared = session.inputInfo[k]?.info.toString()
                    val actual = v.info.toString()
                    println("  input=$k declared=$declared actual=$actual")
                }
            }
        } catch (e: Exception) {
            println("[PocketTtsService] Failed to enforce/print decoder mapping: ${e.message}")
        }
         return inputs
     }

    private fun computeVoiceEmbeddings(encoder: OrtSession): Array<FloatArray> {
         val voicePath = Paths.get(modelsPath).resolve(voice).absolutePathString()
         val file = File(voicePath)
         if (!file.exists()) {
             throw IllegalArgumentException("Voice file not found: $voicePath")
         }

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
             val convertedAis = javax.sound.sampled.AudioSystem.getAudioInputStream(targetFormat, originalAis)
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
