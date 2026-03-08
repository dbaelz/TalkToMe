package de.dbaelz.ttm.onnx

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.slf4j.LoggerFactory
import org.springframework.beans.factory.annotation.Value
import org.springframework.stereotype.Component
import java.nio.file.Paths
import java.util.concurrent.ConcurrentHashMap
import kotlin.io.path.absolutePathString

@Component
class OnnxWrapper(
    @Value("\${onnx.use-gpu:true}") private val useGpu: Boolean,
    @Value("\${onnx.gpu-device:0}") private val gpuDevice: Int
) {
    private val logger = LoggerFactory.getLogger(OnnxWrapper::class.java)
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()

    private val sessions: MutableMap<ModelFileKey, OrtSession> = ConcurrentHashMap()

    fun loadModelFiles(modelGroupName: String, modelsPath: String, modelFiles: List<String>) {
        // TODO: Very simple solution for now. Always unload all models of the previous group.
        sessions.entries.firstOrNull()?.let {
            if (it.key.modelGroupName != modelGroupName) {
                unloadModelsForGroup(it.key.modelGroupName)
            }
        }

        val unloaded =
            modelFiles.filterNot { sessions.containsKey(ModelFileKey(modelGroupName, it)) }

        val basePath = Paths.get(modelsPath)

        val sessionOptions = try {
            if (useGpu) {
                val tryOpts = OrtSession.SessionOptions()
                tryOpts.addCUDA(gpuDevice)
                logger.info("ONNX: CUDA provider successfully initialized for device={}", gpuDevice)
                tryOpts
            } else {
                OrtSession.SessionOptions()
            }
        } catch (_: Throwable) {
            OrtSession.SessionOptions()
        }

        unloaded.forEach { fileName ->
            try {
                val session = env.createSession(
                    basePath.resolve(fileName).absolutePathString(),
                    sessionOptions
                )
                sessions[ModelFileKey(modelGroupName, fileName)] = session
            } catch (e: Exception) {
                logger.error("Failed to load ONNX model file: {}", fileName, e)
            }
        }
    }

    private fun unloadModelsForGroup(modelGroupName: String) {
        val keysToUnload = sessions.keys.filter { key ->
            key.modelGroupName == modelGroupName
        }

        keysToUnload.forEach { key ->
            sessions[key]?.close()
            sessions.remove(key)
        }
    }

    fun getEnvironment() = env

    fun getModel(modelGroupName: String, fileName: String): OrtSession {
        return sessions[ModelFileKey(modelGroupName, fileName)]
            ?: throw IllegalArgumentException("Model file not loaded: $fileName in group $modelGroupName")
    }

    data class ModelFileKey(val modelGroupName: String, val fileName: String)
}