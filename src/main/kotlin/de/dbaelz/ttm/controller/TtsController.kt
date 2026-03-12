package de.dbaelz.ttm.controller

import tools.jackson.databind.JsonNode
import tools.jackson.databind.ObjectMapper
import de.dbaelz.ttm.model.TtsEngine
import de.dbaelz.ttm.model.TtsJob
import de.dbaelz.ttm.service.TtsService
import de.dbaelz.ttm.tts.TtsConfig
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.*

@RestController
@RequestMapping("/api/tts")
class TtsController(
    private val ttsService: TtsService,
    private val mapper: ObjectMapper
) {

    data class GenerateRequest(
        val text: String?,
        val config: JsonNode? = null,
        val engine: String? = null
    )

    @PostMapping
    fun generateAudio(@RequestBody request: GenerateRequest): ResponseEntity<Any> {
        val text = request.text ?: return ResponseEntity.badRequest().build()

        val engine = request.engine?.trim()?.takeIf { it.isNotEmpty() }
            ?.let {
                try {
                    TtsEngine.valueOf(it.uppercase())
                } catch (_: IllegalArgumentException) {
                    return ResponseEntity.badRequest().body(
                        mapOf(
                            "error" to "INVALID_ENGINE",
                            "message" to "Engine '$it' is not supported",
                            "allowedEngines" to TtsEngine.entries.map { e -> e.name }
                        )
                    )
                }
            }
            ?: TtsEngine.POCKET

        val configNode = request.config ?: return ResponseEntity.badRequest().body(
            mapOf("error" to "MISSING_CONFIG", "message" to "Request must include a 'config' object for the chosen engine")
        )

        val config: TtsConfig = try {
            when (engine) {
                TtsEngine.POCKET -> mapper.treeToValue(configNode, de.dbaelz.ttm.tts.PocketTtsConfig::class.java)
                TtsEngine.CHATTERBOX -> mapper.treeToValue(configNode, de.dbaelz.ttm.tts.ChatterboxConfig::class.java)
            }
        } catch (e: Exception) {
            return ResponseEntity.badRequest().body(mapOf("error" to "INVALID_CONFIG_JSON", "message" to e.message))
        }

        when (engine) {
            TtsEngine.POCKET -> {
                val pocket = config as? de.dbaelz.ttm.tts.PocketTtsConfig ?: return ResponseEntity.badRequest().body(
                    mapOf("error" to "INVALID_CONFIG_TYPE", "message" to "Expected PocketTtsConfig for engine POCKET")
                )
                val errors = mutableListOf<String>()
                if (pocket.steps <= 0) errors.add("steps must be > 0")
                if (pocket.seed < 0) errors.add("seed must be >= 0")
                if (errors.isNotEmpty()) {
                    return ResponseEntity.badRequest().body(mapOf("error" to "INVALID_CONFIG", "messages" to errors))
                }
            }

            TtsEngine.CHATTERBOX -> {
                val chatter = config as? de.dbaelz.ttm.tts.ChatterboxConfig ?: return ResponseEntity.badRequest().body(
                    mapOf("error" to "INVALID_CONFIG_TYPE", "message" to "Expected ChatterboxConfig for engine CHATTERBOX")
                )
                val errors = mutableListOf<String>()
                if (chatter.exaggeration < 0) errors.add("exaggeration must be >= 0")
                if (errors.isNotEmpty()) {
                    return ResponseEntity.badRequest().body(mapOf("error" to "INVALID_CONFIG", "messages" to errors))
                }
            }
        }

        val job = ttsService.generate(text, config, engine)
        return ResponseEntity.accepted().body(job)
    }

    @GetMapping("/jobs")
    fun getAllJobs(): ResponseEntity<List<TtsJob>> {
        return ResponseEntity.ok(ttsService.getAllJobs())
    }


    @GetMapping("/jobs/{id}")
    fun getJob(@PathVariable id: String): ResponseEntity<TtsJob> {
        val job = ttsService.getJob(id) ?: return ResponseEntity.notFound().build()
        return ResponseEntity.ok(job)
    }

    @GetMapping("/files/{id}")
    fun download(@PathVariable id: String): ResponseEntity<ByteArray> {
        val audio = ttsService.getFile(id) ?: return ResponseEntity.notFound().build()
        return ResponseEntity.ok()
            .header("Content-Type", "audio/wav")
            .body(audio)
    }
}
