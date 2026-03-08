package de.dbaelz.ttm.controller

import de.dbaelz.ttm.model.TtsJob
import de.dbaelz.ttm.tts.TtsConfig
import de.dbaelz.ttm.tts.pocket.PocketTtsService
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.*

@RestController
@RequestMapping("/api/tts")
class TtsController(private val pocketTtsService: PocketTtsService) {

    data class GenerateRequest(val text: String?, val config: TtsConfig? = null)

    @PostMapping
    fun generateAudio(@RequestBody request: GenerateRequest): ResponseEntity<TtsJob> {
        val text = request.text ?: return ResponseEntity.badRequest().build()
        val job = pocketTtsService.generate(text, request.config ?: TtsConfig())
        return ResponseEntity.accepted().body(job)
    }

    @GetMapping("/jobs/{id}")
    fun getJob(@PathVariable id: String): ResponseEntity<TtsJob> {
        val job = pocketTtsService.getJob(id) ?: return ResponseEntity.notFound().build()
        return ResponseEntity.ok(job)
    }

    @GetMapping("/files/{id}")
    fun download(@PathVariable id: String): ResponseEntity<ByteArray> {
        val audio = pocketTtsService.getFile(id) ?: return ResponseEntity.notFound().build()
        return ResponseEntity.ok()
            .header("Content-Type", "audio/wav")
            .body(audio)
    }
}
