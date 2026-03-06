package de.dbaelz.ttm.controller

import de.dbaelz.ttm.model.TtsJob
import de.dbaelz.ttm.repository.JobRepository
import de.dbaelz.ttm.service.LocalStorageService
import org.hamcrest.Matchers.notNullValue
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc
import org.springframework.http.MediaType
import org.springframework.test.web.servlet.MockMvc
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post
import org.springframework.test.web.servlet.result.MockMvcResultMatchers.*
import tools.jackson.databind.ObjectMapper
import java.time.Instant
import java.util.*

@SpringBootTest
@AutoConfigureMockMvc
class TtsControllerTest @Autowired constructor(
    private val mockMvc: MockMvc,
    private val objectMapper: ObjectMapper
) {
    @Autowired
    lateinit var jobRepository: JobRepository

    @Autowired
    lateinit var storageService: LocalStorageService

    private val authHeader: String =
        "Basic " + Base64.getEncoder().encodeToString("user:password".toByteArray())

    @Test
    fun `generateAudio - success`() {
        val request = mapOf("text" to "Hello")

        mockMvc.perform(
            post("/api/tts")
                .header("Authorization", authHeader)
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request))
        )
            .andExpect(status().isAccepted)
            .andExpect(content().contentTypeCompatibleWith(MediaType.APPLICATION_JSON))
            .andExpect(jsonPath("$.id", notNullValue()))
            .andExpect(jsonPath("$.text").value("Hello"))
    }

    @Test
    fun `generateAudio - bad request missing text`() {
        val request = emptyMap<String, String>()
        mockMvc.perform(
            post("/api/tts")
                .header("Authorization", authHeader)
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request))
        )
            .andExpect(status().isBadRequest)
    }

    @Test
    fun `generateAudio - unauthorized when missing credentials`() {
        mockMvc.perform(
            post("/api/tts")
                .contentType(MediaType.APPLICATION_JSON)
        )
            .andExpect(status().isUnauthorized)
    }

    @Test
    fun `getJob - success`() {
        val job = TtsJob(id = "job2", text = "Hello there", createdAt = Instant.now())
        jobRepository.save(job)

        mockMvc.perform(
            get("/api/tts/jobs/job2")
                .header("Authorization", authHeader)
        )
            .andExpect(status().isOk)
            .andExpect(content().contentTypeCompatibleWith(MediaType.APPLICATION_JSON))
            .andExpect(jsonPath("$.id").value("job2"))
    }

    @Test
    fun `getJob - not found`() {
        mockMvc.perform(
            get("/api/tts/jobs/missing")
                .header("Authorization", authHeader)
        )
            .andExpect(status().isNotFound)
    }

    @Test
    fun `getJob - unauthorized when missing credentials`() {
        mockMvc.perform(get("/api/tts/jobs/job2"))
            .andExpect(status().isUnauthorized)
    }

    @Test
    fun `download - success`() {
        val payload = "audio-bytes".toByteArray()
        val id = storageService.save(payload)

        mockMvc.perform(
            get("/api/tts/files/$id")
                .header("Authorization", authHeader)
        )
            .andExpect(status().isOk)
            .andExpect(header().string("Content-Type", "audio/wav"))
            .andExpect(content().bytes(payload))
    }

    @Test
    fun `download - not found`() {
        mockMvc.perform(
            get("/api/tts/files/missing")
                .header("Authorization", authHeader)
        )
            .andExpect(status().isNotFound)
    }

    @Test
    fun `download - unauthorized when missing credentials`() {
        mockMvc.perform(get("/api/tts/files/file1"))
            .andExpect(status().isUnauthorized)
    }
}
