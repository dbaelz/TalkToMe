package de.dbaelz.ttm.controller

import de.dbaelz.ttm.model.TtsJobEntity
import de.dbaelz.ttm.repository.JobRepository
import de.dbaelz.ttm.service.StorageService
import org.hamcrest.Matchers.hasItems
import org.hamcrest.Matchers.notNullValue
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc
import org.springframework.context.annotation.Import
import org.springframework.http.MediaType
import org.springframework.test.web.servlet.MockMvc
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post
import org.springframework.test.web.servlet.result.MockMvcResultMatchers.*
import tools.jackson.databind.ObjectMapper
import java.time.Instant
import java.util.*
import de.dbaelz.ttm.TestConfig
import de.dbaelz.ttm.model.ChatterboxConfigEntity
import de.dbaelz.ttm.model.PocketTtsConfigEntity
import de.dbaelz.ttm.repository.ChatterboxConfigRepository
import de.dbaelz.ttm.repository.PocketTtsConfigRepository

@SpringBootTest
@AutoConfigureMockMvc
@Import(TestConfig::class)
class TtsControllerTest @Autowired constructor(
    private val mockMvc: MockMvc,
    private val objectMapper: ObjectMapper
) {
    @Autowired
    lateinit var jobRepository: JobRepository

    @Autowired
    lateinit var storageService: StorageService

    @Autowired
    lateinit var pocketConfigRepository: PocketTtsConfigRepository

    @Autowired
    lateinit var chatterboxConfigRepository: ChatterboxConfigRepository

    private val authHeader: String =
        "Basic " + Base64.getEncoder().encodeToString("user:password".toByteArray())

    @Test
    fun `generateAudio - success`() {
        val request = mapOf("text" to "Hello", "config" to mapOf("steps" to 1, "temperature" to 0.5, "seed" to 0))

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
    fun `generateAudio - with engine - success`() {
        val request = mapOf("text" to "Hello", "engine" to "pocket", "config" to mapOf("steps" to 2, "temperature" to 0.6, "seed" to 0))

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
            .andExpect(jsonPath("$.engine").value("POCKET"))
    }

    @Test
    fun `generateAudio - with invalid engine - bad request`() {
        val request = mapOf("text" to "Hello", "engine" to "invalid", "config" to mapOf("steps" to 2))

        mockMvc.perform(
            post("/api/tts")
                .header("Authorization", authHeader)
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request))
        )
            .andExpect(status().isBadRequest)
            .andExpect(content().contentTypeCompatibleWith(MediaType.APPLICATION_JSON))
            .andExpect(jsonPath("$.error").value("INVALID_ENGINE"))
            .andExpect(jsonPath("$.message").value("Engine 'invalid' is not supported"))
            .andExpect(jsonPath("$.allowedEngines").isArray)
            .andExpect(jsonPath("$.allowedEngines[0]").value("POCKET"))
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
        val jobEntity = TtsJobEntity(id = "job2", text = "Hello there", createdAt = Instant.now())
        jobRepository.save(jobEntity)

        mockMvc.perform(
            get("/api/tts/jobs/job2")
                .header("Authorization", authHeader)
        )
            .andExpect(status().isOk)
            .andExpect(content().contentTypeCompatibleWith(MediaType.APPLICATION_JSON))
            .andExpect(jsonPath("$.id").value("job2"))
    }

    @Test
    fun `getAllJobs - success returns list`() {
        jobRepository.deleteAll()
        val a = TtsJobEntity(id = "jobA", text = "A", createdAt = Instant.now())
        val b = TtsJobEntity(id = "jobB", text = "B", createdAt = Instant.now())
        jobRepository.saveAll(listOf(a, b))

        mockMvc.perform(
            get("/api/tts/jobs")
                .header("Authorization", authHeader)
        )
            .andExpect(status().isOk)
            .andExpect(content().contentTypeCompatibleWith(MediaType.APPLICATION_JSON))
            .andExpect(jsonPath("$[*].id").value(hasItems("jobA", "jobB")))
    }

    @Test
    fun `getAllJobs - success includes engine configs when present`() {
        jobRepository.deleteAll()
        pocketConfigRepository.deleteAll()
        chatterboxConfigRepository.deleteAll()

        val jobA = TtsJobEntity(id = "jobA", text = "A", createdAt = Instant.now(), engine = de.dbaelz.ttm.model.TtsEngine.CHATTERBOX)
        val jobB = TtsJobEntity(id = "jobB", text = "B", createdAt = Instant.now(), engine = de.dbaelz.ttm.model.TtsEngine.POCKET)
        jobRepository.saveAll(listOf(jobA, jobB))

        val chatter = ChatterboxConfigEntity(job = jobA, exaggeration = 0.75f)
        val pocket = PocketTtsConfigEntity(job = jobB, steps = 4, temperature = 0.6f, seed = 2)
        chatterboxConfigRepository.save(chatter)
        pocketConfigRepository.save(pocket)

        mockMvc.perform(
            get("/api/tts/jobs").header("Authorization", authHeader)
        )
            .andExpect(status().isOk)
            .andExpect(content().contentTypeCompatibleWith(MediaType.APPLICATION_JSON))
            .andExpect(jsonPath("$[?(@.id=='jobA')].config.exaggeration").exists())
            .andExpect(jsonPath("$[?(@.id=='jobB')].config.steps").exists())
    }

    @Test
    fun `getAllJobs - success returns empty list`() {
        jobRepository.deleteAll()

        mockMvc.perform(
            get("/api/tts/jobs")
                .header("Authorization", authHeader)
        )
            .andExpect(status().isOk)
            .andExpect(content().contentTypeCompatibleWith(MediaType.APPLICATION_JSON))
            .andExpect(jsonPath("$").isArray)
            .andExpect(jsonPath("$").isEmpty)
    }

    @Test
    fun `getAllJobs - unauthorized when missing credentials`() {
        mockMvc.perform(get("/api/tts/jobs"))
            .andExpect(status().isUnauthorized)
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
