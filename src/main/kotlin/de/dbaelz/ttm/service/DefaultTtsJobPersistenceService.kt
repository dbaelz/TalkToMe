package de.dbaelz.ttm.service

import de.dbaelz.ttm.model.*
import de.dbaelz.ttm.repository.ChatterboxConfigRepository
import de.dbaelz.ttm.repository.JobRepository
import de.dbaelz.ttm.repository.PocketTtsConfigRepository
import de.dbaelz.ttm.tts.ChatterboxConfig
import de.dbaelz.ttm.tts.PocketTtsConfig
import de.dbaelz.ttm.tts.TtsConfig
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.Instant
import java.util.*

@Service
class DefaultTtsJobPersistenceService(
    private val jobRepo: JobRepository,
    private val pocketRepo: PocketTtsConfigRepository,
    private val chatterRepo: ChatterboxConfigRepository
) : TtsJobPersistenceService {

    @Transactional
    override fun createJob(text: String, engine: TtsEngine, config: TtsConfig): TtsJob {
        val id = UUID.randomUUID().toString()
        var jobEntity = TtsJobEntity(
            id = id,
            text = text,
            engine = engine,
            createdAt = Instant.now(),
            status = JobStatus.PENDING,
            fileId = null
        )
        jobEntity = jobRepo.saveAndFlush(jobEntity)

        when (config) {
            is PocketTtsConfig -> {
                val p = PocketTtsConfigEntity(
                    job = jobEntity,
                    steps = config.steps,
                    temperature = config.temperature,
                    seed = config.seed
                )
                pocketRepo.save(p)
            }

            is ChatterboxConfig -> {
                val c = ChatterboxConfigEntity(
                    job = jobEntity,
                    exaggeration = config.exaggeration
                )
                chatterRepo.save(c)
            }
        }

        return TtsJob(
            id = id,
            text = text,
            config = config,
            engine = engine,
            createdAt = jobEntity.createdAt,
            status = JobStatus.PENDING,
            fileId = null
        )
    }

    override fun findJob(id: String): TtsJob? {
        val jobEntity = jobRepo.findById(id).orElse(null) ?: return null
        val config: TtsConfig = when (jobEntity.engine) {
            TtsEngine.POCKET -> pocketRepo.findById(id).map { p ->
                PocketTtsConfig(steps = p.steps, temperature = p.temperature, seed = p.seed)
            }.orElse(PocketTtsConfig())

            TtsEngine.CHATTERBOX -> chatterRepo.findById(id).map { c ->
                ChatterboxConfig(exaggeration = c.exaggeration)
            }.orElse(ChatterboxConfig())
        }

        return TtsJob(
            id = jobEntity.id,
            text = jobEntity.text,
            config = config,
            engine = jobEntity.engine,
            createdAt = jobEntity.createdAt,
            status = jobEntity.status,
            fileId = jobEntity.fileId
        )
    }
}
