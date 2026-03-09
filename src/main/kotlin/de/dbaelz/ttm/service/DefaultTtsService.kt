package de.dbaelz.ttm.service

import de.dbaelz.ttm.model.JobStatus
import de.dbaelz.ttm.model.TtsEngine
import de.dbaelz.ttm.model.TtsJob
import de.dbaelz.ttm.model.TtsJobEntity
import de.dbaelz.ttm.repository.JobRepository
import de.dbaelz.ttm.repository.toEntity
import de.dbaelz.ttm.repository.toTtsJob
import de.dbaelz.ttm.tts.TtsConfig
import de.dbaelz.ttm.tts.pocket.PocketTtsExecutor
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import java.util.*
import java.util.concurrent.Executor

@Service
class DefaultTtsService(
    private val storage: StorageService,
    private val repo: JobRepository,
    private val pocketTtsExecutor: PocketTtsExecutor,
    private val executor: Executor,
) : TtsService {
    private val logger = LoggerFactory.getLogger(DefaultTtsService::class.java)

    override fun generate(text: String, config: TtsConfig, engine: TtsEngine): TtsJob {
        val id = UUID.randomUUID().toString()
        val job = TtsJob(id = id, text = text, config = config, engine = engine)
        repo.save(job.toEntity())

        executor.execute {
            process(engine, job)
        }

        return job
    }

    private fun process(engine: TtsEngine, job: TtsJob) {
        try {
            job.status = JobStatus.RUNNING
            updateJob(job) { entity ->
                entity.status = JobStatus.RUNNING
                entity
            }

            logger.info("Executing $job with engine ${engine.name}")

            val audio = when (engine) {
                TtsEngine.POCKET -> pocketTtsExecutor(job)
            }

            logger.info("Executed $job, audio size: ${audio.size} bytes")

            val fileId = storage.save(audio)
            job.fileId = fileId
            job.status = JobStatus.DONE
            updateJob(job, { entity ->
                entity.fileId = fileId
                entity.status = JobStatus.DONE
                entity
            })
        } catch (_: Exception) {
            job.status = JobStatus.FAILED
            updateJob(job) { entity ->
                entity.status = JobStatus.FAILED
                entity
            }
        }
    }

    private fun updateJob(job: TtsJob, onUpdate: (TtsJobEntity) -> TtsJobEntity) {
        repo.findById(job.id).ifPresentOrElse({ entity ->
            onUpdate(entity)
            repo.save(entity)
        }, {
            repo.save(job.toEntity())
        })
    }

    override fun getJob(id: String): TtsJob? = repo.findById(id).map { it.toTtsJob() }.orElse(null)

    override fun getAllJobs(): List<TtsJob> = repo.findAll().map { it.toTtsJob() }.toList()

    override fun getFile(id: String): ByteArray? = storage.load(id)
}
