package de.dbaelz.ttm.service

import de.dbaelz.ttm.model.JobStatus
import de.dbaelz.ttm.model.TtsEngine
import de.dbaelz.ttm.model.TtsJob
import de.dbaelz.ttm.repository.JobRepository
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
        repo.save(job)

        executor.execute {
            process(engine, job)
        }

        return job
    }

    private fun process(engine: TtsEngine, job: TtsJob) {
        try {
            job.status = JobStatus.RUNNING
            repo.save(job)

            logger.info("Executing $job with engine ${engine.name}")

            val audio = when (engine) {
                TtsEngine.POCKET -> pocketTtsExecutor(job)
            }

            logger.info("Executed $job, audio size: ${audio.size} bytes")

            val fileId = storage.save(audio)
            job.fileId = fileId
            job.status = JobStatus.DONE
            repo.save(job)
        } catch (_: Exception) {
            job.status = JobStatus.FAILED
            repo.save(job)
        }
    }

    override fun getJob(id: String): TtsJob? = repo.findById(id)

    override fun getFile(id: String): ByteArray? = storage.load(id)
}
