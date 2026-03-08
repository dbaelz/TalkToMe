package de.dbaelz.ttm.service

import de.dbaelz.ttm.model.JobStatus
import de.dbaelz.ttm.model.TtsJob
import de.dbaelz.ttm.model.TtsProvider
import de.dbaelz.ttm.repository.JobRepository
import de.dbaelz.ttm.tts.TtsConfig
import de.dbaelz.ttm.tts.pocket.PocketTtsExecutor
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import java.util.*
import java.util.concurrent.Executors

@Service
class DefaultTtsService(
    private val storage: StorageService,
    private val repo: JobRepository,
    private val pocketTtsExecutor: PocketTtsExecutor,
) : TtsService {
    private val logger = LoggerFactory.getLogger(DefaultTtsService::class.java)
    private val executor = Executors.newFixedThreadPool(2)

    override fun generate(text: String, config: TtsConfig, provider: TtsProvider): TtsJob {
        val id = UUID.randomUUID().toString()
        val job = TtsJob(id = id, text = text, config = config, provider = provider)
        repo.save(job)

        executor.submit {
            process(provider, job)
        }

        return job
    }

    private fun process(provider: TtsProvider, job: TtsJob) {
        try {
            job.status = JobStatus.RUNNING
            repo.save(job)

            logger.info("Executing $job with provider ${provider.name}")

            val audio = when (provider) {
                TtsProvider.POCKET -> pocketTtsExecutor(job)
            }

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
