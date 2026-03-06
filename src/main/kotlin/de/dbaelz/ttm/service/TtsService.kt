package de.dbaelz.ttm.service

import de.dbaelz.ttm.model.JobStatus
import de.dbaelz.ttm.model.TtsJob
import de.dbaelz.ttm.tts.OnnxWrapper
import de.dbaelz.ttm.repository.JobRepository
import de.dbaelz.ttm.tts.SentencePieceTokenizer
import org.springframework.stereotype.Service
import java.util.UUID
import java.util.concurrent.Executors

@Service
class TtsService(
    private val tokenizer: SentencePieceTokenizer,
    private val engine: OnnxWrapper,
    private val storage: LocalStorageService,
    private val repo: JobRepository
) {
    private val executor = Executors.newFixedThreadPool(2)

    fun submit(text: String): TtsJob {
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
            val tokens = tokenizer.tokenize(job.text)
            val audio = engine.synthesize(tokens)
            val fileId = storage.save(audio)
            job.fileId = fileId
            job.status = JobStatus.DONE
            repo.save(job)
        } catch (e: Exception) {
            job.status = JobStatus.FAILED
            repo.save(job)
        }
    }

    fun generateSync(text: String): ByteArray {
        val tokens = tokenizer.tokenize(text)
        return engine.synthesize(tokens)
    }

    fun getJob(id: String): TtsJob? = repo.findById(id)
    fun getFile(id: String): ByteArray? = storage.load(id)
}

