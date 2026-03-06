package de.dbaelz.ttm.repository

import de.dbaelz.ttm.model.TtsJob
import org.springframework.stereotype.Repository
import java.util.concurrent.ConcurrentHashMap

@Repository
class JobRepository {
    private val store = ConcurrentHashMap<String, TtsJob>()

    fun save(job: TtsJob): TtsJob { store[job.id] = job; return job }
    fun findById(id: String): TtsJob? = store[id]
}

