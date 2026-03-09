package de.dbaelz.ttm.repository

import de.dbaelz.ttm.model.TtsJobEntity
import de.dbaelz.ttm.model.JobStatus
import org.springframework.data.repository.CrudRepository

interface JobRepository : CrudRepository<TtsJobEntity, String> {
    fun findAllByStatus(status: JobStatus): Iterable<TtsJobEntity>
}
