package de.dbaelz.ttm.repository

import de.dbaelz.ttm.model.TtsJobEntity
import org.springframework.data.repository.CrudRepository

interface JobRepository : CrudRepository<TtsJobEntity, String>
