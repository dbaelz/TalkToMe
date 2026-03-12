package de.dbaelz.ttm.repository

import de.dbaelz.ttm.model.TtsJobEntity
import org.springframework.data.jpa.repository.JpaRepository

interface JobRepository : JpaRepository<TtsJobEntity, String>
