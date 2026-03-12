package de.dbaelz.ttm.repository

import de.dbaelz.ttm.model.PocketTtsConfigEntity
import org.springframework.data.repository.CrudRepository

interface PocketTtsConfigRepository : CrudRepository<PocketTtsConfigEntity, String>

