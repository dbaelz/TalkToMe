package de.dbaelz.ttm.repository

import de.dbaelz.ttm.model.ChatterboxConfigEntity
import org.springframework.data.repository.CrudRepository

interface ChatterboxConfigRepository : CrudRepository<ChatterboxConfigEntity, String>

