package de.dbaelz.ttm.startup

import de.dbaelz.ttm.model.JobStatus
import de.dbaelz.ttm.model.TtsJobEntity
import de.dbaelz.ttm.repository.JobRepository
import de.dbaelz.ttm.service.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.ArgumentMatchers
import org.mockito.Mockito
import java.time.Instant

class StartupCleanupTest {
    private val repo: JobRepository = Mockito.mock(JobRepository::class.java)
    private val storage: StorageService = Mockito.mock(StorageService::class.java)
    private lateinit var cleanup: StartupCleanup

    @BeforeEach
    fun setUp() {
        cleanup = StartupCleanup(repo, storage)
    }

    @Test
    fun `DONE job with file present is not deleted`() {
        val job = TtsJobEntity(id = "j1", text = "t", createdAt = Instant.now(), status = JobStatus.DONE, fileId = "f1")
        Mockito.`when`(repo.findAll()).thenReturn(listOf(job))
        Mockito.`when`(storage.exists("f1")).thenReturn(true)

        cleanup.onApplicationReady()

        Mockito.verify(repo, Mockito.never()).deleteById(ArgumentMatchers.anyString())
    }

    @Test
    fun `DONE job with file missing is deleted`() {
        val job = TtsJobEntity(id = "j2", text = "t", createdAt = Instant.now(), status = JobStatus.DONE, fileId = "f2")
        Mockito.`when`(repo.findAll()).thenReturn(listOf(job))
        Mockito.`when`(storage.exists("f2")).thenReturn(false)

        cleanup.onApplicationReady()

        Mockito.verify(repo).deleteById("j2")
    }

    @Test
    fun `PENDING job with file missing is not deleted`() {
        val job = TtsJobEntity(id = "j3", text = "t", createdAt = Instant.now(), status = JobStatus.PENDING, fileId = "f3")
        Mockito.`when`(repo.findAll()).thenReturn(listOf(job))
        Mockito.`when`(storage.exists("f3")).thenReturn(false)

        cleanup.onApplicationReady()

        Mockito.verify(repo, Mockito.never()).deleteById(ArgumentMatchers.anyString())
    }

    @Test
    fun `RUNNING job with file missing is not deleted`() {
        val job = TtsJobEntity(id = "j4", text = "t", createdAt = Instant.now(), status = JobStatus.RUNNING, fileId = "f4")
        Mockito.`when`(repo.findAll()).thenReturn(listOf(job))
        Mockito.`when`(storage.exists("f4")).thenReturn(false)

        cleanup.onApplicationReady()

        Mockito.verify(repo, Mockito.never()).deleteById(ArgumentMatchers.anyString())
    }

    @Test
    fun `FAILED job is deleted regardless of file existence`() {
        val job = TtsJobEntity(id = "j5", text = "t", createdAt = Instant.now(), status = JobStatus.FAILED, fileId = null)
        Mockito.`when`(repo.findAll()).thenReturn(listOf(job))

        cleanup.onApplicationReady()

        Mockito.verify(repo).deleteById("j5")
    }

    @Test
    fun `storage check throws - DONE job is not deleted`() {
        val job = TtsJobEntity(id = "j6", text = "t", createdAt = Instant.now(), status = JobStatus.DONE, fileId = "f6")
        Mockito.`when`(repo.findAll()).thenReturn(listOf(job))
        Mockito.`when`(storage.exists("f6")).thenThrow(RuntimeException("fail"))

        cleanup.onApplicationReady()

        Mockito.verify(repo, Mockito.never()).deleteById(ArgumentMatchers.anyString())
    }
}
