package de.dbaelz.ttm.model

import de.dbaelz.ttm.tts.TtsConfig
import jakarta.persistence.*
import java.time.Instant

@Entity
@Table(name = "tts_job")
class TtsJobEntity(
    @Id
    @Column(name = "id", nullable = false)
    var id: String = "",

    @Column(name = "text", nullable = false, columnDefinition = "TEXT")
    var text: String = "",

    @Convert(converter = TtsConfigConverter::class)
    @Column(name = "config", nullable = false, columnDefinition = "TEXT")
    var config: TtsConfig = TtsConfig(),

    @Enumerated(EnumType.STRING)
    @Column(name = "engine", nullable = false)
    var engine: TtsEngine = TtsEngine.POCKET,

    @Column(name = "created_at", nullable = false)
    var createdAt: Instant = Instant.now(),

    @Enumerated(EnumType.STRING)
    @Column(name = "status", nullable = false)
    var status: JobStatus = JobStatus.PENDING,

    @Column(name = "file_id")
    var fileId: String? = null
)