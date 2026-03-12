package de.dbaelz.ttm.model

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
) {
    @OneToOne(mappedBy = "job", cascade = [CascadeType.ALL], orphanRemoval = true)
    var pocketConfig: PocketTtsConfigEntity? = null

    @OneToOne(mappedBy = "job", cascade = [CascadeType.ALL], orphanRemoval = true)
    var chatterboxConfig: ChatterboxConfigEntity? = null
}
