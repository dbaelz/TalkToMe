package de.dbaelz.ttm.model

import jakarta.persistence.*

@Entity
@Table(name = "pocket_tts_config")
class PocketTtsConfigEntity(
    @Id
    @Column(name = "id", nullable = false)
    var id: String? = null,

    @OneToOne
    @MapsId
    @JoinColumn(name = "id")
    var job: TtsJobEntity? = null,

    @Column(nullable = false)
    var steps: Int = 10,

    @Column(nullable = false)
    var temperature: Float = 0.5f,

    @Column(nullable = false)
    var seed: Int = 0
)
