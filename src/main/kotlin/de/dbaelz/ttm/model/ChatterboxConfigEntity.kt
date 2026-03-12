package de.dbaelz.ttm.model

import jakarta.persistence.*

@Entity
@Table(name = "chatterbox_config")
class ChatterboxConfigEntity(
    @Id
    @Column(name = "id", nullable = false)
    var id: String? = null,

    @OneToOne
    @MapsId
    @JoinColumn(name = "id")
    var job: TtsJobEntity? = null,

    @Column(nullable = false)
    var exaggeration: Float = 0.5f
)
