package de.dbaelz.ttm.tts.pocket

import com.sentencepiece.Model
import com.sentencepiece.Scoring
import com.sentencepiece.SentencePieceAlgorithm
import org.springframework.beans.factory.annotation.Value
import org.springframework.stereotype.Component
import java.nio.file.Paths


@Component
class SentencePieceTokenizer(
    @Value("\${tts.models.pocket-tts}") private val modelsPath: String,
    @Value("\${tts.models.pocket-tts.tokenizer}") private val fileName: String
) {
    private val tokenizer =
        Model.parseFrom(Paths.get(modelsPath).resolve(fileName))
    private val algorithm = SentencePieceAlgorithm(true, Scoring.HIGHEST_SCORE)


    fun tokenize(text: String): List<Int> {
        return tokenizer.encodeNormalized(text, algorithm)
    }
}