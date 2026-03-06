package de.dbaelz.ttm.tts

import org.springframework.stereotype.Component

@Component
class SentencePieceTokenizer {
    fun tokenize(text: String): IntArray {
        // TODO: Use SentencePiece library to tokenize text into token IDs
        return intArrayOf()
    }
}