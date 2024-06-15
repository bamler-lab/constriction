use std::collections::HashMap;

use constriction::{
    backends::Cursor,
    stream::{
        model::DefaultContiguousCategoricalEntropyModel, stack::DefaultAnsCoder, Decode, Encode,
    },
    UnwrapInfallible,
};

#[derive(Debug, PartialEq, Eq)]
struct UncompressedIndex {
    doc: Vec<String>,
}

#[derive(Debug)]
struct CompressedIndex {
    doc: Vec<Vec<u32>>, // Note that constriction represents bit strings in 32-bit chunks by default for performance reasons.
    probs: DefaultContiguousCategoricalEntropyModel, // (for example; you can use any entropy model in `constriction::stream::model`)
    alphabet: Vec<char>, // List of all distinct characters that can appear in a message.
}

impl UncompressedIndex {
    fn compress(
        &self,
        probs: DefaultContiguousCategoricalEntropyModel,
        alphabet: Vec<char>,
    ) -> CompressedIndex {
        let inverse_alphabet = alphabet
            .iter()
            .enumerate()
            .map(|(index, &character)| (character, index))
            .collect::<HashMap<_, _>>();

        let doc = self
            .doc
            .iter()
            .map(|message| {
                let mut coder = DefaultAnsCoder::new();

                // Start with a special EOF symbol so that `CompressedIndex::decompress` knows when to terminate:
                coder.encode_symbol(alphabet.len(), &probs).unwrap();

                // Then encode the message, character by character, in reverse order:
                for character in message.chars().rev() {
                    let char_index = *inverse_alphabet.get(&character).unwrap();
                    coder.encode_symbol(char_index, &probs).unwrap();
                }

                coder.into_compressed().unwrap_infallible()
            })
            .collect();

        CompressedIndex {
            doc,
            probs,
            alphabet,
        }
    }
}

impl CompressedIndex {
    fn decompress(&self) -> UncompressedIndex {
        let doc = self
            .doc
            .iter()
            .map(|data| {
                let mut coder =
                    DefaultAnsCoder::from_compressed(Cursor::new_at_write_end(&data[..])).unwrap();
                core::iter::from_fn(|| {
                    let symbol_id = coder.decode_symbol(&self.probs).unwrap();
                    // Terminate decoding when EOF token `self.alphabet.len()` is decoded.
                    self.alphabet.get(symbol_id).copied()
                })
                .collect()
            })
            .collect();

        UncompressedIndex { doc }
    }
}

#[test]
fn round_trip() {
    let uncompressed = UncompressedIndex {
        doc: vec!["Hello, World!".to_string(), "Goodbye.".to_string()],
    };

    let alphabet = vec![
        'H', 'e', 'l', 'o', ',', ' ', 'W', 'r', 'd', '!', 'G', 'b', 'y', '.',
    ];
    let counts = [1., 2., 3., 4., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2.]; // The last entry is for the EOF token.
    let probs =
        DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities(&counts)
            .unwrap();

    let compressed = uncompressed.compress(probs, alphabet);
    let reconstructed = compressed.decompress();
    assert_eq!(uncompressed, reconstructed);
}
