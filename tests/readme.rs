//! This is the example from `README.md`

#[test]
#[cfg_attr(miri, ignore)]
fn example() {
    use constriction::stream::{model::DefaultLeakyQuantizer, stack::DefaultAnsCoder, Decode};

    // Let's use an ANS Coder in this example. Constriction also provides an Range
    // Coder, a Huffman Coder, and an experimental new "Chain Coder".
    let mut coder = DefaultAnsCoder::new();

    // Define some data and a sequence of entropy models. We use quantized Gaussians
    // here, but you could also use other models or even implement your own.
    let symbols = vec![23i32, -15, 78, 43, -69];
    let quantizer = DefaultLeakyQuantizer::new(-100..=100);
    let means = vec![35.2f64, -1.7, 30.1, 71.2, -75.1];
    let stds = vec![10.1f64, 25.3, 23.8, 35.4, 3.9];
    let models = means.iter().zip(&stds).map(|(&mean, &std)| {
        quantizer.quantize(probability::distribution::Gaussian::new(mean, std))
    });

    // Encode symbols (in *reverse* order, because ANS Coding operates as a stack).
    coder
        .encode_symbols_reverse(symbols.iter().zip(models.clone()))
        .unwrap();

    // Obtain temporary shared access to the compressed bit string. If you want ownership of the
    // compressed bit string, call `.into_compressed()` instead of `.get_compressed()`.
    println!(
        "Encoded into {} bits: {:?}",
        coder.num_bits(),
        &*coder.get_compressed().unwrap()
    );

    // Decode the symbols and verify correctness.
    let reconstructed = coder
        .decode_symbols(models)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(reconstructed, symbols);
}
