#![no_std]
#![no_main]

use constriction::stream::{Decode, Encode};
use core::panic::PanicInfo;

#[global_allocator]
static EMMA: emma::DefaultEmma = emma::DefaultEmma::new();

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    #[allow(clippy::empty_loop)]
    loop {}
}

#[no_mangle]
pub extern "C" fn _start() -> ! {
    let model = constriction::stream::model::DefaultUniformModel::new(10);

    let mut encoder = constriction::stream::stack::DefaultAnsCoder::new();
    encoder.encode_symbol(3usize, model).unwrap();
    encoder.encode_symbol(5usize, model).unwrap();
    let compressed = core::hint::black_box(encoder.into_compressed().unwrap());

    let mut decoder =
        constriction::stream::stack::DefaultAnsCoder::from_compressed(compressed).unwrap();
    assert_eq!(decoder.decode_symbol(model), Ok(5));
    assert_eq!(decoder.decode_symbol(model), Ok(3));

    #[allow(clippy::empty_loop)]
    loop {}
}
