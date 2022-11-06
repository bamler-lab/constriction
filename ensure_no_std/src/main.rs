#![no_std]
#![no_main]
#![feature(alloc_error_handler)]
#![feature(const_mut_refs)]

use core::panic::PanicInfo;
use simple_chunk_allocator::{heap, heap_bitmap, GlobalChunkAllocator, PageAligned};

static mut HEAP: PageAligned<[u8; 1048576]> = heap!();
static mut HEAP_BITMAP: PageAligned<[u8; 512]> = heap_bitmap!();

#[global_allocator]
static ALLOCATOR: GlobalChunkAllocator =
    unsafe { GlobalChunkAllocator::new(HEAP.deref_mut_const(), HEAP_BITMAP.deref_mut_const()) };

#[allow(unused_imports)]
use constriction;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

#[alloc_error_handler]
fn alloc_error_handler(layout: core::alloc::Layout) -> ! {
    panic!("Can't handle allocation: layout = {:?}", layout);
}

#[no_mangle]
pub extern "C" fn _start() -> ! {
    use constriction::stream::{Decode, Encode};

    let model = constriction::stream::model::UniformModel::<u32, 24>::new(10);

    let mut encoder = constriction::stream::stack::DefaultAnsCoder::new();
    encoder.encode_symbol(3u32, model).unwrap();
    encoder.encode_symbol(5u32, model).unwrap();
    let compressed = core::hint::black_box(encoder.into_compressed().unwrap());

    let mut decoder =
        constriction::stream::stack::DefaultAnsCoder::from_compressed(compressed).unwrap();
    assert_eq!(decoder.decode_symbol(model), Ok(5));
    assert_eq!(decoder.decode_symbol(model), Ok(3));

    loop {}
}
