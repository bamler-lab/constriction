#[cfg(not(miri))]
mod augmented_btree;

#[cfg(not(miri))]
mod lookup;

#[cfg(not(miri))]
criterion::criterion_main!(augmented_btree::augmented_btree, lookup::lookup);

#[cfg(miri)]
fn main() {} // miri currently doesn't seem to be able to run criterion benchmarks as tests.
