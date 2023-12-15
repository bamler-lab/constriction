use core::hash::{Hash, Hasher};
use std::any::type_name;
use std::collections::hash_map::DefaultHasher;

use rand::{distributions, seq::SliceRandom, Rng, RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;

use criterion::{black_box, criterion_group, Criterion};

use constriction::{quant::augmented_btree::AugmentedBTree, NonNanFloat};

type F32 = NonNanFloat<f32>;

criterion_group!(augmented_btree, insert, remove, cdf, quantile, shift);

trait Benchmark {
    fn run<const CAP: usize>(name: &str, amt: usize, c: &mut Criterion);
}

fn run_benchmarks<B: Benchmark>(c: &mut Criterion) {
    let name = type_name::<B>()
        .chars()
        .flat_map(char::to_lowercase)
        .collect::<String>();
    for amt in [10_000, 1000, 100] {
        B::run::<8>(&format!("augmented_btree_{name}_8_{amt}"), amt, c);
        B::run::<16>(&format!("augmented_btree_{name}_16_{amt}"), amt, c);
        B::run::<32>(&format!("augmented_btree_{name}_32_{amt}"), amt, c);
        B::run::<64>(&format!("augmented_btree_{name}_64_{amt}"), amt, c);
        B::run::<128>(&format!("augmented_btree_{name}_128_{amt}"), amt, c);
        B::run::<256>(&format!("augmented_btree_{name}_256_{amt}"), amt, c);
    }
}

struct Insert;

impl Benchmark for Insert {
    fn run<const CAP: usize>(name: &str, amt: usize, c: &mut Criterion) {
        let (insertions, _) = create_random_insertions(amt, (20231211, CAP));

        c.bench_function(name, |b| {
            b.iter(|| {
                let mut tree = AugmentedBTree::<F32, u32, CAP>::new();
                for &(pos, count) in &insertions {
                    tree.insert(pos, count);
                }

                black_box(tree.left_cumulative(NonNanFloat::new(0.123).unwrap()));
            })
        });
    }
}

fn insert(c: &mut Criterion) {
    run_benchmarks::<Insert>(c);
}

struct Remove;

impl Benchmark for Remove {
    fn run<const CAP: usize>(name: &str, amt: usize, c: &mut Criterion) {
        let (insertions, mut rng) = create_random_insertions(amt, (202312112, CAP));
        let mut tree = AugmentedBTree::<F32, u32, CAP>::new();
        for &(pos, count) in &insertions {
            tree.insert(pos, count);
        }

        let mut removals = insertions;
        removals.shuffle(&mut rng);

        c.bench_function(name, |b| {
            b.iter(|| {
                let mut tree = tree.clone();
                for &(pos, count) in &removals {
                    assert!(tree.remove(pos, count).is_ok());
                }

                assert_eq!(tree.total(), 0);
                assert_eq!(tree.left_cumulative(NonNanFloat::new(0.3).unwrap()), 0);
            })
        });
    }
}

fn remove(c: &mut Criterion) {
    run_benchmarks::<Remove>(c);
}

struct Cdf;

impl Benchmark for Cdf {
    fn run<const CAP: usize>(name: &str, amt: usize, c: &mut Criterion) {
        let (insertions, mut rng) = create_random_insertions(amt, (202312113, CAP));
        let mut tree = AugmentedBTree::<F32, u32, CAP>::new();
        for &(pos, count) in &insertions {
            tree.insert(pos, count);
        }

        let test_points = (0..amt)
            .map(|_| F32::new(rng.next_u32() as f32 / u32::MAX as f32).unwrap())
            .collect::<Vec<_>>();

        c.bench_function(name, |b| {
            b.iter(|| {
                let mut dummy = 0;
                for &pos in &test_points {
                    dummy ^= tree.left_cumulative(pos);
                }

                black_box(dummy);
            })
        });
    }
}

fn cdf(c: &mut Criterion) {
    run_benchmarks::<Cdf>(c);
}

struct Quantile;

impl Benchmark for Quantile {
    fn run<const CAP: usize>(name: &str, amt: usize, c: &mut Criterion) {
        let (insertions, mut rng) = create_random_insertions(amt, (202312114, CAP));
        let mut tree = AugmentedBTree::<F32, u32, CAP>::new();
        for &(pos, count) in &insertions {
            tree.insert(pos, count);
        }

        let total = tree.total();
        let test_points = (0..amt).map(|_| rng.next_u32() % total).collect::<Vec<_>>();

        c.bench_function(name, |b| {
            b.iter(|| {
                let mut dummy = 0;
                for &quantile in &test_points {
                    let pos = tree.quantile_function(quantile).unwrap().get();

                    #[allow(clippy::transmute_float_to_int)]
                    {
                        dummy ^= unsafe { core::mem::transmute::<f32, u32>(pos) }
                    }
                }

                black_box(dummy);
            })
        });
    }
}

fn quantile(c: &mut Criterion) {
    run_benchmarks::<Quantile>(c);
}

struct Shift;

impl Benchmark for Shift {
    fn run<const CAP: usize>(name: &str, amt: usize, c: &mut Criterion) {
        let mut shift_internals = |near| {
            let (insertions, mut rng) = create_random_insertions(amt, (20231220, CAP, near));
            let mut tree = AugmentedBTree::<F32, u32, CAP>::new();
            for &(pos, count) in &insertions {
                tree.insert(pos, count);
            }
            let mut pmf = tree
                .iter()
                .scan(0, |prev, (pos, accum)| {
                    let mass = accum - *prev;
                    *prev = accum;
                    Some((pos, mass))
                })
                .collect::<Vec<_>>();

            let shifts = (0..amt)
                .map(|_| {
                    let from_index = rng.next_u32() as usize % pmf.len();
                    let (from_pos, from_mass) = &mut pmf[from_index];
                    let from_pos = *from_pos;
                    let partial_move = *from_mass != 1 && rng.next_u32() % 2 == 0;
                    let mass = if partial_move {
                        let mass = 1 + rng.next_u32() % (*from_mass - 1);
                        *from_mass -= mass;
                        mass
                    } else {
                        let mass = *from_mass;
                        pmf.remove(from_index);
                        mass
                    };

                    let move_to_existing = rng.next_u32() % 2 == 0;
                    let near_offset = (amt / 20) as i32;
                    let near_range = (2 * near_offset) as u32;
                    if move_to_existing {
                        let to_index = if near {
                            ((rng.next_u32() as u32 % near_range) as i32 + from_index as i32
                                - near_offset)
                                .clamp(0, pmf.len() as i32 - 1) as usize
                        } else {
                            rng.next_u32() as usize % pmf.len()
                        };
                        let (to_pos, to_mass) = &mut pmf[to_index];
                        *to_mass += mass;
                        (from_pos, *to_pos, mass)
                    } else {
                        // Find a random position where there isn't yet any entry in the tree.
                        let (to_pos, to_index) = loop {
                            let int_to_pos = rng.next_u32() >> 14;
                            let to_pos = F32::new(
                                from_pos.get() - 0.05
                                    + 0.1 * (int_to_pos as f32 / (u32::MAX >> 14) as f32),
                            )
                            .unwrap();
                            let to_index = pmf.partition_point(|&(pos, _)| pos < to_pos);
                            if pmf.get(to_index).map(|&(pos, _)| pos) != Some(to_pos) {
                                break (to_pos, to_index);
                            }
                        };
                        pmf.insert(to_index, (to_pos, mass));
                        (from_pos, to_pos, mass)
                    }
                })
                .collect::<Vec<_>>();

            c.bench_function(
                &format!("{}_{}_naive", name, if near { "near" } else { "far" }),
                |b| {
                    b.iter(|| {
                        for &(from_pos, to_pos, mass) in &shifts {
                            tree.shift(from_pos, to_pos, mass).unwrap();
                        }
                        for &(to_pos, from_pos, mass) in shifts.iter().rev() {
                            tree.shift(from_pos, to_pos, mass).unwrap();
                        }
                    })
                },
            );

            c.bench_function(
                &format!("{}_{}_specialized", name, if near { "near" } else { "far" }),
                |b| {
                    b.iter(|| {
                        for (i, &(from_pos, to_pos, mass)) in shifts.iter().enumerate() {
                            tree.shift(from_pos, to_pos, mass).unwrap();
                        }
                        for (i, &(to_pos, from_pos, mass)) in shifts.iter().rev().enumerate() {
                            tree.shift(from_pos, to_pos, mass).unwrap();
                        }
                    })
                },
            );
        };

        shift_internals(true);
        shift_internals(false);
    }
}

// #[test]
fn shift(c: &mut Criterion) {
    run_benchmarks::<Shift>(c);
}

fn create_random_insertions(amt: usize, h: impl Hash) -> (Vec<(F32, u32)>, impl Rng) {
    let mut hasher = DefaultHasher::new();
    h.hash(&mut hasher);
    (amt as u64).hash(&mut hasher);

    let mut rng = Xoshiro256StarStar::seed_from_u64(hasher.finish());
    let repeat_distribution = distributions::Uniform::from(1..5);
    let count_distribution = distributions::Uniform::from(1..100);

    let mut insertions = Vec::new();
    for _ in 0..amt {
        let repeats = rng.sample(repeat_distribution);
        let int_pos = rng.next_u32();
        let pos = F32::new(int_pos as f32 / u32::MAX as f32).unwrap();
        for _ in 0..repeats {
            insertions.push((pos, rng.sample(count_distribution)));
        }
    }
    insertions.shuffle(&mut rng);
    assert!(insertions.len() > 2 * amt);
    assert!(insertions.len() < 4 * amt);

    (insertions, rng)
}
