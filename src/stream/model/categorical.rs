pub mod contiguous;
pub mod lazy_contiguous;
pub mod lookup_contiguous;
pub mod lookup_noncontiguous;
pub mod non_contiguous;

use core::borrow::Borrow;

use alloc::vec::Vec;

use libm::log1p;
use num_traits::{float::FloatCore, AsPrimitive};

use crate::{generic_static_asserts, wrapping_pow2, BitArray};

fn fast_quantized_cdf<Probability, F, const PRECISION: usize>(
    probabilities: &[F],
    normalization: Option<F>,
) -> Result<impl ExactSizeIterator<Item = Probability> + '_, ()>
where
    F: FloatCore + core::iter::Sum<F> + AsPrimitive<Probability>,
    Probability: BitArray + AsPrimitive<usize>,
    usize: AsPrimitive<Probability> + AsPrimitive<F>,
{
    generic_static_asserts!(
        (Probability: BitArray; const PRECISION: usize);
        PROBABILITY_MUST_SUPPORT_PRECISION: PRECISION <= Probability::BITS;
        PRECISION_MUST_BE_NONZERO: PRECISION > 0;
    );

    if probabilities.len() < 2
        || probabilities.len() >= wrapping_pow2::<usize>(PRECISION).wrapping_sub(1)
    {
        return Err(());
    }

    let free_weight =
        wrapping_pow2::<Probability>(PRECISION).wrapping_sub(&probabilities.len().as_());
    let normalization = normalization.unwrap_or_else(|| probabilities.iter().copied().sum::<F>());
    if !normalization.is_normal() || !normalization.is_sign_positive() {
        return Err(());
    }
    let scale = AsPrimitive::<F>::as_(free_weight.as_()) / normalization;

    let mut cumulative_float = F::zero();
    let mut accumulated_slack = Probability::zero();

    Ok(probabilities.iter().map(move |probability_float| {
        let left_cumulative = (cumulative_float * scale).as_() + accumulated_slack;
        cumulative_float = cumulative_float + *probability_float;
        accumulated_slack = accumulated_slack.wrapping_add(&Probability::one());
        left_cumulative
    }))
}

fn perfectly_quantized_probabilities<Probability, F, const PRECISION: usize>(
    probabilities: &[F],
) -> Result<Vec<Slot<Probability>>, ()>
where
    F: FloatCore + core::iter::Sum<F> + Into<f64>,
    Probability: BitArray + Into<f64> + AsPrimitive<usize>,
    f64: AsPrimitive<Probability>,
    usize: AsPrimitive<Probability>,
{
    generic_static_asserts!(
        (Probability: BitArray; const PRECISION: usize);
        PROBABILITY_MUST_SUPPORT_PRECISION: PRECISION <= Probability::BITS;
        PRECISION_MUST_BE_NONZERO: PRECISION > 0;
    );

    if probabilities.len() < 2 || probabilities.len() > Probability::max_value().as_() {
        return Err(());
    }

    // Start by assigning each symbol weight 1 and then distributing no more than
    // the remaining weight approximately evenly across all symbols.
    let mut remaining_free_weight =
        wrapping_pow2::<Probability>(PRECISION).wrapping_sub(&probabilities.len().as_());
    let normalization = probabilities.iter().map(|&x| x.into()).sum::<f64>();
    if !normalization.is_normal() || !normalization.is_sign_positive() {
        return Err(());
    }
    let scale = remaining_free_weight.into() / normalization;

    let mut slots = probabilities
        .iter()
        .enumerate()
        .map(|(original_index, &prob)| {
            if prob < F::zero() {
                return Err(());
            }
            let prob: f64 = prob.into();
            let current_free_weight = (prob * scale).as_();
            remaining_free_weight = remaining_free_weight - current_free_weight;
            let weight = current_free_weight + Probability::one();

            // How much the cross entropy would decrease when increasing the weight by one.
            let win = prob * log1p(1.0f64 / weight.into());

            // How much the cross entropy would increase when decreasing the weight by one.
            let loss = if weight == Probability::one() {
                f64::infinity()
            } else {
                -prob * log1p(-1.0f64 / weight.into())
            };

            Ok(Slot {
                original_index,
                prob,
                weight,
                win,
                loss,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Distribute remaining weight evenly among symbols with highest wins.
    while remaining_free_weight != Probability::zero() {
        // We can't use `sort_unstable_by` here because we want the result to be reproducible
        // even across updates of the standard library.
        slots.sort_by(|a, b| b.win.partial_cmp(&a.win).unwrap());
        let batch_size = core::cmp::min(remaining_free_weight.as_(), slots.len());
        for slot in &mut slots[..batch_size] {
            slot.weight = slot.weight + Probability::one(); // Cannot end up in `max_weight` because win would otherwise be -infinity.
            slot.win = slot.prob * log1p(1.0f64 / slot.weight.into());
            slot.loss = -slot.prob * log1p(-1.0f64 / slot.weight.into());
        }
        remaining_free_weight = remaining_free_weight - batch_size.as_();
    }

    loop {
        // Find slot where increasing its weight by one would incur the biggest win.
        let (buyer_index, &Slot { win: buyer_win, .. }) = slots
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.win.partial_cmp(&b.win).unwrap())
            .unwrap();
        // Find slot where decreasing its weight by one would incur the smallest loss.
        let (seller_index, seller) = slots
            .iter_mut()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.loss.partial_cmp(&b.loss).unwrap())
            .unwrap();

        if buyer_index == seller_index {
            // This can only happen due to rounding errors. In this case, we can't expect
            // to be able to improve further.
            break;
        }

        if buyer_win <= seller.loss {
            // We've found the optimal solution.
            break;
        }

        // Setting `seller.win = -infinity` and `buyer.loss = infinity` below ensures that the
        // iteration converges even in the presence of rounding errors because each weight can
        // only be continuously increased or continuously decreased, and the range of allowed
        // weights is bounded from both above and below. See unit test `categorical_converges`.
        seller.weight = seller.weight - Probability::one();
        seller.win = f64::neg_infinity(); // Once a weight gets reduced it may never be increased again.
        seller.loss = if seller.weight == Probability::one() {
            f64::infinity()
        } else {
            -seller.prob * log1p(-1.0f64 / seller.weight.into())
        };

        let buyer = &mut slots[buyer_index];
        buyer.weight = buyer.weight + Probability::one();
        buyer.loss = f64::infinity(); // Once a weight gets increased it may never be decreased again.
        buyer.win = buyer.prob * log1p(1.0f64 / buyer.weight.into());
    }

    slots.sort_unstable_by_key(|slot| slot.original_index);
    Ok(slots)
}

struct Slot<Probability> {
    original_index: usize,
    prob: f64,
    weight: Probability,
    win: f64,
    loss: f64,
}

fn iter_extended_cdf<I, Symbol, Probability>(
    mut cdf: I,
) -> impl Iterator<Item = (Symbol, Probability, Probability::NonZero)>
where
    I: Iterator<Item = (Probability, Symbol)>,
    Symbol: Clone,
    Probability: BitArray,
{
    let (mut left_cumulative, mut symbol) = cdf.next().expect("cdf is not empty").clone();

    cdf.map(move |(right_cumulative, next_symbol)| {
        let old_left_cumulative = left_cumulative;
        let old_symbol = core::mem::replace(&mut symbol, next_symbol.clone());
        left_cumulative = right_cumulative;
        let probability = right_cumulative
            .wrapping_sub(&old_left_cumulative)
            .into_nonzero()
            .expect("quantization is leaky");
        (old_symbol, old_left_cumulative, probability)
    })
}

/// Note: does not check if `symbols` is exhausted (this is so that you one can provide an
/// infinite iterator for `symbols` to optimize out the bounds check on it).
fn accumulate_nonzero_probabilities<Symbol, Probability, S, P, Op, const PRECISION: usize>(
    mut symbols: S,
    probabilities: P,
    mut operation: Op,
    infer_last_probability: bool,
) -> Result<S, ()>
where
    Probability: BitArray,
    S: Iterator<Item = Symbol>,
    P: Iterator,
    P::Item: Borrow<Probability>,
    Op: FnMut(Symbol, Probability, Probability) -> Result<(), ()>,
{
    generic_static_asserts!(
        (Probability: BitArray; const PRECISION: usize);
        PROBABILITY_MUST_SUPPORT_PRECISION: PRECISION <= Probability::BITS;
        PRECISION_MUST_BE_NONZERO: PRECISION > 0;
    );

    // We accumulate all validity checks into single branches at the end in order to
    // keep the loop itself branchless.
    let mut laps_or_zeros = 0usize;
    let mut accum = Probability::zero();

    for probability in probabilities {
        let old_accum = accum;
        accum = accum.wrapping_add(probability.borrow());
        laps_or_zeros += (accum <= old_accum) as usize;
        let symbol = symbols.next().ok_or(())?;
        operation(symbol, old_accum, *probability.borrow())?;
    }

    let total = wrapping_pow2::<Probability>(PRECISION);

    if infer_last_probability {
        if accum >= total || laps_or_zeros != 0 {
            return Err(());
        }
        let symbol = symbols.next().ok_or(())?;
        let probability = total.wrapping_sub(&accum);
        operation(symbol, accum, probability)?;
    } else if accum != total || laps_or_zeros != (PRECISION == Probability::BITS) as usize {
        return Err(());
    }

    Ok(symbols)
}
