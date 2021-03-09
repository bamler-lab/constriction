# Range Coding With Lazy Emission of Unresolved Words

## Internal Coder State

### Parameterization by `lower` and `range`

- The interval covers a contiguous (albeit possibly wrapping) region of `range` numbers in
  `State` starting at `lower` (inclusive).
- The interval is nonempty; even more, we maintain the following invariant (see discussion
  below): `range >= 1 << (State::BITS - Word::BITS)`; thus
  - there is always a `point` in the interval that can be written as a single word followed
    by only zero bits (see discussion of "sealing" below); and
  - the value of `range >> PRECISION` (called `scale` below) is nonzero since
    `PRECISION <= Word::BITS <= State::BITS - Word::BITS`.

### Parameterization by `lower` and `upper`

Equivalently, we can specify the interval by `lower` and `upper` instead of `lower` and
`range`. Here, `upper = lower (+) range`, where `(+)` is wrapping addition in `State`. Thus,
`upper` is the (exclusive) upper end of the interval.

While the parameterization with `lower` and `upper` is sometimes easier to reason about, we
can implement the updates described below with fewer operations in the parameterization with
`lower` and `range` can be implemented in fewer operations. Both parameterizations are
equivalent, i.e., we can reason about the algorithm in one representation but actually
implement it in the other.

## First Part of the Update Step: Finding the Subinterval

When encoding a symbol, the encoder updates its internal state. There are two parts to this
update. The first part of the update goes as follows:

### In Parameterization with `lower` and `range`

We start by looking up the left cumulative distribution `left` and the `probability` of the
symbol that we want to encode in the entropy model. These satisfy the invariants 
`left < (1 >> PRECISION)` and `1 < probability <= (1 >> PRECISION)` (all encodable symbols
must have a nonzero probability). We then perform the following updates:

- set `scale = range >> PRECISION`;
- update `new_lower = lower (+) scale * left`
- update `new_range = scale * probability`

### In Parameterization with `lower` and `upper`

In the (equivalent) parameterization with `lower` and `upper` instead of `lower` and
`range`, the above updates induce the following update for `upper`:

```text
new_upper = new_lower (+) new_range
          = lower (+) scale * left (+) scale * probability
          = lower (+) scale * (left (+) probability)   # <-- (+) wraps in `State` ==> 2nd (+) can't wrap
          = lower (+) scale * (left + probability)
          = lower (+) scale * right
```

where `right = left + probability` is the right sided cumulative of the encoded symbol under
the entropy model. Note that all numbers here are represented in `State`, i.e., `right` can
take its extremal value of `1 << PRECISION` without wrapping.

### Lack of Surjectivity of Range Coding

Note that, even if `right == 1 << PRECISION`, it is not guaranteed that `new_upper = upper`
since `scale * right = (range >> PRECISION) << PRECISION`, which may (and typically will)
truncate. The numbers between `new_upper` (inclusive) and `upper` (exclusive) are
irrepresentable and cannot be decoded.

## Second Part of the Update Step: Flushing and Rescaling

Each one of the above update steps decreases `range` (or, at most, keeps it invariant). To
maintain the invariant `range >= 1 << (State::BITS - Word::BITS)`, we eventually have to
increase `range` again. Thus, after each of the above update steps, we do the following:

- If the invariant is violated, i.e., if `new_range < 1 << (State::BITS - Word::BITS)`:
  1. emit the most significant word of `new_lower` to the compressed bit string (see below
     for an exception);
  2. update `new_new_range = new_range << Word::BITS`; this cannot truncate.
  3. update `new_new_lower = lower << Word::BITS`; this will typically truncate, but that's
     Ok because we've already emitted the part of `new_lower` that gets truncated.
     - this step is equivalent to updating `new_new_upper = new_upper << Word::BITS`.

Step 1 above is guaranteed to restore the invariant since the original `range` satisfied the
invariant, and `new_range = (range >> PRECISION) * probability >= range >> PRECISION` since
`probability >= 1` for any valid symbol, and `PRECISION >= Word::BITS`.

## Wrapping Interval: "Inverted" Situation

A complication arises in the above rescaling if the part that gets truncated from
`new_lower` and the part that gets truncated from `new_upper` are different. In this case,
the emitted compressed word may turn out to be wrong. Therefore, we don't actually emit the
word in thic case just yet. We hold it back and emit a possibly corrected word later.

Let us denote the respective truncated parts by
`lower_word := most_significant_word(new_lower)` and
`upper_word := most_significant_word(new_upper)`. We make the following observations:

If `upper_word != lower_word` then:

- `upper_word = lower_word [+] 1` where `[+]` denotes wrapping addition in `Word`. This is
  because `new_upper = new_lower (+) new_range`, and `new_range < 1 >> (State::BITS -
  Word::BITS)`, so they can only differ due to a carry bit.
- `new_new_upper < new_new_lower` because `new_new_upper = new_new_lower (+) new_new_range`
  and
  - we know that the addition wraps around because that's what leads to the carry bit; and
  - `new_new_range < 1 << State::BITS` (because step 2 above cannot truncate), and thus the
    addition wraps around *exactly once*. Thus, we can write out the wrapping addition
    explicitly: `new_new_upper = new_new_lower + new_new_range - 1 << State::Bits` where `+`
    and `-` denote normal (non-wrapping) arithmetic. We can rewrite this as 
    `new_new_upper = new_new_lower - (1 << State::BITS - new_new_range)` where the term in
    the brackets is strictly positive since `new_new_range < 1 << State::BITS`.

We refer to coder states with `upper < lower` as an *inverted situation* (using the word
"situation" in order to avoid overloading the term "state"). We refer to the opposite
situation `upper > lower` as *normal*.

Note that we left out the cas `upper == lower`. It turns out that rescaling can never lead
to this case. Having `new_new_upper == new_new_lower` after rescaling would mean that
`new_new_range = new_new_upper - new_new_lower` is zero modulo `1 << State::BITS`, which is
not possible because `new_new_range >= 1 << (State::Bits - Word::BITS) > 1` since rescaling
restores the invariant, but also `new_new_range = new_range << Word::BITS`, which is
striclty smaller than `(1 << (State::BITS - Word::BITS)) << Word::BITS = 1 << State::BITS`
since otherwise we wouldn't have rescaled.

Thus, rescaling cannot lead to a state with `upper == lower`. Neither can the first part of
the update step where we find the subinterval, since it satisfies `new_range <= range`. So
as long as we don't initialize the coder with `upper == lower`, such a state can never be
reached.

## State Transitions

The following state transitions are possible in the first and second part of the update
step, spectively:

```text
               Part 1:                 Part 2: rescaling
          find subinterval            (only on underflow)
  normal ------------------>  normal  -----------------+->  normal
                           ↗                            \
                          /                              ↘
inverted ----------------+-> inverted -------------------> inverted 
```

In detail:

- **In the first part of the update step** (finding the subinterval), the following
  transitions are possible:
  - transitions from a normal to a normal situation are (trivially) possible;
  - transitions from an inverted to a normal situation are possible if either
    `new_lower = lower (+) scale * left` wraps or if `new_upper = lower (+) scale * right`
    doesn't wrap, both of which are possible; and
  - transitions from inverted to inverted are possible if
    `new_lower = lower (+) scale * left` does not wrap and if at the same time
    `new_upper = lower (+) scale * right` does wrap (which is possible when coming from an
    inverted situation).
  - However, the first part of the update step can't transition from a normal to an inverted
    situation. If we start from a normal situation, i.e., `upper >= lower`, then
    `new_lower = lower (+) scale * left` and `new_upper = lower (+) scale * right` are both
    within the (non-wrapping) interval `[lower, upper]` since `scale = range >> PRECISION`
    and thus `scale * p <= range` for any probability `p <= 1 << PRECISION` (such as
    `p = left` or `p = right`), and thus `lower (+) scale * p` doesn't wrap and is
    `>= lower` and `<= lower + range == upper`. Further, since `left < right` and
    `scale > 0`, we have `scale * left < scale * right` and thus `new_lower < new_upper`.
- **In the second part of the update step** (rescaling), the following transitions are
  possible:
  - transitions from a normal to a normal situation are possible;
  - transitions from a normal to an inverted situation are possible as discussed above; and
  - transitions from an inverted situation to an inverted situation are possible as
    discussed below.
  - However, the rescaling step cannot transition from an inverted situation to a normal
    situation. If we start with an inverted situation, i.e., with `new_upper < new_lower`,
    then the addition in `new_upper = new_lower (+) new_range` wraps around. The step from
    `new_range` to `new_new_range` doesn't truncate, so the addition
    `new_new_lower (+) new_new_range` also has to wrap, meaning that the new situation is
    also inverted.


### Transitioning from an Inverted Situation During Rescaling

Rescaling operations that already start with an inverted situation deserve special
attention. As discussed above, these transitions can only end up in onother inverted
situation. Interestingly, these transitions always have `upper_word == 0` and
`lower_word == Word::max_value()`.

This is because the addition in `new_upper = new_lower (+) new_range` wraps in an inverted
situation, which means that `new_lower < new_range`. Further, we have
`new_range < 1 << (State::BITS - Word::BITS)` as otherwise we wouldn't be rescaling. Thus,
`new_lower < 1 << (State::BITS - Word::Bits)` and thus `new_lower` has only zeros in its
most significant word, i.e., `lower_word = most_significant_word(new_lower) == 0`. As
discussed above, `upper_word = lower_word [-] 1`, which therefore wraps around and contains
only one bits.

Since all transitions from an inverted to an inverted situation have the same `lower_word`
and `upper_word`, we do not need to keep track of them individually. We can just count how
many such transitions we have in sequence. As soon as we transition out of the inverted
situation (either during the first part of the update step or when we're done with
encoding), we flush the words withhold in these transitions as described below.

We thus keep a counter `num_inverted` which is zero in a normal situation and nonzero in an
inverted situation. We further maintain a slot in memory for a single word
`first_inverted_lower_word`, which is the `lower_word` when the first inversion in the
current row of inversions happened. The value of `lower_word` has no meaning in a normal
situation, so it's probably better to keep track of both in an `enum`:

```rust
enum Situation { 
    Normal,
    Inverted(NonZeroUsize, Word) // `num_inverted` and `first_inverted_lower_word`
}
```

## Transitioning out of an Inverted Situation

As per the above diagram, we can only transition out of an inverted situation and back into
a normal situation during part one of the update step (another such transition happens when
encoding finishes while in an inverted step, see discussion below).

Assume we start from an inverted situation, i.e., `upper < lower`. The first part of the
update step defines a new subinterval by setting

- `new_lower = lower (+) scale * left`
- `new_upper = lower (+) scale * right`

One of three things can happen:

1. The addition `lower (+) scale * left` does not wrap but the addition in
   `lower (+) scale * right` does wrap; or
2. one of the above two additions wrap; or
3. both of the above two additions wrap.

(The fourth combination where the addition wraps for `scale * left` but not for
`scale * right` is impossible since `left < right`.)

In case 1 above, we're transitioning from an inverted into an inverted situation, so we
don't do anything special here. If `range` underflows and we have to rescale after this
update, then rescaling will *not* emit any word yet and instead increase the `num_inverted`
counter (which must already have been nonzero because we were already in an inverted
situation).

In cases 2 and 3 above, we're transitioning from an inverted to a normal situation. This
resolves the values of all words we've held back from emitting. In case 2, they should have
been the `lower_words` and in case 3, they should have been the `upper` words. Thus, we can
emit them now and then set the `num_words` counter to zero:

- in case 2 above: emit `first_inverted_lower_word`, followed by `num_inverted - 1` copies
  of `Word::max_value()`, then set `num_inverted` to zero;
- in case 3 above: emit `first_inverted_lower_word [+] 1` (which actually can't wrap in this
  case), followed by `num_inverted - 1` copies of `0`, then set `num_inverted` to zero.


## Putting it All Together

The following diagram summarizes the operations we do in each one of the possible
transitons.

```text
               Part 1:                 Part 2: rescaling
          find subinterval            (only on underflow)

           [no special op]                [emit word]
  normal ------------------>  normal  -------------+---->  normal
                           ↗                        \
                          /[emit held-back           \[don't emit word, set
                         / words and set              \ first_inverted_lower_word
                        / num_inverted=0]              \ to lower_word & num_inverted=1]
                       /                                ↘
inverted -------------+----> inverted ------------------> inverted 
           [no special op]             [don't emit word
                                        but increment
                                        num_inverted]
```

## Finishing Up (Sealing the Coder)

When encoding finishes, we have to make sure that we have emitted enough words to uniquely
identify the current interval. The compressed bit string may represent any number in the
half open and possibly wrapping interval from `lower` (inclusively) to `upper`
(exclusively). For simplicity, we chose a value within this interval that has only zero bits
in all but the most significant word. This is always possible due to the invariant
`range >= 1 << (State::BITS - Word::BITS)`. Therefore, we have to emit only the most
signifciant word of this value (in addition to any possibly still held-back words).

To find such a word, we take the value `point := mask(upper (-) 1)` where `mask()` sets all
bits that are not in the most significant word to zero. This can be written as
`point = upper (-) x` where `x > 0` and `x <= 1 << (State::BITS - Word::BITS)` which is
`<= range` by our invariant. Thus, `point = upper (-) x` is at least one and at most `range`
below `upper`, i.e., it is in the interval `[lower, upper)` as required.

If we're in an inverted situation, then we first flush all not yet emitted words. If
`point < lower` (the likely case in an inverted situation) then we emit
`first_inverted_lower_word [+] 1` followed by `num_inverted - 1` zeros. If `point >= lower`
(which can only happen if `upper == 0`) then we emit `first_inverted_lower_word` followed by
`num_inverted  - 1` copies of `Word::max_value()`.

Finally regardless of whether we were in a normal or inverted situation, we emit the highest
significant word of `point`. This concludes the compressed bit string.
