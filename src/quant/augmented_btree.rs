use core::{
    fmt::{Debug, Display},
    ops::{Add, Deref, DerefMut, Sub},
};

use self::{
    bounded_vec::{BoundedPairOfVecs, BoundedVec},
    child_ptr::ChildPtr,
};
use NodeType::{Leaf, NonLeaf};

use crate::generic_static_asserts;

pub struct AugmentedBTree<P, C, const CAP: usize> {
    total: C,
    root_type: NodeType,
    root: ChildPtr<P, C, CAP>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeType {
    NonLeaf,
    Leaf,
}

struct NonLeafNode<P, C, const CAP: usize> {
    children_type: NodeType,
    first_child: ChildPtr<P, C, CAP>,

    /// Upholds the following invariant:
    /// - if it is the root, then `separated_childern.len() >= 2`
    /// - if it is not the root, then `separated_children.len() >= CAP / 2 >= 2`
    separated_children: BoundedPairOfVecs<Entry<P, C>, ChildPtr<P, C, CAP>, CAP>,
}

struct LeafNode<P, C, const CAP: usize> {
    /// Satisfies `entires.len() >= CAP / 2 >= 2` unless it is the root node
    /// (the root node may have arbitrarily few entries, including zero).
    entries: BoundedVec<Entry<P, C>, CAP>,
}

#[derive(Clone, Copy)]
struct Entry<P, C> {
    pos: P,

    /// Sum of counts within the current subTree from the first entry up until
    /// and including this entry. Thus the total weight of a leaf node is stored
    /// in the `accum` field of the leaf node's last entry.
    accum: C,
}

impl<P, C, const CAP: usize> Drop for AugmentedBTree<P, C, CAP> {
    fn drop(&mut self) {
        unsafe {
            match self.root_type {
                NonLeaf => self.root.drop_non_leaf(),
                Leaf => self.root.drop_leaf(),
            }
        }
    }
}

impl<P, C, const CAP: usize> Drop for NonLeafNode<P, C, CAP> {
    fn drop(&mut self) {
        unsafe {
            match self.children_type {
                NonLeaf => {
                    self.first_child.drop_non_leaf();
                    while let Some((_, mut child)) = self.separated_children.pop() {
                        child.drop_non_leaf()
                    }
                }
                Leaf => {
                    self.first_child.drop_leaf();
                    while let Some((_, mut child)) = self.separated_children.pop() {
                        child.drop_leaf()
                    }
                }
            }
        }
    }
}

impl<P, C, const CAP: usize> AugmentedBTree<P, C, CAP>
where
    P: Ord + Copy,
    C: Ord + Default + Copy + Add<Output = C> + Sub<Output = C>,
{
    pub fn new() -> Self {
        generic_static_asserts!(
            (; const CAP: usize);
            CAP_MUST_BE_AT_LEAST4: CAP >= 4;
        );

        Self {
            total: C::default(),
            root_type: Leaf,
            root: ChildPtr::leaf(LeafNode::new()),
        }
    }

    pub fn total(&self) -> C {
        self.total
    }

    pub fn insert(&mut self, pos: P, count: C) {
        if count == C::default() {
            return;
        }

        self.total = self.total + count;

        let Some((separator_pos, ejected_accum, right_sibling)) = (match self.root_type {
            NonLeaf => unsafe { self.root.as_non_leaf_mut_unchecked() }.insert(pos, count),
            Leaf => unsafe { self.root.as_leaf_mut_unchecked() }.insert(pos, count),
        }) else {
            return;
        };
        // The root node overflowed, and we had to split off a `right_sibling` from it. The
        // splitting operation ejected a separator at position `separator_pos`, and `ejected_accum`
        // is the weight of the left split + separator.

        // Since we cannot move out of `self.root`, we have to first construct a new root with a
        // temporary dummy `first_child`, then replace the old root by the new root, and then
        // replace the new root's dummy `first_child` with the old root.
        let new_root = ChildPtr::non_leaf(NonLeafNode::new(
            self.root_type,
            right_sibling,
            BoundedPairOfVecs::new(),
        ));
        let old_root = core::mem::replace(&mut self.root, new_root);
        let new_root = unsafe { self.root.as_non_leaf_mut_unchecked() };
        let right_sibling = core::mem::replace(&mut new_root.first_child, old_root);
        let separator = Entry {
            pos: separator_pos,
            accum: ejected_accum,
        };
        new_root
            .separated_children
            .try_push(separator, right_sibling)
            .expect("vector is empty and `CAP>0`");
        self.root_type = NonLeaf;
    }

    pub fn remove(&mut self, pos: P, count: C) -> Result<(), ()> {
        if count == C::default() {
            return Ok(());
        }

        match self.root_type {
            NonLeaf => unsafe { self.root.as_non_leaf_mut_unchecked() }.remove(pos, count),
            Leaf => todo!(),
            // Leaf => unsafe { self.root.as_leaf_mut_unchecked() }.remove(pos, count),
        }
    }

    /// Returns the left-sided CDF.
    ///
    /// This is the sum of all counts strictly left of `pos`.
    pub fn left_cumulative(&self, pos: P) -> C {
        // This is implemented non-recursively as a kind of manual tail call optimization, which
        // is possible because we only have to walk down the tree, not up.
        let mut accum = C::default();

        let mut node_ref = &self.root;
        let mut node_type = self.root_type;
        while node_type == NonLeaf {
            let node = unsafe { node_ref.as_non_leaf_unchecked() };
            let separators = node.separated_children.first_as_ref();
            let index = separators.partition_point(|entry| entry.pos < pos);
            if let Some(entry) = separators.get(index.wrapping_sub(1)) {
                accum = accum + entry.accum
            }
            node_type = node.children_type;
            node_ref = node
                .separated_children
                .second_as_ref()
                .get(index.wrapping_sub(1))
                .unwrap_or(&node.first_child);
        }
        let leaf_node = unsafe { node_ref.as_leaf_unchecked() };

        let index = leaf_node.entries.partition_point(|entry| entry.pos < pos);
        accum = accum
            + leaf_node
                .entries
                .get(index.wrapping_sub(1))
                .map(|entry| entry.accum)
                .unwrap_or_default();
        accum
    }

    /// Returns the quantile function (aka percent point function or inverse CDF).
    ///
    /// More precisely, the returned value is the right-sided inverse of the left-sided CDF, i.e.,
    /// it is the right-most position where the left-sided CDF is smaller than or equal to the
    /// argument `accum`. Returns `None` if `accum >= tree.total()` (in this case, the left-sided
    /// CDF is smaller than or equal to `accum` everywhere, so there is no *single* right-most
    /// position that satisfies this criterion).
    ///
    /// The following two relations hold (where `tree` is an `AugmentedBTree`):
    ///
    /// - `tree.left_cumulative(tree.quantile_function(tree.left_cumulative(pos)).unwrap())` is
    ///   equal to `tree.left_cumulative(pos)` (assuming that `unwrap()` succeeds).
    /// - `tree.quantile_function(tree.left_cumulative(tree.quantile_function(accum).unwrap())).unwrap())`
    ///   is equal to `tree.quantile_function(accum).unwrap()` (assuming that the inner `unwrap()`
    ///   succeeds—in which case the outer `unwrap()` is guaranteed to succeed).
    pub fn quantile_function(&self, accum: C) -> Option<P> {
        // This is implemented non-recursively as a kind of manual tail call optimization, which
        // is possible because we only have to walk down the tree, not up.
        // Since `Entry::accum` stores the *right-sided* CDF, we have to find the
        // first entry whose accum is strictly larger than the provided accum.
        let mut remaining = accum;
        let mut node_ref = &self.root;
        let mut node_type = self.root_type;
        let mut right_bound = None;
        while node_type == NonLeaf {
            let node = unsafe { node_ref.as_non_leaf_unchecked() };
            let separators = node.separated_children.first_as_ref();
            let index = separators.partition_point(|entry| entry.accum <= remaining);
            if let Some(right_separator) = separators.get(index) {
                right_bound = Some(right_separator.pos);
            }
            if let Some(left_separator) = separators.get(index.wrapping_sub(1)) {
                remaining = remaining - left_separator.accum;
            }
            node_type = node.children_type;
            node_ref = node
                .separated_children
                .second_as_ref()
                .get(index.wrapping_sub(1))
                .unwrap_or(&node.first_child);
        }
        let leaf_node = unsafe { node_ref.as_leaf_unchecked() };

        let index = leaf_node
            .entries
            .partition_point(|entry| entry.accum <= remaining);
        leaf_node
            .entries
            .get(index)
            .map(|entry| entry.pos)
            .or(right_bound)
    }
}

impl<P, C, const CAP: usize> NonLeafNode<P, C, CAP>
where
    P: Ord + Copy,
    C: Default + Ord + Add<Output = C> + Sub<Output = C> + Copy,
{
    fn new(
        children_type: NodeType,
        first_child: ChildPtr<P, C, CAP>,
        separated_children: BoundedPairOfVecs<Entry<P, C>, ChildPtr<P, C, CAP>, CAP>,
    ) -> Self {
        Self {
            children_type,
            first_child,
            separated_children,
        }
    }

    /// Destructs the node into its `first_child` and `separated_children`.
    ///
    /// Note that dropping these before sticking them into a different `NonLeafNode` would leak
    /// the memory allocated for the children.
    fn leak_raw_parts(
        self,
    ) -> (
        ChildPtr<P, C, CAP>,
        BoundedPairOfVecs<Entry<P, C>, ChildPtr<P, C, CAP>, CAP>,
    ) {
        unsafe {
            let first_child = core::ptr::read(&self.first_child);
            let separated_children = core::ptr::read(&self.separated_children);
            core::mem::forget(self);
            (first_child, separated_children)
        }
    }

    fn child_by_key_mut<X: Ord>(
        &mut self,
        key: X,
        get_key: impl Fn(&Entry<P, C>) -> X,
        update_right: impl Fn(&mut Entry<P, C>),
    ) -> Option<(usize, &mut ChildPtr<P, C, CAP>, NodeType)> {
        let separators: &mut [Entry<P, C>] = self.separated_children.first_as_mut();
        let index = separators.partition_point(|entry| get_key(entry) < key);
        let mut right_iter: core::iter::Skip<core::slice::IterMut<'_, Entry<P, C>>> =
            separators.iter_mut().skip(index);

        if let Some(right_separator) = right_iter.next() {
            let right_key = get_key(right_separator);
            update_right(right_separator);
            for entry in right_iter {
                update_right(entry)
            }
            if right_key == key {
                return None;
            }
        }

        let child_ref = self
            .separated_children
            .second_as_mut()
            .get_mut(index.wrapping_sub(1))
            .unwrap_or(&mut self.first_child);

        Some((index, child_ref, self.children_type))
    }

    fn insert(&mut self, pos: P, count: C) -> Option<(P, C, ChildPtr<P, C, CAP>)> {
        let Some((insert_index, child, child_type)) = self.child_by_key_mut(
            pos,
            |entry| entry.pos,
            |entry| entry.accum = entry.accum + count,
        ) else {
            // The non-leaf node already had an entry at position `pos`, and we've
            // already incremented the accums, so there's nothing else to do.
            return None;
        };

        let Some((separator_pos, ejected_accum, new_child)) = (match child_type {
            NonLeaf => unsafe { child.as_non_leaf_mut_unchecked() }.insert(pos, count),
            Leaf => unsafe { child.as_leaf_mut_unchecked() }.insert(pos, count),
        }) else {
            // Inserting into the subtree rooted at `child` succeeded without overflowing `child`.
            return None;
        };

        // The child node overflowed and was split into two, where `new_child` should become the new
        // neighbor of `child` to the right. The splitting operation ejected a separator at position
        // `separator_pos`, and `ejected_accum` is the weight of the left split + separator.
        let preceding_accum = self
            .separated_children
            .first_as_ref()
            .get(insert_index.wrapping_sub(1))
            .map(|entry| entry.accum)
            .unwrap_or_default();

        let mut separator = Entry {
            pos: separator_pos,
            accum: preceding_accum + ejected_accum,
        };
        let Err((_, new_child)) =
            self.separated_children
                .try_insert(insert_index, separator, new_child)
        else {
            return None;
        };

        // Inserting would overflow the node. Split it into two.
        let mut right_separated_children = BoundedPairOfVecs::new();
        if insert_index <= CAP / 2 {
            // Insert into the left sibling (i.e., `self`).
            let accum_of_ejected = if insert_index == CAP / 2 {
                separator.accum
            } else {
                self.separated_children
                    .first_as_ref()
                    .get(CAP / 2 - 1)
                    .expect("CAP / 2 > index >= 0")
                    .accum
            };
            let chopped_off = self.separated_children.chop(CAP / 2).expect("node is full");
            right_separated_children
                .try_append_transform1(chopped_off, |entry| Entry {
                    pos: entry.pos,
                    accum: entry.accum - accum_of_ejected,
                })
                .expect("can't overflow");
            self.separated_children
                .try_insert(insert_index, separator, new_child)
                .ok()
                .expect("there are `CAP - CAP/2` vacancies, which is >0 because CAP>0.");
        } else {
            // Insert into `right_sibling`.
            let insert_index = insert_index - (CAP / 2 + 1);
            let accum_of_ejected = self
                .separated_children
                .first_as_ref()
                .get(CAP / 2)
                .expect("CAP/2 < original insert_index <= len")
                .accum;

            // Build `right_sibling`'s list of separators.
            let chopped_off = self
                .separated_children
                .chop(CAP / 2 + 1)
                .expect("CAP/2 < original insert_index <= len");
            let (before_insert, after_insert) = chopped_off.split_at_mut(insert_index);
            right_separated_children
                .try_append_transform1(before_insert, |entry| Entry {
                    pos: entry.pos,
                    accum: entry.accum - accum_of_ejected,
                })
                .expect("can't overflow");
            separator.accum = separator.accum - accum_of_ejected;
            right_separated_children
                .try_push(separator, new_child)
                .expect("can't overflow");
            right_separated_children
                .try_append_transform1(after_insert, |entry| Entry {
                    pos: entry.pos,
                    accum: entry.accum - accum_of_ejected,
                })
                .expect("can't overflow");
        };

        let (ejected_separator, right_first_child) = self
            .separated_children
            .pop()
            .expect("there are CAP/2+1 > 0 entries");

        let right_sibling = ChildPtr::non_leaf(NonLeafNode::new(
            self.children_type,
            right_first_child,
            right_separated_children,
        ));

        Some((
            ejected_separator.pos,
            ejected_separator.accum,
            right_sibling,
        ))
    }

    /// Tries to remove `count` items at position `pos`. Returns `Err(())` if there are fewer than
    /// `count` items at `pos`. in this case, the subtree rooted at this node has been restored
    /// into its original state at the time the method returns.
    fn remove(&mut self, pos: P, count: C) -> Result<(), ()> {
        let separators: &mut [Entry<P, C>] = self.separated_children.first_as_mut();
        let index = separators.partition_point(|entry| entry.pos < pos);
        let mut right_iter = separators.iter_mut().skip(index);

        if let Some(right_separator) = right_iter.next() {
            if right_separator.accum < count {
                return Err(());
            }
            let right_pos = right_separator.pos;
            right_separator.accum = right_separator.accum - count;
            for entry in right_iter {
                entry.accum = entry.accum - count;
            }
            if right_pos == pos {
                // TODO: remove entry if its accum is zero
                return Ok(());
            }
        }

        let child = self
            .separated_children
            .second_as_mut()
            .get_mut(index.wrapping_sub(1))
            .unwrap_or(&mut self.first_child);

        match self.children_type {
            NonLeaf => {
                let child = unsafe { child.as_non_leaf_mut_unchecked() };
                if child.remove(pos, count).is_ok() {
                    if child.separated_children.len() < CAP / 2 {
                        // Child node underflowed. Check if we can steal some entries from one of
                        // its neighbors. If not, merge three children into two.

                        let (mut left, mut right) =
                            self.separated_children.get_all_mut().split_at_mut(index);
                        let mut left_iter = left.iter_mut().rev();
                        let (child, left, left_neighbor_len) =
                            if let Some((left_separator, child)) = left_iter.next() {
                                let left_neighbor = unsafe {
                                    left_iter
                                        .next()
                                        .map(|(_, c)| c)
                                        .unwrap_or(&mut self.first_child)
                                        .as_non_leaf_mut_unchecked()
                                };
                                let left_neighbor_len = left_neighbor.separated_children.len();
                                (
                                    unsafe { child.as_non_leaf_mut_unchecked() },
                                    Some((left_neighbor, left_separator)),
                                    left_neighbor_len,
                                )
                            } else {
                                (
                                    unsafe { self.first_child.as_non_leaf_mut_unchecked() },
                                    None,
                                    0,
                                )
                            };
                        core::mem::drop(left_iter);
                        let right = right
                            .iter_mut()
                            .next()
                            .map(|(e, c)| (e, unsafe { c.as_non_leaf_mut_unchecked() }));
                        let right_neighbor_len = right
                            .as_ref()
                            .map(|(_, c)| c.separated_children.len())
                            .unwrap_or_default();

                        if left_neighbor_len > right_neighbor_len {
                            if left_neighbor_len > CAP / 2 {
                                // Steal entries from the end of `left_neighbor`.
                                let (left_neighbor, mut separator) =
                                    left.expect("`left_neighbor_len > 0`, so it exists");
                                let new_left_neighbor_len =
                                    CAP / 2 + (left_neighbor_len - CAP / 2) / 2;
                                let mut stolen = left_neighbor
                                    .separated_children
                                    .chop(new_left_neighbor_len)
                                    .expect("`new_left_neighbor_len < left_neighbor`");
                                core::mem::swap(&mut stolen.first_as_mut()[0], &mut separator);
                                core::mem::swap(
                                    &mut stolen.second_as_mut()[0],
                                    &mut child.first_child,
                                );
                                child
                                    .separated_children
                                    .try_prepend(stolen)
                                    .expect("can't overflow");
                            }
                        } else if right_neighbor_len > CAP / 2 {
                            // Steal entries from the beginning of `right_neighbor``.
                            let (separator, right_neighbor) =
                                right.expect("`right_neighbor_len > 0`, so it exists.");
                            let new_right_neighbor_len =
                                CAP / 2 + (right_neighbor_len - CAP / 2) / 2;
                            let mut stolen = right_neighbor
                                .separated_children
                                .chop_front(right_neighbor_len - new_right_neighbor_len)
                                .expect("`new_right_neighbor_len < right_neighbor_len`");
                            let (last_stolen_entry, last_stolen_grandchild) =
                                stolen.pop().expect("we stole at least one.");
                            let old_separator = core::mem::replace(separator, last_stolen_entry);
                            let old_first_child = core::mem::replace(
                                &mut right_neighbor.first_child,
                                last_stolen_grandchild,
                            );
                            child
                                .separated_children
                                .try_push(old_separator, old_first_child)
                                .expect("can't overflow");
                            child
                                .separated_children
                                .try_append_guarded(stolen)
                                .expect("can't overflow");
                        } else {
                            // Can't steal any entries from neighbors. We need to reduce the number
                            // of child nodes. Let `i` be the index of the node that underflowed.
                            // - if `i` has neighbors to both sides, then we merge the three nodes
                            //   `[i - 1, i, i + 1]` into two;
                            // - if `i` is the first or last child, then we instead merge the three
                            //   nodes `[i, i + 1, i + 2]` or `[i - 2, i - 1, i]`, respectively,
                            //   into two, if they all three exist;
                            // - if there are only two children (which is the minimum possible
                            //   number of children because the root node has at least two children,
                            //   and all non-root nodes have at least `CAP / 2 >= 2` children
                            //   because `CAP >= 4`), then we merge these two children (i.e., either
                            //   `[i, i + 1]` or `[i - 1, i]` into one).
                            //
                            // Thus in general, we merge 2-3 children `[A, B, C?]` (where `C` may or
                            // may not exist). Our strategy is to remove node `B` and split up its
                            // entries (+ separators) across `A` and `C` (if existent) (+ separator)
                            // such that `A` and `C` have equally many entries (up to 1 if the total
                            // number is odd). This always works without moving any entries between
                            // `A` and `C` because, in all cases considered above, either the set
                            // `{A, B}` or the set `{B, C}` (or both) consist of node `i`, which has
                            // exactly `CAP / 2 - 1` entries (because it just underflowed), and one
                            // of its neighbors, which has exactly `CAP / 2` entries (because we
                            // couldn't steal from it any more). Therefore, even if we merged all
                            // entries of these two nodes and the separator into one node, we would
                            // end up with a merged node with `2 * (CAP / 2) - 1 + 1 ∈ {CAP - 1, CAP}`
                            // entries, and the other node, which has at least `CAP / 2` and at most
                            // `CAP` entries.

                            let index_b = index.clamp(1, self.separated_children.len()) - 1;
                            let (separator_ab, child_b) = self
                                .separated_children
                                .remove(index_b)
                                .expect("`index_b < len`");
                            let child_b = unsafe { child_b.into_non_leaf_unchecked() };
                            let (child_b_first_child, mut child_b_separated_children) =
                                child_b.leak_raw_parts();

                            let (separators, children) = self.separated_children.both_as_mut();
                            let (children_until_a, children_from_c) =
                                children.split_at_mut(index_b);
                            let child_a =
                                children_until_a.last_mut().unwrap_or(&mut self.first_child);
                            let child_a = unsafe { child_a.as_non_leaf_mut_unchecked() };
                            let child_c = children_from_c.first_mut();

                            let separated_child_c = child_c.and_then(|c| {
                                let c = unsafe { c.as_non_leaf_mut_unchecked() };
                                if c.separated_children.len() >= 2 * (CAP / 2) {
                                    // Ignore `child_c` for the merger since it wouldn't be affected
                                    // by it anyway but would make the logic below more complicated.
                                    None
                                } else {
                                    Some((&mut separators[index_b], c))
                                }
                            });

                            if let Some((separator_bc, child_c)) = separated_child_c {
                                // We're merging three children `[A, B, C]` into two `[A', C']`.
                                let len_a = child_a.separated_children.len();
                                let len_b = child_b_separated_children.len();
                                let len_c = child_c.separated_children.len();
                                let num_grandchildren = len_a + len_b + len_c + 1;
                                let target_len_a = num_grandchildren / 2;
                                let increase_len_a = target_len_a - len_a;
                                // `increase_len_a <= CAP / 2`, where equality requires that `child_a` underflowed.
                                // Thus, `increase_len_a <= len_b`.

                                if increase_len_a == 0 {
                                    // Special case: `child_a` is not actually a part of the merger.
                                    let separator_bc =
                                        core::mem::replace(separator_bc, separator_ab);
                                    let first_child_c = core::mem::replace(
                                        &mut child_c.first_child,
                                        child_b_first_child,
                                    );
                                    child_b_separated_children
                                        .try_push(separator_bc, first_child_c)
                                        .expect("child_b underflowed");
                                    child_c
                                        .separated_children
                                        .try_prepend(child_b_separated_children.take_all())
                                        .expect("can't overflow");
                                } else {
                                    let tail_b = child_b_separated_children
                                        .three_way_split(
                                            increase_len_a - 1,
                                            separator_bc,
                                            &mut child_c.first_child,
                                        )
                                        .expect("0 < increase_len_a <= len_b");
                                    child_c.separated_children.try_prepend(tail_b);

                                    child_a
                                        .separated_children
                                        .try_push(separator_ab, child_b_first_child)
                                        .expect("increase_len_a > 0");
                                    child_a
                                        .separated_children
                                        .try_append(child_b_separated_children.take_all())
                                        .expect("can't overflow");
                                }
                            } else {
                                child_a
                                    .separated_children
                                    .try_push(separator_ab, child_b_first_child)
                                    .expect("increase_len_a > 0");
                                child_a
                                    .separated_children
                                    .try_append(child_b_separated_children.take_all())
                                    .expect("can't overflow");
                            }
                        }
                    }
                    return Ok(());
                }
                todo!()
            }
            Leaf => {
                let child = unsafe { child.as_leaf_mut_unchecked() };
                // child.remove(pos, count);
                todo!()
            }
        }
    }
}

impl<P, C, const CAP: usize> LeafNode<P, C, CAP>
where
    P: Copy + Ord,
    C: Copy + Default + Add<Output = C> + Sub<Output = C>,
{
    fn new() -> Self {
        Self {
            entries: BoundedVec::new(),
        }
    }

    fn insert(&mut self, pos: P, count: C) -> Option<(P, C, ChildPtr<P, C, CAP>)> {
        // Check if the node already contains an entry with at the given `pos`.
        // If so, increment its accum and all accums to the right, then return.
        let insert_index = self.entries.partition_point(|entry| entry.pos < pos);
        let mut right_iter = self.entries.iter_mut().skip(insert_index);
        match right_iter.next() {
            Some(right_entry) if right_entry.pos == pos => {
                right_entry.accum = right_entry.accum + count;
                for entry in right_iter {
                    entry.accum = entry.accum + count;
                }
                return None;
            }
            _ => {}
        }

        // An entry at position `pos` doesn't exist yet. Create a new one and insert it.
        let old_accum = self
            .entries
            .get(insert_index.wrapping_sub(1))
            .map(|node| node.accum)
            .unwrap_or_default();
        let mut insert_entry = Entry::new(pos, old_accum + count);

        if self
            .entries
            .try_insert_and_accum(insert_index, insert_entry, |entry| {
                Entry::new(entry.pos, entry.accum + count)
            })
            .is_ok()
        {
            return None; // No splitting necessary.
        }

        // Inserting would overflow the leaf node. Split it into two.
        let mut right_sibling = LeafNode::new();

        if insert_index <= CAP / 2 {
            // We're inserting into the left half or the parent.
            let old_weight_before_right_sibling = if CAP / 2 == 0 {
                C::default()
            } else {
                self.entries
                    .get(CAP / 2 - 1)
                    .expect("node is full and CAP/2 > 0")
                    .accum
            };

            let right_entries = self.entries.chop(CAP / 2).expect("node is full");

            right_sibling
                .entries
                .try_append_transform(right_entries, |entry| Entry {
                    pos: entry.pos,
                    accum: entry.accum - old_weight_before_right_sibling,
                })
                .expect("can't overflow");

            self.entries
                .try_insert_and_accum(insert_index, insert_entry, |entry| {
                    Entry::new(entry.pos, entry.accum + count)
                })
                .expect("there are `CAP - CAP/2` vacancies, which is >0 because CAP>0.");
        } else {
            // We're inserting into the right half.
            let weight_before_right_sibling = self
                .entries
                .get(CAP / 2)
                .expect("CAP/2 < insert_index <= len")
                .accum;

            let right_entries = self
                .entries
                .chop(CAP / 2 + 1)
                .expect("CAP/2 < insert_index <= len");

            let (before_insert, after_insert) =
                right_entries.split_at_mut(insert_index - (CAP / 2 + 1));

            right_sibling
                .entries
                .try_append_transform(before_insert, |entry| Entry {
                    pos: entry.pos,
                    accum: entry.accum - weight_before_right_sibling,
                })
                .expect("can't overflow");

            insert_entry.accum = insert_entry.accum - weight_before_right_sibling;
            right_sibling
                .entries
                .try_push(insert_entry)
                .expect("can't overflow");

            right_sibling
                .entries
                .try_append_transform(after_insert, |entry| Entry {
                    pos: entry.pos,
                    accum: entry.accum - weight_before_right_sibling + count,
                })
                .expect("can't overflow");
        };

        let ejected_entry = self.entries.pop().expect("there are CAP/2+1 > 0 entries");
        let right_sibling_ref = ChildPtr::leaf(right_sibling);

        Some((ejected_entry.pos, ejected_entry.accum, right_sibling_ref))
    }
}

impl<P, C> Entry<P, C> {
    fn new(pos: P, accum: C) -> Self {
        Self { pos, accum }
    }
}

impl<P: Display, C: Display + Debug, const CAP: usize> Debug for AugmentedBTree<P, C, CAP> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut f = f.debug_struct("AugmentedBTree");
        let mut f = f.field("total", &self.total);
        match self.root_type {
            NonLeaf => {
                let root = unsafe { self.root.as_non_leaf_unchecked() };
                f = f.field("root", root);
            }
            Leaf => {
                let root = unsafe { self.root.as_leaf_unchecked() };
                f = f.field("root", root);
            }
        }
        f.finish()
    }
}

impl<P: Display, C: Display, const CAP: usize> Debug for NonLeafNode<P, C, CAP> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut f = f.debug_list();

        match self.children_type {
            NonLeaf => {
                let first_child = unsafe { self.first_child.as_non_leaf_unchecked() };
                f.entry(first_child);
                for (separator, child) in self.separated_children.iter() {
                    f.entry(separator);
                    f.entry(unsafe { child.as_non_leaf_unchecked() });
                }
            }
            Leaf => {
                let first_child = unsafe { self.first_child.as_leaf_unchecked() };
                f.entry(first_child);
                for (separator, child) in self.separated_children.iter() {
                    f.entry(separator);
                    f.entry(unsafe { child.as_leaf_unchecked() });
                }
            }
        }

        f.finish()
    }
}

impl<P: Display, C: Display, const CAP: usize> Debug for LeafNode<P, C, CAP> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[")?;
        let mut iter = self.entries.iter();
        if let Some(first) = iter.next() {
            write!(f, "{first}")?;
            for entry in iter {
                write!(f, ", {entry}")?;
            }
        }
        write!(f, "]")?;
        Ok(())
    }
}

impl<P: Display, C: Display> Display for Entry<P, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "({}, {})", &self.pos, &self.accum)
    }
}

impl<P: Display, C: Display> Debug for Entry<P, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}

mod child_ptr {
    use core::{fmt::Debug, mem::ManuallyDrop};

    use alloc::boxed::Box;

    use super::{LeafNode, NonLeafNode};

    /// A (conceptually) owned reference to either a `NonLeafNode` or a `LeafNode`.
    ///
    /// Note that a `ChildPtr` does not automatically drop the child when it is dropped.
    /// This is not possible because the `ChildPtr` doesn't know the type of the
    /// child. Any container that has a `ChildPtr` needs to implement `Drop`, where
    /// it has to call either `.drop_non_leaf()` or `.drop_leaf()` on all of its
    /// fields of type `ChildPtr`.
    pub union ChildPtr<P, C, const CAP: usize> {
        non_leaf: ManuallyDrop<Box<NonLeafNode<P, C, CAP>>>,
        leaf: ManuallyDrop<Box<LeafNode<P, C, CAP>>>,
    }

    impl<P, C, const CAP: usize> Debug for ChildPtr<P, C, CAP> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("ChildPtr").finish_non_exhaustive()
        }
    }

    impl<P, C, const CAP: usize> ChildPtr<P, C, CAP> {
        pub fn non_leaf(child: NonLeafNode<P, C, CAP>) -> ChildPtr<P, C, CAP> {
            Self {
                non_leaf: ManuallyDrop::new(Box::new(child)),
            }
        }

        pub fn leaf(child: LeafNode<P, C, CAP>) -> ChildPtr<P, C, CAP> {
            Self {
                leaf: ManuallyDrop::new(Box::new(child)),
            }
        }

        #[inline(always)]
        pub unsafe fn drop_non_leaf(&mut self) {
            ManuallyDrop::drop(&mut self.non_leaf);
        }

        #[inline(always)]
        pub unsafe fn drop_leaf(&mut self) {
            ManuallyDrop::drop(&mut self.leaf);
        }

        #[inline(always)]
        pub unsafe fn as_non_leaf_unchecked(&self) -> &NonLeafNode<P, C, CAP> {
            &*self.non_leaf
        }

        #[inline(always)]
        pub unsafe fn as_leaf_unchecked(&self) -> &LeafNode<P, C, CAP> {
            &*self.leaf
        }

        #[inline(always)]
        pub unsafe fn as_non_leaf_mut_unchecked(&mut self) -> &mut NonLeafNode<P, C, CAP> {
            &mut *self.non_leaf
        }

        #[inline(always)]
        pub unsafe fn as_leaf_mut_unchecked(&mut self) -> &mut LeafNode<P, C, CAP> {
            &mut *self.leaf
        }

        #[inline(always)]
        pub unsafe fn into_non_leaf_unchecked(self) -> Box<NonLeafNode<P, C, CAP>> {
            ManuallyDrop::into_inner(self.non_leaf)
        }

        #[inline(always)]
        pub unsafe fn into_leaf_unchecked(self) -> Box<LeafNode<P, C, CAP>> {
            ManuallyDrop::into_inner(self.leaf)
        }
    }
}

mod bounded_vec {
    use core::{
        mem::MaybeUninit,
        ops::{Deref, DerefMut, Index},
        slice::SliceIndex,
    };

    /// Semantically equivalent to `BoundedVec<(T1, T2), CAP>`, but with all `T1`s stored in one
    /// array and all `T2` stored in a different array, thus improving memory locality for
    /// algorithms that access one part more often than the other
    #[derive(Debug)]
    pub struct BoundedPairOfVecs<T1, T2, const CAP: usize> {
        len: usize,
        buf1: [MaybeUninit<T1>; CAP],
        buf2: [MaybeUninit<T2>; CAP],
    }

    #[derive(Debug)]
    pub struct BoundedVec<T, const CAP: usize>(BoundedPairOfVecs<T, (), CAP>);

    impl<T1, T2, const CAP: usize> Drop for BoundedPairOfVecs<T1, T2, CAP> {
        fn drop(&mut self) {
            while let Some(pair) = self.pop() {
                core::mem::drop(pair);
            }
        }
    }

    /// A pair of two slices of equal length up to `BOUND` that are conceptually owned, i.e., one
    /// can safely move the entries out (e.g., by passing an `OwnedBoundedPairOfSlices` as an
    /// argument to `BoundedPairOfVecs::try_append`).
    #[derive(Debug)]
    pub struct OwnedBoundedPairOfSlices<'a, T1, T2, const BOUND: usize> {
        slice1: &'a mut [MaybeUninit<T1>],
        slice2: &'a mut [MaybeUninit<T2>],
    }

    #[derive(Debug)]
    pub struct OwnedBoundedSlice<'a, T, const CAP: usize> {
        len: usize,
        slice: &'a mut [MaybeUninit<T>],
    }

    pub struct DropGuard<T, F: FnMut(&mut T)> {
        inner: T,
        cleanup: F,
    }

    impl<T, F: FnMut(&mut T)> Drop for DropGuard<T, F> {
        fn drop(&mut self) {
            (self.cleanup)(&mut self.inner)
        }
    }

    impl<T, F: FnMut(&mut T)> Deref for DropGuard<T, F> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            &self.inner
        }
    }

    impl<T, F: FnMut(&mut T)> DerefMut for DropGuard<T, F> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.inner
        }
    }

    /// A pair of two slices of equal length up to `BOUND`. In contrast to
    /// [`OwnedBoundedPairOfSlices`], the entries of a `BoundedPairVecsViewMut` are not owned by
    /// the `BoundedPairVecsViewMut` and it is therefore not possible without `unsafe` to move them
    /// out of the `BoundedPairVecsViewMut`.
    #[derive(Debug)]
    pub struct BoundedPairOfVecsViewMut<'a, T1, T2, const BOUND: usize>(&'a mut [T1], &'a mut [T2]);

    impl<T1, T2, const CAP: usize> BoundedPairOfVecs<T1, T2, CAP> {
        pub fn new() -> Self {
            let (buf1, buf2) = unsafe {
                // SAFETY: This is taken from an example in the official documentation of `MaybeUninit`.
                // It calls `assume_init` on the *outer* `MaybeUninit`. This is safe because the type we
                // claim to have initialized at this point is `[MaybeUninit<T>; CAP]`, which does not
                // require initialization. See example in the documentation of `MaybeUninit`.
                (
                    MaybeUninit::<[MaybeUninit<T1>; CAP]>::uninit().assume_init(),
                    MaybeUninit::<[MaybeUninit<T2>; CAP]>::uninit().assume_init(),
                )
            };
            Self { len: 0, buf1, buf2 }
        }

        pub const fn len(&self) -> usize {
            self.len
        }

        pub fn first_as_ref(&self) -> &[T1] {
            unsafe { core::mem::transmute(self.buf1.get_unchecked(..self.len)) }
        }

        pub fn second_as_ref(&self) -> &[T2] {
            unsafe { core::mem::transmute(self.buf2.get_unchecked(..self.len)) }
        }

        pub fn both_as_ref(&self) -> (&[T1], &[T2]) {
            (self.first_as_ref(), self.second_as_ref())
        }

        pub fn first_as_mut(&mut self) -> &mut [T1] {
            unsafe { core::mem::transmute(self.buf1.get_unchecked_mut(..self.len)) }
        }

        pub fn second_as_mut(&mut self) -> &mut [T2] {
            unsafe { core::mem::transmute(self.buf2.get_unchecked_mut(..self.len)) }
        }

        pub fn both_as_mut(&mut self) -> (&mut [T1], &mut [T2]) {
            unsafe {
                (
                    core::mem::transmute(self.buf1.get_unchecked_mut(..self.len)),
                    core::mem::transmute(self.buf2.get_unchecked_mut(..self.len)),
                )
            }
        }

        pub fn try_insert(&mut self, index: usize, item1: T1, item2: T2) -> Result<(), (T1, T2)> {
            assert!(index <= self.len);
            if self.len == CAP {
                return Err((item1, item2));
            }

            unsafe {
                self.buf1
                    .get_unchecked_mut(index..self.len + 1)
                    .rotate_right(1);
                self.buf1.get_unchecked_mut(index).write(item1);
                self.buf2
                    .get_unchecked_mut(index..self.len + 1)
                    .rotate_right(1);
                self.buf2.get_unchecked_mut(index).write(item2);
            }
            self.len += 1;

            Ok(())
        }

        pub fn try_insert_and_accum(
            &mut self,
            index: usize,
            item1: T1,
            item2: T2,
            update1: impl Fn(T1) -> T1,
        ) -> Result<(), ()> {
            assert!(index <= self.len);

            if self.len == CAP {
                return Err(());
            }

            unsafe {
                let mut write_index = self.len;
                while write_index > index {
                    let tmp = self.buf1.get_unchecked(write_index - 1).assume_init_read();
                    self.buf1.get_unchecked_mut(write_index).write(update1(tmp));
                    write_index -= 1;
                }
                self.buf1.get_unchecked_mut(index).write(item1);

                self.buf2
                    .get_unchecked_mut(index..self.len + 1)
                    .rotate_right(1);
                self.buf2.get_unchecked_mut(index).write(item2);
            }
            self.len += 1;

            Ok(())
        }

        pub fn try_push(&mut self, item1: T1, item2: T2) -> Result<(), ()> {
            if self.len == CAP {
                Err(())
            } else {
                unsafe {
                    self.buf1.get_unchecked_mut(self.len).write(item1);
                    self.buf2.get_unchecked_mut(self.len).write(item2);
                }
                self.len += 1;
                Ok(())
            }
        }

        pub fn pop(&mut self) -> Option<(T1, T2)> {
            if self.len == 0 {
                None
            } else {
                self.len -= 1;
                let last = unsafe {
                    (
                        self.buf1.get_unchecked_mut(self.len).assume_init_read(),
                        self.buf2.get_unchecked_mut(self.len).assume_init_read(),
                    )
                };
                Some(last)
            }
        }

        pub fn remove(&mut self, index: usize) -> Option<(T1, T2)> {
            if index >= self.len {
                None
            } else {
                let items = unsafe {
                    (
                        self.buf1.get_unchecked_mut(index).assume_init_read(),
                        self.buf2.get_unchecked_mut(index).assume_init_read(),
                    )
                };
                self.len -= 1;
                unsafe {
                    core::ptr::copy(
                        self.buf1.as_ptr().add(index + 1),
                        self.buf1.as_mut_ptr(),
                        self.len - index,
                    );
                    core::ptr::copy(
                        self.buf2.as_ptr().add(index + 1),
                        self.buf2.as_mut_ptr(),
                        self.len - index,
                    );
                }

                Some(items)
            }
        }

        pub fn get(&self, index: usize) -> Option<(&T1, &T2)> {
            if index >= self.len {
                None
            } else {
                let front = unsafe {
                    (
                        self.buf1.get_unchecked(index).assume_init_ref(),
                        self.buf2.get_unchecked(index).assume_init_ref(),
                    )
                };
                Some(front)
            }
        }

        pub fn get_mut<I>(
            &mut self,
            index: I,
        ) -> Option<(
            &mut <I as SliceIndex<[T1]>>::Output,
            &mut <I as SliceIndex<[T2]>>::Output,
        )>
        where
            I: SliceIndex<[T1]> + SliceIndex<[T2]> + Copy,
        {
            unsafe {
                let buf1: &mut [T1] = core::mem::transmute(self.buf1.get_unchecked_mut(..self.len));
                let buf2: &mut [T2] = core::mem::transmute(self.buf2.get_unchecked_mut(..self.len));
                Some((buf1.get_mut(index)?, buf2.get_unchecked_mut(index)))
            }
        }

        pub fn chop(
            &mut self,
            start_index: usize,
        ) -> Option<OwnedBoundedPairOfSlices<'_, T1, T2, CAP>> {
            let slice1 = self.buf1.get_mut(start_index..self.len)?;
            let slice2 = &mut self.buf2[start_index..self.len];
            self.len = start_index;
            Some(OwnedBoundedPairOfSlices { slice1, slice2 })
        }

        /// Asserts that there is at least one more vacancy. Appends `(slot1, slot2)` to the vecs
        /// and replaces them with the entries at position `index`, then chops off part strictly
        /// after `index` (not including the item at `index`) and returns it. This operation
        /// removes all entries at positions including and after `index` (thus, after successfully
        /// calling `vecs.three_way_split(index)`, we have `vecs.len() == index`).
        ///
        /// Returns `None` if `index` is out of bounds.
        ///
        /// # Panics
        ///
        /// If `index` is within bounds but the vecs are full.
        pub fn three_way_split(
            &mut self,
            index: usize,
            slot1: &mut T1,
            slot2: &mut T2,
        ) -> Option<OwnedBoundedPairOfSlices<'_, T1, T2, CAP>> {
            if index >= self.len {
                return None;
            }
            assert!(self.len < CAP);
            unsafe {
                // Append item at position `index` to the end.
                let item1 = self.buf1.get_unchecked_mut(index).assume_init_read();
                self.buf1
                    .get_unchecked_mut(self.len)
                    .write(core::mem::replace(slot1, item1));

                let item2 = self.buf2.get_unchecked_mut(index).assume_init_read();
                self.buf2
                    .get_unchecked_mut(self.len)
                    .write(core::mem::replace(slot2, item2));

                let slice1 = self.buf1.get_unchecked_mut(index + 1..self.len + 1);
                let slice2 = self.buf2.get_unchecked_mut(index + 1..self.len + 1);
                self.len = index;
                Some(OwnedBoundedPairOfSlices { slice1, slice2 })
            }
        }

        pub fn chop_front<'a>(
            &'a mut self,
            len: usize,
        ) -> Option<
            DropGuard<
                OwnedBoundedPairOfSlices<'a, T1, T2, CAP>,
                impl FnMut(&mut OwnedBoundedPairOfSlices<'a, T1, T2, CAP>),
            >,
        > {
            let slice1 = self.buf1.get_mut(..len)?;
            let slice2 = &mut self.buf2[..len];
            self.len -= len;
            let len_after_chop = self.len;

            Some(DropGuard {
                inner: OwnedBoundedPairOfSlices { slice1, slice2 },
                cleanup: move |slices| unsafe {
                    core::ptr::copy(
                        slices.slice1.as_ptr().add(len),
                        slices.slice1.as_mut_ptr(),
                        len_after_chop,
                    );
                    core::ptr::copy(
                        slices.slice2.as_ptr().add(len),
                        slices.slice2.as_mut_ptr(),
                        len_after_chop,
                    );
                },
            })
        }

        pub fn take_all(&mut self) -> OwnedBoundedPairOfSlices<'_, T1, T2, CAP> {
            // TODO: is this still needed?
            let slice1 = &mut self.buf1[..];
            let slice2 = &mut self.buf2[..];
            self.len = 0;
            OwnedBoundedPairOfSlices { slice1, slice2 }
        }

        pub fn get_all_mut(&mut self) -> BoundedPairOfVecsViewMut<'_, T1, T2, CAP> {
            unsafe {
                BoundedPairOfVecsViewMut(
                    core::mem::transmute(&mut self.buf1[..]),
                    core::mem::transmute(&mut self.buf2[..]),
                )
            }
        }

        pub fn try_append<const BOUND: usize>(
            &mut self,
            tail: OwnedBoundedPairOfSlices<'_, T1, T2, BOUND>,
        ) -> Result<(), ()> {
            self.try_append_guarded(DropGuard {
                inner: tail,
                cleanup: |_| (),
            })
        }

        pub fn try_append_guarded<'a, F, const BOUND: usize>(
            &mut self,
            tail: DropGuard<OwnedBoundedPairOfSlices<'a, T1, T2, BOUND>, F>,
        ) -> Result<(), ()>
        where
            F: FnMut(&mut OwnedBoundedPairOfSlices<'a, T1, T2, BOUND>),
        {
            let tail_len = tail.len();
            let dst1 = self.buf1.get_mut(self.len..self.len + tail_len).ok_or(())?;
            let dst2 = &mut self.buf2[self.len..self.len + tail_len];
            unsafe {
                core::ptr::copy_nonoverlapping(tail.slice1.as_ptr(), dst1.as_mut_ptr(), tail_len);
                core::ptr::copy_nonoverlapping(tail.slice2.as_ptr(), dst2.as_mut_ptr(), tail_len);
            }
            self.len += tail_len;

            Ok(())
        }

        pub fn try_prepend<const BOUND: usize>(
            &mut self,
            head: OwnedBoundedPairOfSlices<'_, T1, T2, BOUND>,
        ) -> Result<(), ()> {
            let head_len = head.len();
            let buf1 = self.buf1.get_mut(..self.len + head_len).ok_or(())?;
            buf1.rotate_right(head_len);
            let buf2 = &mut self.buf2[..self.len + head_len];
            buf2.rotate_right(head_len);
            unsafe {
                core::ptr::copy_nonoverlapping(head.slice1.as_ptr(), buf1.as_mut_ptr(), head_len);
                core::ptr::copy_nonoverlapping(head.slice2.as_ptr(), buf2.as_mut_ptr(), head_len);
            }
            self.len += head_len;

            Ok(())
        }

        pub fn try_append_transform1<const BOUND: usize>(
            &mut self,
            tail: OwnedBoundedPairOfSlices<'_, T1, T2, BOUND>,
            transform1: impl Fn(T1) -> T1,
        ) -> Result<(), ()> {
            let tail_len = tail.len();
            let dst1 = self.buf1.get_mut(self.len..self.len + tail_len).ok_or(())?;

            unsafe {
                let dst2 = self.buf2.get_unchecked_mut(self.len..self.len + tail_len);
                for (src_item, dst_item) in tail.slice1.iter().zip(dst1) {
                    dst_item.write(transform1(src_item.assume_init_read()));
                }
                core::ptr::copy_nonoverlapping(tail.slice2.as_ptr(), dst2.as_mut_ptr(), tail_len);
            }
            self.len += tail_len;

            Ok(())
        }

        pub fn iter(&self) -> impl DoubleEndedIterator<Item = (&T1, &T2)> {
            self.first_as_ref().iter().zip(self.second_as_ref())
        }

        pub fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = (&mut T1, &mut T2)> {
            let (first, second) = self.both_as_mut();
            first.iter_mut().zip(second)
        }
    }

    impl<T1, T2, const CAP: usize> Default for BoundedPairOfVecs<T1, T2, CAP> {
        fn default() -> Self {
            Self::new()
        }
    }

    impl<T, const CAP: usize> BoundedVec<T, CAP> {
        pub fn new() -> Self {
            Self(BoundedPairOfVecs::new())
        }

        pub const fn len(&self) -> usize {
            self.0.len()
        }

        pub fn try_insert(&mut self, index: usize, item: T) -> Result<(), T> {
            self.0.try_insert(index, item, ()).map_err(|(item, _)| item)
        }

        pub fn try_insert_and_accum(
            &mut self,
            index: usize,
            item: T,
            update: impl Fn(T) -> T,
        ) -> Result<(), ()> {
            self.0.try_insert_and_accum(index, item, (), update)
        }

        pub fn try_push(&mut self, item: T) -> Result<(), ()> {
            self.0.try_push(item, ())
        }

        pub fn pop(&mut self) -> Option<T> {
            Some(self.0.pop()?.0)
        }

        pub fn chop(&mut self, start_index: usize) -> Option<OwnedBoundedSlice<'_, T, CAP>> {
            let slice = self.0.chop(start_index)?.slice1;
            let len = slice.len();
            Some(OwnedBoundedSlice { len, slice })
        }

        pub fn try_append<const BOUND: usize>(
            &mut self,
            tail: OwnedBoundedSlice<'_, T, BOUND>,
        ) -> Result<(), ()> {
            let tail_len = tail.len();
            let mut tail2: [MaybeUninit<()>; BOUND] =
                unsafe { MaybeUninit::<[MaybeUninit<()>; BOUND]>::uninit().assume_init() };
            let tail = OwnedBoundedPairOfSlices::<'_, _, _, BOUND> {
                slice1: unsafe { tail.slice.get_unchecked_mut(..tail_len) },
                slice2: unsafe { tail2.get_unchecked_mut(..tail_len) },
            };
            self.0.try_append(tail)
        }

        pub fn try_append_transform<const BOUND: usize>(
            &mut self,
            tail: OwnedBoundedSlice<'_, T, BOUND>,
            transform: impl Fn(T) -> T,
        ) -> Result<(), ()> {
            let tail_len = tail.len();
            let mut tail2: [MaybeUninit<()>; BOUND] =
                unsafe { MaybeUninit::<[MaybeUninit<()>; BOUND]>::uninit().assume_init() };
            let tail = OwnedBoundedPairOfSlices::<'_, _, _, BOUND> {
                slice1: unsafe { tail.slice.get_unchecked_mut(..tail_len) },
                slice2: unsafe { tail2.get_unchecked_mut(..tail_len) },
            };
            self.0.try_append_transform1(tail, transform)
        }
    }

    impl<T, const CAP: usize> Default for BoundedVec<T, CAP> {
        fn default() -> Self {
            Self::new()
        }
    }

    impl<'a, T1, T2, const BOUND: usize> BoundedPairOfVecsViewMut<'a, T1, T2, BOUND> {
        pub fn len(&self) -> usize {
            self.0.len()
        }

        pub fn split_at_mut(
            self,
            mid: usize,
        ) -> (
            BoundedPairOfVecsViewMut<'a, T1, T2, BOUND>,
            BoundedPairOfVecsViewMut<'a, T1, T2, BOUND>,
        ) {
            let (left1, right1) = self.0.split_at_mut(mid);
            let (left2, right2) = self.1.split_at_mut(mid);
            (
                BoundedPairOfVecsViewMut(left1, left2),
                BoundedPairOfVecsViewMut(right1, right2),
            )
        }

        pub fn first_as_ref(&self) -> &[T1] {
            self.0
        }

        pub fn second_as_ref(&self) -> &[T2] {
            self.1
        }

        pub fn both_as_ref(&self) -> (&[T1], &[T2]) {
            (self.first_as_ref(), self.second_as_ref())
        }

        pub fn first_as_mut(&mut self) -> &mut [T1] {
            self.0
        }

        pub fn second_as_mut(&mut self) -> &mut [T2] {
            self.1
        }

        pub fn iter(&self) -> impl DoubleEndedIterator<Item = (&T1, &T2)> {
            self.first_as_ref().iter().zip(self.second_as_ref())
        }

        pub fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = (&mut T1, &mut T2)> {
            self.0.iter_mut().zip(self.1.iter_mut())
        }
    }

    impl<'a, T, const BOUND: usize> OwnedBoundedSlice<'a, T, BOUND> {
        pub fn len(&self) -> usize {
            self.len
        }

        pub fn as_ref(&self) -> &[T] {
            unsafe { core::mem::transmute(self.slice.get_unchecked(..self.len)) }
        }

        pub fn as_mut(&mut self) -> &mut [T] {
            unsafe { core::mem::transmute(self.slice.get_unchecked_mut(..self.len)) }
        }

        pub fn split_at_mut(self, mid: usize) -> (Self, Self) {
            assert!(mid <= self.len);
            let (left, right) = self.slice.split_at_mut(mid);
            (
                OwnedBoundedSlice {
                    len: mid,
                    slice: left,
                },
                OwnedBoundedSlice {
                    len: self.len - mid,
                    slice: right,
                },
            )
        }
    }

    impl<'a, T1, T2, const BOUND: usize> OwnedBoundedPairOfSlices<'a, T1, T2, BOUND> {
        pub fn len(&self) -> usize {
            self.slice1.len()
        }

        pub fn pop(&mut self) -> Option<(T1, T2)> {
            if self.len() == 0 {
                None
            } else {
                unsafe {
                    let new_len = self.len() - 1;

                    let item1 = self.slice1.get_unchecked_mut(new_len).assume_init_read();
                    let old_slice1 = std::mem::replace(&mut self.slice1, &mut []);
                    std::mem::replace(&mut self.slice1, old_slice1.get_unchecked_mut(..new_len));

                    let item2 = self.slice2.get_unchecked_mut(new_len).assume_init_read();
                    let old_slice2 = std::mem::replace(&mut self.slice2, &mut []);
                    std::mem::replace(&mut self.slice2, old_slice2.get_unchecked_mut(..new_len));

                    Some((item1, item2))
                }
            }
        }

        pub fn split_at_mut(
            self,
            mid: usize,
        ) -> (
            OwnedBoundedPairOfSlices<'a, T1, T2, BOUND>,
            OwnedBoundedPairOfSlices<'a, T1, T2, BOUND>,
        ) {
            let (left1, right1) = self.slice1.split_at_mut(mid);
            let (left2, right2) = self.slice2.split_at_mut(mid);
            (
                OwnedBoundedPairOfSlices {
                    slice1: left1,
                    slice2: left2,
                },
                OwnedBoundedPairOfSlices {
                    slice1: right1,
                    slice2: right2,
                },
            )
        }

        pub fn first_as_ref(&self) -> &[T1] {
            unsafe { core::mem::transmute(&*self.slice1) }
        }

        pub fn second_as_ref(&self) -> &[T2] {
            unsafe { core::mem::transmute(&*self.slice2) }
        }

        pub fn both_as_ref(&self) -> (&[T1], &[T2]) {
            (self.first_as_ref(), self.second_as_ref())
        }

        pub fn first_as_mut(&mut self) -> &mut [T1] {
            unsafe { core::mem::transmute(&mut *self.slice1) }
        }

        pub fn second_as_mut(&mut self) -> &mut [T2] {
            unsafe { core::mem::transmute(&mut *self.slice2) }
        }

        pub fn both_as_mut(&mut self) -> (&mut [T1], &mut [T2]) {
            unsafe {
                (
                    core::mem::transmute(&mut *self.slice2),
                    core::mem::transmute(&mut *self.slice1),
                )
            }
        }
    }

    impl<'a, T1, T2, const CAP: usize> From<OwnedBoundedPairOfSlices<'a, T1, T2, CAP>>
        for BoundedPairOfVecs<T1, T2, CAP>
    {
        fn from(view: OwnedBoundedPairOfSlices<'a, T1, T2, CAP>) -> Self {
            let mut vec = Self::new();
            vec.try_append(view)
                .expect("capacities match and original vec was empty");
            vec
        }
    }

    impl<'a, T, const CAP: usize> From<OwnedBoundedSlice<'a, T, CAP>> for BoundedVec<T, CAP> {
        fn from(view: OwnedBoundedSlice<'a, T, CAP>) -> Self {
            let mut vec = Self::new();
            vec.try_append(view)
                .expect("capacities match and original vec was empty");
            vec
        }
    }

    impl<'a, T, const BOUND: usize> Index<usize> for OwnedBoundedSlice<'a, T, BOUND> {
        type Output = T;

        fn index(&self, index: usize) -> &Self::Output {
            unsafe { self.slice[index].assume_init_ref() }
        }
    }

    impl<T, const CAP: usize> Deref for BoundedVec<T, CAP> {
        type Target = [T];

        fn deref(&self) -> &Self::Target {
            unsafe { core::mem::transmute(self.0.buf1.get_unchecked(..self.0.len)) }
        }
    }

    impl<T, const CAP: usize> DerefMut for BoundedVec<T, CAP> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            unsafe { core::mem::transmute(self.0.buf1.get_unchecked_mut(..self.0.len)) }
        }
    }
}

#[cfg(test)]
mod tests {
    use core::hash::{Hash, Hasher};
    use std::{collections::hash_map::DefaultHasher, dbg};

    use alloc::vec::Vec;
    use rand::{distributions, seq::SliceRandom, Rng, RngCore, SeedableRng};
    use rand_xoshiro::Xoshiro256StarStar;

    use crate::NonNanFloat;

    use super::AugmentedBTree;

    type F32 = NonNanFloat<f32>;
    type F64 = NonNanFloat<f64>;

    #[test]
    fn minimal() {
        dbg!(minimal_internal::<128>());
        dbg!(minimal_internal::<10>());
        dbg!(minimal_internal::<5>());
        dbg!(minimal_internal::<4>());
        // dbg!(minimal_internal::<3>());
        // dbg!(minimal_internal::<2>());
        // dbg!(minimal_internal::<1>());

        fn minimal_internal<const CAP: usize>() {
            let mut tree = AugmentedBTree::<F32, u32, CAP>::new();
            assert_eq!(tree.total(), 0);

            // Insert { -2.25=>3, 1.5=>10, 2.5=>9, 3.25=>7, 4.75=>3, 5.125=>2, 6.5=>7 } in some
            // random order.
            let insertions = [
                (3.25, 7, 7, 0, None, None),
                (5.125, 2, 9, 7, Some(3.25), None),
                (4.75, 3, 12, 7, Some(3.25), Some(5.125)),
                (1.5, 10, 22, 0, None, Some(3.25)),
                (2.5, 9, 31, 10, Some(1.5), Some(3.25)),
                (6.5, 7, 38, 31, Some(5.125), None),
                (-2.25, 3, 41, 0, None, Some(1.5)),
            ];
            for (pos, count, total, accum_before, left_neighbor_pos, right_neighbor_pos) in
                insertions
            {
                let pos = F32::new(pos).unwrap();
                let right_neighbor_pos = right_neighbor_pos.map(|x| F32::new(x).unwrap());
                let left_neighbor_pos = left_neighbor_pos.map(|x| F32::new(x).unwrap());
                let epsilon = F32::new(1e-5).unwrap();

                assert_eq!(tree.left_cumulative(pos), accum_before);
                assert_eq!(tree.left_cumulative(pos + epsilon), accum_before);
                if let Some(pos_r) = right_neighbor_pos {
                    assert_eq!(tree.left_cumulative(pos_r), accum_before);
                }
                assert_eq!(tree.quantile_function(accum_before), right_neighbor_pos);
                assert_eq!(
                    tree.quantile_function(accum_before.wrapping_sub(1)),
                    left_neighbor_pos
                );

                tree.insert(pos, count);

                assert_eq!(tree.total(), total);
                assert_eq!(tree.left_cumulative(pos), accum_before);
                assert_eq!(tree.left_cumulative(pos + epsilon), accum_before + count);
                if let Some(pos_r) = right_neighbor_pos {
                    assert_eq!(tree.left_cumulative(pos_r), accum_before + count);
                }
                assert_eq!(tree.quantile_function(accum_before), Some(pos));
                assert_eq!(tree.quantile_function(accum_before + count - 1), Some(pos));
                assert_eq!(
                    tree.quantile_function(accum_before + count),
                    right_neighbor_pos
                );
                assert_eq!(
                    tree.quantile_function(accum_before.wrapping_sub(1)),
                    left_neighbor_pos
                );
            }

            let test_points = [
                (-5.7, 0),
                (-2.25, 0),
                (-2.1, 3),
                (1.5, 3),
                (1.8, 13),
                (2.5, 13),
                (2.7, 22),
                (3.25, 22),
                (4.1, 29),
                (4.75, 29),
                (5.1, 32),
                (5.125, 32),
                (5.5, 34),
                (6.5, 34),
                (7.2, 41),
            ];

            for (pos, expected_accum) in test_points {
                let pos = F32::new(pos).unwrap();
                assert_eq!(tree.left_cumulative(pos), expected_accum);
            }
            // Insert { -2.25=>3, 1.5=>10, 2.5=>9, 3.25=>7, 4.75=>3, 5.125=>2, 6.5=>1 } in some

            let test_quantiles = [
                (0, Some(-2.25)),
                (3, Some(1.5)),
                (13, Some(2.5)),
                (22, Some(3.25)),
                (29, Some(4.75)),
                (32, Some(5.125)),
                (34, Some(6.5)),
                (39, Some(6.5)),
                (41, None),
            ];

            let mut last_expected_pos = None;
            for (accum, expected_pos) in test_quantiles {
                let expected_pos = expected_pos.map(|pos| F32::new(pos).unwrap());
                assert_eq!(tree.quantile_function(accum), expected_pos);
                assert_eq!(tree.quantile_function(accum + 1), expected_pos);
                assert_eq!(
                    tree.quantile_function(accum.wrapping_sub(1)),
                    last_expected_pos
                );
                last_expected_pos = expected_pos;
            }
        }
    }

    #[test]
    fn random_data() {
        #[cfg(not(miri))]
        let amts = [100, 1000, 10_000];

        #[cfg(miri)]
        let amts = [100];

        for amt in amts {
            dbg!(amt, random_data_internal::<128>(amt));
            dbg!(amt, random_data_internal::<10>(amt));
            dbg!(amt, random_data_internal::<5>(amt));
            dbg!(amt, random_data_internal::<4>(amt));
            // dbg!(amt, random_data_internal::<3>(amt));
            // dbg!(amt, random_data_internal::<2>(amt));
            // dbg!(amt, random_data_internal::<1>(amt));
        }

        fn random_data_internal<const CAP: usize>(amt: usize) {
            let mut hasher = DefaultHasher::new();
            20231201.hash(&mut hasher);
            (CAP as u64).hash(&mut hasher);
            (amt as u64).hash(&mut hasher);

            let mut rng = Xoshiro256StarStar::seed_from_u64(hasher.finish());
            let repeat_distribution = distributions::Uniform::from(1..5);
            let count_distribution = distributions::Uniform::from(1..100);

            let mut insertions = Vec::new();
            for _ in 0..amt {
                let repeats = rng.sample(repeat_distribution);
                // Deliberately use a somewhat lower precision than what's supported by
                // f64 so that we can always represent a mid-point between two positions.
                let int_pos = rng.next_u64() >> 14;
                let pos = F64::new(int_pos as f64 / (u64::MAX >> 14) as f64).unwrap();
                for _ in 0..repeats {
                    insertions.push((pos, rng.sample(count_distribution)));
                }
            }
            insertions.shuffle(&mut rng);
            let num_insertions = insertions.len();
            assert!(num_insertions > 2 * amt);
            assert!(num_insertions < 4 * amt);

            let mut tree = AugmentedBTree::<F64, u32, CAP>::new();
            let two = F64::new(2.0).unwrap();
            for (i, &(pos, count)) in insertions.iter().enumerate() {
                tree.insert(pos, count);
                if tree.total != tree.left_cumulative(two) {
                    std::eprintln!("AFTER {i}th INSERT ({pos},{count}):");
                    std::dbg!(tree.left_cumulative(two), &tree);
                    panic!();
                }
            }

            let mut sorted = insertions;
            sorted.sort_unstable_by_key(|(pos, _)| *pos);

            let mut last_pos = F64::new(-1.0).unwrap();
            let mut accum = 0;
            let mut cdf = sorted
                .iter()
                .filter_map(|&(pos, count)| {
                    let ret = if pos != last_pos {
                        Some((last_pos, accum))
                    } else {
                        None
                    };
                    accum += count;
                    last_pos = pos;
                    ret
                })
                .collect::<Vec<_>>();
            cdf.push((last_pos, accum));

            assert!(cdf.len() * 2 < num_insertions);
            assert!(cdf.len() * 5 > num_insertions);

            assert_eq!(tree.total, cdf.last().unwrap().1);
            assert_eq!(
                tree.left_cumulative(cdf.last().unwrap().0 + F64::new(0.1).unwrap()),
                tree.total
            );

            let (mut last_pos, mut last_accum) = cdf[0]; // dummy entry at position -1.0 with accum=0
            let half = F64::new(0.5).unwrap();
            for &(pos, accum) in &cdf[1..] {
                let before = half * (last_pos + pos);
                assert_eq!(tree.left_cumulative(before), last_accum);
                assert_eq!(tree.left_cumulative(pos), last_accum);
                assert_eq!(tree.quantile_function(last_accum), Some(pos));
                if accum != last_accum + 1 {
                    assert_eq!(tree.quantile_function(last_accum + 1), Some(pos));
                }
                last_pos = pos;
                last_accum = accum;
            }
            assert_eq!(tree.quantile_function(tree.total), None);
            assert_eq!(tree.quantile_function(tree.total + 1), None);
        }
    }
}
