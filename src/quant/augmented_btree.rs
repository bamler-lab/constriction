use core::{
    cmp::Ordering::{Equal, Greater, Less},
    fmt::{Debug, Display},
    mem::ManuallyDrop,
    ops::{Add, Deref, DerefMut, Sub},
};

use self::{bounded_vec::BoundedPairOfVecs, child_ptr::ChildPtr};
use alloc::vec::Vec;
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

trait Node<P, C, const CAP: usize> {
    type ChildRef;

    /// Tries to remove `count` items at position `pos`. Returns `Err(())` if there are fewer than
    /// `count` items at `pos`. in this case, the subtree rooted at this node has been restored
    /// into its original state at the time the method returns.
    fn remove(&mut self, pos: P, count: C) -> Result<(), ()>;

    fn data_ref(&self) -> &Payload<P, C, Self::ChildRef, CAP>;
    fn data_mut(&mut self) -> &mut Payload<P, C, Self::ChildRef, CAP>;

    /// Returns the node's payload.
    ///
    /// Note that, for a non-leaf node, dropping the payload before sticking its contents into some
    /// different node(s) would leak the memory allocated for the children.
    fn leak_data(self) -> Payload<P, C, Self::ChildRef, CAP>;
}

struct Payload<P, C, L, const CAP: usize> {
    head: L,

    /// Upholds the following invariant:
    /// - if it is the root and a `NonLeafNode`, then `bulk.len() >= 2`
    /// - if it is not the root, then `data.bulk.len() >= CAP / 2 >= 2`
    bulk: BoundedPairOfVecs<Entry<P, C>, L, CAP>,
}

struct NonLeafNode<P, C, const CAP: usize> {
    children_type: NodeType,
    data: ManuallyDrop<Payload<P, C, ChildPtr<P, C, CAP>, CAP>>,
}

struct LeafNode<P, C, const CAP: usize> {
    /// Satisfies `entires.len() >= CAP / 2 >= 2` unless it is the root node
    /// (the root node may have arbitrarily few data.bulk, including zero).
    data: Payload<P, C, (), CAP>,
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
                    self.data.head.drop_non_leaf();
                    while let Some((_, mut child)) = self.data.bulk.pop() {
                        child.drop_non_leaf()
                    }
                }
                Leaf => {
                    self.data.head.drop_leaf();
                    while let Some((_, mut child)) = self.data.bulk.pop() {
                        child.drop_leaf()
                    }
                }
            }
        }
    }
}

impl<P: Copy, C: Copy, const CAP: usize> Clone for NonLeafNode<P, C, CAP> {
    fn clone(&self) -> Self {
        let data = match self.children_type {
            NonLeaf => {
                let head =
                    ChildPtr::non_leaf(unsafe { self.data.head.as_non_leaf_unchecked() }.clone());
                let mut bulk = BoundedPairOfVecs::new();
                for (separator, child) in self.data.bulk.iter() {
                    let child = unsafe { child.as_non_leaf_unchecked() };
                    bulk.try_push(separator.clone(), ChildPtr::non_leaf(child.clone()))
                        .expect("same capacity");
                }
                Payload::new(head, bulk)
            }
            Leaf => {
                let head = ChildPtr::leaf(unsafe { self.data.head.as_leaf_unchecked() }.clone());
                let mut bulk = BoundedPairOfVecs::new();
                for (separator, child) in self.data.bulk.iter() {
                    let child = unsafe { child.as_leaf_unchecked() };
                    bulk.try_push(separator.clone(), ChildPtr::leaf(child.clone()))
                        .expect("same capacity");
                }
                Payload::new(head, bulk)
            }
        };
        Self {
            children_type: self.children_type,
            data: ManuallyDrop::new(data),
        }
    }
}

impl<P: Copy, C: Copy, const CAP: usize> Clone for LeafNode<P, C, CAP> {
    fn clone(&self) -> Self {
        Self {
            data: Payload::new((), self.data.bulk.clone()),
        }
    }
}

impl<P: Copy, C: Copy, const CAP: usize> Clone for AugmentedBTree<P, C, CAP> {
    fn clone(&self) -> Self {
        let root = match self.root_type {
            NonLeaf => ChildPtr::non_leaf(unsafe { self.root.as_non_leaf_unchecked() }.clone()),
            Leaf => ChildPtr::leaf(unsafe { self.root.as_leaf_unchecked() }.clone()),
        };

        Self {
            total: self.total,
            root_type: self.root_type,
            root,
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

    pub unsafe fn from_sorted_unchecked(positions_and_counts: &[(P, C)]) -> Self {
        // let total: C = sorted.iter_mut().fold(C::zero(), |total, (_, count)| {
        //     let old_count = *count;
        //     *count = total;
        //     old_count + total //TODO: this might wrap
        // });

        todo!()
    }

    pub fn total(&self) -> C {
        self.total
    }

    pub fn iter(&self) -> impl Iterator<Item = (P, C)> + '_ {
        Iter::new(self)
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
        // temporary dummy `data.head`, then replace the old root by the new root, and then
        // replace the new root's dummy `data.head` with the old root.
        let new_root = ChildPtr::non_leaf(NonLeafNode::new(
            self.root_type,
            right_sibling,
            BoundedPairOfVecs::new(),
        ));
        let old_root = core::mem::replace(&mut self.root, new_root);
        let new_root = unsafe { self.root.as_non_leaf_mut_unchecked() };
        let right_sibling = core::mem::replace(&mut new_root.data.head, old_root);

        let separator = Entry::new(separator_pos, ejected_accum);
        new_root
            .data
            .bulk
            .try_push(separator, right_sibling)
            .expect("vector is empty and `CAP>0`");
        self.root_type = NonLeaf;
    }

    pub fn remove(&mut self, pos: P, count: C) -> Result<(), ()> {
        if count == C::default() {
            return Ok(());
        }

        match self.root_type {
            NonLeaf => {
                let root = unsafe { self.root.as_non_leaf_mut_unchecked() };
                root.remove(pos, count)?;
                if root.data.bulk.len() == 0 {
                    let old_root = core::mem::replace(&mut self.root, ChildPtr::none());
                    let old_root = unsafe { old_root.into_non_leaf_unchecked() };
                    self.root_type = old_root.children_type;
                    self.root = old_root.leak_data().head;
                }
            }
            Leaf => unsafe { self.root.as_leaf_mut_unchecked() }.remove(pos, count)?,
            // Leaf => unsafe { self.root.as_leaf_mut_unchecked() }.remove(pos, count),
        }

        self.total = self.total - count;
        Ok(())
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
            let separators = node.data.bulk.first_as_ref();
            let index = separators.partition_point(|entry| entry.pos < pos);
            if let Some(entry) = separators.get(index.wrapping_sub(1)) {
                accum = accum + entry.accum
            }
            node_type = node.children_type;
            node_ref = node
                .data
                .bulk
                .second_as_ref()
                .get(index.wrapping_sub(1))
                .unwrap_or(&node.data.head);
        }
        let leaf_node = unsafe { node_ref.as_leaf_unchecked() };

        let index = leaf_node
            .data
            .bulk
            .first_as_ref()
            .partition_point(|entry| entry.pos < pos);
        accum = accum
            + leaf_node
                .data
                .bulk
                .get(index.wrapping_sub(1))
                .map(|(entry, ())| entry.accum)
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
            let separators = node.data.bulk.first_as_ref();
            let index = separators.partition_point(|entry| entry.accum <= remaining);
            if let Some(right_separator) = separators.get(index) {
                right_bound = Some(right_separator.pos);
            }
            if let Some(left_separator) = separators.get(index.wrapping_sub(1)) {
                remaining = remaining - left_separator.accum;
            }
            node_type = node.children_type;
            node_ref = node
                .data
                .bulk
                .second_as_ref()
                .get(index.wrapping_sub(1))
                .unwrap_or(&node.data.head);
        }
        let leaf_node = unsafe { node_ref.as_leaf_unchecked() };

        let index = leaf_node
            .data
            .bulk
            .first_as_ref()
            .partition_point(|entry| entry.accum <= remaining);
        leaf_node
            .data
            .bulk
            .get(index)
            .map(|(entry, ())| entry.pos)
            .or(right_bound)
    }
}

impl<P, C, const CAP: usize> Default for AugmentedBTree<P, C, CAP>
where
    P: Ord + Copy,
    C: Ord + Default + Copy + Add<Output = C> + Sub<Output = C>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<P, C, L, const CAP: usize> Payload<P, C, L, CAP> {
    fn new(head: L, bulk: BoundedPairOfVecs<Entry<P, C>, L, CAP>) -> Self {
        Self { head, bulk }
    }
}

impl<P, C, L, const CAP: usize> Payload<P, C, L, CAP>
where
    P: Copy,
    C: Copy + Default + Add<Output = C> + Sub<Output = C>,
{
    fn by_key_mut_update_right<X: Ord>(
        &mut self,
        key: X,
        get_key: impl Fn(&Entry<P, C>) -> X,
        update_right: impl Fn(&mut Entry<P, C>),
    ) -> Option<(usize, &mut L)> {
        let separators = self.bulk.first_as_mut();
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
            .bulk
            .second_as_mut()
            .get_mut(index.wrapping_sub(1))
            .unwrap_or(&mut self.head);

        Some((index, child_ref))
    }

    fn rebalance_after_underflow_before<CN>(
        &mut self,
        index: usize,
        downcast_mut: impl Fn(&mut L) -> &mut CN,
        into_downcast: impl Fn(L) -> CN,
    ) where
        CN: Node<P, C, CAP>,
    {
        // Child node underflowed. Check if we can steal some entries from one of
        // its neighbors. If not, merge three children into two.

        let (mut left, mut right) = self.bulk.get_all_mut().split_at_mut(index);
        let mut left_iter = left.iter_mut().rev();
        let (child, left, left_neighbor_len, left_left_accum) =
            if let Some((left_separator, child)) = left_iter.next() {
                let (left_left_accum, left_neighbor) = left_iter
                    .next()
                    .map(|(e, c)| (e.accum, c))
                    .unwrap_or((C::default(), &mut self.head));
                let left_neighbor = downcast_mut(left_neighbor);
                let left_neighbor_len = left_neighbor.data_ref().bulk.len();
                (
                    downcast_mut(child),
                    Some((left_neighbor, left_separator)),
                    left_neighbor_len,
                    left_left_accum,
                )
            } else {
                (downcast_mut(&mut self.head), None, 0, C::default())
            };
        core::mem::drop(left_iter);
        let right = right.iter_mut().next().map(|(e, c)| (e, downcast_mut(c)));
        let right_neighbor_len = right
            .as_ref()
            .map(|(_, c)| c.data_ref().bulk.len())
            .unwrap_or_default();

        if left_neighbor_len > right_neighbor_len && left_neighbor_len > CAP / 2 {
            // Steal data from the end of `left_neighbor`.
            let (left_neighbor, mut separator) =
                left.expect("`left_neighbor_len > 0`, so it exists");
            let new_left_neighbor_len = CAP / 2 + (left_neighbor_len - CAP / 2) / 2;

            let mut stolen = left_neighbor
                .data_mut()
                .bulk
                .chop(new_left_neighbor_len)
                .expect("we steal at least one");
            let (first_stolen_entry, new_first_child) =
                stolen.pop_front().expect("we steal at least one");

            let adjust_accums_left = first_stolen_entry.accum;
            let new_separator = Entry::new(
                first_stolen_entry.pos,
                left_left_accum + first_stolen_entry.accum,
            );
            let adjust_accums_right = separator.accum - new_separator.accum;

            let old_separator = core::mem::replace(separator, new_separator);
            let last_stolen_entry = Entry::new(old_separator.pos, adjust_accums_right);
            let old_first_child = core::mem::replace(&mut child.data_mut().head, new_first_child);

            child
                .data_mut()
                .bulk
                .try_prepend_and_update1(
                    stolen,
                    last_stolen_entry,
                    old_first_child,
                    |entry| Entry::new(entry.pos, entry.accum - adjust_accums_left),
                    |entry| Entry::new(entry.pos, entry.accum + adjust_accums_right),
                )
                .expect("can't overflow");
        } else if right_neighbor_len > CAP / 2 {
            // Steal data from the beginning of `right_neighbor``.
            let (separator, right_neighbor) =
                right.expect("`right_neighbor_len > 0`, so it exists.");
            let right_neighbor = right_neighbor.data_mut();
            let new_right_neighbor_len = CAP / 2 + (right_neighbor_len - CAP / 2) / 2;
            let amt_steal = right_neighbor_len - new_right_neighbor_len;

            let adjust_accums_right = right_neighbor
                .bulk
                .get(amt_steal - 1)
                .expect("we steal less than available but at least one")
                .0
                .accum;
            let mut stolen = right_neighbor
                .bulk
                .chop_front(amt_steal, |entry| {
                    Entry::new(entry.pos, entry.accum - adjust_accums_right)
                })
                .expect("we steal at least one");
            let (last_stolen_entry, new_first_child) =
                stolen.pop().expect("we steal at least one.");

            let new_separator = Entry::new(
                last_stolen_entry.pos,
                separator.accum + last_stolen_entry.accum,
            );
            let old_separator = core::mem::replace(separator, new_separator);

            let left_accum = left.map(|(_, entry)| entry.accum).unwrap_or_default();
            let adjust_accums_left = old_separator.accum - left_accum;

            let first_stolen_entry = Entry::new(old_separator.pos, adjust_accums_left);
            let old_first_child = core::mem::replace(&mut right_neighbor.head, new_first_child);

            child
                .data_mut()
                .bulk
                .try_push(first_stolen_entry, old_first_child)
                .expect("can't overflow");
            child
                .data_mut()
                .bulk
                .try_append_guarded_transform1(stolen, |entry| {
                    Entry::new(entry.pos, entry.accum + adjust_accums_left)
                })
                .expect("can't overflow");
        } else {
            // Can't steal any data.bulk from neighbors. We need to reduce the number
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
            // data.bulk (+ separators) across `A` and `C` (if existent) (+ separator)
            // such that `A` and `C` have equally many data.bulk (up to 1 if the total
            // number is odd). This always works without moving any data.bulk between
            // `A` and `C` because, in all cases considered above, either the set
            // `{A, B}` or the set `{B, C}` (or both) consist of node `i`, which has
            // exactly `CAP / 2 - 1` data.bulk (because it just underflowed), and one
            // of its neighbors, which has exactly `CAP / 2` data.bulk (because we
            // couldn't steal from it any more). Therefore, even if we merged all
            // data.bulk of these two nodes and the separator into one node, we would
            // end up with a merged node with `2 * (CAP / 2) - 1 + 1 ∈ {CAP - 1, CAP}`
            // data.bulk, and the other node, which has at least `CAP / 2` and at most
            // `CAP` data.bulk.

            let index_b = index.clamp(1, self.bulk.len()) - 1;
            let (separator_ab, child_b) = self.bulk.remove(index_b).expect("`index_b < len`");
            let accum_before_a = self
                .bulk
                .first_as_ref()
                .get(index_b.wrapping_sub(1))
                .map(|entry| entry.accum)
                .unwrap_or_default();
            let mut child_b = into_downcast(child_b).leak_data();

            let (separators, children) = self.bulk.both_as_mut();
            let (children_until_a, children_from_c) = children.split_at_mut(index_b);
            let child_a = children_until_a.last_mut().unwrap_or(&mut self.head);
            let child_a = downcast_mut(child_a);
            let child_c = children_from_c.first_mut();

            let separated_child_c = child_c.and_then(|c| {
                let c = downcast_mut(c);
                if c.data_ref().bulk.len() >= 2 * (CAP / 2) {
                    // Ignore `child_c` for the merger since it wouldn't be affected
                    // by it anyway but would make the logic below more complicated.
                    None
                } else {
                    Some((&mut separators[index_b], c))
                }
            });

            if let Some((separator_bc, child_c)) = separated_child_c {
                // We're merging three children `[A, B, C]` into two `[A', C']`.
                let len_a = child_a.data_ref().bulk.len();
                let len_b = child_b.bulk.len();
                let len_c = child_c.data_ref().bulk.len();
                let num_grandchildren = len_a + len_b + len_c + 1;
                let target_len_a = num_grandchildren / 2;
                let increase_len_a = target_len_a - len_a;
                // `increase_len_a <= CAP / 2`, where equality requires that `child_a` underflowed.
                // Thus, `increase_len_a <= len_b`.

                if increase_len_a == 0 {
                    // Special case: `child_a` is not actually a part of the merger.
                    let adjust_accums_c = separator_bc.accum - separator_ab.accum;
                    let separator_bc = core::mem::replace(separator_bc, separator_ab);
                    let first_stolen_entry = Entry::new(separator_bc.pos, adjust_accums_c);
                    let first_child_c =
                        core::mem::replace(&mut child_c.data_mut().head, child_b.head);

                    child_c
                        .data_mut()
                        .bulk
                        .try_prepend_and_update1(
                            child_b.bulk.take_all(),
                            first_stolen_entry,
                            first_child_c,
                            |entry| entry,
                            |entry| Entry::new(entry.pos, entry.accum + adjust_accums_c),
                        )
                        .expect("can't overflow");
                } else {
                    let child_c = child_c.data_mut();
                    let mut tail_b = child_b
                        .bulk
                        .chop(increase_len_a - 1)
                        .expect("0 < increase_len_a <= len_b");
                    let (mid_point, new_first_child_c) =
                        tail_b.pop_front().expect("0 < increase_len_a <= len_b");
                    let new_separator =
                        Entry::new(mid_point.pos, mid_point.accum + separator_ab.accum);
                    let adjust_accums_c_left = new_separator.accum - separator_ab.accum;
                    let adjust_accums_c_right = separator_bc.accum - new_separator.accum;
                    let tail_b_last_entry = Entry::new(separator_bc.pos, adjust_accums_c_right);
                    let old_first_child_c =
                        core::mem::replace(&mut child_c.head, new_first_child_c);
                    *separator_bc = new_separator;

                    child_c
                        .bulk
                        .try_prepend_and_update1(
                            tail_b,
                            tail_b_last_entry,
                            old_first_child_c,
                            |entry| Entry::new(entry.pos, entry.accum - adjust_accums_c_left),
                            |entry| Entry::new(entry.pos, entry.accum + adjust_accums_c_right),
                        )
                        .expect("can't overflow");

                    let adjust_accums_a_right = separator_ab.accum - accum_before_a;
                    let head_b_first_entry = Entry::new(separator_ab.pos, adjust_accums_a_right);
                    child_a
                        .data_mut()
                        .bulk
                        .try_push(head_b_first_entry, child_b.head)
                        .expect("increase_len_a > 0");
                    child_a
                        .data_mut()
                        .bulk
                        .try_append_transform1(child_b.bulk.take_all(), |entry| {
                            Entry::new(entry.pos, entry.accum + adjust_accums_a_right)
                        })
                        .expect("can't overflow");
                }
            } else {
                // We're merging two children `[A, B]` into one `A'`.
                let adjust_accums = separator_ab.accum - accum_before_a;
                let first_stolen = Entry::new(separator_ab.pos, adjust_accums);
                child_a
                    .data_mut()
                    .bulk
                    .try_push(first_stolen, child_b.head)
                    .expect("increase_len_a > 0");
                child_a
                    .data_mut()
                    .bulk
                    .try_append_transform1(child_b.bulk.take_all(), |entry| {
                        Entry::new(entry.pos, entry.accum + adjust_accums)
                    })
                    .expect("can't overflow");
            }
        }
    }
}

impl<P, C, const CAP: usize> Node<P, C, CAP> for NonLeafNode<P, C, CAP>
where
    P: Ord + Copy,
    C: Default + Ord + Add<Output = C> + Sub<Output = C> + Copy,
{
    type ChildRef = ChildPtr<P, C, CAP>;

    fn remove(&mut self, pos: P, count: C) -> Result<(), ()> {
        let data = self.data.deref_mut();
        let (separators, children) = data.bulk.both_as_mut();
        let index = separators.partition_point(|entry| entry.pos < pos);
        let child = children
            .get_mut(index.wrapping_sub(1))
            .unwrap_or(&mut data.head);
        let (left_separators, right_separators) = separators.split_at_mut(index);
        let mut right_iter = right_separators.iter_mut();

        let mut found = false;
        if let Some(right_separator) = right_iter.next() {
            if right_separator.accum < count {
                return Err(());
            }
            let new_accum = right_separator.accum - count;

            found = right_separator.pos == pos;
            if found {
                let previous_accum = left_separators
                    .last()
                    .map(|entry| entry.accum)
                    .unwrap_or_default();
                if previous_accum > new_accum {
                    return Err(());
                }
                let new_accum_in_left_subtree = new_accum - previous_accum;
                match self.children_type {
                    NonLeaf => {
                        let child = unsafe { child.as_non_leaf_mut_unchecked() };
                        let last_entry_left =
                            child.get_last_and_remove_if_accum_is(new_accum_in_left_subtree);
                        match last_entry_left.accum.cmp(&new_accum_in_left_subtree) {
                            Equal => {
                                // Removed all counts at `pos`. Replace the separator with the last
                                // entry of the left subtree, which we just removed.
                                right_separator.pos = last_entry_left.pos;
                            }
                            Greater => return Err(()),
                            Less => (),
                        }
                    }
                    Leaf => {
                        let child = unsafe { child.as_leaf_mut_unchecked() };
                        let last_entry_left =
                            child.get_last_and_remove_if_accum_is(new_accum_in_left_subtree);
                        match last_entry_left.accum.cmp(&new_accum_in_left_subtree) {
                            Equal => {
                                // Removed all counts at `pos`. Replace the separator with the last
                                // entry of the left subtree, which we just removed.
                                right_separator.pos = last_entry_left.pos;
                            }
                            Greater => return Err(()),
                            Less => (),
                        }
                    }
                };
            }

            right_separator.accum = new_accum;
            for entry in right_iter {
                entry.accum = entry.accum - count;
            }
        }

        'dispatch: {
            match self.children_type {
                NonLeaf => {
                    let child = unsafe { child.as_non_leaf_mut_unchecked() };
                    if !found && child.remove(pos, count).is_err() {
                        break 'dispatch;
                    }
                    if child.data.bulk.len() < CAP / 2 {
                        self.data.rebalance_after_underflow_before(
                            index,
                            |c| unsafe { c.as_non_leaf_mut_unchecked() },
                            |c| unsafe { *c.into_non_leaf_unchecked() },
                        );
                    }
                    return Ok(());
                }
                Leaf => {
                    let child = unsafe { child.as_leaf_mut_unchecked() };
                    if !found && child.remove(pos, count).is_err() {
                        break 'dispatch;
                    }
                    if child.data.bulk.len() < CAP / 2 {
                        self.data.rebalance_after_underflow_before(
                            index,
                            |c| unsafe { c.as_leaf_mut_unchecked() },
                            |c| unsafe { *c.into_leaf_unchecked() },
                        );
                    }
                    return Ok(());
                }
            }
        }

        // Entry wasn't found. Clean up any modifications we've done optimistically.
        for entry in &mut separators[index..] {
            entry.accum = entry.accum + count;
        }

        Err(())
    }

    fn data_ref(&self) -> &Payload<P, C, Self::ChildRef, CAP> {
        &self.data
    }

    fn data_mut(&mut self) -> &mut Payload<P, C, Self::ChildRef, CAP> {
        &mut self.data
    }

    fn leak_data(mut self) -> Payload<P, C, Self::ChildRef, CAP> {
        unsafe {
            let data = ManuallyDrop::take(&mut self.data);
            core::mem::forget(self);
            data
        }
    }
}

impl<P, C, const CAP: usize> Node<P, C, CAP> for LeafNode<P, C, CAP>
where
    P: Ord + Copy,
    C: Default + Ord + Add<Output = C> + Sub<Output = C> + Copy,
{
    type ChildRef = ();

    fn remove(&mut self, pos: P, count: C) -> Result<(), ()> {
        let entries = self.data.bulk.first_as_mut();
        let index = entries.partition_point(|entry| entry.pos < pos);
        let previous_accum = entries
            .get(index.wrapping_sub(1))
            .map(|entry| entry.accum)
            .unwrap_or_default();

        let mut right_iter = entries.iter_mut().skip(index);
        let entry = right_iter.next().ok_or(())?;
        if entry.pos != pos || entry.accum < previous_accum + count {
            return Err(());
        }
        if entry.accum == previous_accum + count {
            self.data
                .bulk
                .remove_and_update1(index, |entry| Entry::new(entry.pos, entry.accum - count))
                .expect("it exists");
        } else {
            entry.accum = entry.accum - count;
            for entry in right_iter {
                entry.accum = entry.accum - count;
            }
        }

        Ok(())
    }

    fn data_ref(&self) -> &Payload<P, C, Self::ChildRef, CAP> {
        &self.data
    }

    fn data_mut(&mut self) -> &mut Payload<P, C, Self::ChildRef, CAP> {
        &mut self.data
    }

    fn leak_data(self) -> Payload<P, C, Self::ChildRef, CAP> {
        self.data
    }
}

impl<P, C, const CAP: usize> NonLeafNode<P, C, CAP>
where
    P: Ord + Copy,
    C: Default + Ord + Add<Output = C> + Sub<Output = C> + Copy,
{
    fn new(
        children_type: NodeType,
        head: ChildPtr<P, C, CAP>,
        bulk: BoundedPairOfVecs<Entry<P, C>, ChildPtr<P, C, CAP>, CAP>,
    ) -> Self {
        Self {
            children_type,
            data: ManuallyDrop::new(Payload::new(head, bulk)),
        }
    }

    fn insert(&mut self, pos: P, count: C) -> Option<(P, C, ChildPtr<P, C, CAP>)> {
        let (insert_index, child) = self.data.by_key_mut_update_right(
            pos,
            |entry| entry.pos,
            |entry| entry.accum = entry.accum + count,
        )?;

        let Some((separator_pos, ejected_accum, new_child)) = (match self.children_type {
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
            .data
            .bulk
            .first_as_ref()
            .get(insert_index.wrapping_sub(1))
            .map(|entry| entry.accum)
            .unwrap_or_default();

        let mut separator = Entry::new(separator_pos, preceding_accum + ejected_accum);
        let Err((_, new_child)) = self
            .data
            .bulk
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
                self.data
                    .bulk
                    .first_as_ref()
                    .get(CAP / 2 - 1)
                    .expect("CAP / 2 > index >= 0")
                    .accum
            };
            let chopped_off = self.data.bulk.chop(CAP / 2).expect("node is full");
            right_separated_children
                .try_append_transform1(chopped_off, |entry| {
                    Entry::new(entry.pos, entry.accum - accum_of_ejected)
                })
                .expect("can't overflow");
            self.data
                .bulk
                .try_insert(insert_index, separator, new_child)
                .ok()
                .expect("there are `CAP - CAP/2` vacancies, which is >0 because CAP>0.");
        } else {
            // Insert into `right_sibling`.
            let insert_index = insert_index - (CAP / 2 + 1);
            let accum_of_ejected = self
                .data
                .bulk
                .first_as_ref()
                .get(CAP / 2)
                .expect("CAP/2 < original insert_index <= len")
                .accum;

            // Build `right_sibling`'s list of separators.
            let chopped_off = self
                .data
                .bulk
                .chop(CAP / 2 + 1)
                .expect("CAP/2 < original insert_index <= len");
            let (before_insert, after_insert) = chopped_off.split_at_mut(insert_index);
            right_separated_children
                .try_append_transform1(before_insert, |entry| {
                    Entry::new(entry.pos, entry.accum - accum_of_ejected)
                })
                .expect("can't overflow");
            separator.accum = separator.accum - accum_of_ejected;
            right_separated_children
                .try_push(separator, new_child)
                .expect("can't overflow");
            right_separated_children
                .try_append_transform1(after_insert, |entry| {
                    Entry::new(entry.pos, entry.accum - accum_of_ejected)
                })
                .expect("can't overflow");
        };

        let (ejected_separator, right_first_child) = self
            .data
            .bulk
            .pop()
            .expect("there are CAP/2+1 > 0 data.bulk");

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

    /// May return a wrong separator if accum of the last separator is higher than the argument
    /// `accum`. But in this case, the returned entry will also have an accum that is higher than
    /// the argument `accum`.
    fn get_last_and_remove_if_accum_is(&mut self, accum: C) -> Entry<P, C> {
        let len = self.data.bulk.len();
        let (&mut last_separator, last_child) = self
            .data
            .bulk
            .last_mut()
            .expect("invariant bulk.len() >= 2");
        let offset = last_separator.accum;
        if offset > accum {
            last_separator // Not the last separator but already with higher accum.
        } else {
            let accum = accum - offset;
            let mut last = match self.children_type {
                NonLeaf => {
                    let last_child = unsafe { last_child.as_non_leaf_mut_unchecked() };
                    let last = last_child.get_last_and_remove_if_accum_is(accum);
                    if last_child.data.bulk.len() < CAP / 2 {
                        self.data.rebalance_after_underflow_before(
                            len,
                            |c| unsafe { c.as_non_leaf_mut_unchecked() },
                            |c| unsafe { *c.into_non_leaf_unchecked() },
                        );
                    }
                    last
                }
                Leaf => {
                    let last_child = unsafe { last_child.as_leaf_mut_unchecked() };
                    let last = last_child.get_last_and_remove_if_accum_is(accum);
                    if last_child.data.bulk.len() < CAP / 2 {
                        self.data.rebalance_after_underflow_before(
                            len,
                            |c| unsafe { c.as_leaf_mut_unchecked() },
                            |c| unsafe { *c.into_leaf_unchecked() },
                        );
                    }
                    last
                }
            };
            last.accum = last.accum + offset;
            last
        }
    }
}

impl<P, C, const CAP: usize> LeafNode<P, C, CAP>
where
    P: Copy + Ord,
    C: Copy + Default + Add<Output = C> + Sub<Output = C> + PartialEq,
{
    fn new() -> Self {
        Self {
            data: Payload::new((), BoundedPairOfVecs::new()),
        }
    }

    fn insert(&mut self, pos: P, count: C) -> Option<(P, C, ChildPtr<P, C, CAP>)> {
        // Check if the node already contains an entry with at the given `pos`.
        // If so, increment its accum and all accums to the right, then return.
        let insert_index = self
            .data
            .bulk
            .first_as_ref()
            .partition_point(|entry| entry.pos < pos);
        let mut right_iter = self.data.bulk.iter_mut().skip(insert_index);
        match right_iter.next() {
            Some((right_entry, ())) if right_entry.pos == pos => {
                right_entry.accum = right_entry.accum + count;
                for (entry, ()) in right_iter {
                    entry.accum = entry.accum + count;
                }
                return None;
            }
            _ => {}
        }
        core::mem::drop(right_iter);

        // An entry at position `pos` doesn't exist yet. Create a new one and insert it.
        let old_accum = self
            .data
            .bulk
            .get(insert_index.wrapping_sub(1))
            .map(|(node, ())| node.accum)
            .unwrap_or_default();
        let mut insert_entry = Entry::new(pos, old_accum + count);

        if self
            .data
            .bulk
            .try_insert_and_update1(insert_index, insert_entry, (), |entry| {
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
                self.data
                    .bulk
                    .first_as_ref()
                    .get(CAP / 2 - 1)
                    .expect("node is full and CAP/2 > 0")
                    .accum
            };

            let right_entries = self.data.bulk.chop(CAP / 2).expect("node is full");

            right_sibling
                .data
                .bulk
                .try_append_transform1(right_entries, |entry| {
                    Entry::new(entry.pos, entry.accum - old_weight_before_right_sibling)
                })
                .expect("can't overflow");

            self.data
                .bulk
                .try_insert_and_update1(insert_index, insert_entry, (), |entry| {
                    Entry::new(entry.pos, entry.accum + count)
                })
                .expect("there are `CAP - CAP/2` vacancies, which is >0 because CAP>0.");
        } else {
            // We're inserting into the right half.
            let weight_before_right_sibling = self
                .data
                .bulk
                .first_as_ref()
                .get(CAP / 2)
                .expect("CAP/2 < insert_index <= len")
                .accum;

            let right_entries = self
                .data
                .bulk
                .chop(CAP / 2 + 1)
                .expect("CAP/2 < insert_index <= len");

            let (before_insert, after_insert) =
                right_entries.split_at_mut(insert_index - (CAP / 2 + 1));

            right_sibling
                .data
                .bulk
                .try_append_transform1(before_insert, |entry| {
                    Entry::new(entry.pos, entry.accum - weight_before_right_sibling)
                })
                .expect("can't overflow");

            insert_entry.accum = insert_entry.accum - weight_before_right_sibling;
            right_sibling
                .data
                .bulk
                .try_push(insert_entry, ())
                .expect("can't overflow");

            right_sibling
                .data
                .bulk
                .try_append_transform1(after_insert, |entry| {
                    Entry::new(entry.pos, entry.accum - weight_before_right_sibling + count)
                })
                .expect("can't overflow");
        };

        let (ejected_entry, ()) = self
            .data
            .bulk
            .pop()
            .expect("there are CAP/2+1 > 0 data.bulk");
        let right_sibling_ref = ChildPtr::leaf(right_sibling);

        Some((ejected_entry.pos, ejected_entry.accum, right_sibling_ref))
    }

    fn get_last_and_remove_if_accum_is(&mut self, accum: C) -> Entry<P, C> {
        let (&mut last, ()) = self.data.bulk.last_mut().expect("isn't empty");
        if last.accum == accum {
            self.data.bulk.pop();
        }
        last
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
                let head = unsafe { self.data.head.as_non_leaf_unchecked() };
                f.entry(head);
                for (separator, child) in self.data.bulk.iter() {
                    f.entry(separator);
                    f.entry(unsafe { child.as_non_leaf_unchecked() });
                }
            }
            Leaf => {
                let head = unsafe { self.data.head.as_leaf_unchecked() };
                f.entry(head);
                for (separator, child) in self.data.bulk.iter() {
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
        let mut iter = self.data.bulk.iter();
        if let Some((first, ())) = iter.next() {
            write!(f, "{first}")?;
            for (entry, ()) in iter {
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

struct Iter<'a, P, C, const CAP: usize> {
    stack: Vec<(&'a NonLeafNode<P, C, CAP>, C, usize)>,
    leaf: Option<(&'a LeafNode<P, C, CAP>, C, usize)>,
}

impl<'a, P, C: Default + Add<Output = C>, const CAP: usize> Iter<'a, P, C, CAP> {
    fn new(tree: &'a AugmentedBTree<P, C, CAP>) -> Self {
        let mut stack = Vec::new();
        let mut node = &tree.root;
        let mut node_type = tree.root_type;
        while node_type == NonLeaf {
            let non_leaf_node = unsafe { node.as_non_leaf_unchecked() };
            node = &non_leaf_node.data.head;
            node_type = non_leaf_node.children_type;
            stack.push((non_leaf_node, C::default(), 0))
        }

        let leaf_node = unsafe { node.as_leaf_unchecked() };
        Self {
            stack,
            leaf: Some((leaf_node, C::default(), 0)),
        }
    }
}

impl<'a, P, C: Default + Add<Output = C>, const CAP: usize> Iterator for Iter<'a, P, C, CAP>
where
    P: Copy,
    C: Copy,
{
    type Item = (P, C);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((leaf_node, accum, index)) = &mut self.leaf {
            if let Some(&entry) = leaf_node.data.bulk.first_as_ref().get(*index) {
                *index += 1;
                Some((entry.pos, *accum + entry.accum))
            } else {
                self.leaf = None;
                while let Some((non_leaf_node, accum, index)) = self.stack.pop() {
                    if let Some(&entry) = non_leaf_node.data.bulk.first_as_ref().get(index) {
                        self.stack.push((non_leaf_node, accum, index));
                        return Some((entry.pos, accum + entry.accum));
                    }
                }
                None
            }
        } else {
            let (parent, parent_accum, index) = self
                .stack
                .last_mut()
                .expect("the tree has at least one node");
            let mut node_type = parent.children_type;
            let (left_separator, mut node) = parent
                .data
                .bulk
                .get(*index)
                .expect("every separator has a right child");
            let accum = *parent_accum + left_separator.accum;
            *index += 1;

            while node_type == NonLeaf {
                let non_leaf_node = unsafe { node.as_non_leaf_unchecked() };
                node = &non_leaf_node.data.head;
                node_type = non_leaf_node.children_type;
                self.stack.push((non_leaf_node, accum, 0))
            }

            let leaf_node = unsafe { node.as_leaf_unchecked() };
            let entry = leaf_node
                .data
                .bulk
                .first_as_ref()
                .first()
                .expect("can'b be empty because it's not the root.");
            self.leaf = Some((leaf_node, accum, 1));
            Some((entry.pos, accum + entry.accum))
        }
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
        none: (),
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

        pub fn none() -> ChildPtr<P, C, CAP> {
            Self { none: () }
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
        marker::PhantomData,
        mem::MaybeUninit,
        ops::{Deref, DerefMut},
        slice::{from_raw_parts, from_raw_parts_mut, SliceIndex},
    };

    /// Semantically equivalent to `BoundedVec<(T1, T2), CAP>`, but with all `T1`s stored in one
    /// array and all `T2` stored in a different array, thus improving memory locality for
    /// algorithms that access one part more often than the other
    #[derive(Debug)]
    pub struct BoundedPairOfVecs<T1, T2, const CAP: usize> {
        len: usize,
        buf1: [MaybeUninit<T1>; CAP],
        buf2: [MaybeUninit<T2>; CAP],
        phantom: PhantomData<(T1, T2)>,
    }

    impl<T1, T2, const CAP: usize> Drop for BoundedPairOfVecs<T1, T2, CAP> {
        fn drop(&mut self) {
            while let Some(pair) = self.pop() {
                core::mem::drop(pair);
            }
        }
    }

    impl<T1: Copy, T2: Copy, const CAP: usize> Clone for BoundedPairOfVecs<T1, T2, CAP> {
        fn clone(&self) -> Self {
            Self {
                len: self.len,
                buf1: self.buf1,
                buf2: self.buf2,
                phantom: PhantomData,
            }
        }
    }

    /// A pair of two slices of equal length up to `BOUND` that are conceptually owned, i.e., one
    /// can safely move the data.bulk out (e.g., by passing an `OwnedBoundedPairOfSlices` as an
    /// argument to `BoundedPairOfVecs::try_append`).
    ///
    /// Doesn't drop its content when dropped.
    #[derive(Debug)]
    pub struct OwnedBoundedPairOfSlices<'a, T1, T2, const BOUND: usize> {
        len: usize,
        head1: *mut MaybeUninit<T1>,
        head2: *mut MaybeUninit<T2>,
        phantom: PhantomData<(&'a mut T1, &'a mut T2)>,
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

    /// A pair of two slices of equal length up to `BOUND`. In contrast to
    /// [`OwnedBoundedPairOfSlices`], the data.bulk of a `BoundedPairVecsViewMut` are not owned by
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
            Self {
                len: 0,
                buf1,
                buf2,
                phantom: PhantomData,
            }
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

        pub fn try_insert_and_update1(
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
                        self.buf1.get_unchecked(self.len).assume_init_read(),
                        self.buf2.get_unchecked(self.len).assume_init_read(),
                    )
                };
                Some(last)
            }
        }

        pub fn last_mut(&mut self) -> Option<(&mut T1, &mut T2)> {
            if self.len == 0 {
                None
            } else {
                let last = unsafe {
                    (
                        self.buf1.get_unchecked_mut(self.len - 1).assume_init_mut(),
                        self.buf2.get_unchecked_mut(self.len - 1).assume_init_mut(),
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
                        self.buf1.get_unchecked(index).assume_init_read(),
                        self.buf2.get_unchecked(index).assume_init_read(),
                    )
                };

                unsafe {
                    self.buf1.get_unchecked_mut(index..self.len).rotate_left(1);
                    self.buf2.get_unchecked_mut(index..self.len).rotate_left(1);
                }
                self.len -= 1;

                Some(items)
            }
        }

        pub fn remove_and_update1(
            &mut self,
            index: usize,
            update1: impl Fn(T1) -> T1,
        ) -> Option<(T1, T2)> {
            if index >= self.len {
                None
            } else {
                let items = unsafe {
                    (
                        self.buf1.get_unchecked(index).assume_init_read(),
                        self.buf2.get_unchecked(index).assume_init_read(),
                    )
                };

                unsafe {
                    self.buf2.get_unchecked_mut(index..self.len).rotate_left(1);
                    for read_index in index + 1..self.len {
                        let tmp = self.buf1.get_unchecked(read_index).assume_init_read();
                        self.buf1
                            .get_unchecked_mut(read_index - 1)
                            .write(update1(tmp));
                    }
                }

                self.len -= 1;

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
            if start_index > self.len {
                return None;
            }
            let len = self.len - start_index;
            let head1 = unsafe { (&mut self.buf1 as *mut MaybeUninit<T1>).add(start_index) };
            let head2 = unsafe { (&mut self.buf2 as *mut MaybeUninit<T2>).add(start_index) };
            self.len = start_index;
            Some(OwnedBoundedPairOfSlices {
                len,
                head1,
                head2,
                phantom: PhantomData,
            })
        }

        pub fn chop_front<'a>(
            &'a mut self,
            len: usize,
            update1: impl Fn(T1) -> T1,
        ) -> Option<
            DropGuard<
                OwnedBoundedPairOfSlices<'a, T1, T2, CAP>,
                impl FnMut(&mut OwnedBoundedPairOfSlices<'a, T1, T2, CAP>),
            >,
        > {
            if len > self.len {
                return None;
            }

            let head1 = &mut self.buf1 as *mut _;
            let head2 = &mut self.buf2 as *mut _;
            self.len -= len;
            let len_after_chop = self.len;

            Some(DropGuard {
                inner: OwnedBoundedPairOfSlices {
                    len,
                    head1,
                    head2,
                    phantom: PhantomData,
                },
                cleanup: move |slices| unsafe {
                    let head1 = slices.head1;
                    let head2 = slices.head2;
                    core::ptr::copy(head2.add(len), head2, len_after_chop);
                    for write_index in 0..len_after_chop {
                        let tmp = (*head1.add(write_index + len)).assume_init_read();
                        (*head1.add(write_index)).write(update1(tmp));
                    }
                },
            })
        }

        pub fn take_all(&mut self) -> OwnedBoundedPairOfSlices<'_, T1, T2, CAP> {
            let len = self.len;
            let head1 = &mut self.buf1 as *mut _;
            let head2 = &mut self.buf2 as *mut _;

            self.len = 0;

            OwnedBoundedPairOfSlices {
                len,
                head1,
                head2,
                phantom: PhantomData,
            }
        }

        pub fn get_all_mut(&mut self) -> BoundedPairOfVecsViewMut<'_, T1, T2, CAP> {
            unsafe {
                BoundedPairOfVecsViewMut(
                    core::mem::transmute(self.buf1.get_unchecked_mut(..self.len)),
                    core::mem::transmute(self.buf2.get_unchecked_mut(..self.len)),
                )
            }
        }

        pub fn try_append<const BOUND: usize>(
            &mut self,
            tail: OwnedBoundedPairOfSlices<'_, T1, T2, BOUND>,
        ) -> Result<(), ()> {
            let tail_len = tail.len();
            if self.len + tail_len > CAP {
                return Err(());
            }

            let dst1 = unsafe { (&mut self.buf1 as *mut MaybeUninit<T1>).add(self.len) };
            let dst2 = unsafe { (&mut self.buf2 as *mut MaybeUninit<T2>).add(self.len) };

            unsafe {
                core::ptr::copy_nonoverlapping(tail.head1, dst1, tail_len);
                core::ptr::copy_nonoverlapping(tail.head2, dst2, tail_len);
            }
            self.len += tail_len;

            Ok(())
        }

        /// Prepends the concatenation of `head_begin` and `head_end` to the beginning of the pairs
        /// of vecs. Applies `transform1_head_begin` to the `T1` parts of `head_begin` upon
        /// inserting them, inserts `head_end` as is, and applies applies `update1_existing` to the
        /// `T1` parts of existing items in the pair of vecs upon moving them.
        pub fn try_prepend_and_update1<const BOUND: usize>(
            &mut self,
            mut head_begin: OwnedBoundedPairOfSlices<'_, T1, T2, BOUND>,
            head_end1: T1,
            head_end2: T2,
            transform1_head_begin: impl Fn(T1) -> T1,
            update1_existing: impl Fn(T1) -> T1,
        ) -> Result<(), ()> {
            let head_len = head_begin.len() + 1;

            let buf2 = self.buf2.get_mut(..self.len + head_len).ok_or(())?;
            self.len += head_len;
            buf2.rotate_right(head_len);
            unsafe {
                core::ptr::copy_nonoverlapping(head_begin.head2, buf2.as_mut_ptr(), head_len - 1);
                buf2.get_unchecked_mut(head_len - 1).write(head_end2);
            }

            let buf1 = &mut self.buf1;
            let head_begin_buf1 = head_begin.slice1_ref();
            unsafe {
                for write_index in (head_len..self.len).rev() {
                    let tmp = buf1
                        .get_unchecked(write_index - head_len)
                        .assume_init_read();
                    buf1.get_unchecked_mut(write_index)
                        .write(update1_existing(tmp));
                }
                buf1.get_unchecked_mut(head_len - 1).write(head_end1);
                for write_index in 0..head_len - 1 {
                    let tmp = head_begin_buf1
                        .get_unchecked(write_index)
                        .assume_init_read();
                    buf1.get_unchecked_mut(write_index)
                        .write(transform1_head_begin(tmp));
                }
            }

            Ok(())
        }

        pub fn try_append_transform1<const BOUND: usize>(
            &mut self,
            tail: OwnedBoundedPairOfSlices<'_, T1, T2, BOUND>,
            transform1: impl Fn(T1) -> T1,
        ) -> Result<(), ()> {
            let tail = DropGuard {
                inner: tail,
                cleanup: |_| (),
            };
            self.try_append_guarded_transform1(tail, transform1)
        }

        pub fn try_append_guarded_transform1<'a, F, const BOUND: usize>(
            &mut self,
            tail: DropGuard<OwnedBoundedPairOfSlices<'a, T1, T2, BOUND>, F>,
            transform1: impl Fn(T1) -> T1,
        ) -> Result<(), ()>
        where
            F: FnMut(&mut OwnedBoundedPairOfSlices<'a, T1, T2, BOUND>),
        {
            let tail_len = tail.len();
            let dst1 = self.buf1.get_mut(self.len..self.len + tail_len).ok_or(())?;

            unsafe {
                let dst2 = self.buf2.get_unchecked_mut(self.len..self.len + tail_len);
                for (src_item, dst_item) in tail.slice1_ref().iter().zip(dst1) {
                    dst_item.write(transform1(src_item.assume_init_read()));
                }
                core::ptr::copy_nonoverlapping(tail.head2, dst2.as_mut_ptr(), tail_len);
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

    impl<'a, T1, T2, const BOUND: usize> OwnedBoundedPairOfSlices<'a, T1, T2, BOUND> {
        pub fn len(&self) -> usize {
            self.len
        }

        fn slice1_mut(&mut self) -> &mut [MaybeUninit<T1>] {
            unsafe { from_raw_parts_mut(self.head1, self.len) }
        }

        fn slice2_mut(&mut self) -> &mut [MaybeUninit<T2>] {
            unsafe { from_raw_parts_mut(self.head2, self.len) }
        }

        fn slice1_ref(&self) -> &[MaybeUninit<T1>] {
            unsafe { from_raw_parts(self.head1, self.len) }
        }

        fn slice2_ref(&self) -> &[MaybeUninit<T2>] {
            unsafe { from_raw_parts(self.head2, self.len) }
        }

        pub fn pop(&mut self) -> Option<(T1, T2)> {
            if self.len() == 0 {
                None
            } else {
                unsafe {
                    let new_len = self.len() - 1;

                    let item1 = self.slice1_ref().get_unchecked(new_len).assume_init_read();
                    let item2 = self.slice2_ref().get_unchecked(new_len).assume_init_read();
                    self.len = new_len;

                    Some((item1, item2))
                }
            }
        }

        pub fn pop_front(&mut self) -> Option<(T1, T2)> {
            if self.len() == 0 {
                None
            } else {
                unsafe {
                    let item1 = (*self.head1).assume_init_read();
                    let item2 = (*self.head2).assume_init_read();

                    self.head1 = self.head1.add(1);
                    self.head2 = self.head2.add(1);
                    self.len -= 1;

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
            assert!(mid <= self.len);

            (
                OwnedBoundedPairOfSlices {
                    len: mid,
                    head1: self.head1,
                    head2: self.head2,
                    phantom: PhantomData,
                },
                OwnedBoundedPairOfSlices {
                    len: self.len - mid,
                    head1: unsafe { self.head1.add(mid) },
                    head2: unsafe { self.head2.add(mid) },
                    phantom: PhantomData,
                },
            )
        }

        pub fn first_as_ref(&self) -> &[T1] {
            unsafe { core::mem::transmute(&*self.slice1_ref()) }
        }

        pub fn second_as_ref(&self) -> &[T2] {
            unsafe { core::mem::transmute(&*self.slice2_ref()) }
        }

        pub fn both_as_ref(&self) -> (&[T1], &[T2]) {
            (self.first_as_ref(), self.second_as_ref())
        }

        pub fn first_as_mut(&mut self) -> &mut [T1] {
            unsafe { core::mem::transmute(&mut *self.slice1_mut()) }
        }

        pub fn second_as_mut(&mut self) -> &mut [T2] {
            unsafe { core::mem::transmute(&mut *self.slice2_mut()) }
        }

        pub fn both_as_mut(&mut self) -> (&mut [T1], &mut [T2]) {
            unsafe {
                (
                    core::mem::transmute(&mut *self.slice2_mut()),
                    core::mem::transmute(&mut *self.slice1_mut()),
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

    impl<'a, T1, T2, F, const BOUND: usize> DropGuard<OwnedBoundedPairOfSlices<'a, T1, T2, BOUND>, F>
    where
        F: FnMut(&mut OwnedBoundedPairOfSlices<'a, T1, T2, BOUND>),
    {
        /// We allow `pop`ing from the end even within a `DropGuard` because this doesn't change
        /// the head positions of the slices, which are needed for cleanup logic.
        pub fn pop(&mut self) -> Option<(T1, T2)> {
            if self.len() == 0 {
                None
            } else {
                unsafe {
                    let new_len = self.len() - 1;

                    let item1 = self
                        .inner
                        .slice1_ref()
                        .get_unchecked(new_len)
                        .assume_init_read();
                    let item2 = self
                        .inner
                        .slice2_ref()
                        .get_unchecked(new_len)
                        .assume_init_read();
                    self.inner.len = new_len;

                    Some((item1, item2))
                }
            }
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

    use crate::{F32, F64};

    use super::AugmentedBTree;

    #[test]
    fn manual() {
        dbg!(manual_internal::<128>());
        dbg!(manual_internal::<10>());
        dbg!(manual_internal::<5>());
        dbg!(manual_internal::<4>());
        // dbg!(manual_internal::<3>());
        // dbg!(manual_internal::<2>());
        // dbg!(manual_internal::<1>());

        fn manual_internal<const CAP: usize>() {
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

            // Test if `left_cumulative` returns the correct values on all insert positions,
            // between all insert positions, and above and below the range of insert positions.
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

            // Remove some items. Recall that the tree currently has state
            // { -2.25=>3, 1.5=>10, 2.5=>9, 3.25=>7, 4.75=>3, 5.125=>2, 6.5=>7 }.
            let removals = [
                (5.125, 1, true, 32, 34, 33),
                (2.7, 0, true, 22, 22, 22),
                (0.1, 1, false, 3, 3, 3),
                (1.5, 6, true, 3, 13, 7),
                (3.25, 7, true, 16, 23, 16),
                (1.5, 2, true, 3, 7, 5),
                (-2.25, 1, true, 0, 3, 2),
                (1.5, 5, false, 2, 4, 4),
                (2.5, 3, true, 4, 13, 10),
                (2.5, 7, false, 4, 10, 10),
                (2.5, 6, true, 4, 10, 4),
            ];
            let mut total = tree.total;
            for (pos, count, should_work, cdf, cdf_right_before, cdf_right_after) in removals {
                let pos_right = F32::new(pos + 0.001).unwrap();
                let pos = F32::new(pos).unwrap();
                assert_eq!(tree.left_cumulative(pos), cdf);
                assert_eq!(tree.left_cumulative(pos_right), cdf_right_before);

                let worked = tree.remove(pos, count).is_ok();

                assert_eq!(worked, should_work);
                assert_eq!(tree.left_cumulative(pos), cdf);
                assert_eq!(tree.left_cumulative(pos_right), cdf_right_after);
                if should_work {
                    total -= count
                }
                assert_eq!(tree.total, total);
            }

            let expected_tree = [(-2.25, 2), (1.5, 2), (4.75, 3), (5.125, 1), (6.5, 7)];

            let mut accum = 0;
            for &(pos, count) in &expected_tree {
                let pos_right = F32::new(pos + 0.001).unwrap();
                let pos = F32::new(pos).unwrap();
                assert_eq!(tree.left_cumulative(pos), accum);
                accum += count;
                assert_eq!(tree.left_cumulative(pos_right), accum);
            }
            assert_eq!(tree.total(), accum);

            // Removing one more item should make sure the root node is a leaf node for all CAP >= 4.
            for &(pos, count) in &expected_tree {
                let mut tree = tree.clone();
                assert!(tree.remove(F32::new(pos).unwrap(), count).is_ok());
                assert_eq!(tree.total(), accum - count);

                assert_eq!(tree.root_type, super::Leaf);
                let root = unsafe { tree.root.as_leaf_unchecked() };
                let mut data = root
                    .data
                    .bulk
                    .first_as_ref()
                    .iter()
                    .scan(0, |accum, entry| {
                        let count = entry.accum - *accum;
                        *accum = entry.accum;
                        Some((entry.pos.get(), count))
                    })
                    .collect::<Vec<_>>();
                let index = data.partition_point(|&(p, _)| p < pos);
                data.insert(index, (pos, count));

                assert_eq!(data, &expected_tree);
            }

            for &(pos, count) in &expected_tree {
                assert!(tree.remove(F32::new(pos).unwrap(), count).is_ok());
            }

            assert_eq!(tree.total(), 0);
            assert_eq!(tree.root_type, super::Leaf);
            let root = unsafe { tree.root.as_leaf_unchecked() };
            let data = root.data.bulk.first_as_ref();
            assert_eq!(data.len(), 0);
        }
    }

    #[test]
    fn random_data() {
        #[cfg(not(miri))]
        let amts = [100, 1000, 10_000, 100_000];

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

        fn verify_tree<const CAP: usize>(tree: &AugmentedBTree<F64, u32, CAP>, cdf: &[(F64, u32)]) {
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
            assert_eq!(tree.total(), cdf.last().unwrap().1);
            assert_eq!(tree.quantile_function(tree.total()), None);
            assert_eq!(tree.quantile_function(tree.total() + 1), None);
        }

        fn random_data_internal<const CAP: usize>(amt: usize) {
            let mut hasher = DefaultHasher::new();
            20231201.hash(&mut hasher);
            (CAP as u64).hash(&mut hasher);
            (amt as u64).hash(&mut hasher);

            let mut rng: Xoshiro256StarStar = Xoshiro256StarStar::seed_from_u64(hasher.finish());
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

            let mut sorted_insertions = insertions;
            sorted_insertions.sort_unstable_by_key(|(pos, _)| *pos);

            let mut last_pos = F64::new(-1.0).unwrap();
            let mut accum = 0;
            let mut cdf = sorted_insertions
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

            verify_tree(&tree, &cdf);

            let mut removals = sorted_insertions;
            let partial_removal_probability = distributions::Bernoulli::new(0.1).unwrap();
            let mut cdf_iter = cdf.iter_mut();
            let mut cdf_entry = cdf_iter.next().unwrap();

            let mut accumulated_removals = 0;
            for (pos, count) in &mut removals {
                if rng.sample(partial_removal_probability) {
                    *count = rng.sample(distributions::Uniform::from(0..*count));
                }

                if *pos != cdf_entry.0 {
                    cdf_entry.1 -= accumulated_removals;
                    cdf_entry = cdf_iter.next().unwrap();
                    assert_eq!(*pos, cdf_entry.0);
                }

                accumulated_removals += *count;
            }
            cdf_entry.1 -= accumulated_removals;
            core::mem::drop((cdf_iter, cdf_entry));
            let mut last_accum = 0;
            let cdf = cdf
                .iter()
                .filter_map(|&(pos, accum)| {
                    if accum == last_accum {
                        None
                    } else {
                        last_accum = accum;
                        Some((pos, accum))
                    }
                })
                .collect::<Vec<_>>();

            removals.shuffle(&mut rng);
            for (i, &(pos, count)) in removals.iter().enumerate() {
                assert_eq!((i, tree.remove(pos, count)), (i, Ok(())));
            }

            verify_tree(&tree, &cdf);

            // Remove all remaining entries.
            let mut removals = cdf
                .iter()
                .scan(0, |a, &(pos, accum)| {
                    let old_a = *a;
                    *a = accum;
                    Some((pos, accum - old_a))
                })
                .collect::<Vec<_>>();
            removals.shuffle(&mut rng);

            for (i, &(pos, count)) in removals.iter().enumerate() {
                assert_eq!((i, tree.remove(pos, count)), (i, Ok(())));
            }

            assert_eq!(tree.total(), 0);
            assert_eq!(tree.root_type, super::Leaf);
            let root = unsafe { tree.root.as_leaf_unchecked() };
            assert_eq!(root.data.bulk.len(), 0);
        }
    }

    #[test]
    fn iter() {
        #[cfg(not(miri))]
        let amts = [100, 1000, 10_000, 100_000];

        #[cfg(miri)]
        let amts = [100];

        for amt in amts {
            dbg!(amt, iter_internal::<128>(amt));
            dbg!(amt, iter_internal::<10>(amt));
            dbg!(amt, iter_internal::<5>(amt));
            dbg!(amt, iter_internal::<4>(amt));
        }

        fn iter_internal<const CAP: usize>(amt: usize) {
            let mut hasher = DefaultHasher::new();
            20231212.hash(&mut hasher);
            (CAP as u64).hash(&mut hasher);
            (amt as u64).hash(&mut hasher);

            let mut rng: Xoshiro256StarStar = Xoshiro256StarStar::seed_from_u64(hasher.finish());

            let pmf = (0..amt)
                .map(|i| {
                    let pos = F32::new(i as f32).unwrap();
                    let count = 1 + rng.next_u32() % 128;
                    (pos, count)
                })
                .collect::<Vec<_>>();

            let mut insertions = pmf.clone();
            insertions.shuffle(&mut rng);
            let mut tree = AugmentedBTree::<_, _, 5>::new();
            for &(pos, count) in &insertions {
                tree.insert(pos, count);
            }

            let mut accum = 0;
            let pmf_from_iterator = tree
                .iter()
                .map(|(p, c)| {
                    let count = c - accum;
                    accum = c;
                    (p, count)
                })
                .collect::<Vec<_>>();
            assert_eq!(pmf_from_iterator, pmf);
        }
    }
}
