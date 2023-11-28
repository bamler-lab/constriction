use core::{
    fmt::Debug,
    mem::{ManuallyDrop, MaybeUninit},
    ops::{Add, Bound, Deref, DerefMut, Index, IndexMut, Sub},
    ptr::{self, NonNull},
};

use alloc::boxed::Box;

pub struct AugmentedBtree<F, C, const CAP: usize> {
    total: C,
    root_type: NodeType,
    root: ChildRef<F, C, CAP>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeType {
    NonLeaf,
    Leaf,
}

use grove::ModifiableWalker;
use NodeType::{Leaf, NonLeaf};

/// A (conceptually) owned reference to either a `NonLeafNode` or a `LeafNode`.
///
/// Note that a `ChildRef` does not actually the child when it is dropped.
/// This is not possible because the `ChildRef` doesn't know the type of the
/// child. Any container that has a `ChildRef` needs to implement `Drop`, where
/// it has to call either `.drop_non_leaf()` or `.drop_leaf()` on all of its
/// fields of type `ChildRef`.
union ChildRef<F, C, const CAP: usize> {
    non_leaf: ManuallyDrop<Box<NonLeafNode<F, C, CAP>>>,
    leaf: ManuallyDrop<Box<LeafNode<F, C, CAP>>>,
}

impl<F, C, const CAP: usize> Debug for ChildRef<F, C, CAP> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ChildRef").finish_non_exhaustive()
    }
}

/// A non-owned reference to a `NonLeafNode`.
#[derive(Debug, Clone, Copy)]
struct ParentRef<F, C, const CAP: usize>(NonNull<NonLeafNode<F, C, CAP>>);

#[derive(Debug)]
struct NonLeafNode<F, C, const CAP: usize> {
    parent: Option<ParentRef<F, C, CAP>>,
    children_type: NodeType,
    first_child: ChildRef<F, C, CAP>,
    separators: BoundedVec<Entry<F, C>, CAP>,
    remaining_children: BoundedVec<ChildRef<F, C, CAP>, CAP>,
}

#[derive(Debug)]
struct LeafNode<F, C, const CAP: usize> {
    parent: Option<ParentRef<F, C, CAP>>,
    entries: BoundedVec<Entry<F, C>, CAP>,
}

#[derive(Debug)]
struct BoundedVec<T, const CAP: usize> {
    len: usize,
    buf: [MaybeUninit<T>; CAP],
}

#[derive(Clone, Copy, Debug)]
struct Entry<F, C> {
    key: F,

    /// Sum of counts within the current subtree from the first entry up until
    /// and including this entry. Thus the total weight of a leaf node is stored
    /// in the `accum_within_subtree` field of the leaf node's last entry.
    accum_within_subtree: C,
}

// DROP =======================================================================

impl<F, C, const CAP: usize> Drop for AugmentedBtree<F, C, CAP> {
    fn drop(&mut self) {
        unsafe {
            match self.root_type {
                NonLeaf => self.root.drop_non_leaf(),
                Leaf => self.root.drop_leaf(),
            }
        }
    }
}

impl<F, C, const CAP: usize> ChildRef<F, C, CAP> {
    #[inline(always)]
    unsafe fn drop_non_leaf(&mut self) {
        core::mem::drop(ManuallyDrop::take(&mut self.non_leaf));
    }

    #[inline(always)]
    unsafe fn drop_leaf(&mut self) {
        core::mem::drop(ManuallyDrop::take(&mut self.leaf));
    }
}

impl<F, C, const CAP: usize> Drop for NonLeafNode<F, C, CAP> {
    fn drop(&mut self) {
        unsafe {
            match self.children_type {
                NonLeaf => {
                    self.first_child.drop_non_leaf(); // TODO: what if we drop before initializing this?
                    for sc in self.remaining_children.deref_mut() {
                        sc.drop_non_leaf();
                    }
                }
                Leaf => {
                    self.first_child.drop_leaf();
                    for sc in self.remaining_children.deref_mut() {
                        sc.drop_leaf();
                    }
                }
            }
        }
    }
}

// UTILITIES ==================================================================

impl<T, const CAP: usize> Deref for BoundedVec<T, CAP> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { core::mem::transmute(self.buf.get_unchecked(..self.len)) }
    }
}

impl<T, const CAP: usize> DerefMut for BoundedVec<T, CAP> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { core::mem::transmute(self.buf.get_unchecked_mut(..self.len)) }
    }
}

macro_rules! child_call {
    (($child: expr , $discriminator: expr).$fn_name: ident $args: tt) => {
        match $discriminator {
            NonLeaf => ($child.non_leaf).$fn_name$args,
            Leaf => ($child.leaf).$fn_name$args,
        }
    };
}

macro_rules! bind_child {
    (&mut $child: expr, $discriminator: expr, |$child_ident: ident| $expression: expr) => {{
        match $discriminator {
            NonLeaf => match (&mut $child.non_leaf) {
                $child_ident => $expression,
            },
            Leaf => match (&mut $child.leaf) {
                $child_ident => $expression,
            },
        }
    }};
    (&$child: expr, $discriminator: expr, |$child_ident: ident| $expression: expr) => {{
        match $discriminator {
            NonLeaf => match (&mut $child.non_leaf) {
                $child_ident => $expression,
            },
            Leaf => match (&mut $child.leaf) {
                $child_ident => $expression,
            },
        }
    }};
}

// MAIN API ===================================================================

impl<F, C, const CAP: usize> AugmentedBtree<F, C, CAP>
where
    F: Ord + Copy + Unpin,
    C: Ord + Default + Copy + Add<Output = C> + Sub<Output = C> + Unpin,
{
    pub fn new() -> Self {
        Self {
            total: C::default(),
            root_type: Leaf,
            root: ChildRef::leaf(LeafNode::empty(None)),
        }
    }

    pub fn insert(&mut self, key: F, amount: C) {
        let mut node_type = self.root_type;
        let mut node_ref = &mut self.root;

        while node_type == NonLeaf {
            let node = &mut ***unsafe { &mut node_ref.non_leaf };

            // Identify separator that either matches key or is just right to it.
            let index = node.separators.partition_point(|entry| entry.key < key);
            let mut right_iter = node.separators.iter_mut().take(index);

            if let Some(right_separator) = right_iter.next() {
                // Increment all accumulators to the right (including the one we're at, if it already exists).
                let right_key = right_separator.key;
                right_separator.accum_within_subtree =
                    right_separator.accum_within_subtree + amount;
                for entry in right_iter {
                    entry.accum_within_subtree = entry.accum_within_subtree + amount;
                }
                if right_key == key {
                    // An entry with `key` already existed in this non-leaf node, and we've already
                    // incremented its count, so ther's nothing else to do.
                    return;
                }
            }

            node_ref = if index == 0 {
                &mut node.first_child
            } else {
                let slice = node.remaining_children.deref_mut();
                unsafe { slice.get_unchecked_mut(index - 1) }
            };
            node_type = node.children_type;
        }

        // We've arrived at a leaf node.
        let node = &mut ***unsafe { &mut node_ref.leaf };

        // Identify entry that either matches key or is just right to it.
        let insert_index = node.entries.partition_point(|entry| entry.key < key);
        let mut right_iter = node.entries.iter_mut().take(insert_index);

        if let Some(right_entry) = right_iter.next() {
            if right_entry.key == key {
                right_entry.accum_within_subtree = right_entry.accum_within_subtree + amount;
                for entry in right_iter {
                    entry.accum_within_subtree = entry.accum_within_subtree + amount;
                }
                return;
            }
        }

        let old_accum = node
            .entries
            .get(insert_index.wrapping_sub(1))
            .map(|node| node.accum_within_subtree)
            .unwrap_or_default();
        let mut insert_entry = Entry::new(key, old_accum + amount);

        if node
            .entries
            .try_insert_and_accum(insert_index, insert_entry, |entry| {
                Entry::new(entry.key, entry.accum_within_subtree + amount)
            })
            .is_ok()
        {
            return;
        }

        // TODO: try spilling into neighboring siblings first.

        let left_sibling = node;
        let mut right_sibling = LeafNode::empty(left_sibling.parent);

        // `left_child_weight_inclusive` includes the weight of the separator.
        if insert_index <= CAP / 2 {
            // We're inserting into the left half or the parent.
            let weight_before_right_sibling = if insert_index == CAP / 2 {
                insert_entry.accum_within_subtree
            } else {
                left_sibling
                    .entries
                    .get(CAP / 2 - 1)
                    .expect("CAP/2 > insert_index >= 0")
                    .accum_within_subtree
            };

            let right_entries = left_sibling.entries.chop(CAP / 2).expect("node is full");

            right_sibling
                .entries
                .try_append_transform(right_entries, |entry| Entry {
                    key: entry.key,
                    accum_within_subtree: entry.accum_within_subtree - weight_before_right_sibling,
                })
                .expect("can't overflow");

            left_sibling
                .entries
                .try_insert_and_accum(insert_index, insert_entry, |entry| {
                    Entry::new(entry.key, entry.accum_within_subtree + amount)
                })
                .expect("there are `CAP - CAP/2` vacancies, which is >0 because CAP>0.");
        } else {
            // We're inserting into the right half.
            let weight_before_right_sibling = left_sibling
                .entries
                .get(CAP / 2 - 1)
                .expect("CAP/2 > insert_index >= 0")
                .accum_within_subtree;

            let right_entries = left_sibling
                .entries
                .chop(CAP / 2 + 1)
                .expect("CAP/2 < insert_index <= len");

            let (before_insert, after_insert) =
                right_entries.split_at(insert_index - (CAP / 2 + 1));

            right_sibling
                .entries
                .try_append_transform(before_insert, |entry| Entry {
                    key: entry.key,
                    accum_within_subtree: entry.accum_within_subtree - weight_before_right_sibling,
                })
                .expect("can't overflow");

            insert_entry.accum_within_subtree =
                insert_entry.accum_within_subtree - weight_before_right_sibling;
            right_sibling
                .entries
                .try_push(insert_entry)
                .expect("can't overflow");

            right_sibling
                .entries
                .try_append_transform(after_insert, |entry| Entry {
                    key: entry.key,
                    accum_within_subtree: entry.accum_within_subtree - weight_before_right_sibling
                        + amount,
                })
                .expect("can't overflow");
        };

        let ejected_entry = left_sibling
            .entries
            .pop()
            .expect("there are CAP/2+1 > 0 entries");

        let left_child = left_sibling;
        let right_child_ref = ChildRef::leaf(right_sibling);

        while let Some(mut node) = left_child.parent {
            let node = unsafe { node.0.as_mut() };

            if node.separators.len() < CAP {
                // Identify separator that is just right to key (we know that key does not exist in nodes).
                let insert_index = node.separators.partition_point(|entry| entry.key < key);
                let preceeding_accum = node
                    .separators
                    .deref()
                    .get(insert_index.wrapping_sub(1))
                    .map(|entry| entry.accum_within_subtree)
                    .unwrap_or_default();

                node.separators.insert(
                    insert_index,
                    Entry {
                        key: ejected_entry.key,
                        accum_within_subtree: preceeding_accum + ejected_entry.accum_within_subtree,
                    },
                );

                node.remaining_children
                    .insert(insert_index, right_child_ref);

                return;
            }

            // TODO: try to first spill into neighboring siblings.
            // `left_child_weight_inclusive` includes the weight of the separator.
            todo!("port code below");
            // let (key, left_child_weight_inclusive) = if insert_index <= CAP / 2 {
            //     // We're inserting into the left half or the parent.
            //     let right_entries = left_sibling.entries.chop(CAP / 2).expect("node is full");

            //     left_sibling
            //         .entries
            //         .try_insert_and_accum(insert_index, insert_entry, |entry| {
            //             Entry::new(entry.key, entry.accum_within_subtree + amount)
            //         })
            //         .expect("there are `CAP - CAP/2` vacancies, which is >0 because CAP>0.");
            //     let ejected_entry = left_sibling
            //         .entries
            //         .pop()
            //         .expect("we just inserted an entry");
            //     let weight_before_right_sibling = ejected_entry.accum_within_subtree;

            //     right_sibling
            //         .entries
            //         .try_append_transform(right_entries, |entry| Entry {
            //             key: entry.key,
            //             accum_within_subtree: entry.accum_within_subtree
            //                 - weight_before_right_sibling,
            //         })
            //         .expect("can't overflow");

            //     (ejected_entry.key, weight_before_right_sibling)
            // } else {
            //     // We're inserting into the right half.
            //     let right_entries = left_sibling
            //         .entries
            //         .chop(CAP / 2 + 1)
            //         .expect("CAP/2 < insert_index <= len");
            //     let ejected_entry = left_sibling
            //         .entries
            //         .pop()
            //         .expect("there are CAP/2+1 > 0 entries");
            //     let weight_before_right_sibling = ejected_entry.accum_within_subtree;

            //     let (before_insert, after_insert) =
            //         right_entries.split_at_mut(insert_index - (CAP / 2 + 1));

            //     right_sibling
            //         .entries
            //         .try_append_transform(before_insert, |entry| Entry {
            //             key: entry.key,
            //             accum_within_subtree: entry.accum_within_subtree
            //                 - weight_before_right_sibling,
            //         })
            //         .expect("can't overflow");

            //     insert_entry.accum_within_subtree =
            //         insert_entry.accum_within_subtree - weight_before_right_sibling;
            //     right_sibling
            //         .entries
            //         .try_push(insert_entry)
            //         .expect("can't overflow");

            //     right_sibling
            //         .entries
            //         .try_append_transform(after_insert, |entry| Entry {
            //             key: entry.key,
            //             accum_within_subtree: entry.accum_within_subtree
            //                 - weight_before_right_sibling
            //                 + amount,
            //         })
            //         .expect("can't overflow");

            //     (ejected_entry.key, weight_before_right_sibling)
            // };
        }

        todo!("Create new root node");
    }

    pub fn total(&self) -> C {
        self.total
    }
}

impl<F, C, const CAP: usize> ChildRef<F, C, CAP> {
    fn non_leaf(child: NonLeafNode<F, C, CAP>) -> ChildRef<F, C, CAP> {
        Self {
            non_leaf: ManuallyDrop::new(Box::new(child)),
        }
    }

    fn leaf(child: LeafNode<F, C, CAP>) -> ChildRef<F, C, CAP> {
        Self {
            leaf: ManuallyDrop::new(Box::new(child)),
        }
    }
}

impl<F, C, const CAP: usize> LeafNode<F, C, CAP>
where
    F: Copy + Unpin,
    C: Copy + Unpin,
{
    fn empty(parent: Option<ParentRef<F, C, CAP>>) -> Self {
        Self {
            parent,
            entries: BoundedVec::new(),
        }
    }

    fn insert(&mut self, key: F, amount: C) {
        todo!()
    }
}

impl<T: Unpin, const CAP: usize> BoundedVec<T, CAP> {
    fn new() -> Self {
        let buf = unsafe {
            // SAFETY: This is taken from an example in the official documentation of `MaybeUninit`.
            // It calls `assume_init` on the *outer* `MaybeUninit`. This is safe because the type we
            // claim to have initialized at this point is `[MaybeUninit<T>; CAP]`, which does not
            // require initialization. See example in the documentation of `MaybeUninit`.
            MaybeUninit::<[MaybeUninit<T>; CAP]>::uninit().assume_init()
        };
        Self { len: 0, buf }
    }

    const fn len(&self) -> usize {
        self.len
    }

    fn insert(&mut self, index: usize, item: T) -> Result<(), ()> {
        assert!(index <= self.len);
        if self.len == CAP {
            return Err(());
        }

        unsafe {
            self.buf.get_unchecked_mut(index..).rotate_right(1);
            let tmp = self.buf.get_unchecked(index).assume_init_read();
            self.buf.get_unchecked_mut(self.len).write(tmp);
            self.buf.get_unchecked_mut(index).write(item);
        }

        Ok(())
    }

    fn try_insert_and_accum(
        &mut self,
        index: usize,
        item: T,
        update: impl Fn(T) -> T,
    ) -> Result<(), ()> {
        assert!(index <= self.len);

        if self.len == CAP {
            return Err(());
        }

        let mut write_index = self.len;
        unsafe {
            while write_index > index {
                let tmp = self.buf.get_unchecked(write_index - 1).assume_init_read();
                self.buf.get_unchecked_mut(write_index).write(update(tmp));
                write_index -= 1;
            }
            self.buf.get_unchecked_mut(index).write(item);
        }

        Ok(())
    }

    pub(crate) fn try_push(&mut self, item: T) -> Result<(), ()> {
        if self.len == CAP {
            Err(())
        } else {
            unsafe { self.buf.get_unchecked_mut(self.len).write(item) };
            self.len += 1;
            Ok(())
        }
    }

    fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            let last = unsafe { self.buf.get_unchecked_mut(self.len).assume_init_read() };
            Some(last)
        }
    }

    fn chop(&mut self, start_index: usize) -> Option<BoundedVecView<'_, T>> {
        let view = self.buf.get(start_index..self.len)?;
        self.len = start_index;
        Some(BoundedVecView(view))
    }

    pub(crate) fn try_append(&mut self, tail: BoundedVecView<'_, T>) -> Result<(), ()> {
        let dst = self
            .buf
            .get_mut(self.len..self.len + tail.len())
            .ok_or(())?;
        unsafe {
            ptr::copy_nonoverlapping(tail.0.as_ptr(), dst.as_mut_ptr(), tail.len());
        }
        self.len += tail.len();
        Ok(())
    }

    pub(crate) fn try_append_transform(
        &mut self,
        tail: BoundedVecView<'_, T>,
        transform: impl Fn(T) -> T,
    ) -> Result<(), ()> {
        let dst = self
            .buf
            .get_mut(self.len..self.len + tail.len())
            .ok_or(())?;

        for (src_item, dst_item) in tail.0.iter().zip(dst) {
            unsafe { dst_item.write(transform(src_item.assume_init_read())) };
        }

        Ok(())
    }
}

struct BoundedVecView<'a, T>(&'a [MaybeUninit<T>]);

impl<'a, T> Index<usize> for BoundedVecView<'a, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe { self.0[index].assume_init_ref() }
    }
}

impl<'a, T> BoundedVecView<'a, T> {
    fn split_at(self, mid: usize) -> (BoundedVecView<'a, T>, BoundedVecView<'a, T>) {
        let (left, right) = self.0.split_at(mid);
        (BoundedVecView(left), BoundedVecView(right))
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<F, C> Entry<F, C> {
    fn new(key: F, accum_within_subtree: C) -> Self {
        Self {
            key,
            accum_within_subtree,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::NonNanFloat;

    use super::AugmentedBtree;

    type F32 = NonNanFloat<f32>;

    #[test]
    fn create() {
        let tree = AugmentedBtree::<F32, u32, 128>::new();
        assert_eq!(tree.total(), 0);
    }
}
