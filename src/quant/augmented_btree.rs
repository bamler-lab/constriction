use core::{
    fmt::Debug,
    mem::MaybeUninit,
    ops::{Add, Deref, DerefMut, Sub},
};

use self::{
    bounded_vec::BoundedVec,
    tree_refs::{ChildRef, ParentRef},
};
use NodeType::{Leaf, NonLeaf};

pub struct AugmentedBTree<F, C, const CAP: usize> {
    total: C,
    root_type: NodeType,
    root: ChildRef<F, C, CAP>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeType {
    NonLeaf,
    Leaf,
}

#[derive(Debug)]
struct NonLeafNode<F, C, const CAP: usize> {
    parent: Option<ParentRef<F, C, CAP>>,
    children_type: NodeType,
    separators: BoundedVec<Entry<F, C>, CAP>,
    first_child: ChildRef<F, C, CAP>,
    remaining_children: BoundedVec<ChildRef<F, C, CAP>, CAP>,
}

#[derive(Debug)]
struct LeafNode<F, C, const CAP: usize> {
    parent: Option<ParentRef<F, C, CAP>>,
    entries: BoundedVec<Entry<F, C>, CAP>,
}

#[derive(Clone, Copy, Debug)]
struct Entry<F, C> {
    key: F,

    /// Sum of counts within the current subTree from the first entry up until
    /// and including this entry. Thus the total weight of a leaf node is stored
    /// in the `accum` field of the leaf node's last entry.
    accum: C,
}

impl<F, C, const CAP: usize> Drop for AugmentedBTree<F, C, CAP> {
    fn drop(&mut self) {
        unsafe {
            match self.root_type {
                NonLeaf => self.root.drop_non_leaf(),
                Leaf => self.root.drop_leaf(),
            }
        }
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

impl<F, C, const CAP: usize> AugmentedBTree<F, C, CAP>
where
    F: Ord + Copy + Unpin,
    C: Ord + Default + Copy + Add<Output = C> + Sub<Output = C> + Unpin,
{
    pub fn new() -> Self {
        // TODO: statically assert that CAP>0;
        Self {
            total: C::default(),
            root_type: Leaf,
            root: ChildRef::leaf(LeafNode::empty(None)),
        }
    }

    pub fn insert(&mut self, key: F, amount: C) {
        self.total = self.total + amount;

        // Find the leaf node where the key should be inserted, and increment all accums to the
        // right of the path from root to leaf. If the key already exists in a non-leaf node, then
        // incrementing the accums already took care of inserting, so there's nothing else to do.
        let mut node_ref = &mut self.root;
        let mut node_type = self.root_type;
        while node_type == NonLeaf {
            let node = unsafe { node_ref.as_non_leaf_mut_unchecked() };
            let Some((child_ref, child_type)) = node.child_by_key_mut(
                key,
                |entry| entry.key,
                |entry| entry.accum = entry.accum + amount,
            ) else {
                return;
            };

            node_ref = child_ref;
            node_type = child_type;
        }
        let leaf_node = unsafe { node_ref.as_leaf_mut_unchecked() };

        // Insert into the leaf node.
        let Some((mut key, mut weight_before_right_child, mut right_child_ref)) =
            leaf_node.insert(key, amount)
        else {
            return;
        };

        // The leaf node overflew and had to be split into two. Propagate up the tree.
        let mut parent = leaf_node.parent;
        while let Some(mut node) = parent {
            let node = unsafe { node.as_mut() };
            let Some((ejected_key, ejected_weight, right_sibling_ref)) =
                node.insert(key, weight_before_right_child, right_child_ref)
            else {
                return;
            };

            key = ejected_key;
            weight_before_right_child = ejected_weight;
            right_child_ref = right_sibling_ref;
            parent = node.parent;
        }

        // The root node overflew. We have to increase the tree hight by creating a new root node.
        let mut separators = BoundedVec::new();
        separators
            .try_push(Entry {
                key,
                accum: weight_before_right_child,
            })
            .expect("vector is empty and `CAP>0`");

        // Since we cannot move out of `self.root`, we have to first construct a new root with a
        // temporary dummy `first_child`, then replace the old root by the new root, and then
        // replace the new root's dummy `first_child` with the old root.
        let new_root = ChildRef::non_leaf(NonLeafNode::new(
            None,
            self.root_type,
            separators,
            right_child_ref,
            BoundedVec::new(),
        ));
        self.root_type = NonLeaf;
        let old_root = core::mem::replace(&mut self.root, new_root);
        let new_root = unsafe { self.root.as_non_leaf_mut_unchecked() };
        let mut right_child = core::mem::replace(&mut new_root.first_child, old_root);

        let new_root_ref = ParentRef::from_ref(unsafe { self.root.as_non_leaf_unchecked() });
        let new_root = unsafe { self.root.as_non_leaf_mut_unchecked() };
        if self.root_type == NonLeaf {
            unsafe {
                new_root.first_child.as_non_leaf_mut_unchecked().parent = Some(new_root_ref);
                right_child.as_non_leaf_mut_unchecked().parent = Some(new_root_ref);
            }
        } else {
            unsafe {
                new_root.first_child.as_leaf_mut_unchecked().parent = Some(new_root_ref);
                right_child.as_leaf_mut_unchecked().parent = Some(new_root_ref);
            }
        }

        new_root
            .remaining_children
            .try_push(right_child)
            .expect("vector is empty and `CAP>0`");
    }

    pub fn total(&self) -> C {
        self.total
    }
}

impl<F, C, const CAP: usize> NonLeafNode<F, C, CAP>
where
    F: Unpin + Ord + Copy,
    C: Default + Add<Output = C> + Sub<Output = C> + Unpin + Copy,
{
    fn new(
        parent: Option<ParentRef<F, C, CAP>>,
        children_type: NodeType,
        separators: BoundedVec<Entry<F, C>, CAP>,
        first_child: ChildRef<F, C, CAP>,
        remaining_children: BoundedVec<ChildRef<F, C, CAP>, CAP>,
    ) -> Self {
        Self {
            parent,
            children_type,
            separators,
            first_child,
            remaining_children,
        }
    }

    fn child_by_key_mut<X: Ord>(
        &mut self,
        key: X,
        get_key: impl Fn(&Entry<F, C>) -> X,
        update_right: impl Fn(&mut Entry<F, C>),
    ) -> Option<(&mut ChildRef<F, C, CAP>, NodeType)> {
        let index = self
            .separators
            .partition_point(|entry| get_key(entry) < key);
        let mut right_iter = self.separators.iter_mut().take(index);

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
            .remaining_children
            .deref_mut()
            .get_mut(index.wrapping_sub(1))
            .unwrap_or(&mut self.first_child);

        Some((child_ref, self.children_type))
    }

    fn insert(
        &mut self,
        key: F,
        weight_before_right_child: C,
        right_child: ChildRef<F, C, CAP>,
    ) -> Option<(F, C, ChildRef<F, C, CAP>)> {
        // Identify separator that is just right to key (we know that key does not exist in nodes).
        let insert_index = self.separators.partition_point(|entry| entry.key < key);
        let preceding_accum = self
            .separators
            .deref()
            .get(insert_index.wrapping_sub(1))
            .map(|entry| entry.accum)
            .unwrap_or_default();

        let mut separator = Entry {
            key,
            accum: preceding_accum + weight_before_right_child,
        };
        if self.separators.try_insert(insert_index, separator).is_ok() {
            self.remaining_children
                .try_insert(insert_index, right_child)
                .expect("separators and remaining_children always have same len");
            return None;
        }

        // Inserting would overflow the node. Split it into two.
        // TODO: maybe try spilling into neighboring siblings first.
        let right_sibling = match insert_index.cmp(&(CAP / 2)) {
            core::cmp::Ordering::Less => {
                // Insert both `separator` and `right_child` into the left sibling (i.e., `self`).
                let right_separators = self.separators.chop(CAP / 2).expect("node is full").into();
                let right_remaining_children: BoundedVec<_, CAP> = self
                    .remaining_children
                    .chop(CAP / 2)
                    .expect("node is full")
                    .into();
                let right_first_child = self
                    .remaining_children
                    .pop()
                    .expect("len = CAP/2 and CAP > 1");

                // `self.remaining_children` is now one item shorter than `self.separators`, but
                // they become equal in length once we eject the last separator.

                self.separators
                    .try_insert(insert_index, separator)
                    .expect("there are `CAP - CAP/2` vacancies, which is >0 because CAP>0.");
                self.remaining_children
                    .try_insert(insert_index, right_child)
                    .expect("there are `CAP - CAP/2 + 1` vacancies, which is >0 because CAP>0.");

                NonLeafNode::new(
                    self.parent,
                    self.children_type,
                    right_separators,
                    right_first_child,
                    right_remaining_children,
                )
            }
            core::cmp::Ordering::Equal => {
                // Append `separator` to the end of `self.separators` (so it gets ejected afterwards),
                // and set `right_child` to `right_sibling.first_child`.
                let right_separators: BoundedVec<_, CAP> =
                    self.separators.chop(CAP / 2).expect("node is full").into();
                self.separators
                    .try_push(separator)
                    .expect("there are `CAP - CAP/2` vacancies, which is >0 because CAP>0.");
                let right_first_child = right_child;
                let right_remaining_children = self
                    .remaining_children
                    .chop(CAP / 2)
                    .expect("node is full")
                    .into();

                NonLeafNode::new(
                    self.parent,
                    self.children_type,
                    right_separators,
                    right_first_child,
                    right_remaining_children,
                )
            }
            core::cmp::Ordering::Greater => {
                // Insert both `separator` and `right_child` into `right_sibling`.
                let insert_index = insert_index - (CAP / 2 + 1);
                let weight_before_right_sibling = self
                    .separators
                    .get(CAP / 2)
                    .expect("CAP/2 < original insert_index <= len")
                    .accum;

                // Build `right_sibling`'s list of separators.
                let right_separators = self
                    .separators
                    .chop(CAP / 2 + 1)
                    .expect("CAP/2 < original insert_index <= len");
                let (before_insert, after_insert) = right_separators.split_at(insert_index);
                let mut right_separators = BoundedVec::new();
                right_separators
                    .try_append_transform(before_insert, |entry| Entry {
                        key: entry.key,
                        accum: entry.accum - weight_before_right_sibling,
                    })
                    .expect("can't overflow");
                separator.accum = separator.accum - weight_before_right_sibling;
                right_separators
                    .try_push(separator)
                    .expect("can't overflow");
                right_separators
                    .try_append_transform(after_insert, |entry| Entry {
                        key: entry.key,
                        accum: entry.accum - weight_before_right_sibling,
                    })
                    .expect("can't overflow");

                // Build `right_sibling`'s list of children.
                let right_children = self
                    .remaining_children
                    .chop(CAP / 2)
                    .expect("CAP/2 < original insert_index <= len");
                let (mut right_first_child, right_remaining_children) = right_children.split_at(1);
                let right_first_child = right_first_child.pop().expect("contains exactly 1");
                let (before_insert, after_insert) = right_remaining_children.split_at(insert_index);
                let mut right_remaining_children: BoundedVec<_, CAP> = before_insert.into();
                right_remaining_children
                    .try_push(right_child)
                    .expect("can't overflow");
                right_remaining_children
                    .try_append(after_insert)
                    .expect("can't overflow");

                NonLeafNode::new(
                    self.parent,
                    self.children_type,
                    right_separators,
                    right_first_child,
                    right_remaining_children,
                )
            }
        };
        // We're inserting into the left half or the parent.
        // FIXME: this is wrong! if we insert the separator into the parent, then we should insert the
        // right_child into `right_sibling`. Maybe it would all be simpler if we always split tto the left?
        // Actually, it's probably even easier to distinguish all three cases here, because `insert_index == CAP/2` is special anyway because it means inserting the child into `right_sibling.first_child`.

        let ejected_separator = self
            .separators
            .pop()
            .expect("there are CAP/2+1 > 0 separators");
        let right_sibling_ref = ChildRef::non_leaf(right_sibling);

        Some((
            ejected_separator.key,
            ejected_separator.accum,
            right_sibling_ref,
        ))
    }
}

impl<F, C, const CAP: usize> LeafNode<F, C, CAP>
where
    F: Copy + Unpin + Ord,
    C: Copy + Unpin + Default + Add<Output = C> + Sub<Output = C>,
{
    fn empty(parent: Option<ParentRef<F, C, CAP>>) -> Self {
        Self {
            parent,
            entries: BoundedVec::new(),
        }
    }

    fn insert(&mut self, key: F, amount: C) -> Option<(F, C, ChildRef<F, C, CAP>)> {
        // Check if the node already contains an entry with the given `key`.
        // If so, increment its accum and all accums to the right, then return.
        let insert_index = self.entries.partition_point(|entry| entry.key < key);
        let mut right_iter = self.entries.iter_mut().take(insert_index);
        match right_iter.next() {
            Some(right_entry) if right_entry.key == key => {
                right_entry.accum = right_entry.accum + amount;
                for entry in right_iter {
                    entry.accum = entry.accum + amount;
                }
                return None;
            }
            _ => {}
        }

        // An entry with `key` doesn't exist yet. Create a new one and insert it.
        let old_accum = self
            .entries
            .get(insert_index.wrapping_sub(1))
            .map(|node| node.accum)
            .unwrap_or_default();
        let mut insert_entry = Entry::new(key, old_accum + amount);

        if self
            .entries
            .try_insert_and_accum(insert_index, insert_entry, |entry| {
                Entry::new(entry.key, entry.accum + amount)
            })
            .is_ok()
        {
            return None; // No splitting necessary.
        }

        // Inserting would overflow the leaf node. Split it into two.
        // TODO: maybe try spilling into neighboring siblings first.
        let mut right_sibling = LeafNode::empty(self.parent);

        if insert_index <= CAP / 2 {
            // We're inserting into the left half or the parent.
            let weight_before_right_sibling = if insert_index == CAP / 2 {
                insert_entry.accum
            } else {
                self.entries
                    .get(CAP / 2 - 1)
                    .expect("CAP/2 > insert_index >= 0")
                    .accum
            };

            let right_entries = self.entries.chop(CAP / 2).expect("node is full");

            right_sibling
                .entries
                .try_append_transform(right_entries, |entry| Entry {
                    key: entry.key,
                    accum: entry.accum - weight_before_right_sibling,
                })
                .expect("can't overflow");

            self.entries
                .try_insert_and_accum(insert_index, insert_entry, |entry| {
                    Entry::new(entry.key, entry.accum + amount)
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
                right_entries.split_at(insert_index - (CAP / 2 + 1));

            right_sibling
                .entries
                .try_append_transform(before_insert, |entry| Entry {
                    key: entry.key,
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
                    key: entry.key,
                    accum: entry.accum - weight_before_right_sibling + amount,
                })
                .expect("can't overflow");
        };

        let ejected_entry = self.entries.pop().expect("there are CAP/2+1 > 0 entries");
        let right_sibling_ref = ChildRef::leaf(right_sibling);

        Some((ejected_entry.key, ejected_entry.accum, right_sibling_ref))
    }
}

impl<F, C> Entry<F, C> {
    fn new(key: F, accum: C) -> Self {
        Self { key, accum }
    }
}

mod tree_refs {
    use core::{fmt::Debug, mem::ManuallyDrop, ptr::NonNull};

    use alloc::boxed::Box;

    use super::{LeafNode, NonLeafNode};

    /// A (conceptually) owned reference to either a `NonLeafNode` or a `LeafNode`.
    ///
    /// Note that a `ChildRef` does not actually the child when it is dropped.
    /// This is not possible because the `ChildRef` doesn't know the type of the
    /// child. Any container that has a `ChildRef` needs to implement `Drop`, where
    /// it has to call either `.drop_non_leaf()` or `.drop_leaf()` on all of its
    /// fields of type `ChildRef`.
    pub union ChildRef<F, C, const CAP: usize> {
        non_leaf: ManuallyDrop<Box<NonLeafNode<F, C, CAP>>>,
        leaf: ManuallyDrop<Box<LeafNode<F, C, CAP>>>,
    }

    /// A non-owned reference to a `NonLeafNode`.
    #[derive(Debug, Clone, Copy)]
    pub struct ParentRef<F, C, const CAP: usize>(NonNull<NonLeafNode<F, C, CAP>>);

    impl<F, C, const CAP: usize> Debug for ChildRef<F, C, CAP> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("ChildRef").finish_non_exhaustive()
        }
    }

    impl<F, C, const CAP: usize> ChildRef<F, C, CAP> {
        pub fn non_leaf(child: NonLeafNode<F, C, CAP>) -> ChildRef<F, C, CAP> {
            Self {
                non_leaf: ManuallyDrop::new(Box::new(child)),
            }
        }

        pub fn leaf(child: LeafNode<F, C, CAP>) -> ChildRef<F, C, CAP> {
            Self {
                leaf: ManuallyDrop::new(Box::new(child)),
            }
        }

        #[inline(always)]
        pub unsafe fn drop_non_leaf(&mut self) {
            core::mem::drop(ManuallyDrop::take(&mut self.non_leaf));
        }

        #[inline(always)]
        pub unsafe fn drop_leaf(&mut self) {
            core::mem::drop(ManuallyDrop::take(&mut self.leaf));
        }

        #[inline(always)]
        pub unsafe fn as_non_leaf_unchecked(&self) -> &NonLeafNode<F, C, CAP> {
            &self.non_leaf
        }

        #[inline(always)]
        pub unsafe fn as_non_leaf_mut_unchecked(&mut self) -> &mut NonLeafNode<F, C, CAP> {
            &mut self.non_leaf
        }

        #[inline(always)]
        pub unsafe fn as_leaf_unchecked(&self) -> &LeafNode<F, C, CAP> {
            &self.leaf
        }

        #[inline(always)]
        pub unsafe fn as_leaf_mut_unchecked(&mut self) -> &mut LeafNode<F, C, CAP> {
            &mut self.leaf
        }
    }

    impl<F, C, const CAP: usize> ParentRef<F, C, CAP> {
        #[inline(always)]
        pub fn from_ref(node: &NonLeafNode<F, C, CAP>) -> Self {
            let ptr_mut = node as *const NonLeafNode<_, _, CAP> as *mut NonLeafNode<_, _, CAP>;
            ParentRef(unsafe { NonNull::new_unchecked(ptr_mut) })
        }

        #[inline(always)]
        pub unsafe fn as_mut(&mut self) -> &mut NonLeafNode<F, C, CAP> {
            self.0.as_mut()
        }
    }
}

mod bounded_vec {
    use core::{
        mem::MaybeUninit,
        ops::{Deref, DerefMut, Index},
    };

    #[derive(Debug)]
    pub struct BoundedVec<T, const CAP: usize> {
        len: usize,
        buf: [MaybeUninit<T>; CAP],
    }

    pub struct BoundedVecViewMut<'a, T, const BOUND: usize>(&'a mut [MaybeUninit<T>]);

    impl<T: Unpin, const CAP: usize> BoundedVec<T, CAP> {
        pub fn new() -> Self {
            let buf = unsafe {
                // SAFETY: This is taken from an example in the official documentation of `MaybeUninit`.
                // It calls `assume_init` on the *outer* `MaybeUninit`. This is safe because the type we
                // claim to have initialized at this point is `[MaybeUninit<T>; CAP]`, which does not
                // require initialization. See example in the documentation of `MaybeUninit`.
                MaybeUninit::<[MaybeUninit<T>; CAP]>::uninit().assume_init()
            };
            Self { len: 0, buf }
        }

        pub const fn len(&self) -> usize {
            self.len
        }

        pub fn try_insert(&mut self, index: usize, item: T) -> Result<(), ()> {
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

        pub fn try_insert_and_accum(
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

        pub fn try_push(&mut self, item: T) -> Result<(), ()> {
            if self.len == CAP {
                Err(())
            } else {
                unsafe { self.buf.get_unchecked_mut(self.len).write(item) };
                self.len += 1;
                Ok(())
            }
        }

        pub fn pop(&mut self) -> Option<T> {
            if self.len == 0 {
                None
            } else {
                self.len -= 1;
                let last = unsafe { self.buf.get_unchecked_mut(self.len).assume_init_read() };
                Some(last)
            }
        }

        pub fn chop(&mut self, start_index: usize) -> Option<BoundedVecViewMut<'_, T, CAP>> {
            let view = self.buf.get_mut(start_index..self.len)?;
            self.len = start_index;
            Some(BoundedVecViewMut(view))
        }

        pub fn try_append<const BOUND: usize>(
            &mut self,
            tail: BoundedVecViewMut<'_, T, BOUND>,
        ) -> Result<(), ()> {
            let dst = self
                .buf
                .get_mut(self.len..self.len + tail.len())
                .ok_or(())?;
            unsafe {
                core::ptr::copy_nonoverlapping(tail.0.as_ptr(), dst.as_mut_ptr(), tail.len());
            }
            self.len += tail.len();
            Ok(())
        }

        pub fn try_append_transform<const BOUND: usize>(
            &mut self,
            tail: BoundedVecViewMut<'_, T, BOUND>,
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

    impl<'a, T, const BOUND: usize> BoundedVecViewMut<'a, T, BOUND> {
        pub fn len(&self) -> usize {
            self.0.len()
        }

        pub fn pop(&'a mut self) -> Option<T> {
            if self.0.len() == 0 {
                None
            } else {
                unsafe {
                    let item = self.0.get_unchecked_mut(0).assume_init_read();
                    self.0 = self.0.get_unchecked_mut(..self.len() - 1);
                    Some(item)
                }
            }
        }

        pub fn split_at(
            self,
            mid: usize,
        ) -> (
            BoundedVecViewMut<'a, T, BOUND>,
            BoundedVecViewMut<'a, T, BOUND>,
        ) {
            let (left, right) = self.0.split_at_mut(mid);
            (BoundedVecViewMut(left), BoundedVecViewMut(right))
        }
    }

    impl<'a, T, const BOUND: usize> Index<usize> for BoundedVecViewMut<'a, T, BOUND> {
        type Output = T;

        fn index(&self, index: usize) -> &Self::Output {
            unsafe { self.0[index].assume_init_ref() }
        }
    }

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

    impl<'a, T: Unpin, const CAP: usize> From<BoundedVecViewMut<'a, T, CAP>> for BoundedVec<T, CAP> {
        fn from(view: BoundedVecViewMut<'a, T, CAP>) -> Self {
            let mut vec = Self::new();
            vec.try_append(view)
                .expect("capacities match and original vec was empty");
            vec
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::NonNanFloat;

    use super::AugmentedBTree;

    type F32 = NonNanFloat<f32>;

    #[test]
    fn create() {
        let tree = AugmentedBTree::<F32, u32, 128>::new();
        assert_eq!(tree.total(), 0);
    }
}
