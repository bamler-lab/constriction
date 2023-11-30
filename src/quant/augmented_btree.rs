use core::{
    fmt::Debug,
    ops::{Add, Deref, DerefMut, Sub},
};

use self::{
    bounded_vec::BoundedVec,
    tree_refs::{ChildRef, ParentRef},
};
use NodeType::{Leaf, NonLeaf};

#[cfg(test)]
use alloc::string::String;
#[cfg(test)]
use core::fmt::Display;

pub struct AugmentedBTree<P, C, const CAP: usize> {
    #[cfg(test)]
    test: bool,
    total: C,
    root_type: NodeType,
    root: ChildRef<P, C, CAP>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeType {
    NonLeaf,
    Leaf,
}

#[derive(Debug)]
struct NonLeafNode<P, C, const CAP: usize> {
    parent: Option<ParentRef<P, C, CAP>>,
    children_type: NodeType,
    separators: BoundedVec<Entry<P, C>, CAP>,
    first_child: ChildRef<P, C, CAP>,
    remaining_children: BoundedVec<ChildRef<P, C, CAP>, CAP>,
}

#[derive(Debug)]
struct LeafNode<P, C, const CAP: usize> {
    parent: Option<ParentRef<P, C, CAP>>,
    entries: BoundedVec<Entry<P, C>, CAP>,
}

#[derive(Clone, Copy, Debug)]
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

impl<P, C, const CAP: usize> AugmentedBTree<P, C, CAP>
where
    P: Ord + Copy + Unpin,
    C: Ord + Default + Copy + Add<Output = C> + Sub<Output = C> + Unpin,
{
    pub fn new() -> Self {
        // TODO: statically assert that CAP>0;
        Self {
            #[cfg(test)]
            test: false,
            total: C::default(),
            root_type: Leaf,
            root: ChildRef::leaf(LeafNode::empty(None)),
        }
    }

    pub fn total(&self) -> C {
        self.total
    }

    #[cfg(test)]
    pub fn set_test(&mut self, test: bool) {
        self.test = test;
    }

    pub fn insert(&mut self, key: P, count: C)
    where
        P: core::fmt::Display + Debug, // TODO: remove (only for debugging)
        C: core::fmt::Display + Debug, // TODO: remove (only for debugging)
    {
        self.total = self.total + count;

        // Find the leaf node where the key should be inserted, and increment all accums to the
        // right of the path from root to leaf. If the key already exists in a non-leaf node, then
        // incrementing the accums already took care of inserting, so there's nothing else to do.
        let mut node_ref = &mut self.root;
        let mut node_type = self.root_type;
        while node_type == NonLeaf {
            let node = unsafe { node_ref.as_non_leaf_mut_unchecked() };
            let Some((child_ref, child_type)) = node.child_by_key_mut(
                key,
                |entry| entry.pos,
                |entry| entry.accum = entry.accum + count,
            ) else {
                return;
            };

            node_ref = child_ref;
            node_type = child_type;
        }
        let leaf_node = unsafe { node_ref.as_leaf_mut_unchecked() };

        // Insert into the leaf node.
        let Some((mut key, mut weight_before_right_child, mut right_child_ref)) =
            leaf_node.insert(key, count)
        else {
            return;
        };

        #[cfg(test)]
        if self.test {
            let mut s = String::from("SPLITTED LEAF NODE:\n");
            leaf_node.append_to_debug_string(&mut s, "  ", leaf_node.parent);
            s.push_str("\nRIGHT SIBLING:\n");
            unsafe { right_child_ref.as_leaf_unchecked() }.append_to_debug_string(
                &mut s,
                "  ",
                leaf_node.parent,
            );
            std::eprintln!("{}", s);
            std::dbg!(key, weight_before_right_child);
        }

        // The leaf node overflew and had to be split into two. Propagate up the tree.
        let mut parent = leaf_node.parent;
        while let Some(mut node) = parent {
            let node = unsafe { node.as_mut() };
            let Some((ejected_key, ejected_weight, right_sibling_ref)) =
                node.insert(key, weight_before_right_child, right_child_ref)
            else {
                return;
            };

            #[cfg(test)]
            if self.test {
                let mut s = String::from("SPLITTED NON-LEAF NODE:\n");
                node.append_to_debug_string(&mut s, "  ", node.parent);
                s.push_str("\nRIGHT SIBLING:\n");
                unsafe { right_sibling_ref.as_non_leaf_unchecked() }.append_to_debug_string(
                    &mut s,
                    "  ",
                    node.parent,
                );
                std::eprintln!("{}", s);
                std::dbg!(ejected_key, ejected_weight);
            }

            key = ejected_key;
            weight_before_right_child = ejected_weight;
            right_child_ref = right_sibling_ref;
            parent = node.parent;
        }

        // The root node overflew. We have to increase the tree hight by creating a new root node.
        let mut separators = BoundedVec::new();
        separators
            .try_push(Entry {
                pos: key,
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

        let new_root_ref = unsafe { self.root.downgrade_non_leaf_unchecked() };
        let new_root = unsafe { self.root.as_non_leaf_mut_unchecked() };
        if new_root.children_type == NonLeaf {
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

    /// Returns the left-sided CDF, i.e., the sum of all counts strictly left of `pos`.
    pub fn get_by_pos(&self, pos: P) -> C {
        self.get_by_key(pos, |entry| entry.pos)
    }

    pub fn get_by_accum(&self, accum: C) {
        todo!()
    }

    pub fn get_by_key<X: Ord>(&self, key: X, get_key: impl Fn(&Entry<P, C>) -> X) -> C {
        let mut accum = C::default();

        let mut node_ref = &self.root;
        let mut node_type = self.root_type;
        while node_type == NonLeaf {
            let node = unsafe { node_ref.as_non_leaf_unchecked() };
            let index = node
                .separators
                .partition_point(|entry| get_key(entry) < key);
            accum = accum
                + node
                    .separators
                    .get(index.wrapping_sub(1))
                    .map(|entry| entry.accum)
                    .unwrap_or_default();
            node_type = node.children_type;
            node_ref = node
                .remaining_children
                .deref()
                .get(index.wrapping_sub(1))
                .unwrap_or(&node.first_child);
        }
        let leaf_node = unsafe { node_ref.as_leaf_unchecked() };

        let index = leaf_node
            .entries
            .partition_point(|entry| get_key(entry) < key);
        accum = accum
            + leaf_node
                .entries
                .get(index.wrapping_sub(1))
                .map(|entry| entry.accum)
                .unwrap_or_default();
        accum
    }

    #[cfg(test)]
    fn to_debug_string(&self) -> String
    where
        P: Display,
        C: Display,
    {
        let mut s = String::new();
        match self.root_type {
            NonLeaf => {
                let root = unsafe { self.root.as_non_leaf_unchecked() };
                root.append_to_debug_string(&mut s, "", None);
            }
            Leaf => {
                let root = unsafe { self.root.as_leaf_unchecked() };
                root.append_to_debug_string(&mut s, "", None);
            }
        }
        s
    }
}

impl<P, C, const CAP: usize> NonLeafNode<P, C, CAP>
where
    P: Unpin + Ord + Copy,
    C: Default + Add<Output = C> + Sub<Output = C> + Unpin + Copy,
{
    fn new(
        parent: Option<ParentRef<P, C, CAP>>,
        children_type: NodeType,
        separators: BoundedVec<Entry<P, C>, CAP>,
        first_child: ChildRef<P, C, CAP>,
        remaining_children: BoundedVec<ChildRef<P, C, CAP>, CAP>,
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
        get_key: impl Fn(&Entry<P, C>) -> X,
        update_right: impl Fn(&mut Entry<P, C>),
    ) -> Option<(&mut ChildRef<P, C, CAP>, NodeType)> {
        let index = self
            .separators
            .partition_point(|entry| get_key(entry) < key);
        let mut right_iter = self.separators.iter_mut().skip(index);

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
        key: P,
        weight_before_right_child: C,
        right_child: ChildRef<P, C, CAP>,
    ) -> Option<(P, C, ChildRef<P, C, CAP>)> {
        // Identify separator that is just right to key (we know that key does not exist in nodes).
        let insert_index = self.separators.partition_point(|entry| entry.pos < key);
        let preceding_accum = self
            .separators
            .deref()
            .get(insert_index.wrapping_sub(1))
            .map(|entry| entry.accum)
            .unwrap_or_default();

        let mut separator = Entry {
            pos: key,
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
                let accum_of_ejected = self
                    .separators
                    .get(CAP / 2 - 1)
                    .expect("CAP / 2 > index >= 0")
                    .accum;
                let chopped_off_separators = self.separators.chop(CAP / 2).expect("node is full");
                let mut right_separators = BoundedVec::new();
                right_separators
                    .try_append_transform(chopped_off_separators, |entry| Entry {
                        pos: entry.pos,
                        accum: entry.accum - accum_of_ejected,
                    })
                    .expect("can't overflow");

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
                // TODO: ensure test code coverage.
                // Append `separator` to the end of `self.separators` (so it gets ejected afterwards),
                // and set `right_child` to `right_sibling.first_child`.
                let accum_of_ejected = separator.accum;
                let chopped_off_separators = self.separators.chop(CAP / 2).expect("node is full");
                let mut right_separators = BoundedVec::new();
                right_separators
                    .try_append_transform(chopped_off_separators, |entry| Entry {
                        pos: entry.pos,
                        accum: entry.accum - accum_of_ejected,
                    })
                    .expect("can't overflow");

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
                let accum_of_ejected = self
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
                        pos: entry.pos,
                        accum: entry.accum - accum_of_ejected,
                    })
                    .expect("can't overflow");
                separator.accum = separator.accum - accum_of_ejected;
                right_separators
                    .try_push(separator)
                    .expect("can't overflow");
                right_separators
                    .try_append_transform(after_insert, |entry| Entry {
                        pos: entry.pos,
                        accum: entry.accum - accum_of_ejected,
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

        let mut right_sibling_ref = ChildRef::non_leaf(right_sibling);
        let right_sibling_pref = unsafe { right_sibling_ref.downgrade_non_leaf_unchecked() };
        let right_sibling = unsafe { right_sibling_ref.as_non_leaf_mut_unchecked() };
        match self.children_type {
            NonLeaf => {
                let first_nephew = unsafe { right_sibling.first_child.as_non_leaf_mut_unchecked() };
                first_nephew.parent = Some(right_sibling_pref);
                for nephew in right_sibling.remaining_children.iter_mut() {
                    let nephew = unsafe { nephew.as_non_leaf_mut_unchecked() };
                    nephew.parent = Some(right_sibling_pref);
                }
            }
            Leaf => {
                let first_nephew = unsafe { right_sibling.first_child.as_leaf_mut_unchecked() };
                first_nephew.parent = Some(right_sibling_pref);
                for nephew in right_sibling.remaining_children.iter_mut() {
                    let nephew = unsafe { nephew.as_leaf_mut_unchecked() };
                    nephew.parent = Some(right_sibling_pref);
                }
            }
        }

        let ejected_separator = self
            .separators
            .pop()
            .expect("there are CAP/2+1 > 0 separators");

        Some((
            ejected_separator.pos,
            ejected_separator.accum,
            right_sibling_ref,
        ))
    }

    #[cfg(test)]
    fn append_to_debug_string(
        &self,
        s: &mut String,
        ident: &str,
        expected_parent: Option<ParentRef<P, C, CAP>>,
    ) where
        P: Display,
        C: Display,
    {
        use alloc::format;

        if self.parent != expected_parent {
            s.push_str(&ident);
            s.push_str("INVALID PARENT: ");
        }
        let self_ref = Some(ParentRef::from_ref(&self));

        s.push_str(&format!(
            "{}nonleaf with {} separators:\n",
            ident,
            self.separators.len()
        ));
        let mut ident = String::from(ident);
        ident.push_str("  ");
        match self.children_type {
            NonLeaf => {
                let first_child = unsafe { self.first_child.as_non_leaf_unchecked() };
                first_child.append_to_debug_string(s, &ident, self_ref);
                for (separator, child) in self.separators.iter().zip(self.remaining_children.iter())
                {
                    s.push_str(&format!(
                        "{}separator: {} (accum={})\n",
                        &ident, separator.pos, separator.accum
                    ));
                    let child = unsafe { child.as_non_leaf_unchecked() };
                    child.append_to_debug_string(s, &ident, self_ref);
                }
            }
            Leaf => {
                let first_child = unsafe { self.first_child.as_leaf_unchecked() };
                first_child.append_to_debug_string(s, &ident, self_ref);
                for (separator, child) in self.separators.iter().zip(self.remaining_children.iter())
                {
                    s.push_str(&format!(
                        "{}separator: {} (accum={})\n",
                        &ident, separator.pos, separator.accum
                    ));
                    let child = unsafe { child.as_leaf_unchecked() };
                    child.append_to_debug_string(s, &ident, self_ref);
                }
            }
        }
    }
}

impl<P, C, const CAP: usize> LeafNode<P, C, CAP>
where
    P: Copy + Unpin + Ord,
    C: Copy + Unpin + Default + Add<Output = C> + Sub<Output = C>,
{
    fn empty(parent: Option<ParentRef<P, C, CAP>>) -> Self {
        Self {
            parent,
            entries: BoundedVec::new(),
        }
    }

    fn insert(&mut self, key: P, count: C) -> Option<(P, C, ChildRef<P, C, CAP>)> {
        // Check if the node already contains an entry with the given `key`.
        // If so, increment its accum and all accums to the right, then return.
        let insert_index = self.entries.partition_point(|entry| entry.pos < key);
        let mut right_iter = self.entries.iter_mut().skip(insert_index);
        match right_iter.next() {
            Some(right_entry) if right_entry.pos == key => {
                right_entry.accum = right_entry.accum + count;
                for entry in right_iter {
                    entry.accum = entry.accum + count;
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
        let mut insert_entry = Entry::new(key, old_accum + count);

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
        // TODO: maybe try spilling into neighboring siblings first.
        let mut right_sibling = LeafNode::empty(self.parent);

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
                right_entries.split_at(insert_index - (CAP / 2 + 1));

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
        let right_sibling_ref = ChildRef::leaf(right_sibling);

        Some((ejected_entry.pos, ejected_entry.accum, right_sibling_ref))
    }

    #[cfg(test)]
    fn append_to_debug_string(
        &self,
        s: &mut String,
        ident: &str,
        expected_parent: Option<ParentRef<P, C, CAP>>,
    ) where
        P: Display,
        C: Display,
    {
        use alloc::format;

        if self.parent != expected_parent {
            s.push_str(&ident);
            s.push_str("INVALID PARENT: ");
        }

        s.push_str(&ident);
        s.push_str("  leaf:");
        for Entry { pos, accum } in self.entries.iter() {
            s.push_str(&format!(" {pos},{accum} |"));
        }
        s.push_str("\n");
    }
}

impl<P, C> Entry<P, C> {
    fn new(key: P, accum: C) -> Self {
        Self { pos: key, accum }
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
    pub union ChildRef<P, C, const CAP: usize> {
        non_leaf: ManuallyDrop<Box<NonLeafNode<P, C, CAP>>>,
        leaf: ManuallyDrop<Box<LeafNode<P, C, CAP>>>,
    }

    /// A non-owned reference to a `NonLeafNode`.
    #[derive(Debug, Clone, Copy)]
    pub struct ParentRef<P, C, const CAP: usize>(NonNull<NonLeafNode<P, C, CAP>>);

    impl<P, C, const CAP: usize> Debug for ChildRef<P, C, CAP> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("ChildRef").finish_non_exhaustive()
        }
    }

    impl<P, C, const CAP: usize> ChildRef<P, C, CAP> {
        pub fn non_leaf(child: NonLeafNode<P, C, CAP>) -> ChildRef<P, C, CAP> {
            Self {
                non_leaf: ManuallyDrop::new(Box::new(child)),
            }
        }

        pub fn leaf(child: LeafNode<P, C, CAP>) -> ChildRef<P, C, CAP> {
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
        pub unsafe fn as_non_leaf_unchecked(&self) -> &NonLeafNode<P, C, CAP> {
            &self.non_leaf
        }

        #[inline(always)]
        pub unsafe fn as_non_leaf_mut_unchecked(&mut self) -> &mut NonLeafNode<P, C, CAP> {
            &mut self.non_leaf
        }

        #[inline(always)]
        pub unsafe fn as_leaf_unchecked(&self) -> &LeafNode<P, C, CAP> {
            &self.leaf
        }

        #[inline(always)]
        pub unsafe fn as_leaf_mut_unchecked(&mut self) -> &mut LeafNode<P, C, CAP> {
            &mut self.leaf
        }

        #[inline(always)]
        pub unsafe fn downgrade_non_leaf_unchecked(&self) -> ParentRef<P, C, CAP> {
            let ptr_mut = self.as_non_leaf_unchecked() as *const NonLeafNode<_, _, CAP>
                as *mut NonLeafNode<_, _, CAP>;
            ParentRef(unsafe { NonNull::new_unchecked(ptr_mut) })
        }
    }

    impl<P, C, const CAP: usize> ParentRef<P, C, CAP> {
        #[inline(always)]
        pub unsafe fn as_mut(&mut self) -> &mut NonLeafNode<P, C, CAP> {
            self.0.as_mut()
        }

        /// This is dangerous. Prefer calling `ChildRef::downgrade_non_leaf_unchecked()` if possible
        /// as this statically ensures that the ref is at least already on the heap.
        #[cfg(test)]
        pub fn from_ref(node: &NonLeafNode<P, C, CAP>) -> Self {
            let ptr_mut = node as *const NonLeafNode<_, _, CAP> as *mut NonLeafNode<_, _, CAP>;
            ParentRef(unsafe { NonNull::new_unchecked(ptr_mut) })
        }
    }

    impl<P, C, const CAP: usize> PartialEq for ParentRef<P, C, CAP> {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }

    impl<P, C, const CAP: usize> Eq for ParentRef<P, C, CAP> {}
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
                self.buf
                    .get_unchecked_mut(index..self.len + 1)
                    .rotate_right(1);
                self.buf.get_unchecked_mut(index).write(item);
            }
            self.len += 1;

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
            self.len += 1;

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
            self.len += tail.len();

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
        dbg!(minimal_internal::<2>());
        dbg!(minimal_internal::<1>());

        fn minimal_internal<const CAP: usize>() {
            let mut tree = AugmentedBTree::<F32, u32, CAP>::new();
            assert_eq!(tree.total(), 0);

            // Insert { -2.25=>3, 1.5=>10, 2.5=>9, 3.25=>7, 4.75=>3, 5.125=>2, 6.5=>1 } in some
            // random order.
            let insertions = [
                (3.25, 7, 7),
                (5.125, 2, 9),
                (4.75, 3, 12),
                (1.5, 10, 22),
                (2.5, 9, 31),
                (6.5, 1, 32),
                (-2.25, 3, 35),
            ];
            for (pos, count, total) in insertions {
                tree.insert(F32::new(pos).unwrap(), count);
                assert_eq!(tree.total(), total);
                // Uncomment the following for debugging.
                // std::eprintln!(
                //     "after inserting {},{}:\n{}\n",
                //     pos,
                //     count,
                //     tree.to_debug_string()
                // );
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
                (7.2, 35),
            ];

            for (pos, expected_accum) in test_points {
                let pos = F32::new(pos).unwrap();
                assert_eq!(tree.get_by_pos(pos), expected_accum);
            }
        }
    }

    #[test]
    fn random_data() {
        #[cfg(not(miri))]
        let amts = [1000, 10_000];

        #[cfg(miri)]
        let amts = [1000];

        for amt in amts {
            dbg!(amt, random_data_internal::<128>(amt));
            dbg!(amt, random_data_internal::<10>(amt));
            dbg!(amt, random_data_internal::<5>(amt));
            dbg!(amt, random_data_internal::<2>(amt));
            dbg!(amt, random_data_internal::<1>(amt));
        }

        fn random_data_internal<const CAP: usize>(amt: usize) {
            let mut hasher = DefaultHasher::new();
            20231201.hash(&mut hasher);
            (CAP as u64).hash(&mut hasher);
            (amt as u64).hash(&mut hasher);

            let mut rng = Xoshiro256StarStar::seed_from_u64(hasher.finish());
            let repeat_distribution = distributions::Uniform::from(1..5);
            let count_distribution = distributions::Uniform::from(0..100);

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
                if tree.total != tree.get_by_pos(two) {
                    std::eprintln!("AFTER {i}th INSERT ({pos},{count}):");
                    std::dbg!(tree.total, tree.get_by_pos(two));
                    std::eprintln!("{}", tree.to_debug_string());
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
                tree.get_by_pos(cdf.last().unwrap().0 + F64::new(0.1).unwrap()),
                tree.total
            );

            let mut last_pos = F64::new(-1.0).unwrap();
            let mut last_accum = 0;
            let half = F64::new(0.5).unwrap();
            for &(pos, accum) in &cdf {
                let before = half * (last_pos + pos);
                assert_eq!(tree.get_by_pos(before), last_accum);
                assert_eq!(tree.get_by_pos(pos), last_accum);
                last_pos = pos;
                last_accum = accum;
            }
        }
    }
}
