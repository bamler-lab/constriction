use std::{fmt::Debug, marker::PhantomData};

/// TODO: document that, if a `Backend<Item>` implements `AsRef<[Item]>` then the
/// head of the stack must be at the end of the slice returned by `as_ref`.
pub trait Backend<Item> {}

pub trait ReadItems<Item>: Backend<Item> {
    fn pop(&mut self) -> Option<Item>;
    fn peek(&self) -> Option<&Item>;
}

pub trait ReadLookaheadItems<Item>: Backend<Item> {
    fn amt_left(&self) -> usize;

    #[inline(always)]
    fn is_at_end(&self) -> bool {
        self.amt_left() == 0
    }
}

pub trait Seek<Item>: Backend<Item> {
    fn seek(&mut self, pos: usize, must_be_end: bool) -> Result<(), ()>;
}

pub trait Pos<Item>: Backend<Item> {
    fn pos(&self) -> usize;
}

pub trait WriteItems<Item>: Backend<Item> + std::iter::Extend<Item> {
    fn push(&mut self, item: Item);
}

pub trait WriteMutableItems<Item>: WriteItems<Item> {
    fn clear(&mut self);
}

impl<Item> Backend<Item> for Vec<Item> {}

impl<Item> ReadItems<Item> for Vec<Item> {
    #[inline(always)]
    fn pop(&mut self) -> Option<Item> {
        self.pop()
    }

    #[inline(always)]
    fn peek(&self) -> Option<&Item> {
        self.last()
    }
}

impl<Item> ReadLookaheadItems<Item> for Vec<Item> {
    #[inline(always)]
    fn amt_left(&self) -> usize {
        self.len()
    }
}

impl<Item> WriteItems<Item> for Vec<Item> {
    #[inline(always)]
    fn push(&mut self, item: Item) {
        self.push(item)
    }
}

impl<Item> WriteMutableItems<Item> for Vec<Item> {
    fn clear(&mut self) {
        self.clear()
    }
}

impl<Item> Pos<Item> for Vec<Item> {
    fn pos(&self) -> usize {
        self.len()
    }
}

pub struct ReadOwnedFromBack<Item, Buf: AsRef<[Item]>> {
    buf: Buf,

    /// One plus the index of the next item to read.
    /// Satisfies invariant `pos <= buf.as_ref().len()`.
    pos: usize,

    phantom: PhantomData<Item>,
}

impl<Item, Buf: AsRef<[Item]>> ReadOwnedFromBack<Item, Buf> {
    #[inline(always)]
    pub fn new(buf: Buf) -> Self {
        let pos = buf.as_ref().len();
        Self {
            buf,
            pos,
            phantom: PhantomData,
        }
    }
}

impl<'a, Item: Clone, Buf: AsRef<[Item]>> IntoIterator for &'a ReadOwnedFromBack<Item, Buf> {
    type Item = Item;
    type IntoIter = std::iter::Cloned<std::slice::Iter<'a, Item>>;

    fn into_iter(self) -> Self::IntoIter {
        let slice = unsafe {
            // SAFETY: We maintain the invariant `self.pos <= self.buf.len()`.
            self.buf.as_ref().get_unchecked(..self.pos)
        };

        slice.iter().cloned()
    }
}

impl<Item: Clone + Debug, Buf: AsRef<[Item]>> Debug for ReadOwnedFromBack<Item, Buf> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self).finish()
    }
}

impl<Item, Buf: AsRef<[Item]>> Backend<Item> for ReadOwnedFromBack<Item, Buf> {}

impl<Item: Clone, Buf: AsRef<[Item]>> ReadItems<Item> for ReadOwnedFromBack<Item, Buf> {
    #[inline(always)]
    fn pop(&mut self) -> Option<Item> {
        if self.pos == 0 {
            None
        } else {
            self.pos -= 1;
            unsafe {
                // SAFETY: We maintain the invariant `self.pos <=self.buf.as_ref().len()` and we
                // just decreased `self.pos` (making sure it doesn't wrap around), so we now have
                // `self.pos < self.buf.as_ref().len()`.
                Some(self.buf.as_ref().get_unchecked(self.pos).clone())
            }
        }
    }

    fn peek(&self) -> Option<&Item> {
        if self.pos == 0 {
            None
        } else {
            unsafe {
                // SAFETY: We maintain the invariant `self.pos <=self.buf.as_ref().len()` and we
                // ensured that `self.pos != 0`, thus `self.pos - 1 < self.buf.as_ref().len()`.
                Some(self.buf.as_ref().get_unchecked(self.pos - 1))
            }
        }
    }
}

impl<Item: Clone, Buf: AsRef<[Item]>> ReadLookaheadItems<Item> for ReadOwnedFromBack<Item, Buf> {
    fn amt_left(&self) -> usize {
        self.pos
    }
}

impl<Item: Clone, Buf: AsRef<[Item]>> Seek<Item> for ReadOwnedFromBack<Item, Buf> {
    fn seek(&mut self, pos: usize, must_be_end: bool) -> Result<(), ()> {
        if pos > self.buf.as_ref().len() || (must_be_end && pos != 0) {
            Err(())
        } else {
            self.pos = pos;
            Ok(())
        }
    }
}

impl<Item: Clone, Buf: AsRef<[Item]>> Pos<Item> for ReadOwnedFromBack<Item, Buf> {
    fn pos(&self) -> usize {
        self.pos
    }
}

pub struct ReadOwnedFromFront<Item, Buf: AsRef<[Item]>> {
    buf: Buf,

    /// Index of the next item to be read, or `buf.as_ref().len()` if `is_at_end()`.
    pos: usize,

    phantom: PhantomData<Item>,
}

impl<Item, Buf: AsRef<[Item]>> ReadOwnedFromFront<Item, Buf> {
    #[inline(always)]
    pub fn new(buf: Buf) -> Self {
        Self {
            buf,
            pos: 0,
            phantom: PhantomData,
        }
    }
}

impl<'a, Item: Clone, Buf: AsRef<[Item]>> IntoIterator for &'a ReadOwnedFromFront<Item, Buf> {
    type Item = Item;
    type IntoIter = std::iter::Cloned<std::iter::Rev<std::slice::Iter<'a, Item>>>;

    fn into_iter(self) -> Self::IntoIter {
        let slice = unsafe {
            // SAFETY: We maintain the invariant `self.pos <= self.buf.len()`.
            self.buf.as_ref().get_unchecked(self.pos..)
        };

        slice.iter().rev().cloned()
    }
}

impl<Item: Clone + Debug, Buf: AsRef<[Item]>> Debug for ReadOwnedFromFront<Item, Buf> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self).finish()
    }
}

impl<Item, Buf: AsRef<[Item]>> Backend<Item> for ReadOwnedFromFront<Item, Buf> {}

impl<Item: Clone, Buf: AsRef<[Item]>> ReadItems<Item> for ReadOwnedFromFront<Item, Buf> {
    #[inline(always)]
    fn pop(&mut self) -> Option<Item> {
        let item = self.buf.as_ref().get(self.pos)?.clone();
        self.pos += 1;
        Some(item)
    }

    fn peek(&self) -> Option<&Item> {
        self.buf.as_ref().get(self.pos)
    }
}

impl<Item: Clone, Buf: AsRef<[Item]>> ReadLookaheadItems<Item> for ReadOwnedFromFront<Item, Buf> {
    fn amt_left(&self) -> usize {
        // This cannot underflow since we maintain the invariant `pos >= buf.as_ref().len()`.
        self.buf.as_ref().len() - self.pos
    }
}

impl<Item: Clone, Buf: AsRef<[Item]>> Seek<Item> for ReadOwnedFromFront<Item, Buf> {
    fn seek(&mut self, pos: usize, must_be_end: bool) -> Result<(), ()> {
        match (pos.cmp(&self.buf.as_ref().len()), must_be_end) {
            (std::cmp::Ordering::Less, false) | (std::cmp::Ordering::Equal, _) => {
                self.pos = pos;
                Ok(())
            }
            _ => Err(()),
        }
    }
}

impl<Item: Clone, Buf: AsRef<[Item]>> Pos<Item> for ReadOwnedFromFront<Item, Buf> {
    fn pos(&self) -> usize {
        self.pos
    }
}
