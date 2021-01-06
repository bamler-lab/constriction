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

pub trait Direction: 'static {
    const FORWARD: bool;
    type Reverse: Direction;
}

#[derive(Debug, Clone)]
pub struct Forward;

#[derive(Debug, Clone)]
pub struct Backward;

impl Direction for Forward {
    const FORWARD: bool = true;
    type Reverse = Backward;
}

impl Direction for Backward {
    const FORWARD: bool = false;
    type Reverse = Forward;
}

#[derive(Clone)]
pub struct ReadCursor<Item, Buf: AsRef<[Item]>, Dir: Direction> {
    buf: Buf,

    /// If `Dir::FORWARD`: the index of the next item to be read.
    /// else: one plus the index of the next item to read.
    ///
    /// In both cases: satisfies invariant `pos <= buf.as_ref().len()`.
    pos: usize,

    phantom: PhantomData<(Item, Dir)>,
}

pub type ReadCursorForward<Item, Buf> = ReadCursor<Item, Buf, Forward>;
pub type ReadCursorBackward<Item, Buf> = ReadCursor<Item, Buf, Backward>;

impl<Item, Buf: AsRef<[Item]>, Dir: Direction> ReadCursor<Item, Buf, Dir> {
    #[inline(always)]
    pub fn new(buf: Buf) -> Self {
        let pos = if Dir::FORWARD { 0 } else { buf.as_ref().len() };
        Self {
            buf,
            pos,
            phantom: PhantomData,
        }
    }

    pub fn as_view(&self) -> ReadCursor<Item, &[Item], Dir> {
        ReadCursor {
            buf: self.buf.as_ref(),
            pos: self.pos,
            phantom: PhantomData,
        }
    }

    pub fn to_owned(&self) -> ReadCursor<Item, Vec<Item>, Dir>
    where
        Item: Clone,
    {
        ReadCursor {
            buf: self.buf.as_ref().to_vec(),
            pos: self.pos,
            phantom: PhantomData,
        }
    }

    pub fn buf(&self) -> &[Item] {
        self.buf.as_ref()
    }

    pub fn into_buf_and_pos(self) -> (Buf, usize) {
        (self.buf, self.pos)
    }
}

impl<Item, Dir: Direction> ReadCursor<Item, Vec<Item>, Dir> {
    pub fn into_reversed(self) -> ReadCursor<Item, Vec<Item>, Dir::Reverse> {
        let ReadCursor {
            mut buf, mut pos, ..
        } = self;

        buf.reverse();
        pos = buf.len() - pos;
        ReadCursor {
            buf,
            pos,
            phantom: PhantomData,
        }
    }
}

impl<'a, Item: Clone, Buf: AsRef<[Item]>, Dir: Direction> IntoIterator
    for &'a ReadCursor<Item, Buf, Dir>
{
    type Item = Item;
    type IntoIter = std::iter::Cloned<std::slice::Iter<'a, Item>>;

    fn into_iter(self) -> Self::IntoIter {
        let slice = unsafe {
            // SAFETY: We maintain the invariant `self.pos <= self.buf.len()`.
            if Dir::FORWARD {
                self.buf.as_ref().get_unchecked(self.pos..)
            } else {
                self.buf.as_ref().get_unchecked(..self.pos)
            }
        };

        slice.iter().cloned()
    }
}

impl<Item: Clone + Debug, Buf: AsRef<[Item]>, Dir: Direction> Debug for ReadCursor<Item, Buf, Dir> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self).finish()
    }
}

impl<Item, Buf: AsRef<[Item]>, Dir: Direction> Backend<Item> for ReadCursor<Item, Buf, Dir> {}

impl<Item: Clone, Buf: AsRef<[Item]>, Dir: Direction> ReadItems<Item>
    for ReadCursor<Item, Buf, Dir>
{
    #[inline(always)]
    fn pop(&mut self) -> Option<Item> {
        if Dir::FORWARD {
            let item = self.buf.as_ref().get(self.pos)?.clone();
            self.pos += 1;
            Some(item)
        } else {
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
    }

    fn peek(&self) -> Option<&Item> {
        if Dir::FORWARD {
            self.buf.as_ref().get(self.pos)
        } else {
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
}

impl<Item: Clone, Buf: AsRef<[Item]>, Dir: Direction> ReadLookaheadItems<Item>
    for ReadCursor<Item, Buf, Dir>
{
    fn amt_left(&self) -> usize {
        if Dir::FORWARD {
            // This cannot underflow since we maintain the invariant `pos >= buf.as_ref().len()`.
            self.buf.as_ref().len() - self.pos
        } else {
            self.pos
        }
    }
}

impl<Item: Clone, Buf: AsRef<[Item]>, Dir: Direction> Seek<Item> for ReadCursor<Item, Buf, Dir> {
    fn seek(&mut self, pos: usize, must_be_end: bool) -> Result<(), ()> {
        let end_pos = if Dir::FORWARD {
            self.buf.as_ref().len()
        } else {
            0
        };

        if pos > self.buf.as_ref().len() || (must_be_end && pos != end_pos) {
            Err(())
        } else {
            self.pos = pos;
            Ok(())
        }
    }
}

impl<Item: Clone, Buf: AsRef<[Item]>, Dir: Direction> Pos<Item> for ReadCursor<Item, Buf, Dir> {
    fn pos(&self) -> usize {
        self.pos
    }
}
