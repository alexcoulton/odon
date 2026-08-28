//! Compact nullable `f32` storage shared by column loaders and consumers.

#[derive(Debug, Clone)]
pub(crate) struct NullableF32Column {
    values: Vec<f32>,
    validity: Vec<u64>,
}

impl NullableF32Column {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            validity: Vec::with_capacity(capacity.div_ceil(u64::BITS as usize)),
        }
    }

    pub(crate) fn from_optional_values(values: impl IntoIterator<Item = Option<f32>>) -> Self {
        let values = values.into_iter();
        let (lower_bound, _) = values.size_hint();
        let mut column = Self::with_capacity(lower_bound);
        for value in values {
            column.push(value);
        }
        column
    }

    pub(crate) fn push(&mut self, value: Option<f32>) {
        let index = self.values.len();
        if index % u64::BITS as usize == 0 {
            self.validity.push(0);
        }
        if let Some(value) = value {
            self.validity[index / u64::BITS as usize] |= 1u64 << (index % u64::BITS as usize);
            self.values.push(value);
        } else {
            self.values.push(0.0);
        }
    }

    pub(crate) fn get(&self, index: usize) -> Option<f32> {
        let value = *self.values.get(index)?;
        let word = self.validity.get(index / u64::BITS as usize)?;
        ((*word & (1u64 << (index % u64::BITS as usize))) != 0).then_some(value)
    }

    pub(crate) fn iter_present(&self) -> impl Iterator<Item = (usize, f32)> + '_ {
        self.values
            .iter()
            .enumerate()
            .filter_map(|(index, _)| self.get(index).map(|value| (index, value)))
    }

    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    #[cfg(test)]
    pub(crate) fn validity_word_len(&self) -> usize {
        self.validity.len()
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        self.values.capacity() * std::mem::size_of::<f32>()
            + self.validity.capacity() * std::mem::size_of::<u64>()
    }
}
