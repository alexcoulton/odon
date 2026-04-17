pub(crate) fn squeeze_to_2d<T>(
    data: ndarray::ArrayD<T>,
    vertical_dim_orig: usize,
    horizontal_dim_orig: usize,
) -> Option<ndarray::Array2<T>> {
    use ndarray::Axis;

    let mut arr = data;
    let mut vertical_dim = vertical_dim_orig;
    let mut horizontal_dim = horizontal_dim_orig;

    for dim in (0..arr.ndim()).rev() {
        if dim == vertical_dim || dim == horizontal_dim {
            continue;
        }
        if arr.shape().get(dim).copied().unwrap_or(0) != 1 {
            return None;
        }
        arr = arr.index_axis_move(Axis(dim), 0);
        if dim < vertical_dim {
            vertical_dim = vertical_dim.saturating_sub(1);
        }
        if dim < horizontal_dim {
            horizontal_dim = horizontal_dim.saturating_sub(1);
        }
    }

    if arr.ndim() != 2 {
        return None;
    }

    let mut a2 = arr.into_dimensionality::<ndarray::Ix2>().ok()?;
    match (vertical_dim, horizontal_dim) {
        (0, 1) => {}
        (1, 0) => a2.swap_axes(0, 1),
        _ => return None,
    }

    Some(a2)
}

pub(crate) fn squeeze_to_yx<T>(
    data: ndarray::ArrayD<T>,
    y_dim_orig: usize,
    x_dim_orig: usize,
) -> Option<ndarray::Array2<T>> {
    squeeze_to_2d(data, y_dim_orig, x_dim_orig)
}

#[cfg(test)]
mod tests {
    use super::{squeeze_to_2d, squeeze_to_yx};
    use ndarray::{Array, IxDyn};

    #[test]
    fn squeezes_singleton_tczyx_to_yx() {
        let data = Array::from_iter(0u16..12)
            .into_shape_with_order(IxDyn(&[1, 1, 1, 3, 4]))
            .expect("shape");
        let yx = squeeze_to_yx(data, 3, 4).expect("squeezed");
        assert_eq!(yx.shape(), &[3, 4]);
        assert_eq!(yx[(0, 0)], 0);
        assert_eq!(yx[(2, 3)], 11);
    }

    #[test]
    fn squeezes_singleton_czyx_to_zx() {
        let data = Array::from_iter(0u16..12)
            .into_shape_with_order(IxDyn(&[1, 3, 1, 4]))
            .expect("shape");
        let zx = squeeze_to_2d(data, 1, 3).expect("squeezed");
        assert_eq!(zx.shape(), &[3, 4]);
        assert_eq!(zx[(0, 0)], 0);
        assert_eq!(zx[(2, 3)], 11);
    }

    #[test]
    fn rejects_non_singleton_non_spatial_dims() {
        let data = Array::from_iter(0u16..48)
            .into_shape_with_order(IxDyn(&[2, 2, 3, 4]))
            .expect("shape");
        assert!(squeeze_to_yx(data, 2, 3).is_none());
    }
}
