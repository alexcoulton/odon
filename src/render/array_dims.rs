pub(crate) fn squeeze_to_yx<T>(
    data: ndarray::ArrayD<T>,
    y_dim_orig: usize,
    x_dim_orig: usize,
) -> Option<ndarray::Array2<T>> {
    use ndarray::Axis;

    let mut arr = data;
    let mut y_dim = y_dim_orig;
    let mut x_dim = x_dim_orig;

    for dim in (0..arr.ndim()).rev() {
        if dim == y_dim || dim == x_dim {
            continue;
        }
        if arr.shape().get(dim).copied().unwrap_or(0) != 1 {
            return None;
        }
        arr = arr.index_axis_move(Axis(dim), 0);
        if dim < y_dim {
            y_dim = y_dim.saturating_sub(1);
        }
        if dim < x_dim {
            x_dim = x_dim.saturating_sub(1);
        }
    }

    if arr.ndim() != 2 {
        return None;
    }

    let mut a2 = arr.into_dimensionality::<ndarray::Ix2>().ok()?;
    match (y_dim, x_dim) {
        (0, 1) => {}
        (1, 0) => a2.swap_axes(0, 1),
        _ => return None,
    }

    Some(a2)
}

#[cfg(test)]
mod tests {
    use super::squeeze_to_yx;
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
    fn rejects_non_singleton_non_spatial_dims() {
        let data = Array::from_iter(0u16..48)
            .into_shape_with_order(IxDyn(&[2, 2, 3, 4]))
            .expect("shape");
        assert!(squeeze_to_yx(data, 2, 3).is_none());
    }
}
