use super::*;

impl OmeZarrViewerApp {
    pub(super) fn default_tile_loader_threads() -> usize {
        recommended_tile_loader_threads()
    }

    pub(super) fn supports_runtime_tile_loader_tuning(&self) -> bool {
        self.tiff_plane_state.is_none()
            && !self.dataset.source.local_path().is_some_and(|path| {
                classify_local_dataset_path(path) == Some(LocalDatasetKind::Tiff)
            })
    }

    pub(super) fn respawn_tile_loaders(&mut self) -> anyhow::Result<()> {
        if !self.supports_runtime_tile_loader_tuning() {
            anyhow::bail!("runtime tile loading settings are not available for this dataset");
        }
        self.loader = spawn_tile_loader(
            self.store.clone(),
            self.dataset.levels.clone(),
            self.dataset.dims.clone(),
            self.tile_loader_threads,
        )?;
        self.raw_loader = if self.tiles_gl.is_some() {
            Some(spawn_raw_tile_loader(
                self.store.clone(),
                self.dataset.levels.clone(),
                self.dataset.dims.clone(),
                self.tile_loader_threads,
            )?)
        } else {
            None
        };
        self.pending.clear();
        if let Some(tiles_gl) = self.tiles_gl.as_ref() {
            tiles_gl.reset();
        }
        self.cache = TileCache::new(256);
        self.bump_render_id();
        Ok(())
    }

    pub(super) fn try_get_raw_tile_from_pinned_finer(
        &self,
        key: RawTileKey,
        level: &crate::data::ome::LevelInfo,
    ) -> Option<crate::render::tiles_raw::RawTileResponse> {
        if !self.prefer_pinned_finer_levels {
            return None;
        }
        for source_level in 0..key.level {
            let Some(source_info) = self.dataset.levels.get(source_level) else {
                continue;
            };
            if let Some(resp) = self.pinned_levels.try_get_raw_tile_resampled_from_level(
                source_level,
                key,
                &self.dataset.dims,
                level,
                source_info,
            ) {
                return Some(resp);
            }
        }
        None
    }

    pub(super) fn try_get_composited_tile_from_pinned_finer(
        &self,
        key: TileKey,
        channels: &[RenderChannel],
        level: &crate::data::ome::LevelInfo,
    ) -> Option<TileResponse> {
        if !self.prefer_pinned_finer_levels {
            return None;
        }
        for source_level in 0..key.level {
            let Some(source_info) = self.dataset.levels.get(source_level) else {
                continue;
            };
            if let Some(tile) = self
                .pinned_levels
                .try_get_composited_tile_resampled_from_level(
                    source_level,
                    key,
                    channels,
                    &self.dataset.dims,
                    level,
                    source_info,
                )
            {
                return Some(tile);
            }
        }
        None
    }

    pub(super) fn axis_unit_to_um(unit: &str) -> Option<f32> {
        let u = unit.trim().to_ascii_lowercase();
        match u.as_str() {
            "um" | "µm" | "micrometer" | "micrometre" | "micron" | "microns" => Some(1.0),
            "nm" | "nanometer" | "nanometre" | "nanometers" | "nanometres" => Some(0.001),
            "mm" | "millimeter" | "millimetre" | "millimeters" | "millimetres" => Some(1000.0),
            "m" | "meter" | "metre" | "meters" | "metres" => Some(1_000_000.0),
            _ => None,
        }
    }

    pub(super) fn dataset_pixel_size_um(&self) -> Option<f32> {
        let ms = &self.dataset.multiscale;
        let ds0 = ms.datasets.first()?;
        let mut scale: Option<&[f32]> = None;
        for ct in &ds0.coordinate_transformations {
            if let crate::data::ome::CoordTransform::Scale { scale: s } = ct {
                scale = Some(s.as_slice());
                break;
            }
        }
        let scale = scale?;
        if scale.len() != self.dataset.dims.ndim {
            return None;
        }

        let ax_x = ms.axes.get(self.dataset.dims.x)?;
        let ax_y = ms.axes.get(self.dataset.dims.y)?;
        let fx = ax_x.unit.as_deref().and_then(Self::axis_unit_to_um)?;
        let fy = ax_y.unit.as_deref().and_then(Self::axis_unit_to_um)?;
        let sx = scale[self.dataset.dims.x] * fx;
        let sy = scale[self.dataset.dims.y] * fy;
        if !(sx.is_finite() && sy.is_finite() && sx > 0.0 && sy > 0.0) {
            return None;
        }
        Some((sx + sy) * 0.5)
    }

    pub(super) fn z_extent_level0(&self) -> Option<u64> {
        self.dataset.levels.first().and_then(|level0| {
            crate::imaging::plane_selection::level0_z_extent(&self.dataset.dims, level0)
        })
    }

    pub(super) fn view_plane_modes(&self) -> Vec<ViewPlaneMode> {
        supported_modes(&self.dataset.dims)
    }

    pub(super) fn view_plane_is_xy(&self) -> bool {
        self.view_plane_mode == ViewPlaneMode::Xy
    }

    pub(super) fn display_axes(&self) -> Option<crate::imaging::view_plane::DisplayAxes> {
        display_axes_for_mode(&self.dataset.dims, self.view_plane_mode)
    }

    pub(super) fn active_view_slice_level0_unclamped(&self) -> u64 {
        match self.view_plane_mode {
            ViewPlaneMode::Xy => self.current_z_level0,
            ViewPlaneMode::Xz => self.current_y_level0,
            ViewPlaneMode::Yz => self.current_x_level0,
        }
    }

    pub(super) fn view_slice_extent_level0(&self) -> Option<u64> {
        let level0 = self.dataset.levels.first()?;
        slice_extent_level0(&self.dataset.dims, level0, self.view_plane_mode)
    }

    pub(super) fn committed_view_selection(&self) -> ViewPlaneSelection {
        let level0 = self
            .dataset
            .levels
            .first()
            .expect("dataset should always have at least one level");
        clamp_view_selection(
            &self.dataset.dims,
            level0,
            self.view_plane_mode,
            self.active_view_slice_level0_unclamped(),
        )
    }

    pub(super) fn displayed_view_selection(&self) -> ViewPlaneSelection {
        let level0 = self
            .dataset
            .levels
            .first()
            .expect("dataset should always have at least one level");
        clamp_view_selection(
            &self.dataset.dims,
            level0,
            self.view_plane_mode,
            self.draft_view_slice_level0
                .unwrap_or(self.active_view_slice_level0_unclamped()),
        )
    }

    pub(super) fn fallback_view_selection(&self) -> Option<ViewPlaneSelection> {
        let displayed = self.displayed_view_selection();
        let committed = self.committed_view_selection();
        if displayed != committed {
            self.previous_displayed_view_selection
                .filter(|view| *view != displayed)
                .or(Some(committed))
        } else {
            self.previous_view_selection
                .filter(|view| *view != displayed)
        }
    }

    pub(super) fn active_view_selection(&self) -> ViewPlaneSelection {
        self.committed_view_selection()
    }

    pub(super) fn reset_image_view_state(&mut self, message: &str) {
        self.hist = None;
        self.hist_request_id = self.hist_request_id.wrapping_add(1);
        self.hist_request_pending = false;
        self.hist_dirty = true;
        self.hist_navigation_dirty_since = None;
        self.chanmax_request_id = self.chanmax_request_id.wrapping_add(1).max(1);
        for pending in &mut self.chanmax_pending {
            *pending = false;
        }
        self.threshold_region_preview = None;
        self.threshold_region_status.clear();
        self.pinned_levels = PinnedLevels::new();
        self.memory_status = message.to_string();
        self.bump_render_id();
    }

    pub(super) fn active_z_level0(&self) -> u64 {
        self.z_extent_level0()
            .map(|extent| self.current_z_level0.min(extent.saturating_sub(1)))
            .unwrap_or(0)
    }

    pub(super) fn set_view_plane_mode(&mut self, mode: ViewPlaneMode) {
        let Some(level0) = self.dataset.levels.first() else {
            return;
        };
        let supported = self.view_plane_modes();
        let next = if supported.contains(&mode) {
            mode
        } else {
            ViewPlaneMode::Xy
        };
        if next == self.view_plane_mode {
            return;
        }
        let previous_selection = self.committed_view_selection();
        self.view_plane_mode = next;
        let clamped = clamp_view_selection(
            &self.dataset.dims,
            level0,
            next,
            self.active_view_slice_level0_unclamped(),
        );
        self.draft_view_slice_level0 = None;
        self.previous_displayed_view_selection = None;
        self.set_active_view_slice_level0_with_previous(
            clamped.slice_level0,
            previous_selection,
            "Cleared image caches after changing the active view plane.",
        );
        self.clear_spatial_selection_drag();
        if !self.view_plane_is_xy() && self.tool_mode != ToolMode::Pan {
            self.tool_mode = ToolMode::Pan;
        }
        if !self.view_plane_is_xy() && !matches!(self.active_layer, LayerId::Channel(_)) {
            self.active_layer = LayerId::Channel(
                self.selected_channel
                    .min(self.channels.len().saturating_sub(1)),
            );
        }
    }

    pub(super) fn set_active_view_slice_level0(&mut self, slice_level0: u64) {
        let previous_selection = self.committed_view_selection();
        self.set_active_view_slice_level0_with_previous(
            slice_level0,
            previous_selection,
            "Cleared image caches after changing the active slice plane.",
        );
    }

    pub(super) fn set_active_view_slice_level0_with_previous(
        &mut self,
        slice_level0: u64,
        previous_selection: ViewPlaneSelection,
        message: &str,
    ) {
        let Some(level0) = self.dataset.levels.first() else {
            return;
        };
        let selection = clamp_view_selection(
            &self.dataset.dims,
            level0,
            self.view_plane_mode,
            slice_level0,
        );
        let changed = selection != previous_selection;
        match self.view_plane_mode {
            ViewPlaneMode::Xy => self.current_z_level0 = selection.slice_level0,
            ViewPlaneMode::Xz => self.current_y_level0 = selection.slice_level0,
            ViewPlaneMode::Yz => self.current_x_level0 = selection.slice_level0,
        }
        if changed {
            self.previous_view_selection = Some(previous_selection);
            self.reset_image_view_state(message);
        }
    }
}
