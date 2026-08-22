use std::fmt;

use serde::{Deserialize, Serialize};

pub const MAX_VIEWPORTS: usize = 2;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ViewportId(String);

impl ViewportId {
    pub fn new(value: impl Into<String>) -> Result<Self, ViewportError> {
        let value = value.into();
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(ViewportError::InvalidId);
        }
        if trimmed.len() > 128
            || !trimmed
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
        {
            return Err(ViewportError::InvalidId);
        }
        Ok(Self(trimmed.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ViewportId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViewportLayout {
    #[default]
    Single,
    Horizontal,
    Vertical,
}

impl ViewportLayout {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Horizontal => "horizontal",
            Self::Vertical => "vertical",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "single" => Some(Self::Single),
            "horizontal" | "horizontal_split" | "side_by_side" => Some(Self::Horizontal),
            "vertical" | "vertical_split" | "stacked" => Some(Self::Vertical),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViewportLinks {
    pub camera: bool,
    pub plane: bool,
    pub selection: bool,
}

impl Default for ViewportLinks {
    fn default() -> Self {
        Self {
            camera: true,
            plane: true,
            selection: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ViewportSlot<T> {
    pub id: ViewportId,
    pub title: String,
    pub state: T,
    pub navigation_revision: u64,
    pub presentation_revision: u64,
}

#[derive(Debug, Clone)]
pub struct ViewportWorkspace<T> {
    viewports: Vec<ViewportSlot<T>>,
    active: ViewportId,
    layout: ViewportLayout,
    split_ratio: f32,
    links: ViewportLinks,
    next_id: u64,
    revision: u64,
}

impl<T> ViewportWorkspace<T> {
    pub fn new(initial_state: T) -> Self {
        let id = ViewportId("viewport-1".to_string());
        Self {
            viewports: vec![ViewportSlot {
                id: id.clone(),
                title: "View 1".to_string(),
                state: initial_state,
                navigation_revision: 1,
                presentation_revision: 1,
            }],
            active: id,
            layout: ViewportLayout::Single,
            split_ratio: 0.5,
            links: ViewportLinks::default(),
            next_id: 2,
            revision: 1,
        }
    }

    pub fn restore(
        viewports: Vec<ViewportSlot<T>>,
        active: ViewportId,
        layout: ViewportLayout,
        links: ViewportLinks,
    ) -> Result<Self, ViewportError> {
        if viewports.is_empty() {
            return Err(ViewportError::SingleRequiresOne);
        }
        if viewports.len() > MAX_VIEWPORTS {
            return Err(ViewportError::LimitReached(MAX_VIEWPORTS));
        }
        let mut seen = std::collections::HashSet::new();
        for viewport in &viewports {
            if !seen.insert(viewport.id.clone()) {
                return Err(ViewportError::DuplicateId(viewport.id.clone()));
            }
            if viewport.title.trim().is_empty() {
                return Err(ViewportError::InvalidTitle);
            }
        }
        if !viewports.iter().any(|viewport| viewport.id == active) {
            return Err(ViewportError::NotFound(active));
        }
        if viewports.len() == 1 && layout != ViewportLayout::Single {
            return Err(ViewportError::SplitRequiresTwo);
        }
        if viewports.len() == 2 && layout == ViewportLayout::Single {
            return Err(ViewportError::SingleRequiresOne);
        }
        let next_id = viewports
            .iter()
            .filter_map(|viewport| {
                viewport
                    .id
                    .as_str()
                    .strip_prefix("viewport-")?
                    .parse::<u64>()
                    .ok()
            })
            .max()
            .unwrap_or(0)
            .saturating_add(1)
            .max(1);
        Ok(Self {
            viewports,
            active,
            layout,
            split_ratio: 0.5,
            links,
            next_id,
            revision: 1,
        })
    }

    pub fn restore_projection(
        viewports: Vec<ViewportSlot<T>>,
        active: ViewportId,
        layout: ViewportLayout,
        links: ViewportLinks,
        split_ratio: f32,
        revision: u64,
    ) -> Result<Self, ViewportError> {
        let mut workspace = Self::restore(viewports, active, layout, links)?;
        workspace.set_split_ratio(split_ratio)?;
        workspace.revision = revision.max(1);
        Ok(workspace)
    }

    pub fn viewports(&self) -> &[ViewportSlot<T>] {
        &self.viewports
    }

    pub fn viewports_mut(&mut self) -> &mut [ViewportSlot<T>] {
        &mut self.viewports
    }

    pub fn len(&self) -> usize {
        self.viewports.len()
    }

    pub fn active_id(&self) -> &ViewportId {
        &self.active
    }

    pub fn active(&self) -> &ViewportSlot<T> {
        self.get(&self.active)
            .expect("active viewport must always exist")
    }

    pub fn active_mut(&mut self) -> &mut ViewportSlot<T> {
        let active = self.active.clone();
        self.get_mut(&active)
            .expect("active viewport must always exist")
    }

    pub fn get(&self, id: &ViewportId) -> Option<&ViewportSlot<T>> {
        self.viewports.iter().find(|viewport| viewport.id == *id)
    }

    pub fn get_mut(&mut self, id: &ViewportId) -> Option<&mut ViewportSlot<T>> {
        self.viewports
            .iter_mut()
            .find(|viewport| viewport.id == *id)
    }

    pub fn layout(&self) -> ViewportLayout {
        self.layout
    }

    pub fn split_ratio(&self) -> f32 {
        self.split_ratio
    }

    pub fn links(&self) -> ViewportLinks {
        self.links
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn bump_navigation_revision(&mut self, id: &ViewportId) -> Result<u64, ViewportError> {
        let viewport = self
            .get_mut(id)
            .ok_or_else(|| ViewportError::NotFound(id.clone()))?;
        viewport.navigation_revision = viewport.navigation_revision.wrapping_add(1).max(1);
        Ok(viewport.navigation_revision)
    }

    pub fn bump_presentation_revision(&mut self, id: &ViewportId) -> Result<u64, ViewportError> {
        let viewport = self
            .get_mut(id)
            .ok_or_else(|| ViewportError::NotFound(id.clone()))?;
        viewport.presentation_revision = viewport.presentation_revision.wrapping_add(1).max(1);
        Ok(viewport.presentation_revision)
    }

    pub fn set_active(&mut self, id: &ViewportId) -> Result<bool, ViewportError> {
        if self.get(id).is_none() {
            return Err(ViewportError::NotFound(id.clone()));
        }
        if self.active == *id {
            return Ok(false);
        }
        self.active = id.clone();
        self.bump_revision();
        Ok(true)
    }

    pub fn set_layout(&mut self, layout: ViewportLayout) -> Result<bool, ViewportError> {
        if layout != ViewportLayout::Single && self.viewports.len() != 2 {
            return Err(ViewportError::SplitRequiresTwo);
        }
        if layout == ViewportLayout::Single && self.viewports.len() != 1 {
            return Err(ViewportError::SingleRequiresOne);
        }
        if self.layout == layout {
            return Ok(false);
        }
        self.layout = layout;
        self.bump_revision();
        Ok(true)
    }

    pub fn set_split_ratio(&mut self, ratio: f32) -> Result<bool, ViewportError> {
        if !ratio.is_finite() || !(0.1..=0.9).contains(&ratio) {
            return Err(ViewportError::InvalidSplitRatio);
        }
        if (self.split_ratio - ratio).abs() <= f32::EPSILON {
            return Ok(false);
        }
        self.split_ratio = ratio;
        self.bump_revision();
        Ok(true)
    }

    pub fn set_links(&mut self, links: ViewportLinks) -> bool {
        if self.links == links {
            return false;
        }
        self.links = links;
        self.bump_revision();
        true
    }

    pub fn swap_order(&mut self) -> bool {
        if self.viewports.len() != 2 {
            return false;
        }
        self.viewports.swap(0, 1);
        self.bump_revision();
        true
    }

    pub fn rename(&mut self, id: &ViewportId, title: String) -> Result<bool, ViewportError> {
        let title = title.trim();
        if title.is_empty() {
            return Err(ViewportError::InvalidTitle);
        }
        let viewport = self
            .get_mut(id)
            .ok_or_else(|| ViewportError::NotFound(id.clone()))?;
        if viewport.title == title {
            return Ok(false);
        }
        viewport.title = title.to_string();
        self.bump_revision();
        Ok(true)
    }

    pub fn remove(&mut self, id: &ViewportId) -> Result<ViewportSlot<T>, ViewportError> {
        if self.viewports.len() == 1 {
            return Err(ViewportError::CannotRemoveLast);
        }
        let index = self
            .viewports
            .iter()
            .position(|viewport| viewport.id == *id)
            .ok_or_else(|| ViewportError::NotFound(id.clone()))?;
        let removed = self.viewports.remove(index);
        if self.active == *id {
            self.active = self.viewports[0].id.clone();
        }
        self.layout = ViewportLayout::Single;
        self.bump_revision();
        Ok(removed)
    }

    fn allocate_id(&mut self) -> ViewportId {
        loop {
            let id = ViewportId(format!("viewport-{}", self.next_id));
            self.next_id = self.next_id.wrapping_add(1).max(1);
            if self.get(&id).is_none() {
                return id;
            }
        }
    }

    fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
    }
}

impl<T: Clone> ViewportWorkspace<T> {
    pub fn clone_viewport(
        &mut self,
        source_id: &ViewportId,
        title: Option<String>,
        layout: ViewportLayout,
    ) -> Result<ViewportId, ViewportError> {
        if self.viewports.len() >= MAX_VIEWPORTS {
            return Err(ViewportError::LimitReached(MAX_VIEWPORTS));
        }
        if layout == ViewportLayout::Single {
            return Err(ViewportError::SplitRequiresTwo);
        }
        let source_state = self
            .get(source_id)
            .ok_or_else(|| ViewportError::NotFound(source_id.clone()))?
            .state
            .clone();
        let id = self.allocate_id();
        let title = title
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| format!("View {}", self.viewports.len() + 1));
        self.viewports.push(ViewportSlot {
            id: id.clone(),
            title,
            state: source_state,
            navigation_revision: 1,
            presentation_revision: 1,
        });
        self.layout = layout;
        self.active = id.clone();
        self.bump_revision();
        Ok(id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViewportError {
    InvalidId,
    InvalidTitle,
    NotFound(ViewportId),
    DuplicateId(ViewportId),
    LimitReached(usize),
    CannotRemoveLast,
    SplitRequiresTwo,
    SingleRequiresOne,
    InvalidSplitRatio,
}

impl fmt::Display for ViewportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidId => formatter.write_str("viewport ID is invalid"),
            Self::InvalidTitle => formatter.write_str("viewport title must not be empty"),
            Self::NotFound(id) => write!(formatter, "viewport '{id}' was not found"),
            Self::DuplicateId(id) => write!(formatter, "viewport ID '{id}' is duplicated"),
            Self::LimitReached(limit) => write!(formatter, "viewport limit of {limit} reached"),
            Self::CannotRemoveLast => formatter.write_str("the final viewport cannot be removed"),
            Self::SplitRequiresTwo => {
                formatter.write_str("a split layout requires exactly two viewports")
            }
            Self::SingleRequiresOne => {
                formatter.write_str("a single layout requires exactly one viewport")
            }
            Self::InvalidSplitRatio => {
                formatter.write_str("split ratio must be finite and between 0.1 and 0.9")
            }
        }
    }
}

impl std::error::Error for ViewportError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_starts_with_one_stable_active_viewport() {
        let workspace = ViewportWorkspace::new(7usize);

        assert_eq!(workspace.len(), 1);
        assert_eq!(workspace.active_id().as_str(), "viewport-1");
        assert_eq!(workspace.active().state, 7);
        assert_eq!(workspace.layout(), ViewportLayout::Single);
        assert_eq!(workspace.split_ratio(), 0.5);
        assert_eq!(workspace.revision(), 1);
    }

    #[test]
    fn cloning_creates_independent_state_and_split_layout() {
        let mut workspace = ViewportWorkspace::new(vec![1usize]);
        let first = workspace.active_id().clone();

        let second = workspace
            .clone_viewport(
                &first,
                Some("Property B".to_string()),
                ViewportLayout::Horizontal,
            )
            .unwrap();
        workspace.get_mut(&second).unwrap().state.push(2);

        assert_eq!(workspace.get(&first).unwrap().state, vec![1]);
        assert_eq!(workspace.get(&second).unwrap().state, vec![1, 2]);
        assert_eq!(workspace.active_id(), &second);
        assert_eq!(workspace.layout(), ViewportLayout::Horizontal);
    }

    #[test]
    fn workspace_enforces_limit_and_preserves_last_viewport() {
        let mut workspace = ViewportWorkspace::new(1usize);
        let first = workspace.active_id().clone();
        let second = workspace
            .clone_viewport(&first, None, ViewportLayout::Vertical)
            .unwrap();

        assert_eq!(
            workspace
                .clone_viewport(&second, None, ViewportLayout::Horizontal)
                .unwrap_err(),
            ViewportError::LimitReached(2)
        );
        workspace.remove(&second).unwrap();
        assert_eq!(workspace.layout(), ViewportLayout::Single);
        assert_eq!(
            workspace.remove(&first).unwrap_err(),
            ViewportError::CannotRemoveLast
        );
    }

    #[test]
    fn setting_active_layout_links_and_title_updates_revision_only_on_change() {
        let mut workspace = ViewportWorkspace::new(1usize);
        let first = workspace.active_id().clone();
        let second = workspace
            .clone_viewport(&first, None, ViewportLayout::Horizontal)
            .unwrap();
        let revision = workspace.revision();

        assert!(workspace.set_active(&first).unwrap());
        assert!(!workspace.set_active(&first).unwrap());
        assert!(workspace.rename(&second, "Comparison".to_string()).unwrap());
        assert!(!workspace.rename(&second, "Comparison".to_string()).unwrap());
        assert!(workspace.set_layout(ViewportLayout::Vertical).unwrap());
        assert!(!workspace.set_layout(ViewportLayout::Vertical).unwrap());
        assert!(workspace.set_links(ViewportLinks {
            camera: false,
            ..ViewportLinks::default()
        }));
        assert!(workspace.swap_order());
        assert!(workspace.set_split_ratio(0.7).unwrap());
        assert!(!workspace.set_split_ratio(0.7).unwrap());
        assert_eq!(
            workspace.set_split_ratio(0.05).unwrap_err(),
            ViewportError::InvalidSplitRatio
        );
        assert_eq!(workspace.viewports()[0].id, second);
        assert!(workspace.revision() > revision);
    }

    #[test]
    fn ids_and_layout_aliases_are_validated() {
        assert!(ViewportId::new("left-view").is_ok());
        assert!(ViewportId::new("bad view").is_err());
        assert_eq!(
            ViewportLayout::parse("side_by_side"),
            Some(ViewportLayout::Horizontal)
        );
        assert_eq!(
            ViewportLayout::parse("stacked"),
            Some(ViewportLayout::Vertical)
        );
        assert_eq!(ViewportLayout::parse("grid"), None);
    }

    #[test]
    fn restore_validates_ids_layout_and_continues_stable_id_allocation() {
        let first = ViewportId::new("viewport-4").unwrap();
        let second = ViewportId::new("custom-right").unwrap();
        let slots = vec![
            ViewportSlot {
                id: first.clone(),
                title: "Left".to_string(),
                state: 10usize,
                navigation_revision: 7,
                presentation_revision: 8,
            },
            ViewportSlot {
                id: second.clone(),
                title: "Right".to_string(),
                state: 20usize,
                navigation_revision: 9,
                presentation_revision: 10,
            },
        ];
        let mut workspace = ViewportWorkspace::restore(
            slots,
            second.clone(),
            ViewportLayout::Horizontal,
            ViewportLinks::default(),
        )
        .unwrap();
        workspace.remove(&first).unwrap();
        let created = workspace
            .clone_viewport(&second, None, ViewportLayout::Vertical)
            .unwrap();
        assert_eq!(created.as_str(), "viewport-5");

        let duplicate = vec![
            ViewportSlot {
                id: second.clone(),
                title: "One".to_string(),
                state: 1usize,
                navigation_revision: 1,
                presentation_revision: 1,
            },
            ViewportSlot {
                id: second.clone(),
                title: "Two".to_string(),
                state: 2usize,
                navigation_revision: 1,
                presentation_revision: 1,
            },
        ];
        assert_eq!(
            ViewportWorkspace::restore(
                duplicate,
                second,
                ViewportLayout::Horizontal,
                ViewportLinks::default(),
            )
            .unwrap_err(),
            ViewportError::DuplicateId(ViewportId::new("custom-right").unwrap())
        );
    }
}
