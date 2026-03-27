use std::collections::HashSet;

#[derive(Debug, Clone)]
pub enum GroupLayersTarget {
    Channels(Vec<usize>),
    Annotations(Vec<u64>),
}

#[derive(Debug, Clone)]
pub struct GroupLayersDialog {
    pub target: GroupLayersTarget,
    pub name: String,
    pub default_name: String,
    pub focus_name_on_open: bool,
}

impl GroupLayersDialog {
    pub fn new(target: GroupLayersTarget, default_name: String) -> Self {
        Self {
            target,
            name: default_name.clone(),
            default_name,
            focus_name_on_open: true,
        }
    }

    pub fn resolved_name(&self) -> String {
        let s = self.name.trim();
        if s.is_empty() {
            return self.default_name.clone();
        }
        s.to_string()
    }
}

pub fn default_group_name(existing_names: impl IntoIterator<Item = String>) -> String {
    let existing = existing_names.into_iter().collect::<HashSet<_>>();
    for i in 1u32.. {
        let name = format!("Group {i}");
        if !existing.contains(&name) {
            return name;
        }
    }
    "Group".to_string()
}
